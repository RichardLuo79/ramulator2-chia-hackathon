"""Linux Landlock + seccomp launch boundary; fail closed if unavailable.

The launcher's policy is operator-owned, never supplied by an LLM. Compilation
sees exported headers, system tools, and one writable build directory. Runtime
gets a second, tighter Landlock domain before loading candidate code.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import errno
import json
import os
import pathlib
import resource
import signal
import subprocess
import sys

READ = (1 << 0) | (1 << 2) | (1 << 3)
WRITE = sum(1 << bit for bit in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
HANDLED = READ | WRITE


def restrict(read: list[str], write: list[str], *, execute=True, isolate_processes=False):
    libc = ct.CDLL(None, use_errno=True)
    # Boundary machinery belongs to this trusted launcher. Load it before
    # restricting the child view; a replay policy need not expose host libraries
    # merely so the launcher can finish installing its syscall filter.
    sec = ct.CDLL("libseccomp.so.2")
    abi = libc.syscall(444, 0, 0, 1)
    minimum_abi = 6 if isolate_processes else 3
    if abi < minimum_abi:
        raise RuntimeError(f"Landlock ABI >= {minimum_abi} required; got {abi}")

    class Ruleset(ct.Structure):
        _fields_ = [("fs", ct.c_uint64), ("net", ct.c_uint64), ("scoped", ct.c_uint64)]

    # The scoped variant reuses the signal/abstract-IPC isolation of the old
    # native CLI boundary. Network access remains entirely disabled here; the
    # legacy TCP-port filter is not a destination-address allowlist.
    attr = Ruleset(HANDLED, 0, 3) if isolate_processes else ct.c_uint64(HANDLED)
    fd = libc.syscall(444, ct.byref(attr), ct.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ct.get_errno(), "landlock_create_ruleset")

    class Beneath(ct.Structure):
        _pack_ = 1
        _fields_ = [("access", ct.c_uint64), ("parent", ct.c_int)]

    try:
        for path, mask in [(p, READ) for p in read] + [(p, WRITE) for p in write]:
            resolved = pathlib.Path(path).resolve(strict=True)
            if not resolved.is_dir():
                mask &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            handle = os.open(resolved, os.O_PATH | os.O_CLOEXEC)
            rule = Beneath(mask, handle)
            try:
                if libc.syscall(445, fd, 1, ct.byref(rule), 0) != 0:
                    raise OSError(ct.get_errno(), f"landlock_add_rule: {path}")
            finally:
                os.close(handle)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, fd, 0) != 0:
            raise OSError(ct.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)

    sec.seccomp_init.restype = ct.c_void_p
    sec.seccomp_rule_add.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_int, ct.c_uint]
    sec.seccomp_load.argtypes = [ct.c_void_p]
    sec.seccomp_release.argtypes = [ct.c_void_p]
    sec.seccomp_syscall_resolve_name.argtypes = [ct.c_char_p]
    ctx = sec.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    blocked = ["socket", "connect", "bind", "listen", "accept", "accept4",
               "ptrace", "process_vm_readv", "process_vm_writev", "mount",
               "umount2", "pivot_root", "chroot", "unshare", "setns",
               "open_by_handle_at", "io_uring_setup", "bpf", "userfaultfd",
               "perf_event_open", "keyctl", "add_key", "request_key"]
    if not execute:
        blocked += ["execve", "execveat", "fork", "vfork", "clone", "clone3"]
    if isolate_processes:
        # Descendants must remain in the launcher's tracked process group.
        blocked += ["setsid", "setpgid", "pidfd_getfd"]
    try:
        for name in blocked:
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0):
                raise RuntimeError(f"seccomp rule failed: {name}")
        if isolate_processes and execute:
            # clone3 hides namespace flags behind a pointer; ENOSYS allows libc
            # to use ordinary clone for threads/processes. Deny only namespace
            # creation on clone, not normal tools and compiler subprocesses.
            number = sec.seccomp_syscall_resolve_name(b"clone3")
            if number >= 0 and sec.seccomp_rule_add(ctx, 0x50000 | errno.ENOSYS, number, 0):
                raise RuntimeError("seccomp rule failed: clone3")

            class Arg(ct.Structure):
                _fields_ = [("arg", ct.c_uint), ("op", ct.c_int),
                            ("a", ct.c_uint64), ("b", ct.c_uint64)]

            number = sec.seccomp_syscall_resolve_name(b"clone")
            for flag in (0x00020000, 0x02000000, 0x04000000, 0x08000000,
                         0x10000000, 0x20000000, 0x40000000):
                condition = Arg(0, 7, flag, flag)  # SCMP_CMP_MASKED_EQ
                if number >= 0 and sec.seccomp_rule_add(ctx, 0x50000 | errno.EPERM,
                                                       number, 1, condition):
                    raise RuntimeError("seccomp namespace rule failed")
        if sec.seccomp_load(ctx):
            raise RuntimeError("seccomp_load failed")
    finally:
        sec.seccomp_release(ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lease-fd", type=int)
    ap.add_argument("--agent-process", action="store_true")
    ap.add_argument("policy", type=pathlib.Path)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.lease_fd is not None:
        # Keep the recovery lease in this trusted launcher, not in generated
        # code. Landlock cannot revoke access through an inherited open fd.
        # The ordinary child path below still applies all existing restrictions.
        if args.lease_fd < 3:
            raise ValueError("a recovery lease must not be a standard stream")
        os.fstat(args.lease_fd)
        parent = os.getpid()
        libc = ct.CDLL(None, use_errno=True)

        def parent_death_signal():
            # This launcher is single-threaded. Close the fork/prctl race too:
            # a child cannot outlive its lease owner if that owner is killed.
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
                os._exit(127)

        command = [sys.executable, "-I", str(pathlib.Path(__file__).absolute())]
        if args.agent_process:
            command.append("--agent-process")
        command += [str(args.policy), *args.command]
        with subprocess.Popen(command, close_fds=True, preexec_fn=parent_death_signal,
                              env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}) as child:
            return child.wait()
    policy = json.loads(args.policy.read_text())
    if args.agent_process:
        if policy["network"] != "disabled":
            raise ValueError("agent process boundary has no qualified online network profile")
        for key, limit in (("cpu_seconds", resource.RLIMIT_CPU),
                           ("memory_bytes", resource.RLIMIT_AS),
                           ("file_bytes", resource.RLIMIT_FSIZE)):
            value = policy[key]
            if value is not None:
                resource.setrlimit(limit, (value,) * 2)
    else:
        resource.setrlimit(resource.RLIMIT_CPU, (policy["cpu_seconds"],) * 2)
        resource.setrlimit(resource.RLIMIT_AS, (policy["memory_bytes"],) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (policy.get("file_bytes", 256 * 1024 * 1024),) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.chdir(policy["cwd"])
    restrict(policy["read"], policy["write"], isolate_processes=args.agent_process)
    command = args.command[1:] if args.command[0] == "--" else args.command
    environment = policy.get("environment") if args.agent_process else policy.get("environment", {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
        "TMPDIR": policy["cwd"],
        "LD_LIBRARY_PATH": policy.get("library_path", ""),
    })
    if not isinstance(environment, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()):
        raise ValueError("the trusted launch policy must provide a string environment")
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    raise SystemExit(main())
