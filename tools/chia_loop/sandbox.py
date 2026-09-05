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

READ = (1 << 0) | (1 << 2) | (1 << 3)
WRITE = sum(1 << bit for bit in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
HANDLED = READ | WRITE


def restrict(read: list[str], write: list[str], *, execute=True):
    libc = ct.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError(f"Landlock ABI >= 3 required; got {abi}")
    attr = ct.c_uint64(HANDLED)
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

    sec = ct.CDLL("libseccomp.so.2")
    sec.seccomp_init.restype = ct.c_void_p
    sec.seccomp_rule_add.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_int, ct.c_uint]
    sec.seccomp_load.argtypes = [ct.c_void_p]
    sec.seccomp_release.argtypes = [ct.c_void_p]
    sec.seccomp_syscall_resolve_name.argtypes = [ct.c_char_p]
    ctx = sec.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    blocked = ["socket", "connect", "bind", "listen", "accept", "accept4",
               "ptrace", "process_vm_readv", "process_vm_writev", "mount",
               "umount2", "pivot_root", "chroot", "unshare", "setns",
               "open_by_handle_at", "io_uring_setup", "bpf", "userfaultfd",
               "perf_event_open", "keyctl", "add_key", "request_key"]
    if not execute:
        blocked += ["execve", "execveat", "fork", "vfork", "clone", "clone3"]
    try:
        for name in blocked:
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0):
                raise RuntimeError(f"seccomp rule failed: {name}")
        if sec.seccomp_load(ctx):
            raise RuntimeError("seccomp_load failed")
    finally:
        sec.seccomp_release(ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", type=pathlib.Path)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    policy = json.loads(args.policy.read_text())
    resource.setrlimit(resource.RLIMIT_CPU, (policy["cpu_seconds"],) * 2)
    resource.setrlimit(resource.RLIMIT_AS, (policy["memory_bytes"],) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (policy.get("file_bytes", 256 * 1024 * 1024),) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.chdir(policy["cwd"])
    restrict(policy["read"], policy["write"])
    command = args.command[1:] if args.command[0] == "--" else args.command
    os.execve(command[0], command, {
        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
        "TMPDIR": policy["cwd"],
        "LD_LIBRARY_PATH": policy.get("library_path", ""),
    })


if __name__ == "__main__":
    main()
