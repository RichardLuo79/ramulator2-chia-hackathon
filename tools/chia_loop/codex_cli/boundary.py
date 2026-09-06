"""Whole-process Landlock/seccomp boundary, installed BEFORE starting Codex.

This is not Codex's read-only sandbox (which normally allows broad reads).
No host home, repository, /proc, Unix socket, or inherited credential is granted.
Only TCP connections to the per-invocation broker's port are permitted. The
broker accepts one authenticated Responses request, not arbitrary HTTP tunnels.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import errno
import json
import os
import pathlib
import resource
import socket

READ = (1 << 0) | (1 << 2) | (1 << 3)
WRITE = sum(1 << b for b in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))


def restrict(read, write, port):
    libc = ct.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 6:
        raise RuntimeError("Codex boundary requires Landlock ABI >= 6; no permissive fallback")

    class Ruleset(ct.Structure):
        _fields_ = [("fs", ct.c_uint64), ("net", ct.c_uint64), ("scoped", ct.c_uint64)]
    class Path(ct.Structure):
        _pack_ = 1
        _fields_ = [("access", ct.c_uint64), ("parent", ct.c_int)]
    class Net(ct.Structure):
        _fields_ = [("access", ct.c_uint64), ("port", ct.c_uint64)]

    ruleset = Ruleset(READ | WRITE, 3, 3)  # Handle TCP bind/connect; scope IPC/signals.
    fd = libc.syscall(444, ct.byref(ruleset), ct.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ct.get_errno(), "landlock_create_ruleset")
    try:
        for name, access in [(p, READ) for p in read] + [(p, WRITE) for p in write]:
            path = pathlib.Path(name).resolve(strict=True)
            if not path.is_dir():
                access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            handle = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Path(access, handle)
                if libc.syscall(445, fd, 1, ct.byref(rule), 0):
                    raise OSError(ct.get_errno(), "landlock_add_path")
            finally:
                os.close(handle)
        if not 1 <= port <= 65535:
            raise ValueError("invalid broker port")
        rule = Net(2, port)  # Connect only; do not permit binding/listening.
        if libc.syscall(445, fd, 2, ct.byref(rule), 0):
            raise OSError(ct.get_errno(), "landlock_add_network")
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
            raise OSError(ct.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)

    sec = ct.CDLL("libseccomp.so.2")
    sec.seccomp_init.restype = ct.c_void_p
    sec.seccomp_rule_add.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_int, ct.c_uint]
    sec.seccomp_load.argtypes = [ct.c_void_p]
    sec.seccomp_release.argtypes = [ct.c_void_p]
    sec.seccomp_syscall_resolve_name.argtypes = [ct.c_char_p]
    class Arg(ct.Structure):
        _fields_ = [("arg", ct.c_uint), ("op", ct.c_int), ("a", ct.c_uint64), ("b", ct.c_uint64)]
    ctx = sec.seccomp_init(0x7FFF0000)
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    denied = 0x50000 | errno.EPERM
    try:
        for name in ("ptrace", "process_vm_readv", "process_vm_writev", "mount", "umount2",
                     "pivot_root", "chroot", "unshare", "setns", "open_by_handle_at",
                     "io_uring_setup", "bpf", "userfaultfd", "perf_event_open", "keyctl",
                     "add_key", "request_key", "bind", "listen", "accept", "accept4"):
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(ctx, denied, number, 0):
                raise RuntimeError("seccomp rule failed: " + name)
        number = sec.seccomp_syscall_resolve_name(b"socket")
        # No Unix sockets (including the host app server), IPv6, or UDP/DNS.
        for arg in (Arg(0, 1, socket.AF_INET, 0), Arg(1, 7, 0xF, socket.SOCK_DGRAM)):
            if sec.seccomp_rule_add(ctx, denied, number, 1, arg):
                raise RuntimeError("seccomp socket filter failed")
        if sec.seccomp_load(ctx):
            raise RuntimeError("seccomp_load failed")
    finally:
        sec.seccomp_release(ctx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=pathlib.Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (180,) * 2)
    os.chdir(policy["cwd"])
    restrict(policy["read"], policy["write"], policy["port"])
    command = args.command[1:] if args.command[0] == "--" else args.command
    os.execve(command[0], command, policy["environment"])


if __name__ == "__main__":
    main()
