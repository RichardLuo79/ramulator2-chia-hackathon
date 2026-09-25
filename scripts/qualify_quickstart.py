#!/usr/bin/env python3
"""Qualify a copied checkout while the original checkout is unreadable.

Run inside the copy with its own campaign environment. Network access remains
available for the explicit input-download action and localhost test fixtures.
Simulation/agent isolation is tested separately by the retained harness tests.
"""

import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys

from artifact import ROOT


def restrict_filesystem(read, write):
    """Landlock the trusted qualification driver, not a campaign agent."""
    from ramulator_chia.sandbox import READ, WRITE, HANDLED

    libc = ct.CDLL(None, use_errno=True)
    if libc.syscall(444, 0, 0, 1) < 3:
        raise RuntimeError("Landlock ABI 3 required for filesystem qualification")
    attr = ct.c_uint64(HANDLED)
    fd = libc.syscall(444, ct.byref(attr), ct.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ct.get_errno(), "create qualification ruleset")

    class Beneath(ct.Structure):
        _pack_ = 1
        _fields_ = [("access", ct.c_uint64), ("parent", ct.c_int)]

    try:
        for path, mask in [(p, READ) for p in read] + [(p, WRITE) for p in write]:
            path = Path(path).resolve(strict=True)
            if not path.is_dir():
                mask &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            handle = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Beneath(mask, handle)
                if libc.syscall(445, fd, 1, ct.byref(rule), 0):
                    raise OSError(ct.get_errno(), "add qualification path")
            finally:
                os.close(handle)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
            raise OSError(ct.get_errno(), "restrict qualification process")
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "inputs", "tests", "check"])
    parser.add_argument("--deny-path", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("build workers must be 1–16")
    for path in args.deny_path:
        if not path.is_file() or path.is_relative_to(ROOT):
            parser.error("each denied probe must be an existing file outside this copy")
    temporary = ROOT / ".work/qualification-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        TMPDIR=str(temporary),
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(temporary / "matplotlib"),
    )
    read = [ROOT, "/usr", "/etc", "/proc", "/dev"]
    read.extend(p for p in ("/lib", "/lib64", "/bin") if Path(p).exists())
    restrict_filesystem(read, [ROOT, "/dev/null"])
    for path in args.deny_path:
        try:
            path.read_bytes()
        except PermissionError:
            pass
        else:
            raise RuntimeError("original workspace is still accessible")
    print(
        json.dumps(
            {"original_files_denied": len(args.deny_path), "action": args.action}
        ),
        flush=True,
    )

    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)

    if args.action == "build":
        run("scripts/setup", "--component", "runtime", "--workers", args.workers)
        for cores in (1, 4, 8):
            run(
                "scripts/setup",
                "--component",
                "champsim",
                "--cores",
                cores,
                "--workers",
                args.workers,
            )
    elif args.action == "inputs":
        run(
            "scripts/fetch-data",
            "--download",
            "--config",
            "chia-loop/configs/astra.json",
        )
        run("scripts/fetch-data")
    elif args.action == "tests":
        os.environ["CHIA_NATIVE_TEST_RUNTIME"] = str(ROOT / ".work/runtime")
        os.environ["BASELINE_REUSE_RUNTIME"] = str(ROOT / ".work/runtime")
        run("-m", "pytest", "chia-loop/tests", "-q", "--basetemp", temporary / "pytest")
    else:
        from ramulator_chia.preflight import isolation_check, runtime_check

        print(
            json.dumps(
                {
                    "isolation": isolation_check(),
                    "runtime_manifest_sha256": runtime_check(ROOT / ".work/runtime"),
                }
            )
        )


if __name__ == "__main__":
    main()
