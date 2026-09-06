"""Codex-equivalent process boundary plus Bun's own mapped-file inventory.

Only this process's maps file is granted, not /proc, environ, mem, fd or other
processes. Bun's embedded executable loader requires this inventory at startup.
All file/network/IPC restrictions otherwise use the unchanged shared boundary.
"""
import importlib.util
import json
import os
import pathlib
import resource
import sys

spec = importlib.util.spec_from_file_location("chia_native_boundary", pathlib.Path(__file__).parents[1] / "codex_cli/boundary.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def main():
    policy = json.loads(pathlib.Path(sys.argv[1]).read_text())
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (180,) * 2)
    os.chdir(policy["cwd"])
    native.restrict([*policy["read"], f"/proc/{os.getpid()}/maps"], policy["write"], policy["port"])
    command = sys.argv[3:] if sys.argv[2] == "--" else sys.argv[2:]
    os.execve(command[0], command, policy["environment"])


if __name__ == "__main__":
    main()
