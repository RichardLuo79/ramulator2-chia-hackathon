"""Test-only native CLI transport boundary. Not an online campaign launcher.

Reuse the previously qualified Landlock/seccomp implementation, permitting one
local fixture port. A port rule is NOT an IP/destination allowlist. This fixture
has no real provider credential, and does not certify production isolation.
"""

import importlib.util
import json
import os
import resource
import sys
from pathlib import Path


def main():
    policy = json.loads(Path(sys.argv[1]).read_text())
    if os.environ.get("CHIA_INSTALLED_CLI_FIXTURE") != "1":
        raise RuntimeError("offline transport fixture was not explicitly selected")
    boundary = Path(__file__).resolve().parents[2] / "tools/chia_loop/codex_cli/boundary.py"
    spec = importlib.util.spec_from_file_location("native_transport_fixture", boundary)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    environment = dict(os.environ)
    # Never look for an operator profile, even when a native library consults HOME.
    assert environment["HOME"] == policy["private"]
    assert all("OAUTH" not in name for name in environment)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.chdir(policy["cwd"])
    read = list(policy["read"])
    if policy["backend"] == "claude_cli":
        read.append(f"/proc/{os.getpid()}/maps")
        read.append(f"/proc/{os.getpid()}/stat")
    native.restrict(read, policy["write"], policy["port"])
    command = sys.argv[2:]
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
