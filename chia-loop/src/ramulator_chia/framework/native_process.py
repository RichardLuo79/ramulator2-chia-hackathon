"""Install the existing whole-process boundary before executing a native CLI."""

import json
import os
import resource
import sys

from ramulator_chia.codex_cli.boundary import restrict


def main():
    policy = json.loads(open(sys.argv[1]).read())
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    reads = policy["read"]
    if policy["kind"] == "claude_cli":
        reads += [f"/proc/{os.getpid()}/maps", f"/proc/{os.getpid()}/stat"]
    os.chdir(policy["cwd"])
    restrict(reads, policy["write"], policy["ports"])
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    command = sys.argv[2:]
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
