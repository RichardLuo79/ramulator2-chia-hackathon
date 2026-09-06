"""Restricted external-frontend launcher, with no inherited login/environment.

Unlike SimpleO3's in-memory driver, external simulators need their workload
files at runtime. Those specific inputs are readable; other runs, repository
history, conversation logs, credentials, and network are not. Model I/O is
additionally forbidden by the source/compliance contract.
"""
import json
import os
import pathlib
import resource
import sys

from tools.chia_loop.sandbox import restrict


def main():
    policy = json.loads(pathlib.Path(sys.argv[1]).read_text())
    for which, value in ((resource.RLIMIT_CPU, policy["cpu_seconds"]),
                         (resource.RLIMIT_AS, 8 * 1024**3),
                         (resource.RLIMIT_FSIZE, 8 * 1024**3),
                         (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(which, (value, value))
    os.chdir(policy["cwd"])
    restrict(policy["read"], policy["write"])
    os.execve(policy["command"][0], policy["command"], policy["environment"])


if __name__ == "__main__":
    main()
