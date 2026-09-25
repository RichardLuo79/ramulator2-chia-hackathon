"""Native build contract checks; no provider SDK or model calls are needed."""

import json
from pathlib import Path

import pytest

from ramulator_chia.framework import build


@pytest.mark.parametrize("existing", [False, True])
def test_frontend_preparation_reuses_only_a_verified_capture(tmp_path, monkeypatch, existing):
    from types import SimpleNamespace

    from ramulator_chia.framework import inputs

    calls = []

    def verify():
        calls.append("verify")
        return {"frontend": "gem5"}

    host = SimpleNamespace(record=verify)
    monkeypatch.setattr(inputs, "ExternalHost", lambda root, sha: host)

    def prepare(*args):
        calls.append("prepare")
        return host

    monkeypatch.setattr(inputs, "prepare_host", prepare)
    if existing:
        (tmp_path / "host.json").write_text("{}")
    assert inputs._host(tmp_path, "gem5", None, None) is host
    assert calls == ["verify" if existing else "prepare"]
    if existing:
        with pytest.raises(ValueError, match="requested frontend"):
            inputs._host(tmp_path, "champsim", None, None)

        def corrupt():
            raise ValueError("frontend runtime input changed")

        host.record = corrupt
        with pytest.raises(ValueError, match="runtime input changed"):
            inputs._host(tmp_path, "gem5", None, None)


def test_build_command_forwards_the_declared_source_and_cpu_settings(tmp_path, monkeypatch, capsys):
    calls = []

    def prepare(repo, destination, **kwargs):
        calls.append((repo, destination, kwargs))
        return {"translation_units": 130}

    monkeypatch.setattr(build, "prepare_runtime", prepare)
    build.main(
        [
            "--repo",
            str(tmp_path),
            "--output",
            str(tmp_path / "runtime"),
            "--workers",
            "3",
            "--timeout-seconds",
            "600",
        ]
    )
    assert calls == [
        (tmp_path, tmp_path / "runtime", {"cpus": 3, "timeout_seconds": 600, "model_api": True})
    ]
    assert json.loads(capsys.readouterr().out)["translation_units"] == 130


@pytest.mark.parametrize("workers", ["0", "13"])
def test_build_command_rejects_an_invalid_cpu_count(tmp_path, monkeypatch, workers):
    monkeypatch.setattr(build.os, "sched_getaffinity", lambda _: set(range(12)))
    monkeypatch.setattr(build, "prepare_runtime", lambda *args, **kwargs: pytest.fail("launched"))
    with pytest.raises(SystemExit):
        build.main(["--output", str(tmp_path / "runtime"), "--workers", workers])


def source_fixture(tmp_path):
    repo = tmp_path / "repo"
    for relative in build.SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n")
    for relative in build.SOURCE_TREES:
        path = repo / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "fixture.h").write_text("header\n")
    (repo / "local_notes").mkdir()
    (repo / "local_notes/operator.md").write_text("not a build input")
    (repo / ".git").mkdir()
    (repo / ".git/config").write_text("not a build input")
    return repo


def test_stage_contains_only_approved_build_roots(tmp_path):
    repo = source_fixture(tmp_path)
    staged = tmp_path / "stage"
    inventory = build.stage_source(repo, staged)
    assert "src/fixture.h" in inventory
    assert not (staged / "local_notes").exists()
    assert not (staged / ".git").exists()
    assert inventory == json.loads((staged / "source_inventory.json").read_text())
    assert all(build.file_sha256(staged / name) == digest for name, digest in inventory.items())
    with pytest.raises(FileExistsError):
        build.stage_source(repo, staged)


def test_source_symlinks_cannot_expand_the_input_manifest(tmp_path):
    repo = source_fixture(tmp_path)
    (repo / "src/leaked.h").symlink_to(repo / "local_notes/operator.md")
    with pytest.raises(ValueError, match="regular file"):
        build.stage_source(repo, tmp_path / "stage")


def test_source_directory_symlinks_are_rejected(tmp_path):
    repo = source_fixture(tmp_path)
    (repo / "src/leaked").symlink_to(repo / "local_notes", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked build directory"):
        build.stage_source(repo, tmp_path / "stage")


@pytest.mark.parametrize("flags", ["-O0", "-O2", "-O3 -O0", "-Og", ""])
def test_optimization_checks_effective_flags(tmp_path, flags):
    commands = tmp_path / "compile_commands.json"
    commands.write_text(json.dumps([{"file": "model.cpp", "command": f"c++ {flags} -c model.cpp"}]))
    with pytest.raises(ValueError, match="-O3"):
        build.verify_optimized_commands(commands)


def test_optimization_accepts_argument_vector_without_shell_parsing(tmp_path):
    commands = tmp_path / "compile_commands.json"
    commands.write_text(
        json.dumps([{"file": "model.cpp", "arguments": ["c++", "-O3", "-c", "model.cpp"]}])
    )
    assert len(build.verify_optimized_commands(commands)) == 1


@pytest.mark.parametrize("cpus", [0, 13, True, 1.5])
def test_invalid_cpu_reservations_fail_before_staging(tmp_path, monkeypatch, cpus):
    monkeypatch.setattr(build.os, "sched_getaffinity", lambda _: set(range(12)))
    with pytest.raises(ValueError, match="reserved CPUs"):
        build.prepare_runtime(tmp_path, tmp_path / "new", cpus=cpus)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("workers", [16, 32, 48])
def test_build_command_accepts_larger_authorized_hosts(tmp_path, monkeypatch, workers):
    monkeypatch.setattr(build.os, "sched_getaffinity", lambda _: set(range(48)))
    calls = []
    monkeypatch.setattr(build, "prepare_runtime", lambda *args, **kwargs: calls.append(kwargs) or {"translation_units": 130})
    build.main(["--output", str(tmp_path / "runtime"), "--workers", str(workers)])
    assert calls[0]["cpus"] == workers


def test_failed_build_keeps_complete_log(tmp_path):
    import sys

    log = tmp_path / "build.log"
    with pytest.raises(RuntimeError, match="build command failed"):
        build.run_build_command(
            [sys.executable, "-c", "print('failure evidence'); raise SystemExit(7)"],
            log,
            cwd=tmp_path,
            cpus=1,
            timeout_seconds=10,
        )
    assert "failure evidence" in log.read_text()
    with pytest.raises(FileExistsError):
        build.run_build_command(
            [sys.executable, "-c", "print('overwrite')"],
            log,
            cwd=tmp_path,
            cpus=1,
            timeout_seconds=10,
        )


def test_timeout_stops_grandchild_after_group_leader_exits(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    # The leader exits on TERM; its child explicitly ignores TERM. Reaping only
    # the leader used to leave a writer alive after a failed build was recorded.
    child = (
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "print('ready',flush=True); time.sleep(30)"
    )
    leader = (
        "import subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        "print('child-pid',p.pid,flush=True); time.sleep(30)"
    )
    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        build.run_build_command(
            [sys.executable, "-c", leader], log, cwd=tmp_path, cpus=1, timeout_seconds=1
        )
    lines = log.read_text().splitlines()
    pid = int(next(line for line in lines if line.startswith("child-pid ")).split()[1])
    assert "ready" in lines
    deadline = time.monotonic() + 3
    try:
        while time.monotonic() < deadline:
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("timed-out build left its child running")
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_native_lease_survives_worker_exit_without_reaching_generated_code(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    from ramulator_chia.recovery import OperationalPause, exclusive_lock

    sandbox = Path(__file__).resolve().parents[1] / "src/ramulator_chia/sandbox.py"
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "read": ["/usr", "/lib", "/lib64", "/bin", "/dev/null"],
                "write": [],
                "cwd": str(tmp_path),
                "cpu_seconds": 10,
                "memory_bytes": 1024**3,
                "file_bytes": 1024**2,
            }
        )
    )
    # The child must see only standard streams. It stays alive long enough for
    # its CHIA-equivalent owning worker to be killed after acknowledged startup.
    child = """import os,time
for fd in range(3, 64):
    try:
        os.fstat(fd)
    except OSError:
        continue
    raise RuntimeError('unexpected inherited descriptor')
print('ready', os.getpid(), os.getppid(), flush=True)
time.sleep(3)
"""
    worker_code = """import sys
from pathlib import Path
from ramulator_chia.framework.build import run_build_command
from ramulator_chia.recovery import exclusive_lock
root, sandbox, child = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with exclusive_lock(root / 'lease.lock') as lease:
    run_build_command(['/usr/bin/python3', sandbox, '--lease-fd', str(lease.fileno()),
                       str(root / 'policy.json'), '--', '/usr/bin/python3', '-c', child],
                      root / 'lease.log', cwd=root, cpus=1, timeout_seconds=10,
                      lease_fd=lease.fileno())
"""
    worker = subprocess.Popen(
        [sys.executable, "-c", worker_code, str(tmp_path), str(sandbox), child]
    )
    launcher_pid = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            log = tmp_path / "lease.log"
            ready = (
                [line for line in log.read_text().splitlines() if line.startswith("ready ")]
                if log.exists()
                else []
            )
            if ready:
                launcher_pid = int(ready[0].split()[2])
                break
            assert worker.poll() is None, "native launcher failed before readiness"
            time.sleep(0.02)
        else:
            pytest.fail("native launcher did not become ready")
        worker.kill()
        worker.wait(timeout=5)
        with pytest.raises(OperationalPause):
            with exclusive_lock(tmp_path / "lease.lock"):
                pytest.fail("worker exit released a still-running native writer's lease")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                with exclusive_lock(tmp_path / "lease.lock"):
                    break
            except OperationalPause:
                time.sleep(0.02)
        else:
            pytest.fail("completed native command did not release its lease")
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
        if launcher_pid is not None:
            try:
                os.killpg(launcher_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
