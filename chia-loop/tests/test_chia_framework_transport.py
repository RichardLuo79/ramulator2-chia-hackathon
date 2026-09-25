"""Launch-boundary and fixed-destination relay checks; no real provider calls."""

import gzip
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from ramulator_chia.framework.config import CodexBackend
from ramulator_chia.framework.transport import CLI, SYSTEM_READS, relay


def test_fixed_relay_keeps_native_body_and_rejects_other_destinations(tmp_path, monkeypatch):
    backend = CodexBackend(kind="codex_cli", model="fixture", reasoning_effort="high")
    calls = []

    @contextmanager
    def upstream(method, url, **options):
        calls.append((method, url, options))
        yield type(
            "Response",
            (),
            {
                "status_code": 200,
                "headers": {"Content-Type": "text/event-stream"},
                "iter_bytes": lambda self: iter([b"data: native opaque reply\n\n"]),
            },
        )()

    monkeypatch.setattr("ramulator_chia.framework.transport.httpx.stream", upstream)
    with relay(backend, tmp_path, lambda: {"Authorization": "Bearer PRIVATE_CANARY"}) as (
        port,
        token,
    ):
        url = f"http://127.0.0.1:{port}"
        headers = {"X-CHIA-Token": token}
        with httpx.Client(trust_env=False) as client:
            assert client.post(url + "/v1/responses", json={}).status_code == 403
            assert client.post(url + "/other", json={}, headers=headers).status_code == 403
            request = {"model": "fixture", "input": [{"opaque": "native continuation"}]}
            response = client.post(url + "/v1/responses", json=request, headers=headers)
            assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][1] == CLI["codex_cli"]["upstream"]
    assert json.loads(calls[0][2]["content"]) == request
    assert calls[0][2]["headers"]["Authorization"] == "Bearer PRIVATE_CANARY"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            assert b"PRIVATE_CANARY" not in data


def test_native_conversation_wait_has_no_provider_deadline(tmp_path, monkeypatch):
    from ramulator_chia.framework import transport
    real = subprocess.Popen.communicate
    waits = []

    def communicate(self, input=None, timeout=None):
        waits.append(timeout)
        return real(self, input=input, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate)
    result = transport.run_native_process(
        ["/usr/bin/python3", "-c", "import time; time.sleep(.1); print('job completed')"],
        cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, prompt="",
        logs=tmp_path / "logs",
    )
    assert result.returncode == 0 and result.stdout.strip() == "job completed"
    assert waits == [None]


def test_interruption_returns_partial_native_output_to_chia(tmp_path, monkeypatch):
    from ramulator_chia.framework import transport
    import time

    def interrupted(self, input=None, timeout=None):
        time.sleep(.15)
        raise subprocess.TimeoutExpired("fixture", .1)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        transport.run_native_process(
            ["/usr/bin/python3", "-u", "-c",
             "import time; print('{\"type\":\"thread.started\",\"thread_id\":\"saved\"}'); time.sleep(60)"],
            cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, prompt="",
            logs=tmp_path / "logs",
        )
    assert '"thread_id":"saved"' in raised.value.stdout
    assert raised.value.stderr == ""
    assert gzip.decompress((tmp_path / "logs/stdout.jsonl.gz").read_bytes()).decode() == raised.value.stdout


@pytest.mark.parametrize("kind", ["codex_cli", "claude_cli"])
def test_real_process_boundary_and_installed_cli_start_without_provider(tmp_path, kind):
    binary = Path(CLI[kind]["binary"]).resolve()
    if not binary.exists():
        pytest.skip("installed native CLI is absent")
    private = tmp_path / "private"
    private.mkdir()
    (private / "approved.txt").write_text("allowed")
    hidden = tmp_path / "other-campaign.txt"
    hidden.write_text("forbidden")
    policy = {
        "kind": kind,
        "cwd": str(private),
        "ports": [65119, 65120],
        "read": [p for p in SYSTEM_READS if Path(p).exists()]
        + [str(private), str(binary)]
        + [str(binary.with_name(name)) for name in CLI[kind]["companions"]],
        "write": [str(private)],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(private),
        "CODEX_HOME": str(private),
        "CLAUDE_CONFIG_DIR": str(private),
        "TMPDIR": str(private),
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / 'src'),
    }
    prefix = ["/usr/bin/python3", "-m", "ramulator_chia.framework.native_process", str(path)]
    probe = """import errno, pathlib, socket
assert pathlib.Path('approved.txt').read_text() == 'allowed'
try:
    pathlib.Path(%r).read_text()
except PermissionError: pass
else: raise AssertionError('other campaign readable')
try:
    socket.socket().connect(('127.0.0.1', 65121))
except OSError as exc: assert exc.errno in (errno.EPERM, errno.EACCES)
else: raise AssertionError('unapproved port reachable')
print('boundary passed')
""" % str(hidden)
    checked = subprocess.run(
        prefix + ["/usr/bin/python3", "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert checked.returncode == 0, checked.stderr
    version = subprocess.run(
        prefix + [str(binary), "--version"], env=env, capture_output=True, text=True, timeout=20
    )
    assert version.returncode == 0, version.stderr
    for name in CLI[kind]["companions"]:
        helper = subprocess.run(
            prefix + [str(binary.with_name(name)), "--help"],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert helper.returncode == 0, helper.stderr
        started = subprocess.run(
            prefix + [str(binary.with_name(name)), "--listen", "stdio"],
            env=env,
            input="",
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert started.returncode == 0, started.stderr
