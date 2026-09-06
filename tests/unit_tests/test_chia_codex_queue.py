"""Dependency-ordered Astra launches with fake preparation and no model calls."""
import fcntl
from types import SimpleNamespace

import pytest

from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.codex_cli import queue as Q


def predecessor(root, status="completed"):
    root.mkdir()
    atomic_write_json(root / "supervisor_state.json", {"run_id": root.name, "status": status,
        "reason": "PRIVATE_PREDECESSOR_DETAIL", "metrics": "DO_NOT_IMPORT"})
    (root / ".supervisor.lock").touch()
    (root / ".runner.lock").touch()
    return root


def args_for(tmp_path):
    binary = tmp_path / "fixture_codex"
    binary.write_text("offline executable fixture")
    auth = tmp_path / "auth.json"
    auth.write_text("PRIVATE_AUTH_NOT_TO_BE_READ_BY_QUEUE")
    return SimpleNamespace(root=tmp_path / "astra_run", after=[predecessor(tmp_path / "gemini_pro"),
        predecessor(tmp_path / "gemini_flash", "running")], codex_binary=str(binary), auth_file=auth,
        authorize_paid=True, max_iterations=10, usd_cap=100, cpus=6, effort="max")


def forbidden(*args, **kwargs):
    pytest.fail("preparation/generation must not start")


def test_dependencies_require_terminal_state_and_released_worker_locks(tmp_path):
    root = predecessor(tmp_path / "predecessor", "needs_attention")
    before = {p: p.read_bytes() for p in root.iterdir()}
    observed = Q.dependencies([root])
    assert observed[0]["ready"]
    assert "PRIVATE" not in str(observed) and "DO_NOT_IMPORT" not in str(observed)
    with (root / ".runner.lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not Q.dependencies([root])[0]["ready"]
    assert before == {p: p.read_bytes() for p in root.iterdir()}
    atomic_write_json(root / "supervisor_state.json", {"run_id": root.name, "status": "cooldown"})
    assert not Q.dependencies([root])[0]["ready"]


def test_queue_waits_for_both_then_prepares_verifies_and_supervises(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    monkeypatch.setattr(Q, "fingerprint", lambda binary: {"pinned": "offline"})
    calls = []
    def sleep(seconds):
        assert seconds == 30 and not calls and not args.root.exists()
        calls.append("waited")
        atomic_write_json(args.after[1] / "supervisor_state.json", {"run_id": args.after[1].name, "status": "completed"})
    monkeypatch.setattr(Q.time, "sleep", sleep)
    def prepare(got):
        assert got.auth_mode == "chatgpt" and got.wait_for_cpus
        assert got.max_iterations == 10 and got.cpus == 6 and got.effort == "max"
        calls.append("prepare")
        got.root.mkdir()
        Q.B.W.install(got.root)
        Q.B.L.install(got.root)
    monkeypatch.setattr(Q.B, "prepare_with_wait", prepare)
    monkeypatch.setattr(Q.B, "verify", lambda root: calls.append("verify"))
    def supervise(root):
        calls.append("supervise")
        atomic_write_json(root / "supervisor_state.json", {"status": "completed"})
    monkeypatch.setattr(Q.B, "supervise", supervise)
    Q.execute(args)
    assert calls == ["waited", "prepare", "verify", "supervise"]
    saved = Q.R.read_json(tmp_path / "astra_run.queue.json")
    assert saved["status"] == "completed" and saved["generation_authorized"]
    assert not saved["model_generation_started"]
    assert "PRIVATE" not in str(saved) and "DO_NOT_IMPORT" not in str(saved)


@pytest.mark.parametrize("fail_at", ["implementation", "preparation", "verification"])
def test_gate_failure_never_launches_generation(tmp_path, monkeypatch, fail_at):
    args = args_for(tmp_path)
    monkeypatch.setattr(Q, "dependencies", lambda roots: [{"ready": True}])
    snapshots = iter([{"version": 1}, {"version": 2}, {"version": 2}])
    monkeypatch.setattr(Q, "fingerprint", lambda binary: next(snapshots) if fail_at == "implementation" else {})
    def prepare(args):
        if fail_at == "preparation":
            raise RuntimeError("offline preparation failure")
        args.root.mkdir()
        Q.B.W.install(args.root)
        Q.B.L.install(args.root)
    def verify(root):
        raise RuntimeError("offline verification failure")
    monkeypatch.setattr(Q.B, "prepare_with_wait", prepare)
    monkeypatch.setattr(Q.B, "verify", verify)
    monkeypatch.setattr(Q.B, "supervise", forbidden)
    with pytest.raises(RuntimeError):
        Q.execute(args)
    saved = Q.R.read_json(tmp_path / "astra_run.queue.json")
    assert saved["status"] == "needs_attention" and not saved["model_generation_started"]


def test_operator_stop_cancels_queued_job_without_preparation_or_usage(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    monkeypatch.setattr(Q, "fingerprint", lambda binary: {})
    def stop(seconds):
        args.root.with_name(args.root.name + ".preparation_STOP").write_text("operator hold")
    monkeypatch.setattr(Q.time, "sleep", stop)
    monkeypatch.setattr(Q.B, "prepare_with_wait", forbidden)
    monkeypatch.setattr(Q.B, "supervise", forbidden)
    with pytest.raises(Q.R.OperatorStop):
        Q.execute(args)
    assert Q.R.read_json(tmp_path / "astra_run.queue.json")["status"] == "stopped"
    assert not args.root.exists()


def test_unauthorized_queue_is_not_created(tmp_path):
    args = args_for(tmp_path)
    args.authorize_paid = False
    with pytest.raises(ValueError, match="authorization"):
        Q.execute(args)
    assert not (tmp_path / "astra_run.queue.json").exists()


def test_fingerprint_detects_code_changes_but_ignores_run_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(Q.B, "REPO", tmp_path)
    (tmp_path / "CMakeLists.txt").write_text("fixture build")
    (tmp_path / "src").mkdir()
    code = tmp_path / "src/fixture.cpp"
    code.write_text("int a;")
    binary = tmp_path / "codex"
    binary.write_bytes(b"fixture binary")
    before = Q.fingerprint(binary)
    (tmp_path / "eval_out").mkdir()
    (tmp_path / "eval_out/result.py").write_text("not protocol")
    assert Q.fingerprint(binary) == before
    code.write_text("int b;")
    assert Q.fingerprint(binary) != before
