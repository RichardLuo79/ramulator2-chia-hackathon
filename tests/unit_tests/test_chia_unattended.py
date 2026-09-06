"""Unattended recovery/compliance tests. All provider calls are local fakes."""
import json
import pathlib
from types import SimpleNamespace

import httpx
import pytest

from tools.chia_loop import compliance as C, generation as Q, real_core as P, recovery as R
from tools.chia_loop import gemini_loop as G
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.run_records import validate_limits

REPO = pathlib.Path(__file__).resolve().parents[2]


def response(answer, *, signature=None):
    from google.genai import types
    parts = [types.Part(text=json.dumps(answer), thought_signature=signature)]
    return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="STOP",
        content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=50, total_token_count=150))


def fake_client(monkeypatch, answers):
    from google import genai
    calls = []
    answers = iter(answers)
    class Models:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(total_tokens=100)
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            answer = next(answers)
            if isinstance(answer, BaseException):
                raise answer
            return answer
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=Models(), close=lambda: None))
    monkeypatch.setattr(Q, "wait_until", lambda *args: None)
    return calls


def generation_args(root):
    from google.genai import types
    return dict(root=root, ledger=P.Ledger(root / "ledger.json", "flash", run_id=root.name, cap_usd=100),
        policy=G.POLICY, project="no-network-test", backend="flash", purpose="proposal",
        iteration=1, turn=1, operation_key="proposal_001_001", system="contract",
        contents=[types.Content(role="user", parts=[types.Part(text="hello")])])


def test_complete_response_replayed_from_compressed_journal_without_payment(tmp_path, monkeypatch):
    from tools.chia_loop import artifacts
    calls = fake_client(monkeypatch, [response({"status": "no_change"})])
    args = generation_args(tmp_path)
    raw = Q.generate(**args)
    original_ledger = args["ledger"].path.read_bytes()
    artifacts.compress(tmp_path, tmp_path / "aux_archive_manifest.json", min_bytes=1)
    assert Q.generate(**args) == raw
    assert len(calls) == args["ledger"].totals()["api_attempts"] == 1
    assert args["ledger"].path.read_bytes() == original_ledger
    artifacts.verify(tmp_path / "aux_archive_manifest.json")


def test_response_saved_before_settlement_recovers_without_second_call(tmp_path, monkeypatch):
    calls = fake_client(monkeypatch, [response({"status": "no_change"})])
    args = generation_args(tmp_path)
    settle = args["ledger"].settle
    monkeypatch.setattr(args["ledger"], "settle", lambda *a, **kw: (_ for _ in ()).throw(OSError("mock disk fault")))
    with pytest.raises(OSError):
        Q.generate(**args)
    assert args["ledger"].calls()[0]["state"] == "reserved"
    monkeypatch.setattr(args["ledger"], "settle", settle)
    assert P.parse_provider_response(Q.generate(**args))["status"] == "no_change"
    assert len(calls) == 1 and args["ledger"].calls()[0]["state"] == "usage_recorded"


def test_lost_request_reservation_survives_crash_and_retry(tmp_path, monkeypatch):
    calls = fake_client(monkeypatch, [KeyboardInterrupt(), response({"status": "no_change"})])
    args = generation_args(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        Q.generate(**args)
    assert args["ledger"].calls()[0]["state"] == "reserved"
    Q.generate(**args)
    rows = args["ledger"].calls()
    assert len(calls) == len(rows) == 2
    assert rows[0]["state"] == "usage_unknown_reservation_retained"
    assert rows[0]["cap_charge_usd"] == rows[0]["reserved_usd"]
    assert rows[1]["state"] == "usage_recorded"


def test_cooldown_and_resume_preserve_operation_identity(tmp_path, monkeypatch):
    calls = fake_client(monkeypatch, [httpx.RemoteProtocolError("lost")] * 3 + [response({"status": "no_change"})])
    args = generation_args(tmp_path)
    with pytest.raises(R.OperationalPause) as pause:
        Q.generate(**args)
    assert pause.value.retryable
    with pytest.raises(R.OperationalPause, match="cooldown"):
        Q.generate(**args)
    assert len(calls) == 3
    with pytest.raises(RuntimeError, match="identity/context"):
        Q.generate(**{**args, "system": "different contract"})
    path = tmp_path / "checkpoints/proposal_001_001.json"
    saved = R.read_json(path)
    saved["retry_at"] = 0
    atomic_write_json(path, saved)
    Q.generate(**args)
    assert len(calls) == 4 and args["ledger"].totals()["unknown_usage_calls"] == 3


def test_token_count_connection_error_is_recoverable_and_not_billed(tmp_path, monkeypatch):
    from google import genai
    counts = []
    class Models:
        def count_tokens(self, **kwargs):
            counts.append(1)
            raise httpx.ConnectError("count endpoint unavailable")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=Models(), close=lambda: None))
    monkeypatch.setattr(Q, "wait_until", lambda *args: None)
    args = generation_args(tmp_path)
    with pytest.raises(R.OperationalPause):
        Q.generate(**args)
    assert len(counts) == 3 and args["ledger"].totals()["api_attempts"] == 0


def test_auth_fault_stays_blocked_without_free_retries(tmp_path, monkeypatch):
    class AuthError(Exception):
        code = 403
    calls = fake_client(monkeypatch, [AuthError("not authorized")])
    args = generation_args(tmp_path)
    for _ in range(2):
        with pytest.raises(R.OperationalPause) as exc:
            Q.generate(**args)
        assert exc.value.retryable is False
    assert len(calls) == 1 and args["ledger"].totals()["cap_charge_usd"] == 0


def test_review_and_proposer_share_cap_but_use_correct_rates(tmp_path):
    ledger = P.Ledger(tmp_path / "ledger.json", "flash", run_id=tmp_path.name, cap_usd=100)
    call = ledger.reserve("review", 1, 1, input_tokens=100, purpose="review", backend="pro")
    ledger.settle(call, {"prompt_token_count": 1000, "candidates_token_count": 1000})
    assert ledger.totals()["estimated_standard_usd"] == pytest.approx(.014)
    assert ledger.calls()[0]["model"] == P.MODELS["pro"]
    assert json.loads(ledger.path.read_text())["model"] == P.MODELS["flash"]
    with pytest.raises(ValueError, match="proposing"):
        ledger.reserve("wrong proposer", 1, 2, purpose="proposal", backend="pro")
    other = P.Ledger(tmp_path / "small.json", "flash", cap_usd=1)
    with pytest.raises(P.BudgetExhausted):
        other.reserve("review", 1, 1, input_tokens=100, purpose="review", backend="pro")


def decision(source, verdict="pass"):
    return {"source_sha256": P.sha(source), "verdict": verdict,
            "checks": {key: {"verdict": verdict, "reason": "Source-specific fixture evidence"} for key in C.CHECKS}}


@pytest.mark.parametrize("verdict", ["pass", "reject", "uncertain", "malformed"])
def test_automatic_review_never_waits_for_person_and_fails_closed(tmp_path, monkeypatch, verdict):
    source = "example candidate source"
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts/compliance_v1.md").write_text((REPO / "tools/chia_loop/prompts/compliance_v1.md").read_text())
    directory = tmp_path / "candidates/flash_001/draft_001"
    directory.mkdir(parents=True)
    answer = decision(source, verdict) if verdict != "malformed" else {"approved": True}
    calls = fake_client(monkeypatch, [response(answer)] * 3)
    ledger = P.Ledger(tmp_path / "ledger.json", "flash", run_id=tmp_path.name, cap_usd=100)
    args = dict(root=tmp_path, source=source, proposal={"evidence": "hidden scores should not be sent",
        "private_history": "not visible"}, directory=directory, ledger=ledger, policy=G.POLICY,
        project="no-network-test", iteration=1, turn=1)
    result = C.automatic_review(**args)
    assert result["approved"] is (verdict == "pass")
    assert result["human_intervention"] is False and result["reviewer_kind"] == "api_agent"
    assert all(call["model"] == P.MODELS["pro"] for call in calls)
    assert "hidden scores" not in calls[0]["contents"][0].parts[0].text
    assert len(calls) == (3 if verdict == "malformed" else 1)
    assert C.automatic_review(**args) == result
    assert len(calls) == ledger.totals()["api_attempts"]
    with pytest.raises(RuntimeError, match="changed"):
        C.automatic_review(**{**args, "source": "different source"})


def search_fixture(root):
    atomic_write_json(root / "preparation_manifest.json", {"limits": validate_limits(2, 100, 6)})
    (root / "seed").mkdir()
    (root / "seed/atomic_controller.cpp").write_text("seed")
    (root / "prompts").mkdir()
    (root / "prompts/system_v1.md").write_text("contract")
    metrics = {"aggregate": {"cycle_macro_mae_pct": 50, "request_macro_mae_over_L": .5}}
    atomic_write_json(root / "training/reports/seed.json", {"models": {"seed": metrics}})
    return metrics


def test_search_pause_does_not_freeze_or_consume_iteration_and_resumes_parent(tmp_path, monkeypatch):
    metrics = search_fixture(tmp_path)
    calls = []
    def propose(root, arm, iteration, system, prompt, parent):
        calls.append((iteration, prompt, parent["id"]))
        if len(calls) == 1:
            return {"status": "operational_pause", "retryable": True, "reason": "lost connection"}
        return {"status": "evaluated", "candidate": {"sha256": "fixture", "source_path": "fixture",
            "metrics": metrics, "label": f"flash_{iteration:03d}"}}
    monkeypatch.setattr(G, "make_prompt", lambda *args: "frozen first prompt")
    monkeypatch.setattr(G, "propose", SimpleNamespace(chia_remote=propose))
    monkeypatch.setattr(G, "get", lambda result: result)
    paused = G.run_model(tmp_path, "flash")
    assert paused["status"] == "paused" and paused["history"] == []
    assert not (tmp_path / "selection_frozen.json").exists() and not (tmp_path / "test").exists()
    monkeypatch.setattr(G, "make_prompt", lambda *args: "later prompt")
    finished = G.run_model(tmp_path, "flash")
    assert [c[0] for c in calls] == [1, 1, 2]
    assert calls[0] == calls[1]
    assert finished["status"] == "frozen" and len(finished["history"]) == 2
    assert finished["termination"] == "iteration_limit"
    with pytest.raises(RuntimeError, match="frozen"):
        G.run_model(tmp_path, "flash")


def test_committed_terminal_decision_is_not_lost_on_restart(tmp_path, monkeypatch):
    metrics = search_fixture(tmp_path)
    state = {"run_id": tmp_path.name, "model": P.MODELS["pro"], "status": "running",
        "incumbent": "seed", "history": [{"status": "stopped", "iteration": 1}],
        "termination": "budget_stop", "budget": {},
        "candidates": {"seed": {"source_path": "seed", "metrics": metrics}}}
    atomic_write_json(tmp_path / "state.json", state)
    monkeypatch.setattr(G, "make_prompt", lambda *args: pytest.fail("terminal search restarted"))
    assert G.run_model(tmp_path, "pro")["termination"] == "budget_stop"


def test_proposal_resume_retains_signatures_and_does_not_repeat_diagnostics(tmp_path, monkeypatch):
    from google.genai import types
    search_fixture(tmp_path)
    signature = b"\x00\x99\xff signed reasoning"
    inspect = response({"status": "inspect", "requests": [{"tool": "fixture"}]}, signature=signature)
    calls = fake_client(monkeypatch, [inspect] + [httpx.RemoteProtocolError("lost")] * 3
                        + [response({"status": "no_change"})])
    diagnostics = []
    monkeypatch.setattr(G, "inspect_tool", lambda *args: diagnostics.append(1) or "fixture evidence")
    args = (str(tmp_path), "flash", 1, "contract", "initial", {})
    assert G.propose._chia_original(*args)["status"] == "operational_pause"
    saved_path = tmp_path / "checkpoints/proposal_001_002.json"
    saved = R.read_json(saved_path)
    saved["retry_at"] = 0
    atomic_write_json(saved_path, saved)
    assert G.propose._chia_original(*args)["status"] == "no_change"
    assert len(diagnostics) == 1 and len(calls) == 5
    for call in calls[1:]:
        assert call["contents"][1].parts[0].thought_signature == signature
    cp = R.read_json(tmp_path / "candidates/flash_001/proposal_state.json")
    assert cp["diagnostics"] == 1 and cp["turn"] == 2


def test_same_run_and_cpu_leases_prevent_parallel_overcommit(tmp_path):
    with R.exclusive_lock(tmp_path / "run.lock"):
        with pytest.raises(R.OperationalPause, match="another worker"):
            with R.exclusive_lock(tmp_path / "run.lock"):
                pass
    with R.cpu_lease(tmp_path / "cpus", 6), R.cpu_lease(tmp_path / "cpus", 6):
        with pytest.raises(R.OperationalPause, match="twelve"):
            with R.cpu_lease(tmp_path / "cpus", 3):
                pass
    with R.cpu_lease(tmp_path / "cpus", 12):
        pass


@pytest.mark.parametrize("marker", ["test_started.json", "test", "selection_frozen.json"])
def test_any_final_test_or_freeze_marker_forbids_optimization(tmp_path, marker):
    (tmp_path / marker).touch()
    with pytest.raises(RuntimeError, match="optimization cannot resume"):
        R.assert_training_open(tmp_path)


def test_transport_stop_is_not_a_final_test_condition(tmp_path):
    with pytest.raises(RuntimeError, match="explicit scientific"):
        G.final_test(tmp_path, "pro", {"policy": G.POLICY}, {"status": "frozen", "termination": "operational_pause"})
    assert not (tmp_path / "test_started.json").exists()


def test_outage_limit_stops_without_additional_billing(tmp_path, monkeypatch):
    calls = fake_client(monkeypatch, [httpx.ReadTimeout("offline")] * 3)
    args = generation_args(tmp_path)
    with pytest.raises(R.OperationalPause):
        Q.generate(**args)
    path = tmp_path / "checkpoints/proposal_001_001.json"
    cp = R.read_json(path)
    cp.update(retry_at=0, outage_started=1)
    atomic_write_json(path, cp)
    with pytest.raises(R.OperationalPause, match="window exhausted") as error:
        Q.generate(**args)
    assert not error.value.retryable and len(calls) == 3


def test_run_wall_deadline_and_disk_floor_fail_closed(tmp_path, monkeypatch):
    atomic_write_json(tmp_path / "run_manifest.json", {"started_at": 1,
        "policy": {"maximum_run_wall_seconds": 1}})
    with pytest.raises(R.OperationalPause, match="wall-time"):
        R.check_run_deadline(tmp_path)
    monkeypatch.setattr(R.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(R.OperationalPause, match="disk space") as error:
        R.check_storage(tmp_path)
    assert error.value.retryable is False


def pinned_execution_fixture(root, monkeypatch):
    monkeypatch.setattr(G, "verify_pinned_run", lambda *a: None)
    monkeypatch.setattr(G, "ray", SimpleNamespace(init=lambda **kw: None, shutdown=lambda: None))
    monkeypatch.setattr(G, "start_collector", lambda **kw: None)
    monkeypatch.setattr(G, "stop_collector", lambda: None)
    monkeypatch.setattr(G, "get_collector", lambda: None)
    monkeypatch.setattr(G.E, "verify_run_archives", lambda *a: 0)
    monkeypatch.setattr(G.artifacts, "compress", lambda *a, **kw: None)
    monkeypatch.setattr(G.artifacts, "verify", lambda *a: None)
    import time
    return {"status": "running", "phase": "training", "started_at": time.time(), "policy": {
        **G.POLICY, **validate_limits(25, 100, 6)}, "protocol_hashes": {}}


def test_runner_does_not_test_when_search_is_operationally_paused(tmp_path, monkeypatch):
    manifest = pinned_execution_fixture(tmp_path, monkeypatch)
    paused = {"status": "paused", "pause": {"reason": "offline", "retryable": True}}
    monkeypatch.setattr(G, "run_model", lambda *a: paused)
    monkeypatch.setattr(G, "final_test", lambda *a: pytest.fail("outage opened final test"))
    assert G.execute_pinned(tmp_path, "pro", manifest) == 75
    assert R.read_json(tmp_path / "run_manifest.json")["status"] == "paused"
    assert not (tmp_path / "selection_frozen.json").exists()


def test_runner_recovers_frozen_evaluation_without_reentering_search(tmp_path, monkeypatch):
    manifest = pinned_execution_fixture(tmp_path, monkeypatch)
    frozen = {"status": "frozen", "termination": "iteration_limit"}
    atomic_write_json(tmp_path / "state.json", frozen)
    (tmp_path / "test_started.json").touch()
    monkeypatch.setattr(G, "run_model", lambda *a: pytest.fail("optimization resumed after freeze"))
    monkeypatch.setattr(G, "final_test", lambda *a: {"aggregate": {"fixture": True}})
    assert G.execute_pinned(tmp_path, "pro", manifest) == 0
    assert R.read_json(tmp_path / "run_manifest.json")["state"] == frozen


def test_supervisor_recovers_paused_process_without_changing_cap(tmp_path, monkeypatch):
    from tools.chia_loop import supervise_gemini as S
    atomic_write_json(tmp_path / "preflight_pass.json", {})
    atomic_write_json(tmp_path / "preparation_manifest.json", {"limits": validate_limits(25, 100, 6)})
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
        def wait(self):
            if len(calls) == 1:
                atomic_write_json(tmp_path / "run_manifest.json", {"status": "paused",
                    "policy": G.configured_policy(tmp_path),
                    "pause": {"retryable": True, "reason": "transient failure", "retry_at": 0}})
                return 75
            atomic_write_json(tmp_path / "run_manifest.json", {"status": "completed", "archives_verified": True})
            return 0
    monkeypatch.setattr(S.subprocess, "Popen", Process)
    monkeypatch.setattr(S.R, "wait_until", lambda *a: None)
    assert S.supervise(tmp_path, "flash") == 0
    assert len(calls) == 2 and "--resume" not in calls[0] and "--resume" in calls[1]
    assert "--usd-cap" not in calls[1]
    assert R.read_json(tmp_path / "supervisor_state.json")["launches"] == 2


def test_supervisor_does_not_restart_auth_or_integrity_fault(tmp_path, monkeypatch):
    from tools.chia_loop import supervise_gemini as S
    atomic_write_json(tmp_path / "preflight_pass.json", {})
    calls = []
    class Process:
        def __init__(self, *a, **kw):
            calls.append(1)
        def wait(self):
            atomic_write_json(tmp_path / "run_manifest.json", {
                "status": "needs_attention", "pause": {"retryable": False, "reason": "403"}})
            return 78
    monkeypatch.setattr(S.subprocess, "Popen", Process)
    assert S.supervise(tmp_path, "pro") == 78
    assert len(calls) == 1


def test_supervisor_bounds_abrupt_process_restarts(tmp_path, monkeypatch):
    from tools.chia_loop import supervise_gemini as S
    atomic_write_json(tmp_path / "preflight_pass.json", {})
    calls = []
    class Process:
        def __init__(self, *a, **kw):
            calls.append(1)
        def wait(self):
            atomic_write_json(tmp_path / "run_manifest.json", {"status": "running"})
            return -9
    monkeypatch.setattr(S.subprocess, "Popen", Process)
    monkeypatch.setattr(S.R, "wait_until", lambda *a: None)
    assert S.supervise(tmp_path, "pro") == -9
    assert len(calls) == 4


def test_archive_does_not_unlink_live_stdout(tmp_path, monkeypatch):
    from tools.chia_loop import artifacts
    logs = tmp_path / "logs"
    logs.mkdir()
    active, closed = logs / "active.log", logs / "closed.log"
    active.write_text("active output")
    closed.write_text("completed output")
    monkeypatch.setattr(artifacts.os, "readlink", lambda path: str(active) if path.endswith("/1") else "pipe:[1]")
    result = artifacts.compress(tmp_path, tmp_path / "archives.json", min_bytes=1)
    assert active.exists() and not closed.exists()
    assert set(result["artifacts"]) == {"logs/closed.log"}
