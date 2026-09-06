"""Bounded Gemini cancellations and explicit transport-only migration; no LLM calls."""
import copy
import json
import time
from types import SimpleNamespace

import pytest

from tools.chia_loop import generation as Q, recovery as R
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.maintenance import amend_retry499 as M
from test_chia_unattended import fake_client, generation_args, response


class Cancelled(Exception):
    code = 499


@pytest.mark.parametrize("purpose", ["proposal", "review"])
def test_499_retries_same_request_in_proposal_and_review(tmp_path, monkeypatch, purpose):
    calls = fake_client(monkeypatch, [Cancelled("provider cancellation"), response({"status": "no_change"})])
    args = generation_args(tmp_path)
    args.update(purpose=purpose, backend="pro" if purpose == "review" else "flash")
    raw = Q.generate(**args)
    assert len(calls) == 2 and calls[0] == calls[1]
    rows = args["ledger"].calls()
    assert rows[0]["http_status"] == 499 and rows[1]["state"] == "usage_recorded"
    assert rows[0]["payload_sha256"] == rows[1]["payload_sha256"]
    assert Q.generate(**args) == raw and len(calls) == 2
    assert args["ledger"].cap_usd == 100


def test_repeated_499_exhausts_eight_failures_across_restarts(tmp_path, monkeypatch):
    calls = fake_client(monkeypatch, [Cancelled("provider cancellation")] * 9)
    args = generation_args(tmp_path)
    checkpoint = tmp_path / "checkpoints/proposal_001_001.json"
    for count, retryable in ((3, True), (6, True), (8, False)):
        with pytest.raises(R.OperationalPause) as error:
            Q.generate(**args)
        assert error.value.retryable is retryable and len(calls) == count
        saved = R.read_json(checkpoint)
        assert saved["failures"] == count and "blocked" not in saved
        saved["retry_at"] = 0  # Simulate elapsed cooldown, not a production reset.
        atomic_write_json(checkpoint, saved)
    with pytest.raises(R.OperationalPause, match="attempt limit"):
        Q.generate(**args)
    assert len(calls) == args["ledger"].totals()["api_attempts"] == 8


@pytest.mark.parametrize("when", ["before_dispatch", "during_request", "during_backoff"])
def test_stop_is_authoritative_even_when_provider_returns_499(tmp_path, monkeypatch, when):
    from google import genai
    calls = []
    class Models:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(total_tokens=100)
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            if when == "during_request":
                (tmp_path / "STOP").touch()
            raise Cancelled("provider cancellation")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=Models(), close=lambda: None))
    def backoff(root, deadline):
        (root / "STOP").touch()
        R.check_stop(root)
    monkeypatch.setattr(Q, "wait_until", backoff)
    if when == "before_dispatch":
        (tmp_path / "STOP").touch()
    args = generation_args(tmp_path)
    with pytest.raises(R.OperatorStop):
        Q.generate(**args)
    assert len(calls) == (0 if when == "before_dispatch" else 1)


def test_cancelled_token_count_is_retried_without_extra_generation_charge(tmp_path, monkeypatch):
    from google import genai
    counts, generations = [], []
    class Models:
        def count_tokens(self, **kwargs):
            counts.append(1)
            if len(counts) == 1:
                raise Cancelled("count endpoint cancellation")
            return SimpleNamespace(total_tokens=100)
        def generate_content(self, **kwargs):
            generations.append(1)
            return response({"status": "no_change"})
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=Models(), close=lambda: None))
    monkeypatch.setattr(Q, "wait_until", lambda *args: None)
    args = generation_args(tmp_path)
    Q.generate(**args)
    assert len(counts) == 2 and len(generations) == args["ledger"].totals()["api_attempts"] == 1


def test_499_outage_window_is_not_reset_on_resume(tmp_path, monkeypatch):
    calls = fake_client(monkeypatch, [Cancelled("cancelled")] * 4)
    args = generation_args(tmp_path)
    with pytest.raises(R.OperationalPause):
        Q.generate(**args)
    checkpoint = tmp_path / "checkpoints/proposal_001_001.json"
    saved = R.read_json(checkpoint)
    saved.update(retry_at=0, outage_started=time.time() - 3601)
    atomic_write_json(checkpoint, saved)
    with pytest.raises(R.OperationalPause, match="recovery window"):
        Q.generate(**args)
    assert len(calls) == 3


def migration_fixture(tmp_path, monkeypatch):
    root, repo = tmp_path / "run", tmp_path / "repo"
    root.mkdir()
    policy = copy.deepcopy(M.G.POLICY)
    # This fixture exercises only the historical v8 -> v9 transport amendment,
    # never an upgrade to a later scientific evaluation protocol.
    policy["version"] = "gemini_individual_run_v9"
    policy.pop("evaluation", None)
    before_policy = {**policy, "version": "gemini_individual_run_v8",
        "transport_retry_policy": policy["transport_retry_policy"].replace(
            "network, timeout, remote-protocol and unexpected HTTP 499 errors;",
            "network, timeout and remote-protocol errors;")}
    hashes = {}
    for name, replacements in M.EDITS.items():
        before = "\n".join(old for old, new in replacements)
        source = root / "protocol" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(before)
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[name] = M.P.sha(source.read_bytes())
    atomic_write_json(root / "run_manifest.json", {"run_id": root.name, "policy": before_policy, "protocol_hashes": hashes})
    atomic_write_json(root / "state.json", {"status": "needs_attention", "history": [{"iteration": 19}], "active": {"iteration": 20}})
    error = "Cancelled: provider cancellation"
    row = {"id": 59, "operation_key": "review_020_draft_001_00", "http_status": 499,
           "error": error, "payload_sha256": "exact_request_hash", "settled_at": 100}
    atomic_write_json(root / "ledger.json", {"cap_usd": 100, "calls": [row]})
    checkpoint = root / "checkpoints/review_020_draft_001_00.json"
    atomic_write_json(checkpoint, {"identity": {"operation_key": row["operation_key"], "payload_sha256": row["payload_sha256"]},
                                 "blocked": error, "last_error": error, "failures": 0})
    atomic_write_json(root / "supervisor_state.json", {"status": "needs_attention"})
    (root / "STOP").write_text("maintenance")
    monkeypatch.setattr(M.G, "REPO", repo)
    monkeypatch.setattr(M.G, "verify_pinned_run", lambda *args: None)
    monkeypatch.setattr(M.G, "configured_policy", lambda root: policy)
    M.capture(root)
    for name in M.EDITS:
        (repo / name).write_bytes(M.transformed(name, (root / "protocol" / name).read_bytes()))
    return root, repo, checkpoint


def test_amendment_preserves_history_budget_and_request_identity(tmp_path, monkeypatch):
    root, repo, checkpoint = migration_fixture(tmp_path, monkeypatch)
    protected = {name: (root / name).read_bytes() for name in ("state.json", "ledger.json")}
    before = R.read_json(checkpoint)
    record = M.apply(root)
    after = R.read_json(checkpoint)
    assert after["identity"] == before["identity"] and "blocked" not in after
    assert after["failures"] == 1 and after["retry_at"] > after["outage_started"]
    assert protected == {name: (root / name).read_bytes() for name in protected}
    assert R.read_json(root / "run_manifest.json")["human_intervention"] is True
    assert record["human_modeling_hints"] is False and (root / "STOP").exists()
    assert M.apply(root) == record  # Never reset retry counters by applying twice.


def test_amendment_rejects_unrelated_source_changes(tmp_path, monkeypatch):
    root, repo, checkpoint = migration_fixture(tmp_path, monkeypatch)
    path = repo / "tools/chia_loop/generation.py"
    path.write_bytes(path.read_bytes() + b"unrelated change")
    before = checkpoint.read_bytes()
    with pytest.raises(RuntimeError, match="unexpected implementation"):
        M.apply(root)
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("status", [401, 403, 500])
def test_amendment_cannot_clear_other_blocked_errors(status):
    saved = {"identity": {"operation_key": "review", "payload_sha256": "same"}, "blocked": "failure", "failures": 0}
    rows = [{"id": 1, "operation_key": "review", "payload_sha256": "same", "http_status": status, "error": "failure"}]
    with pytest.raises(RuntimeError, match="not the saved provider-499"):
        M.rearm_checkpoint(saved, rows, 1000)
