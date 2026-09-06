"""Offline usage accounting, failed-stream journaling, and preparation queue."""
import json
import pathlib
from types import SimpleNamespace

import httpx
import pytest

from tools.chia_loop import recovery as R
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.codex_cli import transport as T, usage as U, runner as B, auth as A


def response(status="completed", *, details=True):
    usage = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}
    if details:
        usage.update(input_tokens_details={"cached_tokens": 400, "cache_write_tokens": 100},
                     output_tokens_details={"reasoning_tokens": 300})
    return {"id": "resp_fixture", "model": T.MODEL, "status": status, "reasoning": {"effort": "max"},
            "output": [], "usage": usage, "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None}


def bind_call(root, operation, role, answer):
    relative = f"interactions/{operation}/attempt_001"
    attempt = root / relative
    attempt.mkdir(parents=True)
    request = {"model": T.MODEL, "reasoning": {"effort": "max"}, "input": "PRIVATE_PROMPT_TEXT"}
    atomic_write_json(attempt / "provider_request.json", request)
    ledger = T.Ledger(root, 100)
    call = ledger.reserve(request, role, operation, attempt_path=relative, attempt_number=1, auth_mode="chatgpt")
    atomic_write_json(attempt / "reservation.json", {"call_id": call})
    if answer is not None:
        raw = ("data: " + json.dumps({"type": "response." + answer["status"], "response": answer}) + "\n\n").encode()
        (attempt / "provider_response.sse").write_bytes(raw)
        ledger.settle(call, answer, response_sha256=T.P.sha(raw))
    return ledger, attempt


def test_reasoning_is_not_double_counted_and_cache_writes_are_priced(tmp_path):
    answer = response()
    tokens = U.normalized(answer["usage"])
    assert tokens["non_reasoning_output_tokens"] == 200
    assert tokens["total_tokens"] == 1500
    cost, assumptions = U.priced(tokens)
    assert cost == pytest.approx(.03165) and assumptions == []
    ledger, _ = bind_call(tmp_path, "proposal_001_002", "proposal", answer)
    assert ledger.totals()["known_standard_usd"] == pytest.approx(.03165)
    assert ledger.totals()["cap_charge_usd"] == pytest.approx(.0625)
    long = U.normalized({"input_tokens": 300_000, "output_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 50_000, "cache_write_tokens": 100_000}})
    assert U.priced(long)[0] == pytest.approx(5.675)


def test_missing_details_and_lost_usage_are_unknown_not_zero(tmp_path):
    bind_call(tmp_path, "proposal_001_001", "proposal", response(details=False))
    bind_call(tmp_path, "review_astra_001_d01_00", "review", None)
    report = U.write_report(tmp_path)
    totals = report["totals"]
    assert totals["usage_reported_calls"] == totals["usage_unknown_calls"] == 1
    assert totals["token_totals"]["input_tokens"] == {"known_sum": 1000, "reported_calls": 1, "unknown_calls": 1}
    assert totals["token_totals"]["reasoning_output_tokens"] == {"known_sum": 0, "reported_calls": 0, "unknown_calls": 2}
    assert report["by_role"]["review"]["generation_attempts"] == 1
    assert report["by_iteration"]["1"]["generation_attempts"] == 2
    assert totals["conservative_guard_usd"] > totals["known_standard_usd"]
    assert report["invoice"] is False and report["account_quota_measured"] is False
    assert "PRIVATE_PROMPT_TEXT" not in json.dumps(report)
    assert "PRIVATE_PROMPT_TEXT" not in (tmp_path / "reports/llm_usage/calls.csv").read_text()


@pytest.mark.parametrize("patch", [
    {"input_tokens": True}, {"input_tokens": -1}, {"total_tokens": 1501},
    {"input_tokens_details": {"cached_tokens": 950, "cache_write_tokens": 100}},
    {"output_tokens_details": {"reasoning_tokens": 501}},
])
def test_contradictory_usage_is_rejected(patch):
    with pytest.raises(RuntimeError):
        U.normalized({**response()["usage"], **patch})


def test_incomplete_generation_usage_counts_but_response_is_not_accepted(tmp_path):
    ledger, attempt = bind_call(tmp_path, "proposal_001_003", "proposal", response("incomplete"))
    assert ledger.totals()["unknown_usage_calls"] == 0
    assert ledger.totals()["known_standard_usd"] > 0
    with pytest.raises(RuntimeError, match="incomplete"):
        T.completed_response((attempt / "provider_response.sse").read_bytes())
    report = U.report(tmp_path)
    assert report["calls"][0]["response_status"] == "incomplete"
    assert report["audit_issues"] == []


def test_usage_report_detects_changed_evidence_and_survives_compression(tmp_path):
    _, attempt = bind_call(tmp_path, "proposal_002_001", "proposal", response())
    assert U.report(tmp_path)["audit_issues"] == []
    B.artifacts.compress(tmp_path, tmp_path / "archive.json", min_bytes=1)
    B.artifacts.verify(tmp_path / "archive.json")
    assert U.report(tmp_path)["audit_issues"] == []
    atomic_write_json(attempt / "provider_request.json", {"model": "mutated"})
    assert U.report(tmp_path)["audit_issues"] == [{"call": 0, "issue": "request_hash_mismatch"}]


def test_provider_stream_failure_keeps_partial_bytes_and_safe_metadata(tmp_path, monkeypatch):
    native_client = httpx.AsyncClient
    partial = b'data: {"type":"response.created","response":{"id":"resp_partial"}}\n\n'
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield partial
            raise httpx.ReadError("PRIVATE_ERROR_BODY")
    def receive(request):
        assert request.headers["Authorization"] == "Bearer PRIVATE_ACCESS"
        assert request.headers["X-Client-Request-ID"]
        return httpx.Response(200, headers={"x-request-id": "provider_fixture", "set-cookie": "PRIVATE_COOKIE"}, stream=BrokenStream())
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: native_client(transport=httpx.MockTransport(receive), **kwargs))
    monkeypatch.setattr(A, "headers", lambda *args: {"Authorization": "Bearer PRIVATE_ACCESS"})
    upstream = T.Upstream("chatgpt", tmp_path / "unused_auth.json", "/unused_binary")
    upstream.journal_to(tmp_path)
    upstream.prepare_auth()
    with pytest.raises(httpx.ReadError):
        upstream({"model": T.MODEL, "reasoning": {"effort": "max"}})
    assert (tmp_path / "provider_response.sse").read_bytes() == partial
    metadata = R.read_json(tmp_path / "provider_exchange.json")
    assert metadata["status"] == "failed" and metadata["error_kind"] == "ReadError"
    assert metadata["received_bytes"] == len(partial) and metadata["http_status"] == 200
    assert metadata["wall_seconds"] >= metadata["first_byte_seconds"] >= 0
    assert "PRIVATE" not in json.dumps(metadata)


def test_native_cli_usage_mismatch_is_detected_without_double_counting(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 500, "reasoning_output_tokens": 300}}) + "\n")
    assert T.cli_usage(path, response()["usage"])["cli_usage_check"] == "matched"
    assert T.cli_usage(path, {**response()["usage"], "output_tokens": 600, "total_tokens": 1600})["cli_usage_check"] == "mismatch"


def test_preparation_queue_waits_without_model_calls(tmp_path, monkeypatch):
    args = SimpleNamespace(root=tmp_path / "astra_xhigh", effort="xhigh", max_iterations=10, cpus=6, wait_for_cpus=True)
    calls = []
    def prepare(args):
        calls.append("prepare")
        if len(calls) == 1:
            raise R.OperationalPause("CPU slots busy")
        args.root.mkdir()
    monkeypatch.setattr(B, "prepare", prepare)
    monkeypatch.setattr(R, "wait_until", lambda *args: None)
    B.prepare_with_wait(args)
    status = R.read_json(tmp_path / "astra_xhigh.preparation_status.json")
    assert calls == ["prepare", "prepare"] and status["status"] == "prepared"
    assert status["maximum_iterations"] == 10 and status["generation_calls"] == 0
    assert status["auto_launch_generation"] is False
