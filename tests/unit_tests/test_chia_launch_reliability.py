"""Fresh uncapped authorization and transport regressions; no live providers."""

import json
import pathlib
import subprocess
import threading
import time

import pytest

from tests.unit_tests.test_chia_claude_cli import configure, credentials, fake_sse
from tools.chia_loop import artifacts
from tools.chia_loop import gemini_loop as G
from tools.chia_loop import prompt_cache as K
from tools.chia_loop import real_core as P
from tools.chia_loop import run_records as N
from tools.chia_loop.claude_cli import transport as T
from tools.chia_loop.claude_cli import usage as U
from tools.chia_loop.core import atomic_write_json


def test_uncapped_fresh_gemini_keeps_usage_and_cannot_convert_existing_cap(tmp_path):
    root = tmp_path / "fresh"
    K.install(root)
    limits = N.validate_limits(20, None, 3, iteration_guard=True)
    atomic_write_json(root / "preparation_manifest.json", {"limits": limits})
    ledger = G.run_ledger(root, "flash")
    ledger.initialize()
    index = ledger.reserve("fixture", 1, 1, input_tokens=100)
    ledger.settle(index, None, "unknown result retained")
    assert ledger.totals()["unknown_usage_calls"] == 1
    assert ledger.totals()["cap_charge_usd"] > 0
    saved = json.loads(ledger.path.read_text())
    assert saved["cap_usd"] is None and saved["guard_mode"] == "iterations"
    # Uncapped is not implemented as an enormous made-up dollar ceiling.
    with pytest.raises(ValueError, match="guard mode"):
        P.Ledger(ledger.path, "flash", run_id=root.name, cap_usd=100).calls()
    other = tmp_path / "finite"
    finite = P.Ledger(other / "ledger.json", "flash", run_id=other.name, cap_usd=100)
    finite.initialize()
    before = finite.path.read_bytes()
    with pytest.raises(ValueError, match="guard mode"):
        P.Ledger(finite.path, "flash", run_id=other.name, iteration_guard=True).reserve(
            "fixture", 1, 1
        )
    assert finite.path.read_bytes() == before
    for values in ((20, None, 3, False), (20, 100, 3, True), (20, None, 3, "yes")):
        with pytest.raises(ValueError):
            N.validate_limits(*values)


def test_uncapped_still_enforces_context_and_frozen_limits(tmp_path):
    ledger = P.Ledger(tmp_path / "ledger.json", "flash", run_id=tmp_path.name, iteration_guard=True)
    with pytest.raises(P.BudgetExhausted, match="input token"):
        ledger.reserve("fixture", 1, 1, input_tokens=P.MAX_INPUT_TOKENS + 1)
    limits = N.validate_limits(20, None, 3, True)
    atomic_write_json(tmp_path / "preparation_manifest.json", {"limits": limits})
    atomic_write_json(
        tmp_path / "run_manifest.json", {"record_type": "optimization_run", "policy": limits}
    )
    assert N.execution_limits(tmp_path) == limits
    atomic_write_json(
        tmp_path / "preparation_manifest.json", {"limits": N.validate_limits(21, None, 3, True)}
    )
    with pytest.raises(ValueError, match="changed"):
        N.execution_limits(tmp_path)


def test_typed_rate_headers_do_not_log_credentials_or_treat_allowed_as_blocked():
    headers = {
        "retry-after": "18000",
        "authorization": "Bearer PRIVATE_SECRET",
        "x-request-debug": "PRIVATE_SECRET",
        "anthropic-ratelimit-unified-5h-status": "rejected",
        "anthropic-ratelimit-unified-5h-reset": "20000",
        "anthropic-ratelimit-unified-7d-status": "allowed",
        "anthropic-ratelimit-unified-7d-reset": "500000",
        "anthropic-ratelimit-unified-7d-utilization": "0.3",
        "anthropic-ratelimit-unified-status": "PRIVATE_SECRET",
    }
    result = T.retry_metadata(headers, 429, now=1000)
    assert result["retry_at"] == 20000
    assert "PRIVATE" not in json.dumps(result)
    assert T.retry_metadata({}, 429, now=1000)["retry_at"] == 1900
    assert (
        T.retry_metadata({"retry-after": "Thu, 01 Jan 1970 01:00:00 GMT"}, 429, now=1000)[
            "retry_at"
        ]
        == 3600
    )


def test_upstream_relays_and_journals_before_full_response(tmp_path, monkeypatch):
    import httpx

    original_client = httpx.AsyncClient
    received = []
    raw = fake_sse({"status": "no_change"})
    split = len(raw) // 2

    class Bytes(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield raw[:split]
            assert received == [raw[:split]]
            assert (tmp_path / "provider_response.sse").read_bytes() == raw[:split]
            yield raw[split:]

    def response(request):
        assert str(request.url) == "https://api.anthropic.com/v1/messages?beta=true"
        assert request.headers["Authorization"] == "Bearer PRIVATE_TEST_TOKEN"
        return httpx.Response(
            200,
            stream=Bytes(),
            headers={"request-id": "fixture", "authorization": "PRIVATE_SECRET"},
        )

    def client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return original_client(**kwargs, transport=httpx.MockTransport(response))

    monkeypatch.setattr(httpx, "AsyncClient", client)
    result = T.Upstream().stream(
        {"model": T.MODEL},
        {"Authorization": "Bearer PRIVATE_TEST_TOKEN"},
        tmp_path,
        on_chunk=received.append,
    )
    assert result == raw == b"".join(received)
    metadata = json.loads((tmp_path / "provider_exchange.json").read_text())
    assert metadata["status"] == "stream_received" and metadata["received_sha256"] == P.sha(raw)
    assert "PRIVATE" not in json.dumps(metadata)


def test_zero_error_receipt_is_not_nonzero_contradictory_usage(tmp_path):
    path = tmp_path / "events.jsonl"
    response = T.terminal_response(fake_sse({"status": "no_change"}))
    receipt = {
        "type": "result",
        "is_error": True,
        "num_turns": 1,
        "modelUsage": {},
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }
    path.write_text(json.dumps(receipt) + "\n")
    assert T.cli_receipt(path, response)["cli_usage_check"] == "unavailable_or_interrupted"
    receipt["usage"]["output_tokens"] = 1
    path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(RuntimeError, match="disagree"):
        T.cli_receipt(path, response)


def test_early_cli_exit_waits_for_producer_and_replays_without_payment(tmp_path, monkeypatch):
    import http.client

    root = tmp_path / "run"
    configure(root)
    credential = tmp_path / "auth.json"
    credentials(credential)
    started, consumer_done = threading.Event(), threading.Event()
    requests = []
    raw = fake_sse({"status": "no_change"})
    # This test isolates producer/consumer ordering. Native framing and process
    # isolation are independently covered by the real-CLI tests.
    monkeypatch.setattr(T, "validate_request", lambda *a: None)

    def upstream(request, headers, attempt):
        requests.append(request)
        with (attempt / "provider_response.sse").open("xb") as stream:
            stream.write(raw[:40])
            stream.flush()
            started.set()
            time.sleep(0.2)
            stream.write(raw[40:])
        return raw

    def native(args, **kwargs):
        policy = json.loads(pathlib.Path(args[2]).read_text())

        def send():
            try:
                connection = http.client.HTTPConnection("127.0.0.1", policy["port"], timeout=5)
                secret = policy["environment"]["ANTHROPIC_CUSTOM_HEADERS"].split(": ", 1)[1]
                request = {"model": T.MODEL, "output_config": {"effort": "xhigh"}}
                connection.request(
                    "POST",
                    "/v1/messages",
                    json.dumps(request),
                    {"X-CHIA-Token": secret, "Content-Type": "application/json"},
                )
                assert connection.getresponse().read()
                connection.close()
            finally:
                consumer_done.set()

        threading.Thread(target=send, daemon=True).start()
        assert started.wait(5)
        receipt = {
            "type": "result",
            "is_error": True,
            "num_turns": 1,
            "modelUsage": {},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return subprocess.CompletedProcess(
            args, 1, json.dumps(receipt) + "\n", "consumer interrupted"
        )

    monkeypatch.setattr(T.subprocess, "run", native)
    args = dict(
        root=root,
        operation="proposal_001_001",
        system="fixture",
        conversation=[],
        effort="xhigh",
        role="proposal",
        cap=None,
        upstream=upstream,
        binary=pathlib.Path("/unused"),
        auth_file=credential,
    )
    assert T.invoke(**args) == {"status": "no_change"}
    assert consumer_done.wait(5)
    receipt = T.R.read_json(root / "interactions/proposal_001_001/attempt_001/receipt.json")
    assert receipt["cli_usage_check"] == "unavailable_or_interrupted"
    assert not receipt.get("cli_receipt_conflict") and not receipt.get("provider_stream_incomplete")
    artifacts.compress(root, root / "packed.json", min_bytes=1)
    assert T.invoke(**args) == {"status": "no_change"}
    assert len(requests) == 1 and U.Ledger(root).totals()["attempts"] == 1


def test_provider_cooldown_survives_restart_without_new_reservation(tmp_path, monkeypatch):
    configure(tmp_path)
    operation = "proposal_001_001"
    directory = tmp_path / "interactions" / operation
    attempt = directory / "attempt_001"
    attempt.mkdir(parents=True)
    atomic_write_json(
        attempt / "receipt.json", {"retry_at": time.time() + 18000, "retryable": True}
    )

    def forbidden(*a, **kw):
        pytest.fail("cooldown must not use credentials or dispatch")

    monkeypatch.setattr(T.auth, "check", forbidden)
    with pytest.raises(T.R.OperationalPause) as exc:
        T.invoke(
            tmp_path,
            operation,
            "fixture",
            [],
            effort="xhigh",
            role="proposal",
            cap=None,
            upstream=forbidden,
            binary=pathlib.Path("/unused"),
            auth_file="/unused",
        )
    assert exc.value.retry_at > time.time() + 17900
    assert not (tmp_path / "ledger.json").exists()
