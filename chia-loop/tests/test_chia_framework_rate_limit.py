"""HTTP fixtures only: no provider calls or actual backoff waits."""

import httpx
import pytest

from ramulator_chia.framework.rate_limit import RateLimitPolicy, RateLimitTransport


def invoke(statuses, *, headers=None, record=None, jitter=1):
    clock, waits, records, bodies = [0.0], [], [], []

    def request(req):
        bodies.append(req.read())
        return httpx.Response(statuses.pop(0), headers=headers)

    def sleep(delay):
        clock[0] += delay
        waits.append(delay)

    transport = RateLimitTransport(
        httpx.MockTransport(request), RateLimitPolicy(), record or records.append,
        sleep=sleep, clock=lambda: clock[0], jitter=lambda low, high: jitter,
    )
    with httpx.Client(transport=transport) as client:
        response = client.post("https://fixture.invalid/model:generateContent", content=b"unchanged")
    return response, waits, records, bodies


def test_bounded_exponential_retries_preserve_request_and_do_not_repeat_success():
    response, waits, records, bodies = invoke([429, 429, 429, 429, 200])
    assert response.status_code == 200
    assert [r["delay_seconds"] for r in records] == [30, 60, 120, 240]
    assert sum(waits) == 450 and max(waits) <= 55
    assert bodies == [b"unchanged"] * 5
    assert len({r["request_id"] for r in records}) == 1
    assert not any("unchanged" in str(r) for r in records)


def test_retry_limit_and_other_http_errors():
    response, _, records, bodies = invoke([429] * 6)
    assert response.status_code == 429 and len(bodies) == 5 and len(records) == 4
    for code in (200, 400, 401, 403, 500):
        response, waits, records, bodies = invoke([code])
        assert response.status_code == code and not waits and not records and len(bodies) == 1


def test_jitter_retry_after_and_unbounded_server_wait():
    _, _, records, _ = invoke([429, 200], jitter=1.2)
    assert records[0]["delay_seconds"] == 36
    _, _, records, _ = invoke([429, 200], headers={"Retry-After": "90"})
    assert records[0]["delay_seconds"] == 90
    response, waits, _, bodies = invoke([429], headers={"Retry-After": "3600"})
    assert response.status_code == 429 and not waits and len(bodies) == 1


def test_transport_failure_and_failed_receipt_never_trigger_retry():
    def disconnected(request):
        raise httpx.ReadTimeout("ambiguous")

    def denied(event):
        raise OSError("receipt storage unavailable")

    transport = RateLimitTransport(httpx.MockTransport(disconnected), RateLimitPolicy(), denied)
    with httpx.Client(transport=transport) as client, pytest.raises(httpx.ReadTimeout):
        client.post("https://fixture.invalid", content=b"request")
    with pytest.raises(OSError, match="receipt storage"):
        invoke([429, 200], record=denied)


def test_google_sdk_uses_transport_for_generation_and_token_counting():
    from google import genai
    from google.genai import types
    from google.auth.credentials import AnonymousCredentials

    seen, receipts, now = [], [], [0.0]

    def serve(request):
        seen.append(request.url.path)
        if len(seen) == 1:
            return httpx.Response(429, json={"error": {"code": 429, "message": "busy"}})
        if request.url.path.endswith(":countTokens"):
            return httpx.Response(200, json={"totalTokens": 42})
        return httpx.Response(200, json={"candidates": [{
            "content": {"role": "model", "parts": [{"text": "fixture"}]},
            "finishReason": "STOP",
        }]})

    transport = RateLimitTransport(
        httpx.MockTransport(serve), RateLimitPolicy(), receipts.append,
        sleep=lambda delay: now.__setitem__(0, now[0] + delay),
        clock=lambda: now[0], jitter=lambda low, high: 1,
    )
    credentials = AnonymousCredentials()
    credentials.token = "fixture-not-a-credential"
    with genai.Client(vertexai=True, project="fixture-project", location="global",
                      credentials=credentials, http_options=types.HttpOptions(
                          retry_options=types.HttpRetryOptions(attempts=1),
                          client_args={"transport": transport},
                      )) as client:
        assert client.models.generate_content(model="fixture", contents="test").text == "fixture"
        assert client.models.count_tokens(model="fixture", contents="test").total_tokens == 42
    assert len(seen) == 3 and len(receipts) == 1


@pytest.mark.parametrize("kwargs", [
    {"attempts": 0}, {"attempts": True}, {"attempts": 11},
    {"initial_delay_seconds": 0}, {"initial_delay_seconds": 301},
    {"maximum_delay_seconds": float("inf")}, {"jitter_fraction": 2},
])
def test_policy_rejects_unbounded_or_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        RateLimitPolicy(**kwargs)
