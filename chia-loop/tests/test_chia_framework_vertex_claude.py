"""Real Anthropic Vertex SDK and CHIA loop against inert HTTP/MCP fixtures.

No provider sockets, credentials, model processes or simulator runs. Native
blocks in these fixtures are deliberately synthetic, not previous research.
"""

import asyncio
import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from google.oauth2.credentials import Credentials
from test_chia_framework_campaign import no_services as no_services
from test_chia_framework_model_adapters import vertex_fixture

from ramulator_chia.framework.agent import NativeAgent
from ramulator_chia.framework.config import VertexClaudeBackend
from ramulator_chia.framework.model_sessions import create_session
from ramulator_chia.framework.transport import NativeTransport
from ramulator_chia.framework.usage import InvalidUsage, Tariff, normalize, quote

pytestmark = pytest.mark.usefixtures("no_services")
MODEL = "claude-fable-5-1"
THINKING = {"type": "thinking", "thinking": "Fixture reasoning.", "signature": "opaque-signature"}


def usage():
    return dict(input_tokens=10, output_tokens=8, cache_read_input_tokens=70,
                cache_creation_input_tokens=20,
                cache_creation=dict(ephemeral_5m_input_tokens=20, ephemeral_1h_input_tokens=0))


def reply(*, tool=False, compaction=False, stop=None, identifier="msg_fixture"):
    content = [dict(THINKING)]
    if compaction:
        content.insert(0, {"type": "compaction", "content": "Native fixture summary."})
    content.append(
        {"type": "tool_use", "id": "toolu_fixture", "name": "dram__inspect", "input": {"draft": "current"}}
        if tool else {"type": "text", "text": "Fixture complete."}
    )
    counts = usage()
    counts["iterations"] = [{"type": "message", **usage()}]
    if compaction:
        counts["iterations"].insert(0, {"type": "compaction", **usage()})
    return dict(id=identifier, type="message", role="assistant", model=MODEL, content=content,
                stop_reason=stop or ("tool_use" if tool else "end_turn"), stop_sequence=None,
                usage=counts)


def sse(message, *, incomplete=False):
    events = [{"type": "message_start", "message": {
        **message, "content": [], "stop_reason": None, "usage": {**message["usage"], "output_tokens": 0},
    }}]
    for index, block in enumerate(message["content"]):
        events.append(dict(type="content_block_start", index=index, content_block=block))
        events.append(dict(type="content_block_stop", index=index))
    if not incomplete:
        events += [dict(type="message_delta", delta=dict(stop_reason=message["stop_reason"], stop_sequence=None),
                        usage=message["usage"]), dict(type="message_stop")]
    return "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events).encode()


def fixture(tmp_path, monkeypatch, responses, callback=None):
    mcp = vertex_fixture(monkeypatch, [], mock_sdk=False)
    calls, events, states = [], [], []

    async def respond(request):
        assert str(request.url) == (
            "https://aiplatform.googleapis.com/v1/projects/fixture-project/"
            "locations/global/publishers/anthropic/models/claude-fable-5-1:streamRawPredict"
        )
        assert request.headers["authorization"] == "Bearer PRIVATE_FIXTURE_CANARY"
        assert request.headers["anthropic-beta"] == "compact-2026-01-12"
        calls.append(json.loads(request.content))
        item = responses.pop(0)
        if isinstance(item, int):
            return httpx2.Response(item, json={"error": {"message": "Fixture service failure"}})
        if isinstance(item, BaseException):
            raise item
        return httpx2.Response(200, content=item if isinstance(item, bytes) else sse(item),
                               headers={"Content-Type": "text/event-stream"})

    def observe(event, state):
        events.append(event)
        states.append(state)
        if callback:
            callback(event, state)

    def model(identity="fixture-role", timeout=30):
        private = tmp_path / identity
        private.mkdir(exist_ok=True)
        return create_session(
            VertexClaudeBackend(kind="vertex_claude", project="fixture-project"), session_id=identity,
            private_directory=private, workspace=tmp_path, system_message="Fixture tools only.",
            timeout_seconds=timeout,
            anthropic_client_kwargs={
                "credentials": Credentials("PRIVATE_FIXTURE_CANARY"),
                "http_client": httpx2.AsyncClient(transport=httpx2.MockTransport(respond), trust_env=False),
            },
            anthropic_event_callback=observe,
        )

    return SimpleNamespace(model=model, calls=calls, events=events, states=states,
                           tools=mcp.tools, tool=mcp.tool)


def test_routing_thinking_cache_and_native_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_VERTEX_BASE_URL", "http://forbidden.invalid")
    f = fixture(tmp_path, monkeypatch, [reply(tool=True), 503, reply(), reply()])
    model = f.model()
    result = model.prompt_once("explore", [f.tool])
    assert not result.success and result.error.error_type == "server_error"
    assert len(f.calls) == 2 and len(f.tools) == 1  # No SDK retries.
    checkpoint = model.checkpoint(result)
    assert b"PRIVATE_FIXTURE_CANARY" not in checkpoint.files["contents.json"]
    resumed = f.model()
    resumed.restore(checkpoint)
    assert resumed.prompt_once("explore", [f.tool]).success
    assert f.calls[1] == f.calls[2] and len(f.tools) == 1
    state = resumed.model.session_state()
    resumed = f.model()
    resumed.model.restore_session(state)
    assert resumed.prompt_once("reflect", [f.tool]).success
    request = f.calls[0]
    assert request["anthropic_version"] == "vertex-2023-10-16"
    assert request["max_tokens"] == 128000
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "xhigh"}
    assert request["cache_control"] == {"type": "ephemeral"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["context_management"] == {"edits": [{
        "type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 800000},
    }]}
    assert f.calls[-1]["messages"][1]["content"][0] == THINKING
    assert f.calls[-1]["messages"][2]["content"][0]["tool_use_id"] == "toolu_fixture"
    with pytest.raises(ValueError, match="different settings or session"):
        f.model("reviewer").restore(checkpoint)


def test_compaction_round_trip_and_separate_accounting(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, [reply(tool=True, compaction=True), reply()])
    result = f.model().prompt_once("explore", [f.tool])
    assert result.success
    assert f.calls[1]["messages"][1]["content"] == reply(tool=True, compaction=True)["content"]
    # Full original history remains; no manual reconstruction of signed blocks.
    assert f.calls[1]["messages"][0] == {"role": "user", "content": "explore"}
    raw = reply(compaction=True)["usage"]
    counts = normalize(raw, "anthropic_messages")
    assert counts["tokens"]["input_tokens"] == 200
    assert counts["tokens"]["output_tokens"] == 16
    assert counts["tokens"]["cached_input_tokens"] == 140
    tariff = Tariff.model_validate_json((Path(__file__).parents[1]/"configs/tariffs/fable-fixture.json").read_text())
    cost = quote(counts, tariff, scope="provider_request")
    # Each sampling estimate is rounded outward to whole micro-USD.
    assert cost["lower_micro_usd"] == 1534 and cost["upper_micro_usd"] == 1536
    assert [row["type"] for row in cost["iterations"]] == ["compaction", "message"]
    assert normalize(usage(), "anthropic_messages")["tokens"]["input_tokens"] == 100
    with pytest.raises(InvalidUsage):
        normalize({"iterations": [{"type": "unknown"}]}, "anthropic_messages")


def test_interrupted_tool_cannot_be_replayed(tmp_path, monkeypatch):
    def stop(event, state):
        if event["kind"] == "tool_request":
            raise RuntimeError("injected interruption before tool settlement")

    f = fixture(tmp_path, monkeypatch, [reply(tool=True)], stop)
    model = f.model()
    result = model.prompt_once("explore", [f.tool])
    assert not result.success and result.session_state["pending_tools"]
    resumed = f.model()
    resumed.restore(model.checkpoint(result))
    failed = resumed.prompt_once("explore", [f.tool])
    assert not failed.success and "unresolved tools" in str(failed.error)
    assert len(f.calls) == 1 and not f.tools


def test_committed_tool_result_checkpoint_is_recoverable(tmp_path, monkeypatch):
    def stop(event, state):
        if event["kind"] == "tools_complete":
            raise RuntimeError("injected interruption after durable result")

    f = fixture(tmp_path, monkeypatch, [reply(tool=True), reply()], stop)
    model = f.model()
    result = model.prompt_once("explore", [f.tool])
    assert not result.success and not result.session_state["pending_tools"]
    resumed = f.model()
    resumed.restore(model.checkpoint(result))
    assert resumed.prompt_once("explore", [f.tool]).success
    assert len(f.tools) == 1 and len(f.calls) == 2


@pytest.mark.parametrize("status,retryable", [(401, False), (400, False), (503, True), (429, True)])
def test_http_failure_preserves_state_and_has_no_sdk_retry(tmp_path, monkeypatch, status, retryable):
    from ramulator_chia.framework.failures import transient_error

    f = fixture(tmp_path, monkeypatch, [status])
    result = f.model().prompt_once("explore")
    assert not result.success and len(f.calls) == 1
    assert result.session_state["open_user_message"] == "explore"
    assert transient_error(result.error) == retryable


def test_stream_disconnect_retains_partial_blocks_as_evidence(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, [sse(reply(), incomplete=True)])
    result = f.model().prompt_once("explore")
    assert not result.success
    partial = [e for e in f.events if e["kind"] == "partial_response"]
    assert partial[0]["payload"]["content"][0] == THINKING
    assert len(result.session_state["messages"]) == 1  # No incomplete response fed back.


def test_no_fixed_tool_round_cap(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, [reply(tool=True)] * 101 + [reply()])
    result = f.model().prompt_once("explore", [f.tool])
    assert result.success and len(f.calls) == 102 and len(f.tools) == 101


def test_provider_deadline_does_not_limit_tool_wait(tmp_path, monkeypatch):
    import mcp

    f = fixture(tmp_path, monkeypatch, [reply(tool=True), reply()])
    original = mcp.ClientSession.call_tool

    async def slow(self, *args):
        await asyncio.sleep(0.04)
        return await original(self, *args)

    monkeypatch.setattr(mcp.ClientSession, "call_tool", slow)
    model = f.model()
    model.model.timeout_seconds = 0.02  # Request deadline only; fixture tools take longer.
    assert model.prompt_once("explore", [f.tool]).success


def test_adc_only_in_trusted_transport(monkeypatch, tmp_path):
    import google.auth

    credentials = object()
    monkeypatch.setattr(google.auth, "default", lambda **_: (credentials, "ignored-project"))
    view = SimpleNamespace(research=SimpleNamespace(configuration=SimpleNamespace(
        backend=VertexClaudeBackend(kind="vertex_claude", project="fixture-project"))))
    with NativeTransport()(view, tmp_path, {}) as options:
        assert options == {"anthropic_client_kwargs": {"credentials": credentials}}


def test_native_agent_accounting_and_compressed_evidence(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, [reply(compaction=True)])
    model = f.model()
    research = SimpleNamespace(root=tmp_path)
    workspace = SimpleNamespace(research=research, iteration=1, role="proposer")
    tariff = Tariff.model_validate_json((Path(__file__).parents[1]/"configs/tariffs/fable-fixture.json").read_text())
    agent = NativeAgent(workspace, model, None, tariff=tariff)
    attempt = agent.root / "fixture-attempt"
    attempt.mkdir()
    agent.active_attempt = attempt
    result = model.prompt_once("fixture")
    receipt = agent.save_result(result, attempt)
    assert result.success and len(receipt["usage"]) == 1
    assert receipt["usage"][0]["usage"]["tokens"]["input_tokens"] == 200
    assert receipt["usage"][0]["cost"]["lower_micro_usd"] == 1534
    restored = NativeAgent(workspace, f.model(), None, tariff=tariff)
    assert restored.model.model.session_state() == model.model.session_state()
    records = [gzip.decompress(path.read_bytes()).decode() for path in attempt.rglob("*.gz")]
    assert all("PRIVATE_FIXTURE_CANARY" not in row for row in records)
    assert any("opaque-signature" in row for row in records)
