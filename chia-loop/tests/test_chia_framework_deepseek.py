"""DeepSeek wire-format fixtures through CHIA, the real SDK and common hooks.

No network, real credentials, CLI processes or scientific measurements.
"""

import gzip
import json
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from test_chia_framework_campaign import no_services as no_services
from test_chia_framework_model_adapters import LONG, vertex_fixture

from ramulator_chia.framework.agent import NativeAgent
from ramulator_chia.framework.config import DeepSeekBackend
from ramulator_chia.framework.model_sessions import create_session
from ramulator_chia.framework.transport import NativeTransport
from ramulator_chia.framework.usage import Tariff, normalize, quote

pytestmark = pytest.mark.usefixtures("no_services")


def reply(*, tools=False, finish=None):
    message = {"role": "assistant", "content": "fixture answer", "reasoning_content": LONG}
    if tools:
        message["tool_calls"] = [
            {
                "id": "call-fixture",
                "type": "function",
                "function": {
                    "name": "dram__inspect",
                    "arguments": '{"draft":"current"}',
                },
            }
        ]
    return {
        "id": "chat-fixture",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish or ("tool_calls" if tools else "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 70,
            "prompt_cache_miss_tokens": 30,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 40},
            "total_tokens": 150,
        },
    }


def fixture(tmp_path, monkeypatch, responses, callback=None):
    # Share the inert MCP fixture used to qualify Gemini. The SDK and its
    # request serialization are real; MockTransport cannot open a socket.
    mcp = vertex_fixture(monkeypatch, [], mock_sdk=False)
    events, calls = [], []

    async def respond(request):
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["authorization"] == "Bearer PRIVATE_FIXTURE_CANARY"
        calls.append(json.loads(request.content))
        item = responses.pop(0)
        if isinstance(item, int):
            return httpx.Response(item, json={"error": {"message": "fixture service error"}})
        return httpx.Response(200, json=item)

    def observe(event, state):
        events.append(event)
        if callback:
            callback(event, state)

    def model(identity="one-role", model_id="deepseek-v4-flash"):
        private = tmp_path / identity
        private.mkdir(exist_ok=True)
        return create_session(
            DeepSeekBackend(kind="deepseek_api", model=model_id, reasoning_effort="max"),
            session_id=identity,
            private_directory=private,
            workspace=tmp_path,
            system_message="Fixture tools only.",
            timeout_seconds=30,
            chat_client_kwargs={
                "api_key": "PRIVATE_FIXTURE_CANARY",
                "http_client": httpx.AsyncClient(
                    transport=httpx.MockTransport(respond), trust_env=False
                ),
            },
            chat_event_callback=observe,
        )

    return SimpleNamespace(model=model, events=events, calls=calls, tools=mcp.tools, tool=mcp.tool)


def test_reasoning_tools_usage_and_exact_resume(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, [reply(tools=True), 503, reply(), reply()])
    model = f.model()
    failed = model.prompt_once("explore", [f.tool])
    assert not failed.success and failed.error.error_type == "server_error"
    assert len(f.calls) == 2 and len(f.tools) == 1  # SDK makes no hidden retries.
    checkpoint = model.checkpoint(failed)
    model = f.model()
    model.restore(checkpoint)
    assert model.prompt_once("explore", [f.tool]).success
    assert len(f.tools) == 1 and f.calls[1] == f.calls[2]  # No repeated edit or user prompt.
    assert f.calls[2]["messages"][2]["reasoning_content"] == LONG
    assert f.calls[2]["messages"][3]["tool_call_id"] == "call-fixture"
    # Rebuild the SDK client for the next native phase, as the production
    # adapter does; the injected fixture client's lifetime is one call.
    native_state = model.model.session_state()
    model = f.model()
    model.model.restore_session(native_state)
    assert model.prompt_once("reflect", [f.tool]).success
    assert all(
        m["reasoning_content"] == LONG for m in f.calls[-1]["messages"] if m["role"] == "assistant"
    )
    assert f.calls[0]["messages"] == [
        {"role": "system", "content": "Fixture tools only."},
        {"role": "user", "content": "explore"},
    ]
    assert f.calls[0]["thinking"] == {"type": "enabled"}
    assert f.calls[0]["reasoning_effort"] == "max"
    assert "max_tokens" not in f.calls[0]
    assert model.model.max_tool_iterations is None
    assert "PRIVATE_FIXTURE_CANARY" not in json.dumps(f.events)
    assert "PRIVATE_FIXTURE_CANARY" not in json.dumps(checkpoint.metadata)


def test_current_flash_name_is_preserved_on_wire(tmp_path, monkeypatch):
    response = reply()
    response["model"] = "deepseek-flash"
    f = fixture(tmp_path, monkeypatch, [response])
    assert f.model(model_id="deepseek-flash").prompt_once("explore").success
    assert f.calls[0]["model"] == "deepseek-flash"
    assert f.calls[0]["reasoning_effort"] == "max"
    assert f.calls[0]["thinking"] == {"type": "enabled"}
    assert "max_tokens" not in f.calls[0]


def test_unsettled_tool_checkpoint_is_not_replayed(tmp_path, monkeypatch):
    def fail(event, state):
        if event["kind"] == "tool_response":
            raise OSError("fixture storage failed")

    f = fixture(tmp_path, monkeypatch, [reply(tools=True)], fail)
    model = f.model()
    result = model.prompt_once("explore", [f.tool])
    assert not result.success and len(f.tools) == 1
    assert result.session_state["pending_tools"] is True
    replacement = f.model()
    replacement.restore(model.checkpoint(result))
    again = replacement.prompt_once("explore", [f.tool])
    assert not again.success and "unresolved tools" in str(again.error)
    assert len(f.calls) == len(f.tools) == 1


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_unusable_output_retains_reasoning_and_usage(tmp_path, monkeypatch, finish):
    f = fixture(tmp_path, monkeypatch, [reply(finish=finish)])
    result = f.model().prompt_once("explore")
    assert not result.success
    assert result.session_state["messages"][-1]["reasoning_content"] == LONG
    assert f.events[1]["payload"]["usage"]["completion_tokens"] == 50


def test_role_identity_is_not_transferable(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, [reply()])
    model = f.model()
    result = model.prompt_once("explore")
    with pytest.raises(ValueError, match="different settings or session"):
        f.model("reviewer").restore(model.checkpoint(result))


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_effort_is_explicit_without_silent_aliases(effort):
    assert DeepSeekBackend(kind="deepseek_api", reasoning_effort=effort).reasoning_effort == effort


@pytest.mark.parametrize(
    "extra",
    [
        {"reasoning_effort": "xhigh"},
        {"api_key": "not-config"},
        {"base_url": "https://other-provider.invalid"},
    ],
)
def test_backend_rejects_ignored_settings_and_embedded_credentials(extra):
    with pytest.raises(ValueError):
        DeepSeekBackend(kind="deepseek_api", **extra)


def test_transport_fails_closed_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    view = SimpleNamespace(
        research=SimpleNamespace(
            configuration=SimpleNamespace(backend=DeepSeekBackend(kind="deepseek_api"))
        )
    )
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        with NativeTransport()(view, tmp_path, {}):
            pytest.fail("missing key was silently replaced")


def test_cache_and_reasoning_accounting_are_disjoint():
    tokens = normalize(reply()["usage"], "deepseek_chat")["tokens"]
    assert tokens["input_tokens"] == 100
    assert tokens["cached_input_tokens"] == 70 and tokens["uncached_input_tokens"] == 30
    assert tokens["output_tokens"] == 50 and tokens["reasoning_output_tokens"] == 40
    assert tokens["non_reasoning_output_tokens"] == 10 and tokens["total_tokens"] == 150


def test_common_agent_saves_events_checkpoint_and_cost(tmp_path, monkeypatch):
    from test_chia_framework_workspace import workspace

    view = workspace(tmp_path / "campaign")
    f = fixture(tmp_path, monkeypatch, [reply()])

    @contextmanager
    def tools(view):
        yield []

    tariff = Tariff.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "model": "deepseek-v4-flash",
                "source": "fixture",
                "checked_date": "2026-09-09",
                "units": "USD per million tokens",
                "tiers": [
                    {
                        "maximum_input_tokens": None,
                        "rates": {
                            "input": "0.44",
                            "cached_input": "0.014",
                            "cache_write": "0",
                            "output": "1.32",
                        },
                    }
                ],
            }
        )
    )
    agent = NativeAgent(view, f.model(), tools, tariff=tariff)
    receipt = agent.turn("explore", {})
    assert receipt["success"] and len(receipt["usage"]) == 1
    cost = receipt["usage"][0]["cost"]
    assert cost == quote(
        normalize(reply()["usage"], "deepseek_chat"), tariff, scope="provider_request"
    )
    assert cost["lower_micro_usd"] == 80 and cost["upper_micro_usd"] == 81
    files = list((tmp_path / "campaign/native-evidence").rglob("*.gz"))
    assert files and any(p.name == "contents.json.gz" for p in files)
    for p in files:
        assert b"PRIVATE_FIXTURE_CANARY" not in gzip.decompress(p.read_bytes())


def test_deferred_compaction_resumes_native_tools_and_usage_once(tmp_path, monkeypatch):
    from test_chia_framework_workspace import workspace
    from ramulator_chia.framework.agent import INSTRUCTIONS
    from ramulator_chia.framework.compaction import AdkCompactor, MEMORY_PREFIX
    from ramulator_chia.framework.config import ContextCompaction
    from ramulator_chia.framework.identity import canonical_json

    view = workspace(tmp_path / "campaign")
    f = fixture(tmp_path, monkeypatch, [reply(tools=True), 503, reply()])
    prompt = INSTRUCTIONS["explore"] + "\n\n" + canonical_json({})
    policy = ContextCompaction(adk_python="/unused", input_limit_tokens=5000,
                               token_counter="utf8_upper_bound", retain_tool_rounds=1)

    @contextmanager
    def tools(view):
        yield [f.tool]

    model = f.model()
    model.model.context_compactor = AdkCompactor(policy, False)
    model.model.restore_session({
        "messages": [
            {"role": "system", "content": "Fixture tools only."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": MEMORY_PREFIX + "older learned facts"},
        ],
        "open_user_message": prompt,
        "pending_tools": False,
    })
    agent = NativeAgent(view, model, tools)
    with pytest.raises(RuntimeError, match="native turn failed"):
        agent.turn("explore", {})
    saved_path = next(agent.root.glob("explore-*/receipt.json"))
    saved_bytes = saved_path.read_bytes()
    saved = json.loads(saved_bytes)
    assert [u["status"] for u in saved["usage"]] == [
        "response_recorded", "failed_usage_unknown"
    ]
    events = json.loads(gzip.decompress(
        (saved_path.parent / "result.json.gz").read_bytes()
    ))["native_events"]
    assert any(e["kind"] == "compaction_deferred" for e in events)
    assert not any(e["kind"] == "compaction_started" for e in events)

    resumed = f.model()
    resumed.model.context_compactor = AdkCompactor(policy, False)
    receipt = NativeAgent(view, resumed, tools).turn("explore", {})
    assert receipt["success"]
    assert len(f.calls) == 3 and len(f.tools) == 1
    assert f.calls[1] == f.calls[2]
    assert len(receipt["usage"]) == 1
    assert receipt["usage"][0]["status"] == "response_recorded"
    assert saved_path.read_bytes() == saved_bytes
