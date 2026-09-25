"""Context compaction uses real ADK; all inference is an inert local fixture."""

import asyncio
import copy
import json
import os
import socket
from types import SimpleNamespace

import pytest
from google.genai import types

from ramulator_chia.framework import compaction
from ramulator_chia.framework.compaction import AdkCompactor, native_dict, tool_groups
from ramulator_chia.framework.config import ContextCompaction


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("compaction tests cannot contact a provider")

    monkeypatch.setattr(socket.socket, "connect", denied)


def policy(vertex=True, **changes):
    return ContextCompaction.model_validate(
        {
            "adk_python": os.environ.get("CHIA_TEST_ADK_PYTHON", "/unused/adk/python"),
            "input_limit_tokens": 10_000,
            "token_counter": "vertex_count_tokens" if vertex else "utf8_upper_bound",
            "retain_tool_rounds": 2,
            **changes,
        }
    )


def history(vertex):
    if not vertex:
        values = [
            {"role": "system", "content": "unchanged system"},
            {"role": "user", "content": "unchanged task"},
        ]
        for i in range(8):
            values += [
                {
                    "role": "assistant",
                    "reasoning_content": "reasoning " + str(i),
                    "tool_calls": [
                        {
                            "id": str(i),
                            "type": "function",
                            "function": {"name": "inspect", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": str(i), "content": "x" * 2000},
            ]
        return values
    values = [types.Content(role="user", parts=[types.Part(text="unchanged task")])]
    for i in range(8):
        values += [
            types.Content(
                role="model",
                parts=[
                    types.Part(text="exposed thought " + str(i), thought=True),
                    types.Part(
                        thought_signature=b"opaque-native-signature",
                        function_call=types.FunctionCall(
                            id=str(i), name="inspect", args={"candidate": i}
                        ),
                    ),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=str(i), name="inspect", response={"full": "x" * 2000}
                        )
                    )
                ],
            ),
        ]
    return values


def adapter():
    events = []
    return SimpleNamespace(
        model="fixture",
        timeout_seconds=20,
        _pending_tools=False,
        system_message="system instructions",
        events=events,
        _native_event=lambda kind, payload: events.append(
            {"kind": kind, "payload": copy.deepcopy(payload)}
        ),
    )


def client(fail=False):
    def response(**kwargs):
        if fail:
            raise RuntimeError("fixture provider failure")
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="fact summary")]),
                    finish_reason=types.FinishReason.STOP,
                )
            ],
            model_version="fixture",
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100, candidates_token_count=10
            ),
        )

    async def chat(**kwargs):
        from openai.types.chat import ChatCompletion

        if fail:
            raise RuntimeError("fixture provider failure")
        return ChatCompletion.model_validate(
            {
                "id": "fixture",
                "created": 1,
                "model": "fixture",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "fact summary"},
                    }
                ],
            }
        )

    return SimpleNamespace(
        models=SimpleNamespace(
            generate_content=response,
            count_tokens=lambda **kw: SimpleNamespace(
                total_tokens=len(json.dumps([native_dict(x) for x in kw["contents"]]))
            ),
        ),
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat)),
    )


async def fake_adk(policy, model, records, request, timeout):
    response = await request(
        {"contents": [{"role": "user", "parts": [{"text": "\n".join(records)}]}]}
    )
    return {"actions": {"compaction": {"compacted_content": response["content"]}}}


@pytest.mark.parametrize("vertex", [True, False])
def test_summary_keeps_native_suffix_and_uses_same_settings(monkeypatch, vertex):
    monkeypatch.setattr(compaction, "adk_summary", fake_adk)
    values, original = history(vertex), copy.deepcopy(history(vertex))
    model = adapter()
    config = (
        types.GenerateContentConfig(
            system_instruction="fixed", thinking_config={"thinking_level": "HIGH"}
        )
        if vertex
        else {
            "model": model.model,
            "messages": values,
            "reasoning_effort": "max",
            "extra_body": {"thinking": {"type": "enabled"}},
            "tools": [],
        }
    )
    asyncio.run(AdkCompactor(policy(vertex), vertex)(model, client(), values, config))
    assert [native_dict(x) for x in values[-4:]] == [native_dict(x) for x in original[-4:]]
    assert native_dict(values[0]) == native_dict(original[0])
    if not vertex:
        assert values[1] == original[1]
    assert len(values) < len(original)
    requests = [e["payload"] for e in model.events if e["kind"] == "request"]
    assert requests and all(
        r["model"] == model.model and r["purpose"] == "context_compaction" for r in requests
    )
    for r in requests:
        assert not r.get("tools")
        if vertex:
            assert r["config"]["thinking_config"]["thinking_level"] == "HIGH"
            assert not r["config"].get("tools")
        else:
            assert r["reasoning_effort"] == "max"
    assert len(requests) == sum(e["kind"] == "response" for e in model.events)
    assert any(e["kind"] == "compaction_completed" for e in model.events)


@pytest.mark.parametrize("vertex", [True, False])
def test_failure_does_not_remove_any_history(monkeypatch, vertex):
    monkeypatch.setattr(compaction, "adk_summary", fake_adk)
    values = history(vertex)
    before = copy.deepcopy(values)
    config = types.GenerateContentConfig() if vertex else {"messages": values, "model": "fixture"}
    model = adapter()
    with pytest.raises(RuntimeError, match="provider failure"):
        asyncio.run(AdkCompactor(policy(vertex), vertex)(model, client(True), values, config))
    assert values == before
    assert model.events[-1]["kind"] == "request_error"


@pytest.mark.parametrize("vertex", [True, False])
def test_below_threshold_makes_no_summary_call(vertex):
    values = history(vertex)[:1]
    config = types.GenerateContentConfig() if vertex else {"messages": values}
    model = adapter()
    asyncio.run(AdkCompactor(policy(vertex), vertex)(model, client(), values, config))
    assert [e["kind"] for e in model.events] == ["context_count"]


@pytest.mark.parametrize("vertex", [True, False])
def test_pending_or_incomplete_tool_groups_are_not_discarded(vertex):
    values = history(vertex)
    with pytest.raises(ValueError, match="unresolved"):
        tool_groups(values[:-1], vertex)
    model = adapter()
    model._pending_tools = True
    with pytest.raises(ValueError, match="pending tool"):
        asyncio.run(AdkCompactor(policy(vertex), vertex)(model, client(), values, None))


def test_unicode_chunking_is_lossless():
    text = "ASCII ★ 中文 " * 100
    chunks = list(compaction.text_chunks(text, 31))
    assert "".join(chunks) == text
    assert all(len(c.encode()) <= 31 for c in chunks)


def test_native_loop_compacts_then_resumes_without_replaying_tools(monkeypatch):
    import test_chia_framework_model_adapters as fixtures
    from google import genai

    monkeypatch.setattr(compaction, "adk_summary", fake_adk)
    monkeypatch.setattr(fixtures, "LONG", "short tool result")
    call = types.Part(
        function_call=types.FunctionCall(id="new-call", name="dram__inspect", args={}),
        thought_signature=b"keep-this-signature",
    )
    replies = [
        fixtures.response([call]),
        RuntimeError("fixture disconnection"),
        fixtures.response(),
    ]
    f = fixtures.vertex_fixture(monkeypatch, [], mock_sdk=False)
    counts, requests = [], []

    def generate(**kw):
        requests.append(kw)
        if kw["contents"][0].parts[0].text != "unchanged task":
            return fixtures.response(text="fact summary")
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def count(**kw):
        counts.append(1)
        return client().models.count_tokens(**kw)

    monkeypatch.setattr(
        genai,
        "Client",
        lambda **kw: SimpleNamespace(
            models=SimpleNamespace(generate_content=generate, count_tokens=count),
            close=lambda: None,
        ),
    )
    model = f.model()
    model.context_compactor = AdkCompactor(
        policy(), True, anchor=lambda: {"draft/model.cpp": "exact-source"}
    )
    model.restore_session(
        {
            "contents": [native_dict(x) for x in history(True)],
            "open_user_message": "explore",
            "pending_tools": False,
        }
    )
    failed = model.prompt_once("explore", [f.tool])
    assert not failed.success and len(f.tools) == 1
    assert any(e["kind"] == "compaction_completed" for e in failed.native_events)
    assert "exact-source" in json.dumps(failed.session_state)
    assert failed.session_state["pending_tools"] is False
    replacement = f.model()
    replacement.context_compactor = AdkCompactor(policy(), True)
    replacement.restore_session(failed.session_state)
    completed = replacement.prompt_once("explore", [f.tool])
    assert completed.success and len(f.tools) == 1
    assert len(counts) >= len(requests)
    assert any(
        p.thought_signature == b"keep-this-signature"
        for c in replacement._session_contents
        for p in c.parts
    )


def test_threshold_is_not_a_second_hard_context_cap():
    model = adapter()
    values = [types.Content(role="user", parts=[types.Part(text="x" * 8500)])]
    asyncio.run(
        AdkCompactor(policy(), True)(model, client(), values, types.GenerateContentConfig())
    )
    assert len(values) == 1 and not any(e["kind"] == "request" for e in model.events)


def test_conservative_context_bound_cannot_reject_protected_review_history():
    values = [
        {"role": "system", "content": "fixed contract"},
        {"role": "user", "content": "review evidence " * 2000},
        {"role": "assistant", "content": "partial", "reasoning_content": "native " * 3000},
    ]
    before = copy.deepcopy(values)
    model = adapter()
    asyncio.run(AdkCompactor(policy(False), False)(model, client(), values, {"messages": values}))
    assert values == before
    assert [e["kind"] for e in model.events] == ["context_count", "compaction_deferred"]
    assert model.events[-1]["payload"]["history_unchanged"] is True


def test_exact_provider_count_still_rejects_oversized_protected_history():
    model = adapter()
    values = [types.Content(role="user", parts=[types.Part(text="x" * 20000)])]
    with pytest.raises(ValueError, match="protected context"):
        asyncio.run(AdkCompactor(policy(), True)(model, client(), values, types.GenerateContentConfig()))
    assert len(values) == 1


@pytest.mark.parametrize("vertex", [True, False])
def test_summary_capacity_rejection_requires_an_exact_count(monkeypatch, vertex):
    async def large_summary_prompt(policy, model, records, request, timeout):
        response = await request(
            {"contents": [{"role": "user", "parts": [{"text": "x" * 12000}]}]}
        )
        return {"actions": {"compaction": {"compacted_content": response["content"]}}}

    monkeypatch.setattr(compaction, "adk_summary", large_summary_prompt)
    values = history(vertex)
    before = copy.deepcopy(values)
    model = adapter()
    config = types.GenerateContentConfig() if vertex else {"messages": values, "model": "fixture"}
    operation = AdkCompactor(policy(vertex), vertex)(model, client(), values, config)
    if vertex:
        with pytest.raises(ValueError, match="summary input exceeds provider window"):
            asyncio.run(operation)
        assert values == before
        assert not any(e["kind"] == "request" for e in model.events)
    else:
        asyncio.run(operation)
        assert len(values) < len(before)
        assert any(e["kind"] == "compaction_completed" for e in model.events)


@pytest.mark.parametrize("vertex", [True, False])
@pytest.mark.parametrize("over_limit", [False, True])
def test_prior_summary_alone_is_not_resummarized(monkeypatch, vertex, over_limit):
    def forbidden(*args, **kwargs):
        pytest.fail("a prior summary alone must not generate another summary")

    monkeypatch.setattr(compaction, "adk_summary", forbidden)
    values = history(vertex)
    memory = compaction.MEMORY_PREFIX + "retained facts"
    replacement = (
        types.Content(role="user", parts=[types.Part(text=memory)])
        if vertex else {"role": "user", "content": memory}
    )
    values = values[:1 if vertex else 2] + [replacement] + values[-4:]
    config = types.GenerateContentConfig() if vertex else {"messages": values}
    size = (
        client().models.count_tokens(contents=values).total_tokens if vertex
        else len(compaction.canonical_json(config).encode()) + 32 * len(values)
    )
    limit = int(size * (0.9 if over_limit else 1.1))
    before = copy.deepcopy(values)
    model = adapter()
    run = lambda: asyncio.run(AdkCompactor(policy(vertex, input_limit_tokens=limit), vertex)(
        model, client(), values, config))
    if vertex and over_limit:
        with pytest.raises(ValueError, match="provider limit"):
            run()
    else:
        run()
        assert model.events[-1]["kind"] == "compaction_deferred"
        assert model.events[-1]["payload"]["reason"] == "only_prior_summary_is_eligible"
    assert values == before
    assert not any(e["kind"] == "request" for e in model.events)


@pytest.mark.parametrize("vertex", [True, False])
@pytest.mark.parametrize("over_limit", [False, True])
def test_nonshrinking_summary_retains_history(monkeypatch, vertex, over_limit):
    calls = []

    async def growing_summary(*args):
        calls.append(1)
        return {"actions": {"compaction": {"compacted_content": {
            "parts": [{"text": "s" * 1000}]
        }}}}

    monkeypatch.setattr(compaction, "adk_summary", growing_summary)
    head = 1 if vertex else 2
    values = history(vertex)[:head + 8]
    for i in (head + 1, head + 3):
        if vertex:
            values[i].parts[0].function_response.response = {"full": "old"}
        else:
            values[i]["content"] = "old"
    config = types.GenerateContentConfig() if vertex else {"messages": values}
    size = (
        client().models.count_tokens(contents=values).total_tokens if vertex
        else len(compaction.canonical_json(config).encode()) + 32 * len(values)
    )
    before = copy.deepcopy(values)
    model = adapter()
    run = lambda: asyncio.run(AdkCompactor(policy(
        vertex, input_limit_tokens=int(size * (0.9 if over_limit else 1.1))
    ), vertex)(model, client(), values, config))
    if vertex and over_limit:
        with pytest.raises(ValueError, match="provider limit"):
            run()
    else:
        run()
        event = model.events[-1]
        assert event["kind"] == "compaction_deferred"
        assert event["payload"]["reason"] == "replacement_did_not_shrink"
        assert event["payload"]["proposed_tokens"] > event["payload"]["tokens_before"]
    assert calls == [1]
    assert values == before
    assert not any(e["kind"] == "compaction_completed" for e in model.events)


@pytest.mark.skipif(not os.environ.get("CHIA_TEST_ADK_PYTHON"), reason="needs isolated ADK venv")
def test_real_adk_receives_untruncated_tools_and_returns_summary():
    observed = []

    async def request(value):
        observed.append(value)
        return {"content": {"role": "model", "parts": [{"text": "summary from fixture"}]}}

    result = asyncio.run(
        compaction.adk_summary(
            policy(), "fixture", ["x" * 5000 + "TAIL-SCORE:0.123456789"], request, 30
        )
    )
    assert len(observed) == 1
    assert "TAIL-SCORE:0.123456789" in observed[0]["contents"][0]["parts"][0]["text"]
    assert (
        result["actions"]["compaction"]["compacted_content"]["parts"][0]["text"]
        == "summary from fixture"
    )
