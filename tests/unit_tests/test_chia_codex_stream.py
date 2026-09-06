"""Synthetic Codex-auth SSE regressions; no model calls or historical designs."""
import copy
import json

import pytest

from tools.chia_loop.codex_cli import transport as T, stream as S
from test_chia_codex_cli import fake_sse


def encode(events):
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()


def fixture(answer=None, embedded=False):
    return list(S.events(fake_sse(answer or {"status": "inspect", "requests": []},
                                 summary="Exposed fixture summary", terminal_output=embedded)))


@pytest.mark.parametrize("answer", [{"status": "inspect", "requests": []},
    {"status": "proposal", "regions": {"CODE": "fixture"}}, {"status": "no_change"},
    {"verdict": "pass", "source_sha256": "fixture", "checks": {}}])
@pytest.mark.parametrize("embedded", [False, True])
def test_finalized_items_are_the_answer_when_terminal_output_is_empty(answer, embedded):
    raw = encode(fixture(answer, embedded))
    before = raw
    response = T.completed_response(raw)
    assert T.action(response) == answer
    assert response["usage"]["output_tokens"] == 50
    assert [item["type"] for item in response["output"]] == ["reasoning", "message"]
    assert response["output"][0]["summary"][0]["text"] == "Exposed fixture summary"
    assert raw == before  # Assembly never edits the captured SSE evidence.


def test_terminal_only_api_response_remains_supported():
    events = fixture(embedded=True)
    raw = encode([events[-1]])
    assert T.action(T.completed_response(raw))["status"] == "inspect"


def test_sse_crlf_comments_and_multiline_data():
    raw = b": keepalive\r\nevent: response.completed\r\ndata:" + json.dumps(fixture(embedded=True)[-1], indent=2).replace(
        "\n", "\r\ndata: ").encode() + b"\r\n\r\ndata: [DONE]\r\n\r\n"
    assert T.action(T.completed_response(raw))["status"] == "inspect"


@pytest.mark.parametrize("corruption", ["duplicate_item", "missing_item", "conflicting_terminal",
    "changed_response_id", "unfinished_item", "text_mismatch", "terminal_status", "duplicate_terminal"])
def test_inconsistent_stream_is_not_a_model_response(corruption):
    events = fixture()
    if corruption == "duplicate_item":
        events.insert(2, copy.deepcopy(events[1]))
    elif corruption == "missing_item":
        events[2]["output_index"] = 2
    elif corruption == "conflicting_terminal":
        events[-1]["response"]["output"] = [e["item"] for e in fixture()[1:-1]]
        events[-1]["response"]["output"][1]["content"][0]["text"] = "contradiction"
    elif corruption == "changed_response_id":
        events[-1]["response"]["id"] = "different_response"
    elif corruption == "unfinished_item":
        events.insert(1, {"type": "response.output_item.added", "output_index": 2,
                          "item": {"type": "message", "id": "unfinished"}})
    elif corruption == "text_mismatch":
        events.insert(-1, {"type": "response.output_text.done", "output_index": 1, "content_index": 0,
                           "item_id": "msg_mock", "text": "contradiction"})
    elif corruption == "terminal_status":
        events[-1]["response"]["status"] = "incomplete"
    else:
        events.append(copy.deepcopy(events[-1]))
    with pytest.raises(RuntimeError):
        T.completed_response(encode(events))


def test_native_tool_cannot_hide_in_a_stream_with_empty_terminal_output():
    events = fixture()
    events.insert(-1, {"type": "response.output_item.done", "output_index": 2,
                      "item": {"type": "function_call", "name": "exec_command", "id": "tool_fixture"}})
    with pytest.raises(RuntimeError, match="native tool"):
        T.completed_response(encode(events))


@pytest.mark.parametrize("mode", ["no_terminal", "incomplete", "failed", "delta_only", "empty"])
def test_partial_or_missing_answers_never_become_evolution_feedback(mode):
    events = fixture()
    if mode == "no_terminal":
        events.pop()
    elif mode in {"incomplete", "failed"}:
        events[-1]["type"] = "response." + mode
        events[-1]["response"]["status"] = mode
        assert T.terminal_response(encode(events))["usage"]["output_tokens"] == 50
    else:
        events = [events[0], events[-1]]
        if mode == "delta_only":
            events.insert(1, {"type": "response.output_text.delta", "delta": '{"status":"no_change"}'})
    with pytest.raises(RuntimeError):
        T.completed_response(encode(events))
