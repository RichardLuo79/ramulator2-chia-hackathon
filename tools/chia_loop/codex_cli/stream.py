"""Assemble finalized Responses output without trusting unfinished deltas.

Some Codex-auth streams leave the terminal response's output array empty;
response.output_item.done carries the complete items instead. Raw SSE bytes
remain the evidence: this module only creates a validated in-memory view.
"""
from __future__ import annotations

import json


def events(raw):
    # SSE frames may use CRLF, comments, event labels and multi-line data.
    data = []
    for line in raw.decode("utf-8").splitlines() + [""]:
        if not line:
            if data:
                payload = "\n".join(data)
                data = []
                if payload != "[DONE]":
                    value = json.loads(payload)
                    if not isinstance(value, dict):
                        raise RuntimeError("provider SSE event is not an object")
                    yield value
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


def terminal_response(raw):
    terminal, added, done, response_id = None, {}, {}, None
    text_done = {}
    for event in events(raw):
        kind = event.get("type", "")
        if terminal is not None:
            raise RuntimeError("provider emitted events after its terminal response")
        if kind == "error":
            raise RuntimeError("provider stream contains an error event")
        if kind in {"response.created", "response.in_progress", "response.completed",
                    "response.incomplete", "response.failed"}:
            response = event.get("response", {})
            current_id = response.get("id")
            if current_id is not None:
                if response_id is not None and current_id != response_id:
                    raise RuntimeError("provider stream response identity changed")
                response_id = current_id
            if kind in {"response.completed", "response.incomplete", "response.failed"}:
                if response.get("status") != kind.removeprefix("response."):
                    raise RuntimeError("terminal event and response status disagree")
                terminal = response
        elif kind in {"response.output_item.added", "response.output_item.done"}:
            index, item = event.get("output_index"), event.get("item")
            if type(index) is not int or index < 0 or not isinstance(item, dict):
                raise RuntimeError("invalid streamed output item/index")
            target = added if kind.endswith("added") else done
            if index in target:
                raise RuntimeError("duplicate streamed output item")
            target[index] = item
        elif kind == "response.output_text.done":
            key = (event.get("output_index"), event.get("content_index"))
            if any(type(n) is not int or n < 0 for n in key) or not isinstance(event.get("text"), str):
                raise RuntimeError("invalid finalized output text")
            if key in text_done:
                raise RuntimeError("duplicate finalized output text")
            text_done[key] = event
    if terminal is None:
        raise RuntimeError("provider did not return exactly one terminal response")
    output = terminal.get("output", [])
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        raise RuntimeError("invalid terminal output array")
    complete = terminal["status"] == "completed"
    for index, item in done.items():
        if index in added and any(added[index].get(k) != item.get(k) for k in ("id", "type")):
            raise RuntimeError("streamed item identity changed")
        if output and (index >= len(output) or output[index] != item):
            raise RuntimeError("streamed and terminal output disagree")
    if not output and done:
        if complete and sorted(done) != list(range(len(done))):
            raise RuntimeError("completed stream has missing output items")
        output = [done[index] for index in sorted(done)]
    if complete:
        for index, item in added.items():
            if index >= len(output) or any(output[index].get(k) != item.get(k) for k in ("id", "type")):
                raise RuntimeError("completed stream has an unfinished output item")
        for (index, part), event in text_done.items():
            if (index >= len(output) or output[index].get("id") != event.get("item_id")
                    or part >= len(output[index].get("content", []))
                    or output[index]["content"][part].get("type") != "output_text"
                    or output[index]["content"][part].get("text") != event["text"]):
                raise RuntimeError("finalized text and output item disagree")
        for item in output:
            if item.get("type") == "message" and (
                    item.get("role") != "assistant" or item.get("status") != "completed"):
                raise RuntimeError("completed stream contains an incomplete/non-assistant message")
    return {**terminal, "output": output}
