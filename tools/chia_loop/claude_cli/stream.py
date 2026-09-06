"""Reconcile Anthropic SSE blocks; never treat a truncated stream as an answer."""
from __future__ import annotations

import json


def terminal_response(raw):
    text = raw.decode("utf-8").replace("\r\n", "\n")
    message, blocks, opened, closed = None, {}, set(), set()
    terminal = False
    for frame in text.split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
        if not data:
            continue
        try:
            event = json.loads(data)
        except ValueError:
            raise RuntimeError("invalid or truncated SSE frame") from None
        kind = event.get("type")
        if kind == "ping":
            continue
        if terminal:
            raise RuntimeError("events after terminal message")
        if kind == "message_start":
            if message is not None:
                raise RuntimeError("duplicate message start")
            message = event["message"]
            if message.get("type") != "message" or message.get("role") != "assistant" or message.get("content"):
                raise RuntimeError("unexpected message start")
        elif message is None:
            raise RuntimeError("stream has no message start")
        elif kind == "content_block_start":
            index = event["index"]
            if type(index) is not int or index != len(blocks):
                raise RuntimeError("duplicate or unordered block")
            block = dict(event["content_block"])
            if block.get("type") not in {"text", "thinking", "redacted_thinking"}:
                raise RuntimeError("native tool output is forbidden")
            blocks[index] = block
            opened.add(index)
        elif kind == "content_block_delta":
            index, delta = event["index"], event["delta"]
            if index not in opened or index in closed:
                raise RuntimeError("delta outside open block")
            field = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}.get(delta.get("type"))
            expected = "text" if field == "text" else "thinking"
            if field is None or blocks[index]["type"] != expected:
                raise RuntimeError("unexpected block delta")
            blocks[index][field] = blocks[index].get(field, "") + delta[field]
        elif kind == "content_block_stop":
            index = event["index"]
            if index not in opened or index in closed:
                raise RuntimeError("invalid block stop")
            closed.add(index)
        elif kind == "message_delta":
            delta = event.get("delta", {})
            if "stop_reason" in delta and message.get("stop_reason") is not None:
                raise RuntimeError("duplicate stop reason")
            message.update(delta)
            message["usage"] = {**message.get("usage", {}), **event.get("usage", {})}
        elif kind == "message_stop":
            if opened != closed or not message.get("stop_reason"):
                raise RuntimeError("unfinished message")
            terminal = True
        else:
            raise RuntimeError("unexpected provider event")
    if not terminal:
        raise RuntimeError("missing terminal message_stop")
    message["content"] = [blocks[i] for i in range(len(blocks))]
    return message


def action(response):
    if response.get("stop_reason") != "end_turn":
        raise RuntimeError("incomplete or nonfinal provider answer")
    text = "".join(b["text"] for b in response["content"] if b["type"] == "text").strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        # The model receives a formatting error, not an invented valid patch.
        return {"status": "invalid_response", "error": "Return exactly one complete JSON object."}
    return value if isinstance(value, dict) else {"status": "invalid_response", "error": "JSON object required."}


def readable_thinking(response):
    parts = [b["thinking"] for b in response["content"] if b["type"] == "thinking" and b.get("thinking")]
    opaque = any(b["type"] == "redacted_thinking" or b.get("signature") for b in response["content"])
    return {"availability": "returned" if parts else "not_returned", "text": parts,
            "opaque_blocks_present": opaque, "hidden_reasoning_accessed": False}
