"""Same-run observable action index; never synthesizes hidden model reasoning."""
from __future__ import annotations

import pathlib

from tools.chia_loop import recovery as R
from tools.chia_loop.core import atomic_write_json
from . import usage as U
from .stream import terminal_response, readable_thinking
from .transport import response_bytes


def report(root):
    root = pathlib.Path(root)
    operations = []
    for directory in sorted((root / "interactions").glob("*")):
        if not R.exists(directory / "identity.json"):
            continue
        row = {"operation": directory.name, **R.read_json(directory / "identity.json"),
               **U.dimensions(directory.name), "input": str((directory / "input.json").relative_to(root)),
               "attempts": []}
        for attempt in sorted(directory.glob("attempt_*")):
            item = {"path": str(attempt.relative_to(root)), "thinking": {"availability": "not_returned"}}
            if R.exists(attempt / "receipt.json"):
                item["receipt"] = R.read_json(attempt / "receipt.json")
            if R.exists(attempt / "provider_response.sse"):
                try:
                    item["thinking"] = readable_thinking(terminal_response(response_bytes(attempt / "provider_response.sse")))
                except (RuntimeError, ValueError):
                    item["thinking"]["availability"] = "incomplete_stream"
            row["attempts"].append(item)
        if R.exists(directory / "result.json"):
            row["action"] = R.read_json(directory / "result.json")["answer"]
        operations.append(row)
    return {"record_type": "individual_fable_retrospective", "run_id": root.name,
            "operations": operations, "hidden_reasoning_accessed": False,
            "drafts": [str(p.relative_to(root)) for p in sorted((root / "candidates").glob("*/draft_*/proposal.json"))],
            "history": R.read_json(root / "state.json").get("history", []) if (root / "state.json").exists() else []}


def write_report(root):
    root = pathlib.Path(root)
    payload = report(root)
    path = root / "reports/retrospective"
    atomic_write_json(path / "summary.json", payload)
    lines = ["# Fable retrospective", "", f"Run: `{root.name}`.", "",
             "Only provider-exposed thinking and observable actions are recorded. Opaque signatures are not decoded.", "",
             "Operation inputs, raw provider streams, CLI events, diagnostics, drafts, reviews and scores remain in this run.", ""]
    for row in payload["operations"]:
        states = ", ".join(a["thinking"]["availability"] for a in row["attempts"])
        lines.append(f"- `{row['operation']}`: {row.get('action', {}).get('status', row.get('action', {}).get('verdict', 'unfinished'))}; exposed thinking: {states}.")
    (path / "summary.md").write_text("\n".join(lines) + "\n")
    return payload
