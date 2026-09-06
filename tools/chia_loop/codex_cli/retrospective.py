"""Private, deterministic iteration records; never calls or feeds an LLM.

Provider summaries and model explanations are claims, not hidden chain of
thought or evidence that a proposed mechanism caused a measured improvement.
Raw requests, streams, and evaluator receipts remain the primary evidence.
"""
from __future__ import annotations

import json
import pathlib

from tools.chia_loop import real_core as P, recovery as R
from tools.chia_loop.core import atomic_write_json
from . import transport as T, usage as U


def reasoning_summaries(raw):
    """Extract only explicitly exposed summary text, including partial streams.

    A terminal/item snapshot supersedes its streaming deltas. Encrypted
    reasoning content is deliberately not decoded or presented as readable.
    """
    parts, terminal, malformed = {}, None, 0

    def save(item_id, index, text, rank, source):
        if not isinstance(text, str):
            return
        key = (item_id, index)
        old = parts.get(key)
        if old is None or rank >= old["rank"]:
            parts[key] = {"item_id": item_id, "summary_index": index,
                          "text": text, "rank": rank, "source": source}

    def item(value, rank, source):
        if value.get("type") == "reasoning":
            for index, part in enumerate(value.get("summary", [])):
                if part.get("type") == "summary_text":
                    save(value.get("id"), index, part.get("text"), rank, source)

    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        try:
            event = json.loads(line[6:])
            kind = event.get("type")
            if kind in {"response.completed", "response.incomplete", "response.failed"}:
                terminal = event["response"].get("status")
                for value in event["response"].get("output", []):
                    item(value, 3, kind)
            elif kind in {"response.output_item.added", "response.output_item.done"}:
                item(event["item"], 2 if kind.endswith("done") else 1, kind)
            elif kind in {"response.reasoning_summary_part.added", "response.reasoning_summary_part.done"}:
                if event["part"].get("type") == "summary_text":
                    save(event.get("item_id"), event.get("summary_index", 0), event["part"].get("text"),
                         2 if kind.endswith("done") else 1, kind)
            elif kind == "response.reasoning_summary_text.done":
                save(event.get("item_id"), event.get("summary_index", 0), event.get("text"), 2, kind)
            elif kind == "response.reasoning_summary_text.delta":
                key = (event.get("item_id"), event.get("summary_index", 0))
                old = parts.get(key)
                if isinstance(event.get("delta"), str) and (old is None or old["rank"] <= 1):
                    save(*key, (old["text"] if old else "") + event["delta"], 1, kind)
        except (ValueError, KeyError, TypeError, AttributeError):
            malformed += 1  # Interrupted JSON is evidence, not a fabricated summary.
    texts = [{k: v for k, v in part.items() if k != "rank"} for part in parts.values() if part["text"]]
    return {"availability": "returned" if texts and terminal else "partial" if texts else
            "not_returned" if terminal else "unavailable", "terminal_status": terminal,
            "summaries": texts, "malformed_or_truncated_events": malformed,
            "hidden_reasoning_recorded": False}


def evidence(root, path):
    """References never follow another run's files or escaping symlinks."""
    for candidate in (path, path.with_name(path.name + ".gz")):
        if not candidate.resolve().is_relative_to(root):
            raise RuntimeError("retrospective evidence escapes this run")
        if candidate.is_file():
            return str(candidate.relative_to(root))
    return None


def optional_json(root, path):
    return R.read_json(path) if evidence(root, path) else {}


def report(root):
    """Read one run's own iteration evidence, including failures; no mutations."""
    root = pathlib.Path(root).resolve()
    if not root.is_dir():
        raise ValueError("run directory does not exist")
    config = optional_json(root, root / "codex_config.json")
    state = optional_json(root, root / "state.json")
    if config.get("run_id", root.name) != root.name or state.get("run_id", root.name) != root.name:
        raise RuntimeError("retrospective run owner mismatch")
    iterations = {}

    def iteration_record(number):
        if number not in iterations:
            directory = root / "candidates" / f"astra_{number:03d}"
            saved = optional_json(root, directory / "proposal_state.json")
            feedback = {}
            conversation = saved.get("conversation", [])
            for offset in range(1, len(conversation) - 1, 2):
                feedback[str((offset + 1) // 2)] = conversation[offset + 1]
            drafts = []
            for draft in sorted(directory.glob("draft_*")):
                paths = {name: evidence(root, draft / name) for name in (
                    "proposal.json", "atomic_controller.cpp", "static_checks.json", "build/build.json",
                    "review_input.json", "review.json", "rejection.json", "result.json")}
                drafts.append({"draft": draft.name, "evidence": {k: v for k, v in paths.items() if v},
                               "review": optional_json(root, draft / "review.json"),
                               "rejection": optional_json(root, draft / "rejection.json")})
            iterations[number] = {"iteration": number, "parent": saved.get("parent_id"),
                "parent_sha256": saved.get("parent_sha256"), "operations": [], "drafts": drafts,
                "feedback_by_proposal_turn": feedback, "proposal_state": evidence(root, directory / "proposal_state.json"),
                "outcome": next((h for h in state.get("history", []) if h["iteration"] == number), None)}
        return iterations[number]

    for history in state.get("history", []):
        iteration_record(history["iteration"])
    unassigned = []
    for directory in sorted((root / "interactions").glob("*")):
        if not directory.is_dir():
            continue
        identity = optional_json(root, directory / "identity.json")
        if identity.get("run_id", root.name) != root.name:
            raise RuntimeError("interaction owner mismatch")
        dims = U.dimensions(directory.name)
        result = optional_json(root, directory / "result.json")
        answer = result.get("answer")
        attempts = []
        for attempt in sorted(directory.glob("attempt_*")):
            paths = {name: evidence(root, attempt / name) for name in (
                "cli_request.json", "provider_request.json", "provider_response.sse", "cli_events.jsonl",
                "receipt.json", "provider_exchange.json")}
            receipt = optional_json(root, attempt / "receipt.json")
            request = optional_json(root, attempt / "provider_request.json")
            raw = T.response_bytes(attempt / "provider_response.sse") if paths["provider_response.sse"] else b""
            summaries = reasoning_summaries(raw)
            expected = receipt.get("response_sha256")
            attempts.append({"attempt": attempt.name, "started_at": receipt.get("started_at"),
                "requested_summary": (request.get("reasoning") or {}).get("summary"),
                "stage": receipt.get("stage"), "error_kind": receipt.get("error_kind"),
                "response_sha256": P.sha(raw) if raw else None,
                "receipt_hash_matches": P.sha(raw) == expected if raw and expected else None,
                "evidence": {k: v for k, v in paths.items() if v}, **summaries})
        record = {"operation": directory.name, **dims, "role": identity.get("role"),
            "effort": identity.get("effort"), "input": evidence(root, directory / "input.json"),
            "result": evidence(root, directory / "result.json"), "attempts": attempts,
            "action_and_stated_rationale": {k: v for k, v in answer.items() if k != "regions"}
                if isinstance(answer, dict) else None}
        if "iteration" in dims:
            iteration_record(dims["iteration"])["operations"].append(record)
        else:
            unassigned.append(record)
    for record in iterations.values():
        record["operations"].sort(key=lambda op: (
            min((a["started_at"] for a in op["attempts"] if a["started_at"] is not None), default=float("inf")),
            op["operation"]))
    return {"schema_version": 1, "run_id": root.name, "model": config.get("model", T.MODEL),
        "effort": config.get("effort"), "iterations": [iterations[n] for n in sorted(iterations)],
        "loop_configuration": config.get("policy", {}).get("loop_configuration"),
        "loop_configuration_sha256": config.get("policy", {}).get("loop_configuration_sha256"),
        "synthetic_diagnostic_calls": [
            {"evidence": evidence(root, p), **{k: v for k, v in optional_json(root, p).items()
                if k != "response"}}
            for p in sorted((root / "diagnostics/synthetic_calls").glob("call_*.json"))],
        "training_lineage": optional_json(root, root / "continuation.json"),
        "unassigned_operations": unassigned,
        "notes": ["Private experiment data; not automatically published or fed to an agent.",
            "Summaries and stated explanations are model claims, not a hidden chain-of-thought transcript.",
            "not_returned means no readable summary was returned, not that no reasoning occurred.",
            "A null receipt_hash_matches means no comparable receipt, not verified integrity.",
            "Use evaluator receipts and source changes to test the stated explanations."]}


def write_report(root):
    payload = report(root)
    output = pathlib.Path(root) / "reports/retrospective"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "index.json", payload)
    lines = ["# Iteration retrospective", "", f"Run: `{payload['run_id']}`. Effort: `{payload['effort']}`.", "",
             *payload["notes"], ""]
    lineage = payload["training_lineage"]
    if lineage:
        lines += [f"This is a continuation of `{lineage['source_run_id']}` after "
                  f"{lineage['completed_designs']} evaluated designs. Earlier full interactions remain "
                  "in that preserved run; missing local operations for inherited iterations are not lost evidence. "
                  "See `../../continuation.json` for source hashes, reproduced training scores and provenance. "
                  "The operator saw earlier test results; this is an exploratory extension, not a fresh untouched test.", ""]

    def block(value):
        # JSON escaping prevents model text from injecting Markdown/HTML here.
        return ["```json", json.dumps(value, indent=2, ensure_ascii=True).replace("`", "\\u0060").replace("<", "\\u003c"), "```", ""]

    def link(label, relative):
        return f"[{label}](<../../{relative}>)" if relative else f"{label}: unavailable"

    for row in payload["iterations"]:
        lines += [f"## Iteration {row['iteration']}", "", f"Parent: `{row['parent']}` / `{row['parent_sha256']}`.", "",
                  link("Proposal checkpoint and diagnostic feedback", row["proposal_state"]), "", "Recorded outcome:", "",
                  *block(row["outcome"])]
        for op in row["operations"]:
            lines += [f"### {op['operation']}", "", f"Role: `{op['role']}`; effort: `{op['effort']}`.", "",
                      link("Full input", op["input"]) + "; " + link("Full result", op["result"]), "",
                      "Action and stated rationale (region bodies remain in the linked original):", "",
                      *block(op["action_and_stated_rationale"])]
            for attempt in op["attempts"]:
                lines += [f"Attempt `{attempt['attempt']}`: `{attempt['availability']}`; requested summary: `{attempt['requested_summary']}`.", "",
                          *block(attempt["summaries"]),
                          "; ".join(link(k, v) for k, v in attempt["evidence"].items()), ""]
        lines += ["Drafts, reviews and rejections:", "", *block(row["drafts"])]
    (output / "index.md").write_text("\n".join(lines))
    return payload
