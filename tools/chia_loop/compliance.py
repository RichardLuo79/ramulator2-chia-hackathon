"""Automatic compliance gate, not an accuracy judge or a source-code editor."""
from __future__ import annotations

import json
import pathlib
import time

from tools.chia_loop import generation, real_core as P
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.recovery import check_stop, read_json

CHECKS = ("immutable_departures", "bounded_causal_state", "no_command_scheduler",
          "generic_configuration", "no_hidden_access", "explainable_rules")
EXPLANATION_FIELDS = ("hypothesis", "mechanism", "genericity_and_atomicity",
                      "complexity", "expected_effects", "limitations")


def review_input(source, proposal):
    # No metrics, oracle output, other candidates, workload paths or history.
    # Text supplied by the proposer is explicitly untrusted evidence.
    return {"source_sha256": P.sha(source), "source": source,
            "explanation": {key: proposal.get(key, "") for key in EXPLANATION_FIELDS}}


def validate_decision(decision, digest):
    if not isinstance(decision, dict) or decision.get("source_sha256") != digest:
        raise ValueError("review must identify the exact submitted source hash")
    checks = decision.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise ValueError("review must address every compliance rule")
    for key, result in checks.items():
        if (not isinstance(result, dict) or result.get("verdict") not in {"pass", "reject", "uncertain"}
                or not isinstance(result.get("reason"), str) or not result["reason"].strip()):
            raise ValueError("missing verdict/source-specific reason for " + key)
    verdicts = {result["verdict"] for result in checks.values()}
    overall = "reject" if "reject" in verdicts else "uncertain" if "uncertain" in verdicts else "pass"
    if decision.get("verdict") != overall:
        raise ValueError("review verdict disagrees with its rule checks")
    return overall


def automatic_review(root, source, proposal, directory, *, ledger, policy, project, iteration, turn):
    from google.genai import types
    root, directory = pathlib.Path(root), pathlib.Path(directory)
    check_stop(root)
    rubric = (root / "prompts/compliance_v1.md").read_text()
    submitted = review_input(source, proposal)
    encoded = json.dumps(submitted, sort_keys=True)
    binding = {"source_sha256": P.sha(source), "review_input_sha256": P.sha(encoded),
               "rubric_sha256": P.sha(rubric), "reviewer": P.MODELS[policy["reviewer_backend"]]}
    destination = directory / "review.json"
    if destination.exists():
        existing = read_json(destination)
        if any(existing.get(key) != value for key, value in binding.items()):
            raise RuntimeError("cached compliance review source/input/rubric/model changed")
        if existing.get("reviewer_kind") != "api_agent" or existing.get("human_intervention") is not False:
            raise RuntimeError("unattended run cannot silently accept an external review")
        if existing.get("approved") != (validate_decision(existing, binding["source_sha256"]) == "pass"):
            raise RuntimeError("cached review approval contradicts its decision")
        return existing

    atomic_write_json(directory / "review_input.json", submitted)
    contents = [types.Content(role="user", parts=[types.Part(text=encoded)])]
    explanation = ""
    for attempt in range(policy["review_response_attempts"]):
        # Replaying earlier attempts reconstructs the exact review conversation,
        # without charging again. It also retains returned thought signatures.
        raw = generation.generate(root, ledger, policy, project=project,
            backend=policy["reviewer_backend"], purpose="review", iteration=iteration, turn=turn,
            operation_key=f"review_{iteration:03d}_{directory.name}_{attempt:02d}",
            system=rubric, contents=contents)
        check_stop(root)
        decision = P.parse_provider_response(raw)
        try:
            verdict = validate_decision(decision, binding["source_sha256"])
            break
        except ValueError as exc:
            explanation = str(exc)
            content = (raw.get("candidates") or [{}])[0].get("content")
            if content and content.get("parts"):
                contents.append(types.Content.model_validate(content))
            contents.append(types.Content(role="user", parts=[types.Part(text=json.dumps({
                "schema_error": explanation, "instruction": "Return a complete review matching the fixed schema; do not change the source or rubric."}))]))
    else:
        verdict = "uncertain"
        decision = {"source_sha256": binding["source_sha256"], "verdict": verdict,
            "checks": {key: {"verdict": "uncertain", "reason": "Reviewer did not return a valid complete decision: " + explanation}
                       for key in CHECKS}}
    result = {**decision, **binding, "approved": verdict == "pass",
        "reason": "; ".join(f"{key}: {value['reason']}" for key, value in decision["checks"].items()
                            if verdict == "pass" or value["verdict"] != "pass"),
        "reviewer_kind": "api_agent", "human_intervention": False, "time": time.time(),
        "scope": "compliance only; no modeling repair, tuning advice, or accuracy selection",
        "limitation": "LLM semantic assessment plus deterministic gates; not a formal C++ correctness proof"}
    atomic_write_json(destination, result)
    return result
