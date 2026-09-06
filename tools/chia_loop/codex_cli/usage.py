"""Provider-reported usage and a content-free per-run audit/report.

Output tokens INCLUDE reasoning tokens. Missing details stay unknown. The
standard-tariff equivalent is not a ChatGPT invoice or a quota measurement.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import time

from tools.chia_loop import recovery as R
from tools.chia_loop.core import atomic_write_json

SCHEMA = 2
TARIFF = {"source": "https://developers.openai.com/api/docs/models/gpt-6-astra",
          "checked_utc": "2026-09-05", "units": "USD per million tokens",
          "input": 10.0, "cached_input": 1.0, "cache_write_input": 12.5, "output": 50.0,
          "long_input_threshold": 272_000, "long_input_multiplier": 2.0, "long_output_multiplier": 1.5,
          "guard_input": 25.0, "guard_output": 75.0}
TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens",
                "reasoning_output_tokens", "non_reasoning_output_tokens", "total_tokens")


def normalized(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuntimeError("invalid provider usage object")
    def count(value, required=False):
        if value is None and not required:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("invalid provider token count")
        return value
    inp, out = count(raw.get("input_tokens"), True), count(raw.get("output_tokens"), True)
    details_in, details_out = raw.get("input_tokens_details"), raw.get("output_tokens_details")
    details_in = {} if details_in is None else details_in
    details_out = {} if details_out is None else details_out
    if not isinstance(details_in, dict) or not isinstance(details_out, dict):
        raise RuntimeError("invalid provider usage details")
    cached = count(details_in.get("cached_tokens"))
    written = count(details_in.get("cache_write_tokens"))
    reasoning = count(details_out.get("reasoning_tokens"))
    total = count(raw.get("total_tokens"))
    if (cached or 0) + (written or 0) > inp or (reasoning or 0) > out or total not in (None, inp + out):
        raise RuntimeError("inconsistent provider token counts")
    return {"input_tokens": inp, "output_tokens": out, "cached_input_tokens": cached,
            "cache_write_input_tokens": written, "reasoning_output_tokens": reasoning,
            "non_reasoning_output_tokens": None if reasoning is None else out - reasoning,
            "total_tokens": inp + out, "total_tokens_derived": total is None}


def priced(tokens):
    if tokens is None:
        return None, ["usage_not_reported"]
    inp, out = tokens["input_tokens"], tokens["output_tokens"]
    cached, written = tokens["cached_input_tokens"], tokens["cache_write_input_tokens"]
    assumptions = [name + "_not_reported_assumed_zero" for name in ("cached_input_tokens", "cache_write_input_tokens")
                   if tokens[name] is None]
    factor = TARIFF["long_input_multiplier"] if inp > TARIFF["long_input_threshold"] else 1
    output_factor = TARIFF["long_output_multiplier"] if factor != 1 else 1
    uncached = inp - (cached or 0) - (written or 0)
    estimate = ((uncached * TARIFF["input"] + (cached or 0) * TARIFF["cached_input"]
                 + (written or 0) * TARIFF["cache_write_input"]) * factor
                + out * TARIFF["output"] * output_factor) / 1e6
    return estimate, assumptions


def dimensions(operation):
    proposal = re.fullmatch(r"proposal_(\d+)_(\d+)", operation)
    review = re.fullmatch(r"review_astra_(\d+)_d(\d+)_(\d+)", operation)
    if proposal:
        return {"iteration": int(proposal[1]), "turn": int(proposal[2])}
    if review:
        return {"iteration": int(review[1]), "draft": int(review[2]), "review_format_attempt": int(review[3]) + 1}
    return {}  # Fixtures/auxiliary operation IDs need not resemble iterations.


def summarize(rows):
    result = {"generation_attempts": len(rows), "usage_reported_calls": 0, "usage_unknown_calls": 0,
              "known_standard_usd": 0.0, "conservative_guard_usd": 0.0, "token_totals": {},
              "standard_estimate_assumptions": {}}
    for field in TOKEN_FIELDS:
        values = [(row.get("tokens") or {}).get(field) for row in rows]
        result["token_totals"][field] = {"known_sum": sum(v for v in values if v is not None),
            "reported_calls": sum(v is not None for v in values), "unknown_calls": sum(v is None for v in values)}
    for row in rows:
        result["usage_reported_calls" if row.get("tokens") is not None else "usage_unknown_calls"] += 1
        result["known_standard_usd"] += row.get("known_standard_usd") or 0
        result["conservative_guard_usd"] += row["cap_charge_usd"]
        for assumption in row.get("cost_assumptions", []):
            result["standard_estimate_assumptions"][assumption] = result["standard_estimate_assumptions"].get(assumption, 0) + 1
    return result


def combine_totals(current, previous):
    """Keep current-run call IDs separate while retaining prior financial use."""
    result = {}
    for key in ("generation_attempts", "usage_reported_calls", "usage_unknown_calls",
                "known_standard_usd", "conservative_guard_usd"):
        result[key] = current[key] + previous[key]
    result["token_totals"] = {key: {field: current["token_totals"][key][field] + previous["token_totals"][key][field]
                                   for field in ("known_sum", "reported_calls", "unknown_calls")}
                              for key in TOKEN_FIELDS}
    result["standard_estimate_assumptions"] = {
        key: current["standard_estimate_assumptions"].get(key, 0) + previous["standard_estimate_assumptions"].get(key, 0)
        for key in current["standard_estimate_assumptions"] | previous["standard_estimate_assumptions"]}
    return result


def report(root):
    """Read only this run; no account history, prompts, code, or held-out scores."""
    from . import transport as T
    root = pathlib.Path(root)
    config = R.read_json(root / "codex_config.json") if (root / "codex_config.json").exists() else {}
    ledger = R.read_json(root / "ledger.json") if (root / "ledger.json").exists() else {"run_id": root.name, "calls": []}
    if ledger["run_id"] != root.name or ledger.get("model", T.MODEL) != T.MODEL:
        raise RuntimeError("usage report owner/model mismatch")
    if ledger.get("carryover") != T.F.load(root, T.MODEL, ledger.get("cap_usd", config.get("usd_cap", 100))):
        raise RuntimeError("usage report financial carryover mismatch")
    rows, issues = [], []
    for entry in ledger["calls"]:
        if entry["id"] != len(rows):
            raise RuntimeError("ledger IDs are not unique and consecutive")
        row = {key: entry.get(key) for key in ("id", "operation", "role", "attempt_number", "model", "effort", "reasoning_summary",
            "auth_mode", "iteration", "turn", "draft", "review_format_attempt", "state", "reserved_at", "settled_at",
            "response_model", "response_effort", "response_status", "incomplete_reason", "service_tier", "action_status", "request_sha256",
            "response_sha256", "cap_charge_usd", "known_standard_usd", "cost_assumptions") if key in entry}
        row["tokens"] = normalized(entry.get("usage"))
        relative = entry.get("attempt_path")
        if relative:
            attempt = (root / relative).resolve()
            if not attempt.is_relative_to(root.resolve() / "interactions"):
                raise RuntimeError("attempt path escapes this run")
            if R.exists(attempt / "provider_request.json"):
                request = R.read_json(attempt / "provider_request.json")
                digest = T.P.sha(json.dumps(request, ensure_ascii=False, sort_keys=True).encode())
                if digest != row["request_sha256"]:
                    issues.append({"call": entry["id"], "issue": "request_hash_mismatch"})
            if R.exists(attempt / "provider_response.sse"):
                raw = T.response_bytes(attempt / "provider_response.sse")
                if entry.get("response_sha256") and T.P.sha(raw) != entry["response_sha256"]:
                    issues.append({"call": entry["id"], "issue": "response_hash_mismatch"})
                try:
                    response = T.terminal_response(raw)
                    if entry.get("usage") != response.get("usage"):
                        issues.append({"call": entry["id"], "issue": "saved_usage_not_settled"})
                except (RuntimeError, ValueError):
                    if entry.get("usage") is not None:
                        issues.append({"call": entry["id"], "issue": "settled_usage_without_terminal_response"})
            elif entry.get("usage") is not None:
                issues.append({"call": entry["id"], "issue": "settled_response_missing"})
            for file in ("provider_exchange.json", "receipt.json"):
                if R.exists(attempt / file):
                    metadata = R.read_json(attempt / file)
                    for key in ("started_at", "finished_at", "wall_seconds", "first_byte_seconds", "http_status",
                                "error_kind", "cli_exit_code", "cli_wall_seconds", "cli_usage_check", "cli_timeout"):
                        if key in metadata:
                            row[key] = metadata[key]
                    if "stage" in metadata:
                        row["last_recorded_stage"] = metadata["stage"]
        rows.append(row)
    attempts = list((root / "interactions").glob("*/attempt_*"))
    by_role = {role: summarize([row for row in rows if row["role"] == role]) for role in ("proposal", "review")}
    by_iteration = {str(i): summarize([row for row in rows if row.get("iteration") == i])
                    for i in sorted({row["iteration"] for row in rows if row.get("iteration") is not None})}
    cli = R.read_json(root / "cli_identity.json") if (root / "cli_identity.json").exists() else {}
    current = summarize(rows)
    carryover = ledger.get("carryover")
    return {"record_type": "individual_astra_usage_report", "schema_version": SCHEMA, "generated_at": time.time(),
            "run_id": root.name, "model": T.MODEL, "effort": config.get("effort"), "auth_mode": config.get("auth_mode"),
            "invoice": False, "account_quota_measured": False, "tariff": ledger.get("tariff", TARIFF),
            "human_intervention_policy": config.get("policy", {}).get("human_intervention"),
            "maximum_iterations": config.get("maximum_iterations"), "guard_cap_usd": config.get("usd_cap"),
            "guard_mode": config.get("guard_mode", "usd"),
            "scientific_continuation": bool(config.get("continuation_requested")),
            "cli_version": cli.get("version"), "cli_binary_sha256": cli.get("binary_sha256"),
            "reviewer_effort": config.get("policy", {}).get("reviewer_effort"),
            "cli_attempts": len(attempts), "pre_dispatch_attempts": sum(not R.exists(p / "reservation.json") for p in attempts),
            "totals": current, "carryover": carryover,
            "authorization_totals": combine_totals(current, carryover["totals"] if carryover else summarize([])),
            "by_role": by_role, "by_iteration": by_iteration, "calls": rows,
            "audit_issues": issues, "optimization_feedback": False,
            "notes": ["Reasoning is a subset of output tokens; do not add it again.",
                      "Known sums exclude unreported fields/calls; unknown is not zero.",
                      "Only provider-exposed reasoning summaries/content are retained, not hidden reasoning.",
                      "ChatGPT quota is shared with other account use; these are per-run token receipts, not quota deltas."]}


def write_report(root):
    root = pathlib.Path(root)
    payload = report(root)
    output = root / "reports/llm_usage"
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", payload)
    totals = payload["totals"]
    lines = ["# Astra LLM usage", "", f"Run: `{payload['run_id']}`; model: `{payload['model']}`; effort: `{payload['effort']}`.",
        "", "This report contains no prompt, response text, credential, or host account/session identifier.", "",
        "| Role | Attempts | Usage known | Usage unknown | Standard API-equivalent USD | Conservative guard USD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for role, values in payload["by_role"].items():
        lines.append(f"| {role} | {values['generation_attempts']} | {values['usage_reported_calls']} | {values['usage_unknown_calls']} | "
                     f"{values['known_standard_usd']:.6f} | {values['conservative_guard_usd']:.6f} |")
    lines += ["", "| Tokens | Known sum | Calls with field | Calls without field |", "| --- | ---: | ---: | ---: |"]
    for name, values in totals["token_totals"].items():
        lines.append(f"| {name} | {values['known_sum']} | {values['reported_calls']} | {values['unknown_calls']} |")
    lines += ["", *payload["notes"], "", "Dollar figures are tariff-equivalent estimates, not subscription charges or invoices.",
              f"Audit issues: {len(payload['audit_issues'])}. Full call/iteration detail and assumptions: `summary.json`.", ""]
    if payload["guard_mode"] == "iterations":
        lines += [f"Stopping authorization: at most {payload['maximum_iterations']} total evaluated designs; "
                  "no USD stopping guard. Conservative reservations and all reported usage remain recorded. "
                  "Provider quotas and operational safety limits still apply.", ""]
    if payload["carryover"]:
        total = payload["authorization_totals"]
        lines += ["This table covers the current run only. Including financial carryover, "
                  f"known API-equivalent usage is ${total['known_standard_usd']:.6f} and the conservative "
                  f"guard charge is ${total['conservative_guard_usd']:.6f}. " +
                  ("Own-run training history is separately imported under the continuation receipt."
                   if payload["scientific_continuation"] else "Prior scientific history is not imported."), ""]
    (output / "summary.md").write_text("\n".join(lines))
    flat = [{**{k: v for k, v in row.items() if k not in {"tokens", "cost_assumptions"}},
             **(row.get("tokens") or {}), "cost_assumptions": ";".join(row.get("cost_assumptions") or [])} for row in payload["calls"]]
    columns = sorted({k for row in flat for k in row}) or ["id", "operation", "role", *TOKEN_FIELDS]
    with (output / "calls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    return payload
