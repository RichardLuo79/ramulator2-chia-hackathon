"""Fable receipts, cache-aware tariff estimates, and per-iteration retrospection.

Anthropic input_tokens EXCLUDES cache reads/writes; output includes thinking.
No subscription invoice or quota-consumption inference is made here.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import time

from tools.chia_loop import recovery as R, real_core as P
from tools.chia_loop.core import atomic_write_json

MODEL = "claude-fable-5-1"
SCHEMA = 1
TARIFF = {"source": "https://platform.claude.com/docs/en/models/fable-5-1/overview",
          "checked_utc": "2026-09-06", "units": "USD per million tokens",
          "input": 10.0, "output": 50.0, "cache_read": 0.25,
          "cache_write_5m": 12.5, "cache_write_1h": 20.0,
          "guard_input": 20.0, "guard_output": 50.0}


def normalized(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuntimeError("invalid provider usage")
    def count(key, source=raw, required=False):
        value = source.get(key)
        if value is None and not required:
            return None
        if type(value) is not int or value < 0:
            raise RuntimeError("invalid token count")
        return value
    inp, out = count("input_tokens", required=True), count("output_tokens", required=True)
    read, write = count("cache_read_input_tokens"), count("cache_creation_input_tokens")
    details = raw.get("cache_creation") or {}
    short = count("ephemeral_5m_input_tokens", details)
    long = count("ephemeral_1h_input_tokens", details)
    if short is not None and long is not None and write != short + long:
        raise RuntimeError("cache token counts disagree")
    return {"uncached_input_tokens": inp, "output_tokens": out,
            "cached_input_tokens": read, "cache_write_input_tokens": write,
            "cache_write_5m_tokens": short, "cache_write_1h_tokens": long,
            "input_tokens": None if read is None or write is None else inp + read + write,
            "reasoning_output_tokens": None}


def priced(tokens):
    if tokens is None:
        return None, ["usage_not_reported"]
    assumptions = []
    if any(tokens[k] is None for k in ("cached_input_tokens", "cache_write_input_tokens")):
        return None, ["cache_counters_missing; total_cost_unknown"]
    written, short, long = (tokens[k] for k in ("cache_write_input_tokens", "cache_write_5m_tokens", "cache_write_1h_tokens"))
    if short is None or long is None:
        # Claude uses five-minute caching by default, but don't assert it for
        # an unreported TTL. Report a conservative upper estimate explicitly.
        short, long = 0, written
        if written:
            assumptions.append("unreported_cache_TTL_priced_at_1h_upper_rate")
    cost = (tokens["uncached_input_tokens"] * TARIFF["input"]
            + tokens["cached_input_tokens"] * TARIFF["cache_read"]
            + short * TARIFF["cache_write_5m"] + long * TARIFF["cache_write_1h"]
            + tokens["output_tokens"] * TARIFF["output"]) / 1e6
    return cost, assumptions


def dimensions(operation):
    match = re.fullmatch(r"proposal_(\d+)_(\d+)", operation)
    if match:
        return {"iteration": int(match[1]), "turn": int(match[2])}
    match = re.fullmatch(r"review_fable_(\d+)_d(\d+)_(\d+)", operation)
    return {"iteration": int(match[1]), "draft": int(match[2]), "review_format_attempt": int(match[3]) + 1} if match else {}


class Ledger:
    def __init__(self, root, cap=None):
        self.root, self.path = pathlib.Path(root), pathlib.Path(root) / "ledger.json"
        if cap is not None:
            raise ValueError("Fable v1 is explicitly iteration-guarded, not USD-guarded")

    def transaction(self, update):
        with R.exclusive_lock(self.path.with_suffix(".lock")):
            config = R.read_json(self.root / "claude_config.json")
            if (config.get("model") != MODEL or config.get("run_id") != self.root.name
                    or config.get("guard_mode") != "iterations" or config.get("usd_cap") is not None
                    or config.get("auth_mode") != "claude_subscription"
                    or type(config.get("maximum_iterations")) is not int or config["maximum_iterations"] < 1):
                raise RuntimeError("iteration-only usage needs explicit run authorization")
            data = R.read_json(self.path) if self.path.exists() else {
                "run_id": self.root.name, "model": MODEL, "effort": config["effort"], "schema_version": SCHEMA,
                "tariff": TARIFF, "cap_usd": None, "invoice": False, "calls": []}
            if any(data.get(k) != v for k, v in {"run_id": self.root.name, "model": MODEL,
                    "effort": config["effort"], "tariff": TARIFF, "cap_usd": None}.items()):
                raise RuntimeError("ledger owner/model/effort/tariff changed")
            result = update(data)
            atomic_write_json(self.path, data)
            return result

    def reserve(self, request, role, operation, *, attempt_path, attempt_number, **unused):
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True).encode()
        from .transport import MAX_INPUT_BYTES, MAX_OUTPUT
        if len(encoded) > MAX_INPUT_BYTES:
            raise P.BudgetExhausted("input byte safety limit; no silent compaction")
        reservation = ((len(encoded) + 8192) * TARIFF["guard_input"] + MAX_OUTPUT * TARIFF["guard_output"]) / 1e6
        def update(data):
            index = len(data["calls"])
            data["calls"].append({"id": index, "operation": operation, "role": role,
                "attempt_path": attempt_path, "attempt_number": attempt_number,
                "model": request["model"], "effort": request["output_config"]["effort"],
                "request_sha256": P.sha(encoded), "reserved_at": time.time(),
                "reserved_usd": reservation, "cap_charge_usd": reservation,
                "known_standard_usd": None, "state": "unknown_reservation_retained", **dimensions(operation)})
            return index
        return self.transaction(update)

    def settle(self, index, response, *, response_sha256):
        tokens = normalized(response.get("usage"))
        cost, assumptions = priced(tokens)
        def update(data):
            row = data["calls"][index]
            fields = {"response_id": response.get("id"), "response_model": response.get("model"),
                      "stop_reason": response.get("stop_reason"), "usage": response.get("usage"),
                      "response_sha256": response_sha256}
            if "settled_at" in row:
                if any(row.get(k) != v for k, v in fields.items()):
                    raise RuntimeError("cannot change settled response or usage")
                return
            charge = row["reserved_usd"] if tokens is None or tokens["input_tokens"] is None else (
                tokens["input_tokens"] * TARIFF["guard_input"] + tokens["output_tokens"] * TARIFF["guard_output"]) / 1e6
            if charge > row["reserved_usd"]:
                raise RuntimeError("usage exceeds reservation")
            row.update(fields, tokens=tokens, cost_assumptions=assumptions, cap_charge_usd=charge,
                       known_standard_usd=cost, settled_at=time.time(),
                       state="usage_recorded" if cost is not None else "unknown_reservation_retained")
        self.transaction(update)

    def totals(self):
        data = R.read_json(self.path) if self.path.exists() else {"calls": []}
        if data.get("run_id", self.root.name) != self.root.name or data.get("model", MODEL) != MODEL:
            raise RuntimeError("usage owner changed")
        return {"attempts": len(data["calls"]), "cap_usd": None, "guard_mode": "iterations",
                "known_standard_usd": sum(r.get("known_standard_usd") or 0 for r in data["calls"]),
                "cap_charge_usd": sum(r["cap_charge_usd"] for r in data["calls"]),
                "unknown_usage_calls": sum(r["state"] != "usage_recorded" for r in data["calls"])}


def report(root):
    root = pathlib.Path(root)
    config = R.read_json(root / "claude_config.json")
    totals = Ledger(root).totals()
    ledger = R.read_json(root / "ledger.json")
    from .transport import response_bytes
    from .stream import terminal_response
    issues = []
    for index, row in enumerate(ledger["calls"]):
        if row.get("id") != index:
            raise RuntimeError("nonconsecutive usage ledger")
        attempt = (root / row["attempt_path"]).resolve()
        if not attempt.is_relative_to(root.resolve() / "interactions"):
            raise RuntimeError("usage attempt escapes this run")
        request_path, response_path = attempt / "provider_request.json", attempt / "provider_response.sse"
        if R.exists(request_path):
            request = R.read_json(request_path)
            if P.sha(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()) != row["request_sha256"]:
                issues.append({"call": index, "issue": "request_hash_mismatch"})
        else:
            issues.append({"call": index, "issue": "request_missing"})
        if row.get("settled_at"):
            try:
                raw = response_bytes(response_path)
                response = terminal_response(raw)
                if P.sha(raw) != row["response_sha256"] or response.get("usage") != row["usage"]:
                    issues.append({"call": index, "issue": "settled_response_mismatch"})
            except (OSError, RuntimeError, ValueError):
                issues.append({"call": index, "issue": "settled_response_unreadable"})
    return {"record_type": "individual_fable_usage_report", "run_id": root.name,
            "model": MODEL, "effort": config["effort"], "tariff": TARIFF,
            "invoice": False, "quota_measured": False, "totals": totals, "calls": ledger["calls"], "audit_issues": issues,
            "by_iteration": {str(i): {"calls": len(rows), "known_standard_usd": sum(r.get("known_standard_usd") or 0 for r in rows),
                             "unknown_usage_calls": sum(r["state"] != "usage_recorded" for r in rows)}
                             for i in sorted({r["iteration"] for r in ledger["calls"] if "iteration" in r})
                             for rows in [[r for r in ledger["calls"] if r.get("iteration") == i]]},
            "by_role": {role: {"calls": len(rows), "known_standard_usd": sum(r.get("known_standard_usd") or 0 for r in rows)}
                        for role in ("proposal", "review") for rows in [[r for r in ledger["calls"] if r["role"] == role]]}}


def write_report(root):
    root = pathlib.Path(root)
    result = report(root)
    target = root / "reports/llm_usage"
    atomic_write_json(target / "summary.json", result)
    (target / "summary.md").write_text(f"# Fable LLM usage\n\nRun: `{root.name}`; effort: `{result['effort']}`.\n\n"
        f"Known API-equivalent estimate: ${result['totals']['known_standard_usd']:.6f}; "
        f"unknown-usage calls: {result['totals']['unknown_usage_calls']}.\n\n"
        "No USD stopping cap. Estimates are not subscription charges or quota measurements. "
        "Cache input is counted separately; thinking is included in output, not added again. "
        "Per-call cache-TTL assumptions and reviewer usage are retained in summary.json.\n")
    rows = [{**{k: v for k, v in r.items() if k not in {"usage", "tokens"}}, **(r.get("tokens") or {})} for r in result["calls"]]
    with (target / "calls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}) or ["id"])
        writer.writeheader()
        writer.writerows(rows)
    return result
