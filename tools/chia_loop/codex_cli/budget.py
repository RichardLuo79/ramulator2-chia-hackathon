"""Financial-only continuity for an explicitly authorized clean Astra restart.

Read no prior prompts, source, actions, scores or test outcomes. The successor
receives only usage totals and provenance; each predecessor has one successor.
"""
from __future__ import annotations

import contextlib
import math
import pathlib
import time

from tools.chia_loop import real_core as P, recovery as R
from tools.chia_loop.core import atomic_write_json
from . import usage as U


def limits(maximum_iterations, cap, cpus, *, iteration_guard=False, auth_mode="chatgpt"):
    """Only an explicit ChatGPT experiment can replace the dollar stop."""
    from tools.chia_loop.run_records import validate_limits
    if iteration_guard:
        if cap is not None or auth_mode != "chatgpt":
            raise ValueError("iteration-only authorization requires ChatGPT mode and no USD cap")
        result = validate_limits(maximum_iterations, 1.0, cpus)
        result["usd_cap"] = None
        return result
    return validate_limits(maximum_iterations, cap, cpus)


def check_iteration_authorization(root):
    path = pathlib.Path(root) / "codex_config.json"
    if not path.is_file():
        raise ValueError("iteration-only accounting needs an explicit run authorization")
    config = R.read_json(path)
    if (config.get("run_id") != pathlib.Path(root).name or config.get("guard_mode") != "iterations"
            or config.get("auth_mode") != "chatgpt" or config.get("usd_cap") is not None):
        raise ValueError("iteration-only accounting needs an explicit run authorization")
    limits(config.get("maximum_iterations"), None, config.get("cpu_budget"), iteration_guard=True)


def validate(receipt, root, model, cap):
    if (receipt.get("schema_version") != 1 or receipt.get("run_id") != root.name
            or receipt.get("model") != model or receipt.get("cap_usd") != cap):
        raise RuntimeError("financial carryover owner/model/cap mismatch")
    totals = receipt["totals"]
    for key in ("known_standard_usd", "conservative_guard_usd"):
        value = totals.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError("invalid financial carryover amount")
    if cap is not None and totals["conservative_guard_usd"] > cap:
        raise RuntimeError("financial carryover already exceeds the authorization")
    for key in ("generation_attempts", "usage_reported_calls", "usage_unknown_calls"):
        if type(totals.get(key)) is not int or totals[key] < 0:
            raise RuntimeError("invalid financial carryover attempt count")
    if totals["generation_attempts"] != totals["usage_reported_calls"] + totals["usage_unknown_calls"]:
        raise RuntimeError("financial carryover attempt counts disagree")
    return receipt


def load(root, model, cap):
    root = pathlib.Path(root)
    path = root / "budget_carryover.json"
    config_path = root / "codex_config.json"
    config = R.read_json(config_path) if config_path.exists() else {}
    expected = config.get("budget_carryover_sha256")
    if expected and (not path.exists() or P.sha(path.read_bytes()) != expected):
        raise RuntimeError("pinned financial carryover changed or disappeared")
    if not path.exists():
        return None
    receipt = validate(R.read_json(path), root, model, cap)
    if config and receipt["effort"] != config["effort"]:
        raise RuntimeError("financial carryover effort mismatch")
    return receipt


def prepare(root, source, *, model, effort, cap, iteration_guard=False):
    """Bind a quiescent predecessor's ledger, without modifying its evidence."""
    root, source = pathlib.Path(root).resolve(), pathlib.Path(source).resolve(strict=True)
    if cap is None and not iteration_guard:
        raise ValueError("an iteration-only successor needs explicit authorization")
    if iteration_guard and cap is not None:
        raise ValueError("iteration-only authorization cannot have a USD cap")
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("financial predecessor must be a separate run directory")
    with contextlib.ExitStack() as stack:
        for path in (source / ".supervisor.lock", source / ".runner.lock",
                     source.with_name(source.name + ".queue.lock")):
            if not path.exists():
                if path.name.endswith(".queue.lock"):
                    continue
                raise RuntimeError("financial predecessor has no lifecycle lock")
            stack.enter_context(R.exclusive_lock(path))
        supervisor = R.read_json(source / "supervisor_state.json")
        config = R.read_json(source / "codex_config.json")
        ledger = R.read_json(source / "ledger.json")
        if (supervisor.get("run_id") != source.name
                or supervisor.get("status") not in {"completed", "stopped", "needs_attention"}):
            raise RuntimeError("financial predecessor is not quiescent")
        if supervisor["status"] != "completed" and not (source / "STOP").exists():
            raise RuntimeError("unfinished financial predecessor needs a persistent STOP")
        if (config.get("run_id") != source.name or ledger.get("run_id") != source.name
                or config.get("model") != model or ledger.get("model") != model
                or config.get("effort") != effort
                or ledger.get("cap_usd") != config.get("usd_cap")
                or (not iteration_guard and config.get("usd_cap") != cap)
                or (iteration_guard and config.get("auth_mode") != "chatgpt")
                or ledger.get("tariff") != U.TARIFF):
            raise RuntimeError("cannot transfer another effort/model/budget/tariff")
        rows = ledger["calls"]
        if [r["id"] for r in rows] != list(range(len(rows))):
            raise RuntimeError("financial predecessor ledger IDs are inconsistent")
        current = U.summarize([{**r, "tokens": U.normalized(r.get("usage"))} for r in rows])
        previous = load(source, model, config["usd_cap"])
        if ledger.get("carryover") != previous:
            raise RuntimeError("financial predecessor carryover was altered")
        totals = U.combine_totals(current, previous["totals"] if previous else U.summarize([]))
        claim = {"successor_run_id": root.name, "successor_root": str(root),
                 "source_ledger_sha256": P.sha((source / "ledger.json").read_bytes())}
        claim_path = source / "budget_successor.json"
        if claim_path.exists() and R.read_json(claim_path) != claim:
            raise RuntimeError("financial predecessor already has a different successor")
        receipt = {"schema_version": 1, "run_id": root.name, "model": model, "effort": effort,
            "cap_usd": cap, "source_run_id": source.name, "source_ledger_sha256": claim["source_ledger_sha256"],
            "source_config_sha256": P.sha((source / "codex_config.json").read_bytes()),
            "source_supervisor_sha256": P.sha((source / "supervisor_state.json").read_bytes()),
            "totals": totals, "created_at": time.time(), "financial_only": True,
            "scientific_history_imported": False, "authorization": "operator requested clean restart retaining usage"}
        if iteration_guard:
            receipt["authorization"] = "operator replaced USD stopping with a finite iteration ceiling; retain all usage"
            receipt["previous_cap_usd"] = config["usd_cap"]
            receipt["guard_mode"] = "iterations"
        validate(receipt, root, model, cap)
        if (root / "budget_carryover.json").exists() or (root / "ledger.json").exists():
            raise RuntimeError("financial carryover must precede the fresh ledger")
        atomic_write_json(claim_path, claim)
        atomic_write_json(root / "budget_carryover.json", receipt)
        return P.sha((root / "budget_carryover.json").read_bytes())
