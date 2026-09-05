"""Single-model records and a read-only adapter for historical paired records.

The adapter creates an in-memory reporting view, never a combined run on disk.
Historical evidence must retain its original bytes and execution semantics.
"""
from __future__ import annotations

import copy
import json
import pathlib
import re


def run_policy(policy):
    result = copy.deepcopy(policy)
    for old, new in (("iterations_per_arm", "maximum_iterations"),
                     ("usd_cap_per_arm", "usd_cap")):
        if old in result:
            result[new] = result.pop(old)
    return result


def model_pricing(pricing, backend):
    result = {key: pricing[key] for key in ("source", "checked_utc", "units") if key in pricing}
    result["standard_rates"] = pricing[backend]
    result["conservative_cap_rates"] = pricing["conservative_cap_rates"][backend]
    if backend == "flash" and "flash_introductory_pricing_until" in pricing:
        result["introductory_pricing_until"] = pricing["flash_introductory_pricing_until"]
    return result


def reporting_view(root):
    """Normalize one new run or an old paired execution for existing diagnostics."""
    root = pathlib.Path(root)
    raw = json.loads((root / "run_manifest.json").read_text())
    view = copy.deepcopy(raw)
    view["policy"] = run_policy(raw["policy"])
    if raw.get("record_type") == "optimization_run":
        if "arms" in raw or "models" in raw:
            raise ValueError("an individual run cannot contain multiple-model records")
        backend = raw["backend"]
        view["models"] = {backend: raw["model"]}
        view["arms"] = {backend: raw["state"]} if "state" in raw else {}
        view["final_test_metrics"] = {backend: raw["final_test_metrics"]} if "final_test_metrics" in raw else {}
        view["budget_carryover"] = {backend: raw.get("budget_carryover", {})}
        view["pricing"] = {backend: raw["pricing"]["standard_rates"]}
        view["run_ids"] = {backend: raw["run_id"]}
        view["run_directories"] = {backend: root}
        view["freeze_file"] = root / "selection_frozen.json"
        view["test_start_event"] = "frozen_test_started"
    else:
        view["run_ids"] = {backend: f"{root.name}__{model}"
                           for backend, model in raw["models"].items()}
        view["run_directories"] = {backend: root / "arms" / backend for backend in raw["models"]}
        view["freeze_file"] = root / "both_frozen.json"
        view["test_start_event"] = "both_frozen_test_started"
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) for run_id in view["run_ids"].values()):
        raise ValueError("unsafe run identifier")
    if len(set(view["run_ids"].values())) != len(view["run_ids"]):
        raise ValueError("duplicate run identifiers")
    return view


def frozen_selections(view):
    frozen = json.loads(view["freeze_file"].read_text())
    if view.get("record_type") == "optimization_run":
        if frozen.get("run_id") != view["run_id"]:
            raise ValueError("selection freeze belongs to a different run")
        return {view["backend"]: frozen}
    return frozen
