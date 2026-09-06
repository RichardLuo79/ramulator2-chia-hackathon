"""Operator-owned feature/feedback ablations shared by both CHIA backends.

These settings never relax atomicity, isolation, scoring, accounting or the
held-out gate. A resolved profile is immutable before the first model call.
Old roots without a profile retain the historical diagnostic interface.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib

from tools.chia_loop.core import atomic_write_json

DEFAULT = pathlib.Path(__file__).with_name("configs") / "loop_default_v1.json"
DEFAULTS = {
    "schema_version": 1, "name": "full_diagnostics_v1",
    "features": {
        "synthetic_diagnostics": True, "training_statistics": True,
        "logical_trace_diagnostics": True, "controller_trace_diagnostics": True,
        "request_extremes": True, "per_workload_feedback": True,
        "comparison_feedback": True, "evolution_history": True,
        "oracle_source": True, "comparison_source": True,
    },
    "search": {"parent_selection": "pareto_round_robin", "promotion": "two_objective_pareto"},
    "limits": {"model_turns_per_proposal": 48, "diagnostic_calls_per_proposal": 192,
               "drafts_per_iteration": 12, "initial_extreme_rows": 3},
    "synthetic": {"default_requests_per_stream": 50_000, "max_total_reads_per_case": 200_000,
                  "max_cases_per_call": 4, "max_unique_cases_per_run": 128,
                  "max_trace_rows": 100, "cpu_seconds_per_simulation": 120},
}
DIAGNOSTICS = {"stats": "training_statistics", "logical": "logical_trace_diagnostics",
               "controller": "controller_trace_diagnostics", "extremes": "request_extremes"}
ORACLE_FILES = {"src/ramulator/controller/impl/generic_ddr_controller.cpp",
                "src/ramulator/controller/scheduler/impl/frfcfs_rowhit.cpp",
                "src/ramulator/controller/controller_base.h", "src/ramulator/controller/controller_base.cpp"}
COMPARISON_FILES = {"src/ramulator/controller/impl/" + name + "_controller.cpp"
                    for name in ("fixed_lat", "md1", "wmg1", "mess")} | {
                    "src/ramulator/controller/impl/zoo_probe.h"}


def _merge(base, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError(prefix + " must be an object")
    unknown = set(override) - set(base)
    if unknown:
        raise ValueError("unknown loop configuration fields: " + prefix + ", ".join(sorted(unknown)))
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(base[key], value, prefix + key + ".") if isinstance(base[key], dict) else value
    return result


def resolve(value):
    config = _merge(DEFAULTS, value)
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("unsupported loop configuration schema")
    if not isinstance(config["name"], str) or not 1 <= len(config["name"]) <= 120:
        raise ValueError("loop configuration needs a short experiment name")
    if any(type(v) is not bool for v in config["features"].values()):
        raise ValueError("feature switches must be JSON booleans")
    if config["search"]["parent_selection"] not in ("pareto_round_robin", "incumbent"):
        raise ValueError("unsupported parent-selection policy")
    if config["search"]["promotion"] != "two_objective_pareto":
        raise ValueError("core-cycle and request-MAE Pareto promotion is mandatory")
    bounds = {"model_turns_per_proposal": (3, 96), "diagnostic_calls_per_proposal": (0, 512),
              "drafts_per_iteration": (1, 24), "initial_extreme_rows": (0, 40),
              "default_requests_per_stream": (1_000, 200_000), "max_total_reads_per_case": (1_000, 1_000_000),
              "max_cases_per_call": (1, 8), "max_unique_cases_per_run": (1, 512),
              "max_trace_rows": (0, 200), "cpu_seconds_per_simulation": (5, 600)}
    for group in ("limits", "synthetic"):
        for key, value in config[group].items():
            lo, hi = bounds[key]
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    if config["synthetic"]["default_requests_per_stream"] > config["synthetic"]["max_total_reads_per_case"]:
        raise ValueError("default synthetic population exceeds the case limit")
    return config


def read_source(path=None):
    return resolve(json.loads(pathlib.Path(path or DEFAULT).read_text()))


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def install(root, source=None):
    root = pathlib.Path(root)
    path = root / "loop_config.json"
    if path.exists() or (root / "loop_config_identity.json").exists():
        raise RuntimeError("cannot replace a prepared loop configuration")
    config = read_source(source)
    atomic_write_json(path, config)
    atomic_write_json(root / "loop_config_identity.json", {"schema_version": 1,
        "resolved_sha256": identity(config), "configuration": config})
    return config


def load(root):
    root = pathlib.Path(root)
    path, receipt = root / "loop_config.json", root / "loop_config_identity.json"
    if not path.exists() and not receipt.exists():
        for name in ("preparation_manifest.json", "run_manifest.json"):
            manifest = root / name
            if manifest.exists():
                recorded = json.loads(manifest.read_text())
                if "loop_configuration" in recorded or "loop_configuration" in recorded.get("policy", {}):
                    raise RuntimeError("prepared loop configuration is missing")
        config = copy.deepcopy(DEFAULTS)
        config["name"] = "legacy_no_synthetic"
        config["features"]["synthetic_diagnostics"] = False
        return config
    if not path.exists() or not receipt.exists():
        raise RuntimeError("incomplete frozen loop configuration")
    config, saved = resolve(json.loads(path.read_text())), json.loads(receipt.read_text())
    if saved.get("resolved_sha256") != identity(config) or saved.get("configuration") != config:
        raise RuntimeError("frozen loop configuration changed")
    return config


def enabled(root, feature):
    return load(root)["features"][feature]


def policy(root, base):
    config = load(root)
    limits = {k: v for k, v in config["limits"].items() if k != "initial_extreme_rows"}
    return {**base, **limits, "loop_configuration": config, "loop_configuration_sha256": identity(config),
            "parent_selection": config["search"]["parent_selection"]}


def visible_files(root, files):
    config = load(root)["features"]
    excluded = (set() if config["oracle_source"] else ORACLE_FILES) | (
        set() if config["comparison_source"] else COMPARISON_FILES)
    return [p for p in files if p not in excluded]


def diagnostic_kinds(root):
    features = load(root)["features"]
    return [name for name, feature in DIAGNOSTICS.items() if features[feature]]


def tools(root):
    names = ["read_file", "search_file"]
    if diagnostic_kinds(root):
        names.append("training_diagnostics")
    if enabled(root, "synthetic_diagnostics"):
        names.append("synthetic_diagnostics")
    return names


def feedback_metrics(root, metrics):
    return copy.deepcopy(metrics if enabled(root, "per_workload_feedback") else {"aggregate": metrics["aggregate"]})


def comparisons(root):
    if not enabled(root, "comparison_feedback"):
        return {}
    models = json.loads((pathlib.Path(root) / "training/reports/comparisons.json").read_text())["models"]
    return {name: feedback_metrics(root, metrics) for name, metrics in models.items()}


def tool_manifest(root, files):
    result = {"files": visible_files(root, files), "tools": tools(root),
              "training_diagnostic_kinds": diagnostic_kinds(root),
              "submission": "complete includes/code region bodies, no diff or boundary markers"}
    if enabled(root, "synthetic_diagnostics"):
        from tools.chia_loop.synthetic_diagnostics import contract
        result["synthetic_diagnostics"] = contract(root)
    return result


def initial_diagnostics(root, parent):
    from tools.chia_loop import real_eval as E
    rows = load(root)["limits"]["initial_extreme_rows"]
    if not rows or not enabled(root, "request_extremes"):
        return {}
    return {w: E.training_diagnostics(root, parent["label"], w, limit=rows) for w in E.workloads(root, "training")}


def parent(root, candidates, incumbent, iteration):
    from tools.chia_loop import real_core as P
    if load(root)["search"]["parent_selection"] == "incumbent":
        return incumbent
    archive = P.pareto(candidates)
    return archive[(iteration - 1) % len(archive)]
