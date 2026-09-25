"""Pure helpers for the Atomic CHIA loop; no Ray or simulator dependency."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import tempfile

EXPECTED_COMPARISONS = ("fixedlat", "md1", "wmg1", "mess")
AGENT_MUTABLE_PATHS = {"src/ramulator/controller/impl/atomic_controller.cpp"}


def atomic_write_json(path: pathlib.Path, payload: object) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(payload, stream, indent=1, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist the rename as well as the file contents for durable journals.
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_config(path: pathlib.Path) -> dict:
    config = json.loads(path.read_text())
    required = {
        "schema_version", "name", "backend", "frontend", "standard",
        "training_workloads", "validation_workloads", "comparison_models",
        "iterations", "insts_per_core", "workers", "initial_candidate",
        "dummy_max_latency_step_fraction", "archive_aux_min_bytes",
    }
    missing = required - set(config) if isinstance(config, dict) else required
    if missing:
        raise ValueError(f"loop config lacks {sorted(missing)}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported loop config schema")
    if config["backend"] != "dummy":
        raise ValueError("the proof-of-concept runner supports backend='dummy' only")
    if config["frontend"] != "simpleo3" or config["standard"] != "DDR5":
        raise ValueError("the proof-of-concept scope is exactly SimpleO3 + DDR5")
    train = config["training_workloads"]
    validation = config["validation_workloads"]
    if (not isinstance(train, list) or not train or len(train) != len(set(train)) or
            not isinstance(validation, list) or not validation or
            len(validation) != len(set(validation)) or set(train) & set(validation)):
        raise ValueError("training/validation workloads must be nonempty, unique, and disjoint")
    if tuple(config["comparison_models"]) != EXPECTED_COMPARISONS:
        raise ValueError(
            "comparison_models must preserve FixedLat, MD1, WMG1, and MESS")
    for key in ("iterations", "insts_per_core", "workers", "archive_aux_min_bytes"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["iterations"] < 2:
        raise ValueError("the smoke loop needs at least two feedback iterations")
    if not 1 <= config["workers"] <= 12:
        raise ValueError("workers must be in [1, 12]")
    initial = config["initial_candidate"]
    if (not isinstance(initial, dict) or set(initial) != {"latency"} or
            isinstance(initial["latency"], bool) or
            not isinstance(initial["latency"], int) or initial["latency"] <= 0):
        raise ValueError("initial_candidate must contain one positive integer latency")
    fraction = config["dummy_max_latency_step_fraction"]
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
            not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("dummy_max_latency_step_fraction must be in (0, 1]")
    return config


def dummy_proposal(iteration: int, current: dict, prior_result: dict | None,
                   max_step_fraction: float) -> dict:
    """Adjust only the constant intercept from median signed residual.

    The candidate skeleton has one latency degree of freedom.  Subtracting the
    median per-workload signed latency residual is therefore a direct intercept
    correction.  Bounding the step avoids chasing one short smoke sample.
    """
    latency = current.get("latency")
    if isinstance(latency, bool) or not isinstance(latency, int) or latency <= 0:
        raise ValueError("current candidate latency must be a positive integer")
    if iteration == 0:
        if prior_result is not None:
            raise ValueError("iteration zero must not have prior feedback")
        return {
            "iteration": 0,
            "candidate": {"latency": latency},
            "change_kind": "seed",
            "principle": (
                "Initialize the skeleton at the DRAM preset's intrinsic read "
                "latency; no workload-derived rule is present yet."),
            "evidence": {"prior_feedback": None},
        }
    if prior_result is None:
        raise ValueError("feedback iteration requires a prior training result")
    label = prior_result["candidate_label"]
    per_workload = prior_result["summary"]["models"][label]["per_workload"]
    residuals = []
    for workload, row in per_workload.items():
        request = row["requests"]
        residual = request["signed_mean_over_L"] * request["L_oracle_mean_lat"]
        if not math.isfinite(residual):
            raise ValueError(f"{workload}: non-finite signed residual")
        residuals.append(float(residual))
    median_residual = statistics.median(residuals)
    requested_step = int(round(median_residual))
    max_step = max(1, int(round(latency * max_step_fraction)))
    applied_step = max(-max_step, min(max_step, requested_step))
    new_latency = max(1, latency - applied_step)
    return {
        "iteration": iteration,
        "candidate": {"latency": new_latency},
        "change_kind": "bounded_intercept_correction",
        "principle": (
            "The seed has only a constant latency intercept. Subtract the median "
            "signed per-workload request residual, bounded to a fixed fraction "
            "of the current intercept to avoid reacting sharply to a smoke sample."),
        "evidence": {
            "workload_signed_residual_cycles": residuals,
            "median_signed_residual_cycles": median_residual,
            "requested_latency_step_cycles": requested_step,
            "maximum_step_cycles": max_step,
            "applied_latency_step_cycles": applied_step,
            "old_latency_cycles": latency,
            "new_latency_cycles": new_latency,
        },
    }


def objectives(training_result: dict) -> dict[str, float]:
    label = training_result["candidate_label"]
    aggregate = training_result["summary"]["models"][label]["aggregate"]
    values = {
        "cycle_macro_mae_pct": aggregate["cycle_macro_mae_pct"],
        "request_macro_mae_over_L": aggregate["request_macro_mae_over_L"],
    }
    for key, value in values.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid objective {key}={value!r}")
    return {key: float(value) for key, value in values.items()}


def select_candidate(results: list[dict]) -> dict:
    """Retain the incumbent unless a later candidate Pareto-dominates it."""
    if not results:
        raise ValueError("cannot select from an empty candidate set")
    incumbent = results[0]
    comparisons = []
    for challenger in results[1:]:
        incumbent_obj = objectives(incumbent)
        challenger_obj = objectives(challenger)
        dominates = (
            all(challenger_obj[key] <= incumbent_obj[key] for key in incumbent_obj)
            and any(challenger_obj[key] < incumbent_obj[key] for key in incumbent_obj)
        )
        comparisons.append({
            "incumbent": incumbent["candidate_label"],
            "challenger": challenger["candidate_label"],
            "incumbent_objectives": incumbent_obj,
            "challenger_objectives": challenger_obj,
            "challenger_pareto_dominates": dominates,
        })
        if dominates:
            incumbent = challenger
    return {
        "selected_label": incumbent["candidate_label"],
        "selected_candidate": incumbent["proposal"]["candidate"],
        "selected_objectives": objectives(incumbent),
        "policy": "incumbent_retention_unless_pareto_dominated",
        "policy_note": (
            "Core-cycle MAE and request MAE/L are co-equal. No scalar weighting "
            "trades one against the other; incomparable candidates retain the "
            "earlier incumbent in this smoke run."),
        "comparisons": comparisons,
    }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_snapshot(repo: pathlib.Path) -> dict:
    """Hash every tracked file except the explicitly agent-owned candidate."""
    output = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        check=True, capture_output=True,
    ).stdout
    paths = [os.fsdecode(item) for item in output.split(b"\0") if item]
    files = {}
    for relative in paths:
        if relative in AGENT_MUTABLE_PATHS:
            continue
        path = repo / relative
        files[relative] = {
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return {
        "schema_version": 1,
        "agent_mutable_paths": sorted(AGENT_MUTABLE_PATHS),
        "protected_files": files,
    }


def assert_protected_unchanged(repo: pathlib.Path, expected: dict) -> None:
    actual = protected_snapshot(repo)
    if actual != expected:
        expected_files = expected["protected_files"]
        actual_files = actual["protected_files"]
        changed = sorted(
            set(expected_files) ^ set(actual_files) |
            {path for path in set(expected_files) & set(actual_files)
             if expected_files[path] != actual_files[path]}
        )
        raise RuntimeError(
            "protected evaluator/source files changed during the loop: "
            + ", ".join(changed))
