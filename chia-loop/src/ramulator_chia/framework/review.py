"""Trusted score projections and review decisions, not an LLM or execution loop."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import Field

from .config import StrictRecord
from .identity import file_sha256
from .scoring import aggregate

PROMPTS = Path(__file__).parent / "prompts/staged_v1"
PROMPT_FILES = ("task.md", "single_core.md", "multicore.md", "promotion_review.md", "reflection.md")
REQUEST_FIELDS = (
    "mae",
    "sgn",
    "tail",
    "p999_over_L",
    "mae_cycles",
    "signed_mean_cycles",
    "paired_p99_cycles",
    "paired_p999_cycles",
    "extreme_min_cycles",
    "extreme_max_cycles",
    "extreme_min_over_L",
    "extreme_max_over_L",
    "oracle_read_mean_latency",
    "matched",
    "n_oracle",
    "n_model",
    "cov_o",
    "cov_m",
    "stable_eligible_o",
    "stable_eligible_m",
    "address_mismatch_pairs",
)
POPULATION_FIELDS = (
    "recorded_reads",
    "matched_reads",
    "unmatched_reads",
    "coverage",
    "without_stable_id",
    "stable_id_without_counterpart",
    "admitted_foreground_reads",
    "admissions_without_recorded_latency",
    "total_admitted_reads",
)


def prompt_inventory() -> dict:
    return {name: file_sha256(PROMPTS / name) for name in PROMPT_FILES}


def _numeric(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("score projection accepts finite numbers or explicit missing values only")
    return value


def case_view(row: dict) -> dict:
    """Positive numeric projection. Never copy a receipt or arbitrary nested dictionary."""
    cycles = row["cycles"]
    per_core = [_numeric(v) for v in cycles["per_core_dev_pct"]]
    if not per_core:
        raise ValueError("missing core measurements")
    request = row.get("request")
    result = {
        "core_error_pct": _numeric(cycles["mean_abs_per_core_pct"]),
        "signed_core_error_pct": sum(per_core) / len(per_core),
        "per_core_signed_error_pct": per_core,
        "worst_core_error_pct": max(abs(v) for v in per_core),
        "makespan_signed_error_pct": _numeric(cycles["makespan_dev_pct"]),
        "request": None,
    }
    if request is not None:
        result["request"] = {key: _numeric(request[key]) for key in REQUEST_FIELDS}
        result["request"]["absolute_drift_over_L"] = abs(request["sgn"])
        result["request"]["tail_thresholds"] = {
            threshold: {
                key: _numeric(request["tail_thresholds"][threshold][key])
                for key in ("count", "fraction", "absolute_error_share")
            }
            for threshold in ("1L", "5L")
        }
    populations = row.get("request_pairing", {})
    result["populations"] = {
        label: {key: _numeric(populations.get(label, {}).get(key)) for key in POPULATION_FIELDS}
        for label in ("oracle", "model")
    }
    return result


def group_means(rows: dict) -> dict:
    """Equal case weight; no pooling across core counts or matched request populations."""
    values = list(rows.values())
    result = {
        key: sum(row[key] for row in values) / len(values)
        for key in (
            "core_error_pct",
            "signed_core_error_pct",
            "worst_core_error_pct",
            "makespan_signed_error_pct",
        )
    }
    result["request"] = None
    if all(row["request"] is not None for row in values):
        result["request"] = {
            key: sum(row["request"][key] for row in values) / len(values)
            for key in ("mae", "sgn", "absolute_drift_over_L", "tail", "p999_over_L")
        }
    return result


def grouped_measurement(
    rows: dict, *, names: tuple, core_counts: tuple, cases: dict, stage: str
) -> dict:
    if set(rows) != set(names):
        raise ValueError("staged measurements must cover exactly the required cohort")
    groups = {}
    for cores in core_counts:
        selected = tuple(name for name in names if len(cases[name].programs) == cores)
        if not selected:
            raise ValueError("staged measurement has an empty core-count group")
        group = aggregate(
            {name: rows[name] for name in selected},
            expected_workloads=list(selected),
            stage=stage,
            request_objective="champsim_foreground_stable_pairs",
        )
        for name in selected:
            if len(rows[name]["cycles"]["per_core_dev_pct"]) != cores:
                raise ValueError("measured core vector differs from declared topology")
        views = {name: case_view(rows[name]) for name in selected}
        group["review_means"] = group_means(views)
        groups[str(cores)] = group
    return {
        "schema_version": 2,
        "stage": stage,
        "precision": "unrounded_binary64",
        "eligible_for_promotion": all(g["eligible_for_promotion"] for g in groups.values()),
        "groups": groups,
    }


def score_view(receipt: dict, names: tuple, *, anonymous: bool) -> dict:
    """Named training or anonymous validation; failed observations remain missing."""
    aliases = {name: f"val{i}" if anonymous else name for i, name in enumerate(names, 1)}
    measurement = receipt.get("measurement")
    if measurement is None:
        return {"available": False, "groups": {}}
    groups = {}
    for cores, group in measurement["groups"].items():
        rows = {aliases[name]: case_view(row) for name, row in group["workloads"].items()}
        groups[cores] = {"means": group_means(rows), "cases": rows}
    return {"available": True, "groups": groups}


def eligible(receipt: dict | None) -> bool:
    return bool(
        receipt
        and receipt.get("measurement")
        and receipt["measurement"].get("eligible_for_promotion") is True
    )


class PromotionDecision(StrictRecord):
    decision: str = Field(pattern=r"^(promote|keep)$")
    rationale: str = Field(min_length=1)
    improvements: list[str]
    accepted_regressions: list[str]
    contract_findings: list[str]
    contract_violation: bool
    uncertainty: list[str]


def _decision_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```json\n") and stripped.endswith("\n```"):
        stripped = stripped[8:-4]

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate decision field")
            result[key] = value
        return result

    result = json.loads(stripped, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("review decision must be an object")
    return result


def parse_decision(text: str, *, repair_of: str | None = None) -> dict:
    """Accept JSON, not prose; a format repair cannot flip an explicit judgment."""
    decision = PromotionDecision.model_validate(_decision_object(text))
    if repair_of is not None:
        try:
            original = _decision_object(repair_of)
        except (ValueError, TypeError):
            original = {}  # Unparseable prose is not evidence of a verdict.
        if (
            original.get("decision") in {"promote", "keep"}
            and original["decision"] != decision.decision
        ):
            raise ValueError("format-only repair changed the original decision")
    if decision.decision == "promote" and (
        decision.contract_violation or not decision.improvements
    ):
        raise ValueError("promotion needs an improvement and no demonstrated contract violation")
    return decision.model_dump(mode="json")
