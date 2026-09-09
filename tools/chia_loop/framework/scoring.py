"""Scientific selection over existing unrounded cycle and request calculations.

This module has no provider, scheduler, filesystem or held-out selection logic.
The evaluator owns source/input provenance; these functions own objective
eligibility and the established strict two-objective Pareto rule.
"""

from __future__ import annotations

import math

from tools.chia_loop.pareto import dominates, objectives, pareto
from tools.eval.metrics import (
    aggregate_request_metrics,
    cycles_metrics,
    request_count,
    request_payload_from_match,
    validate_request_payload,
)


def score_workload(
    oracle: dict,
    model: dict,
    request_result: dict,
    *,
    request_trace_scope: str,
    request_trace_timebase: str,
    minimum_oracle_owner_reads: int,
    population_verified: bool = True,
) -> dict:
    if type(population_verified) is not bool:
        raise ValueError("population verification must be explicit boolean evidence")
    request_count(minimum_oracle_owner_reads, "minimum oracle owner reads", positive=True)
    owner_reads = request_count(
        oracle.get("controller_stats", {}).get("num_read_reqs"), "oracle owner reads"
    )
    cycles = cycles_metrics(oracle, model)
    reasons = []
    if not population_verified:
        reasons.append("complete observation population has not been verified")
    matched = len(request_result["dv"])
    counts = (
        matched,
        request_result["n_o"],
        request_result["n_m"],
        request_result["stable_eligible_o"],
        request_result["stable_eligible_m"],
    )
    if request_result["match_mode"] != "stable_id":
        reasons.append("request pairing is not stable-ID based")
    if not matched or len(set(counts)) != 1:
        reasons.append("request population lacks exact bidirectional stable-ID coverage")
    if owner_reads < minimum_oracle_owner_reads:
        reasons.append("oracle owner-read population is below the declared minimum")
    request = (
        request_payload_from_match(
            request_result,
            request_trace_scope=request_trace_scope,
            request_trace_timebase=request_trace_timebase,
        )
        if matched
        else None
    )
    return {
        "cycles": cycles,
        "request": request,
        "request_headline_eligible": not reasons,
        "request_population_verified": population_verified,
        "ineligibility_reasons": reasons,
        "minimum_oracle_owner_reads": minimum_oracle_owner_reads,
        "oracle_owner_reads": owner_reads,
        "observation": {"scope": request_trace_scope, "timebase": request_trace_timebase},
        "request_populations": {
            "matched": matched,
            "oracle": request_result["n_o"],
            "model": request_result["n_m"],
        },
    }


def aggregate(workloads: dict[str, dict], *, expected_workloads: list[str], stage: str) -> dict:
    """Equal workload weight; a missing/ineligible case cannot improve the mean."""
    if not expected_workloads or len(set(expected_workloads)) != len(expected_workloads):
        raise ValueError("the expected workload population must be nonempty and unique")
    if set(workloads) != set(expected_workloads):
        raise ValueError("measurements do not cover exactly the declared workload cohort")
    if stage not in {"training", "test", "transfer"}:
        raise ValueError("unknown evaluation stage")
    observations = {tuple(sorted(value["observation"].items())) for value in workloads.values()}
    if len(observations) != 1:
        raise ValueError("different request populations/timebases need separate reports")
    ordered = [workloads[name] for name in expected_workloads]
    for item in ordered:
        request = item["request"]
        if request is None:
            if item["request_headline_eligible"]:
                raise ValueError("missing request measurements cannot be headline eligible")
            continue
        validate_request_payload(
            request,
            request_trace_scope=item["observation"]["scope"],
            request_trace_timebase=item["observation"]["timebase"],
        )
        counts = [
            request[key]
            for key in ("matched", "n_oracle", "n_model", "stable_eligible_o", "stable_eligible_m")
        ]
        owner_reads = request_count(item["oracle_owner_reads"], "oracle owner reads")
        minimum = request_count(
            item["minimum_oracle_owner_reads"], "minimum oracle owner reads", positive=True
        )
        population_verified = item.get("request_population_verified", True)
        if type(population_verified) is not bool:
            raise ValueError("invalid population verification evidence")
        expected_eligibility = (
            population_verified
            and request["match_mode"] == "stable_id"
            and len(set(counts)) == 1
            and owner_reads >= minimum
        )
        if item["request_headline_eligible"] is not expected_eligibility:
            raise ValueError("request eligibility contradicts its recorded population")
    cycle_errors = [item["cycles"]["mean_abs_per_core_pct"] for item in ordered]
    if any(not math.isfinite(value) or value < 0 for value in cycle_errors):
        raise ValueError("invalid workload cycle objective")
    eligible = all(item["request_headline_eligible"] for item in ordered)
    requests = (
        aggregate_request_metrics({name: workloads[name]["request"] for name in expected_workloads})
        if eligible
        else None
    )
    return {
        "schema_version": 1,
        "stage": stage,
        "precision": "unrounded_binary64",
        "eligible_for_promotion": eligible and stage == "training",
        "aggregate": {
            "cycle_macro_mae_pct": sum(cycle_errors) / len(cycle_errors),
            "request_macro_mae_over_L": requests["mean_mae"] if requests else None,
            "request_macro_absolute_signed_drift_over_L": requests["mean_abs_sgn"]
            if requests
            else None,
        },
        "request_summary": requests,
        "workloads": {name: workloads[name] for name in expected_workloads},
    }


def valid_objectives(report: dict) -> tuple[float, float]:
    if (
        report.get("eligible_for_promotion") is not True
        or report.get("precision") != "unrounded_binary64"
        or report.get("stage") != "training"
    ):
        raise ValueError("selection requires eligible, unrounded training measurements")
    values = objectives(report)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("selection objectives must be finite nonnegative numbers")
    return values


def should_promote(candidate: dict, incumbent: dict, *, mechanically_valid: bool) -> bool:
    """Semantic feedback is deliberately not an argument or a promotion veto."""
    if type(mechanically_valid) is not bool:
        raise ValueError("mechanical validity must be a boolean")
    incumbent_values = valid_objectives(incumbent)
    if not mechanically_valid or candidate.get("eligible_for_promotion") is not True:
        return False
    return dominates(valid_objectives(candidate), incumbent_values)


def select_parent(
    candidates: dict[str, dict], incumbent: str, *, iteration: int, policy: str
) -> str:
    if type(iteration) is not int or iteration < 1:
        raise ValueError("iteration must be positive")
    if incumbent not in candidates:
        raise ValueError("incumbent is absent from the candidate catalog")
    for candidate in candidates.values():
        valid_objectives(candidate["metrics"])
    if policy == "incumbent":
        return incumbent
    if policy == "pareto_round_robin":
        archive = pareto(candidates)
        return archive[(iteration - 1) % len(archive)]
    raise ValueError(f"unknown parent-selection policy: {policy}")
