"""Scientific selection over existing unrounded cycle and request calculations.

This module has no provider, scheduler, filesystem or held-out selection logic.
The evaluator owns source/input provenance; these functions own objective
eligibility and the established strict two-objective Pareto rule.
"""

from __future__ import annotations

import math

from ramulator_chia.pareto import dominates, objectives, pareto
from ramulator_chia.eval.metrics import (
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
    include_tails: bool = False,
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
            include_tails=include_tails,
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


def aggregate(
    workloads: dict[str, dict], *, expected_workloads: list[str], stage: str,
    request_objective: str = "complete_stable_id",
) -> dict:
    """Equal workload weight; a missing/ineligible case cannot improve the mean."""
    if not expected_workloads or len(set(expected_workloads)) != len(expected_workloads):
        raise ValueError("the expected workload population must be nonempty and unique")
    if set(workloads) != set(expected_workloads):
        raise ValueError("measurements do not cover exactly the declared workload cohort")
    if stage not in {"training", "validation", "test", "transfer"}:
        raise ValueError("unknown evaluation stage")
    if request_objective not in {"complete_stable_id", "champsim_exact_physical_filter", "champsim_foreground_stable_pairs"}:
        raise ValueError("unknown request objective population")
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
    if request_objective in {"champsim_exact_physical_filter", "champsim_foreground_stable_pairs"}:
        # Explicitly selected for ChampSim search. This is an intersection
        # objective, not evidence of complete admission or callback coverage.
        # Never drop an ineligible workload or pretend its error is zero.
        eligible = all(
            item["observation"] == {
                "scope": ("champsim_dram_controller_foreground_admissions"
                          if request_objective == "champsim_foreground_stable_pairs"
                          else "champsim_dram_controller_lifecycle"),
                "timebase": "ramulator_controller_cycles",
            }
            and item["oracle_owner_reads"] >= item["minimum_oracle_owner_reads"]
            and item["request"] is not None
            and item["request"]["match_mode"] == "stable_id"
            and item["request"]["matched"] > 0
            and item["request"].get("trusted_pair_policy") == "champsim_exact_physical_filter"
            and (request_objective != "champsim_foreground_stable_pairs"
                 or item["request"].get("address_mismatch_pairs") == 0)
            for item in ordered
        )
    requests = (
        aggregate_request_metrics({name: workloads[name]["request"] for name in expected_workloads})
        if eligible
        else None
    )
    report = {
        "schema_version": 1,
        "stage": stage,
        "precision": "unrounded_binary64",
        "eligible_for_promotion": eligible and stage in {"training", "validation"},
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
    if request_objective != "complete_stable_id":
        report["request_objective"] = request_objective
        report["normalization"] = "full_recorded_oracle_read_mean_latency_per_workload"
    if stage == "transfer" and ordered[0]["observation"]["scope"] == "champsim_dram_controller_lifecycle":
        # This is an intersection diagnostic, never a substitute for the
        # full-population objective above. Keep every declared workload.
        unavailable = [
            name for name in expected_workloads
            if not workloads[name]["request"]
            or workloads[name]["request"].get("trusted_pair_policy") != "champsim_exact_physical_filter"
        ]
        matched = None
        if not unavailable:
            payloads = {name: workloads[name]["request"] for name in expected_workloads}
            matched = aggregate_request_metrics(payloads)
            matched["mean_paired_p99_over_L"] = sum(
                value["tail"] for value in payloads.values()
            ) / len(payloads)
        report["matched_request_diagnostics"] = {
            "population": "recorded reads with the same stable ID and exact physical address",
            "normalization": "full_recorded_oracle_read_mean_latency_per_workload",
            "weighting": "equal_workload",
            "eligible_for_promotion": False,
            "unavailable_workloads": unavailable,
            "below_minimum_read_workloads": [
                name for name in expected_workloads
                if workloads[name]["oracle_owner_reads"] < workloads[name]["minimum_oracle_owner_reads"]
            ],
            "summary": matched,
        }
    return report


def valid_objectives(report: dict, *, stage: str = "training") -> tuple[float, float]:
    if (
        report.get("eligible_for_promotion") is not True
        or report.get("precision") != "unrounded_binary64"
        or report.get("stage") != stage
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


def should_promote(
    candidate: dict, incumbent: dict, *, mechanically_valid: bool,
    candidate_validation: dict | None = None, incumbent_validation: dict | None = None,
) -> bool:
    """Semantic feedback is deliberately not an argument or a promotion veto."""
    if type(mechanically_valid) is not bool:
        raise ValueError("mechanical validity must be a boolean")
    incumbent_values = valid_objectives(incumbent)
    if not mechanically_valid or candidate.get("eligible_for_promotion") is not True:
        return False
    if not dominates(valid_objectives(candidate), incumbent_values):
        return False
    if incumbent_validation is not None:
        if not candidate_validation or candidate_validation.get("eligible_for_promotion") is not True:
            return False
        return all(
            new <= old for new, old in zip(
                valid_objectives(candidate_validation, stage="validation"),
                valid_objectives(incumbent_validation, stage="validation"),
            )
        )
    if candidate_validation is not None:
        raise ValueError("validation gating needs the incumbent's validation measurements")
    return True


def validation_view(receipt: dict, names: tuple[str, ...]) -> dict:
    """The entire agent-visible validation surface: aliases and two errors only.

    Raw validation receipts, names, counts, traces and failure details stay in
    the operator's store. Missing measurements remain null, never zero.
    """
    measurement = receipt.get("measurement")

    def values(cycle=None, request=None):
        return {"core_error_pct": cycle, "request_mae_over_L": request}

    aggregate_scores = measurement["aggregate"] if measurement else {}
    workloads = measurement["workloads"] if measurement else {}
    return {
        "aggregate": values(
            aggregate_scores.get("cycle_macro_mae_pct"),
            aggregate_scores.get("request_macro_mae_over_L"),
        ),
        "workloads": {
            f"val{index}": values(
                workloads[name]["cycles"]["mean_abs_per_core_pct"] if name in workloads else None,
                workloads[name]["request"]["mae"]
                if name in workloads and workloads[name]["request"] else None,
            )
            for index, name in enumerate(names, 1)
        },
    }


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
