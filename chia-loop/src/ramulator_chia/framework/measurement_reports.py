"""Recompute existing measurements without importing an execution backend.

Live CHIA tasks and relocated archives use these same checks and metric
functions. Paths are explicit input locations; historical process paths inside
receipts are preserved as evidence but are not followed. Nothing here executes
a candidate, starts a cluster or calls a model.
"""

from __future__ import annotations

import json
from pathlib import Path

from ramulator_chia.eval import matchlib, simpleo3

from .archive import Member, describe_payload, safe_name
from .identity import digest_json, file_sha256
from .scoring import score_workload

SIMPLEO3_OBSERVATIONS = ("logical.csv.ch0", "controller.csv.ch0")
CONTROLLER_OBSERVATIONS = ("controller.csv.ch0",)


def _champsim_pairing_counts(paired: dict) -> dict:
    """Partition recorded reads, including cases with no trusted latency pairs."""
    logical = paired.get("stable_logical_pairs")
    mismatches = paired.get("address_mismatch_pairs")
    matched = len(paired["dv"])
    result = {"stable_logical_pairs": logical, "address_mismatch_pairs": mismatches}
    for label, suffix in (("oracle", "o"), ("model", "m")):
        total = paired["n_" + suffix]
        eligible = paired["stable_eligible_" + suffix]
        result[label] = {
            "recorded_reads": total,
            "matched_reads": matched,
            "unmatched_reads": total - matched,
            "coverage": matched / total if total else None,
            # Identity failure makes this partition unknown, not zero.
            "without_stable_id": total - eligible if logical is not None else None,
            "stable_id_without_counterpart": eligible - logical if logical is not None else None,
        }
        if logical is not None and not (0 <= matched <= logical <= eligible <= total):
            raise ValueError("ChampSim request population counts are inconsistent")
    if logical is not None and logical != matched + mismatches:
        raise ValueError("ChampSim logical/physical pair counts are inconsistent")
    return result


def verify_measurement(directory: Path, expected_sha256: str) -> dict:
    """Verify settled SimpleO3 evidence before reuse or metric recomputation."""
    return verify_native_measurement(
        directory, expected_sha256, frontend="SimpleO3", observations=SIMPLEO3_OBSERVATIONS
    )


def verify_native_measurement(
    directory: Path, expected_sha256: str, *, frontend: str, observations: tuple[str, ...]
) -> dict:
    """Require the caller's actual observation population, not a relabeled receipt."""
    if file_sha256(directory / "measurement.json") != expected_sha256:
        raise ValueError("measurement receipt identity changed")
    receipt = json.loads((directory / "measurement.json").read_text())
    if receipt.get("complete") is not True or receipt.get("execution") != "native":
        raise ValueError("only complete native measurements can be scored")
    if receipt.get("case_sha256") != digest_json(receipt["case"]):
        raise ValueError("measurement case identity is inconsistent")
    if receipt["case"]["frontend"] != frontend:
        raise ValueError("measurement belongs to another frontend")
    if set(receipt["traces"]) != set(observations):
        raise ValueError("measurement is missing an observation population")
    for name, digest in receipt["files"].items():
        safe_name(name)
        if file_sha256(directory / name) != digest:
            raise ValueError(f"measurement evidence changed: {name}")
    for entry in receipt["traces"].values():
        member = Member(**entry)
        if (
            describe_payload(directory / member.name, member.name, codec=member.codec).member
            != member
        ):
            raise ValueError("measurement trace identity changed")
    return receipt


def compare_simpleo3(
    oracle: Path,
    model: Path,
    *,
    oracle_receipt_sha256: str,
    model_receipt_sha256: str,
    minimum_oracle_owner_reads: int,
) -> dict:
    """Use the established exact matcher and unrounded metrics on verified traces."""
    a = verify_measurement(oracle, oracle_receipt_sha256)
    b = verify_measurement(model, model_receipt_sha256)
    if (
        a["model"] != "oracle"
        or a["case_sha256"] != b["case_sha256"]
        or a["runtime_sha256"] != b["runtime_sha256"]
    ):
        raise ValueError("oracle/model measurements use different cases or runtimes")
    if a["case"]["stage"] == "qualification":
        raise ValueError("infrastructure fixtures are not accuracy measurements")
    paired = matchlib.match(
        oracle / a["traces"]["logical.csv.ch0"]["name"],
        model / b["traces"]["logical.csv.ch0"]["name"],
    )
    return score_workload(
        a,
        b,
        paired,
        request_trace_scope=simpleo3.REQUEST_TRACE_SCOPE,
        request_trace_timebase=simpleo3.REQUEST_TRACE_TIMEBASE,
        minimum_oracle_owner_reads=minimum_oracle_owner_reads,
    )


def compare_transfer(
    oracle: Path,
    model: Path,
    *,
    oracle_receipt_sha256: str,
    model_receipt_sha256: str,
    frontend: str,
    minimum_oracle_owner_reads: int,
    include_tails: bool = False,
) -> dict:
    """Keep core results when request identity is unusable; never invent a score."""
    if frontend not in {"champsim", "gem5"}:
        raise ValueError("unsupported transfer frontend")
    a = verify_native_measurement(
        oracle, oracle_receipt_sha256, frontend=frontend, observations=CONTROLLER_OBSERVATIONS
    )
    b = verify_native_measurement(
        model, model_receipt_sha256, frontend=frontend, observations=CONTROLLER_OBSERVATIONS
    )
    if a["model"] != "oracle" or any(
        a[key] != b[key] for key in ("case_sha256", "runtime_sha256", "host", "core_metric")
    ):
        raise ValueError("external comparison changed its input, runtime or metric")
    paths = [
        directory / result["traces"][CONTROLLER_OBSERVATIONS[0]]["name"]
        for directory, result in ((oracle, a), (model, b))
    ]
    pairing_error = None
    if frontend == "gem5":
        # O3 dynamic sequence numbers are not cross-run instruction identities.
        # gem5 transfer is core-only; do not spend time attempting a misleading join.
        paired = {
            "dv": (),
            "match_mode": "disabled",
            "n_o": a["controller_stats"]["observed_read_records"],
            "n_m": b["controller_stats"]["observed_read_records"],
            "stable_eligible_o": 0,
            "stable_eligible_m": 0,
        }
    else:
        try:
            windows = [r.get("frontend_stats", {}).get("admission_windows") for r in (a, b)]
            if a["case"].get("request_window") and not all(windows):
                raise ValueError("foreground measurement is missing its admission windows")
            paired = matchlib.match(*paths,
                champsim_filter_physical_mismatches=True,
                oracle_windows=windows[0], model_windows=windows[1])
        except matchlib.RequestIdentityError as exc:
            # Identity failure is not simulator failure. Malformed CSVs and
            # corrupt archives still raise rather than becoming a core-only score.
            pairing_error = str(exc)
            paired = {
                "dv": (),
                "match_mode": "invalid_identity",
                "n_o": a["controller_stats"]["observed_read_records"],
                "n_m": b["controller_stats"]["observed_read_records"],
                "stable_eligible_o": 0,
                "stable_eligible_m": 0,
            }
    # An observed row count is not an admission counter. Pass it only for the
    # declared traffic screen, alongside an explicit unverified-population flag.
    a = {**a, "controller_stats": {"num_read_reqs": paired["n_o"]}}
    result = score_workload(
        a,
        b,
        paired,
        request_trace_scope=frontend + ("_dram_controller_foreground_admissions" if a["case"].get("request_window")
                                        else "_dram_controller_lifecycle"),
        request_trace_timebase="ramulator_controller_cycles",
        minimum_oracle_owner_reads=minimum_oracle_owner_reads,
        population_verified=a["request_population_verified"] and b["request_population_verified"],
        include_tails=include_tails,
    )
    result["pairing_error"] = pairing_error
    result["core_metric"] = a["core_metric"]
    if frontend == "champsim":
        result["request_pairing"] = _champsim_pairing_counts(paired)
        if a["case"].get("request_window"):
            for label, receipt, count in (("oracle", a, paired["n_o"]), ("model", b, paired["n_m"])):
                windows = receipt["frontend_stats"]["admission_windows"]
                admitted = sum(w["admitted_reads"] for w in windows)
                result["request_pairing"][label].update(
                    admitted_foreground_reads=admitted,
                    admissions_without_recorded_latency=admitted-count,
                    total_admitted_reads=sum(w["total_admitted_reads"] for w in windows))
    else:
        result["request_matching"] = "disabled: gem5 transfer reports core-time accuracy only"
    return result
