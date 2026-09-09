"""Recompute existing measurements without importing an execution backend.

Live CHIA tasks and relocated archives use these same checks and metric
functions. Paths are explicit input locations; historical process paths inside
receipts are preserved as evidence but are not followed. Nothing here executes
a candidate, starts a cluster or calls a model.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.eval import matchlib, simpleo3

from .archive import Member, describe_payload, safe_name
from .identity import digest_json, file_sha256
from .scoring import score_workload

SIMPLEO3_OBSERVATIONS = ("logical.csv.ch0", "controller.csv.ch0")
CONTROLLER_OBSERVATIONS = ("controller.csv.ch0",)


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
) -> dict:
    """Keep core results when request identity is unusable; never invent a score."""
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
    try:
        paired = matchlib.match(*paths, champsim_filter_physical_mismatches=frontend == "champsim")
    except matchlib.RequestIdentityError as exc:
        # Invalid cross-run identity is not a simulator failure. Malformed raw
        # observations still raise; only this typed identity failure is retained
        # as an unavailable request result beside the independent core metric.
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
        request_trace_scope=frontend + "_dram_controller_lifecycle",
        request_trace_timebase="ramulator_controller_cycles",
        minimum_oracle_owner_reads=minimum_oracle_owner_reads,
        population_verified=a["request_population_verified"] and b["request_population_verified"],
    )
    result["pairing_error"] = pairing_error
    result["core_metric"] = a["core_metric"]
    return result
