"""Shared, model-neutral request-metric validation and caching.

This module contains no candidate policy or search logic.  It turns two raw
request traces into a checksum-bound metric payload and rejects partial,
stale, or internally inconsistent results.
"""

import json
import math

from . import artifacts as A
from . import config as C

PERCENTILE_METHOD = "linear"


def paired_error_statistics(delta, oracle_mean_latency):
    """Summarize already paired errors; the caller declares the population of L."""
    import numpy as np

    values = np.asarray(delta)
    scale = finite_float(oracle_mean_latency, "oracle mean read latency")
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or scale <= 0:
        raise ValueError("paired errors and their positive normalization must be finite")
    mae = float(np.abs(values).mean())
    signed = float(values.mean())
    p99 = float(np.percentile(np.abs(values), 99, method=PERCENTILE_METHOD))
    return {
        "mae": mae / scale,
        "sgn": signed / scale,
        "tail": p99 / scale,
        "percentile_method": PERCENTILE_METHOD,
        "mae_cycles": mae,
        "signed_mean_cycles": signed,
        "paired_p99_cycles": p99,
        "extreme_min_cycles": int(values.min()),
        "extreme_max_cycles": int(values.max()),
        "extreme_min_over_L": float(values.min()) / scale,
        "extreme_max_over_L": float(values.max()) / scale,
    }


def cycles_metrics(oracle, model):
    """Machine-precision cycle metrics; presentation rounding is a caller choice."""
    import numpy as np

    o, m = oracle["per_core_cycles"], model["per_core_cycles"]
    if not o or len(o) != len(m):
        raise ValueError("oracle/model per-core cycle vectors must be non-empty and equal length")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
        for value in (*o, *m)
    ):
        raise ValueError("per-core cycles must be finite positive numbers")
    per_core = [100 * (mc - oc) / oc for oc, mc in zip(o, m)]
    return {
        "per_core_dev_pct": per_core,
        "mean_abs_per_core_pct": float(np.mean(np.abs(per_core))),
        "makespan_dev_pct": 100 * (max(m) - max(o)) / max(o),
        "legacy_signed_mean_pct": float(np.mean(per_core)),
    }


def finite_float(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def request_count(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def request_coverage(value, label):
    coverage = finite_float(value, label)
    if not 0 <= coverage <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return coverage


def matcher_schema_version():
    from . import matchlib

    return getattr(matchlib, "MATCHER_SCHEMA_VERSION", 1)


def _trace_signature(path):
    return A.raw_provenance(path)


def _require_ratio(actual, numerator, denominator, label):
    expected = numerator / denominator
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}={actual} is inconsistent with {expected}")


def _request_semantics(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_request_payload(
    payload,
    *,
    request_trace_scope,
    request_trace_timebase,
    signatures=None,
    expected_matcher_schema=None,
):
    """Validate every machine-consumed request-metric invariant."""
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    if payload.get("request_metric_schema_version") != C.REQUEST_METRIC_SCHEMA_VERSION:
        raise ValueError("request metric schema is stale")
    if expected_matcher_schema is None:
        expected_matcher_schema = matcher_schema_version()
    if payload.get("matcher_schema_version") != expected_matcher_schema:
        raise ValueError("matcher schema is stale")
    if signatures is not None and payload.get("_trace_inputs") != signatures:
        raise ValueError("request trace signatures do not match")
    if payload.get("normalization") != "full_oracle_read_mean_latency":
        raise ValueError("request metrics must use full-oracle read normalization")

    scope = _request_semantics(request_trace_scope, "request_trace_scope")
    timebase = _request_semantics(request_trace_timebase, "request_trace_timebase")
    if payload.get("request_trace_scope") != scope:
        raise ValueError(f"request metric trace scope differs from {scope!r}")
    if payload.get("request_trace_timebase") != timebase:
        raise ValueError(f"request metric trace timebase differs from {timebase!r}")

    for key in ("mae", "tail"):
        if finite_float(payload.get(key), key) < 0:
            raise ValueError(f"{key} must be non-negative")
    finite_float(payload.get("sgn"), "sgn")
    oracle_mean = finite_float(payload.get("oracle_read_mean_latency"), "oracle_read_mean_latency")
    if oracle_mean <= 0:
        raise ValueError("oracle_read_mean_latency must be positive")

    matched = request_count(payload.get("matched"), "matched", positive=True)
    n_oracle = request_count(payload.get("n_oracle"), "n_oracle", positive=True)
    n_model = request_count(payload.get("n_model"), "n_model", positive=True)
    if matched > min(n_oracle, n_model):
        raise ValueError("matched exceeds a full request population")
    eligible_o = request_count(payload.get("stable_eligible_o"), "stable_eligible_o")
    eligible_m = request_count(payload.get("stable_eligible_m"), "stable_eligible_m")
    if eligible_o > n_oracle or eligible_m > n_model:
        raise ValueError("stable eligible population exceeds full population")

    cov_o = request_coverage(payload.get("cov_o"), "cov_o")
    cov_m = request_coverage(payload.get("cov_m"), "cov_m")
    stable_cov_o = request_coverage(payload.get("stable_cov_o"), "stable_cov_o")
    stable_cov_m = request_coverage(payload.get("stable_cov_m"), "stable_cov_m")
    eligibility_o = request_coverage(
        payload.get("stable_eligibility_fraction_o"), "stable_eligibility_fraction_o"
    )
    eligibility_m = request_coverage(
        payload.get("stable_eligibility_fraction_m"), "stable_eligibility_fraction_m"
    )
    _require_ratio(cov_o, matched, n_oracle, "cov_o")
    _require_ratio(cov_m, matched, n_model, "cov_m")
    if "cov" in payload:
        compatibility_cov = request_coverage(payload["cov"], "cov")
        if not math.isclose(compatibility_cov, cov_o, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("compatibility cov differs from cov_o")

    mode = payload.get("match_mode")
    fallback = payload.get("legacy_fallback_reason")
    if mode == "stable_id":
        if fallback is not None:
            raise ValueError("stable-ID metrics cannot carry a legacy fallback reason")
        if eligible_o <= 0 or eligible_m <= 0:
            raise ValueError("stable-ID metrics require eligible rows on both sides")
        if matched > min(eligible_o, eligible_m):
            raise ValueError("matched exceeds a stable-eligible population")
        _require_ratio(stable_cov_o, matched, eligible_o, "stable_cov_o")
        _require_ratio(stable_cov_m, matched, eligible_m, "stable_cov_m")
    elif mode == "legacy_occurrence":
        if not isinstance(fallback, str) or not fallback:
            raise ValueError("legacy metrics require a fallback reason")
        if stable_cov_o != 0.0 or stable_cov_m != 0.0:
            raise ValueError("legacy metrics must report zero stable coverage")
    else:
        raise ValueError(f"invalid match_mode {mode!r}")
    _require_ratio(eligibility_o, eligible_o, n_oracle, "stable_eligibility_fraction_o")
    _require_ratio(eligibility_m, eligible_m, n_model, "stable_eligibility_fraction_m")
    return payload


def _request_cache_is_current(
    payload,
    request_path,
    signatures,
    *,
    request_trace_scope,
    request_trace_timebase,
    expected_matcher_schema=None,
):
    if expected_matcher_schema is None:
        expected_matcher_schema = matcher_schema_version()
    try:
        validate_request_payload(
            payload,
            signatures=signatures,
            request_trace_scope=request_trace_scope,
            request_trace_timebase=request_trace_timebase,
            expected_matcher_schema=expected_matcher_schema,
        )
    except (TypeError, ValueError):
        return False
    newest_trace = max(item["mtime_ns"] for item in signatures.values())
    return request_path.stat().st_mtime_ns >= newest_trace


def load_or_compute_request(
    request_path,
    trace_path,
    oracle_trace_path,
    *,
    force=False,
    matcher=None,
    expected_matcher_schema=None,
    request_trace_scope,
    request_trace_timebase,
):
    """Load metrics only for the exact raw inputs, scope, and clock domain."""
    try:
        request_trace_scope = _request_semantics(request_trace_scope, "request_trace_scope")
        request_trace_timebase = _request_semantics(
            request_trace_timebase, "request_trace_timebase"
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    for label, path in (("model", trace_path), ("oracle", oracle_trace_path)):
        if not A.exists(path):
            raise RuntimeError(
                f"missing {label} raw request trace {path}; rerun the producing job with --force"
            )
    signatures = {
        "model": _trace_signature(trace_path),
        "oracle": _trace_signature(oracle_trace_path),
    }
    if expected_matcher_schema is None:
        expected_matcher_schema = matcher_schema_version()
    if request_path.is_file() and not force:
        try:
            payload = json.loads(request_path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
        if _request_cache_is_current(
            payload,
            request_path,
            signatures,
            request_trace_scope=request_trace_scope,
            request_trace_timebase=request_trace_timebase,
            expected_matcher_schema=expected_matcher_schema,
        ):
            return payload

    if matcher is None:
        from . import matchlib

        matcher = matchlib.match
    result = matcher(oracle_trace_path, trace_path)
    result_schema = result.get("matcher_schema_version", expected_matcher_schema)
    if result_schema != expected_matcher_schema:
        raise RuntimeError(
            f"matcher returned schema {result_schema}, expected {expected_matcher_schema}"
        )
    dv = result["dv"]
    if not len(dv):
        raise RuntimeError(
            f"no matched read requests for {trace_path}; refusing a cycles-only score"
        )
    if signatures != {
        "model": _trace_signature(trace_path),
        "oracle": _trace_signature(oracle_trace_path),
    }:
        raise RuntimeError("raw request trace changed while metrics were being computed")
    try:
        payload = request_payload_from_match(
            result,
            request_trace_scope=request_trace_scope,
            request_trace_timebase=request_trace_timebase,
            signatures=signatures,
            expected_matcher_schema=expected_matcher_schema,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"matcher produced an invalid request metric payload for {trace_path}: {exc}"
        ) from exc
    C.atomic_write_json(request_path, payload)
    return payload


def request_payload_from_match(
    result,
    *,
    request_trace_scope,
    request_trace_timebase,
    signatures=None,
    expected_matcher_schema=None,
):
    """The existing normalization, exposed without cache or checkout writes.

    Partial matching is still a valid conditional diagnostic. Promotion eligibility
    is a separate exact-coverage decision, never inferred from a low error value.
    """
    if expected_matcher_schema is None:
        expected_matcher_schema = matcher_schema_version()
    if result.get("matcher_schema_version", expected_matcher_schema) != expected_matcher_schema:
        raise ValueError("matcher schema differs from the requested metric contract")
    dv = result["dv"]
    if not len(dv):
        raise ValueError("request traces contain no trusted read pairs")
    full_oracle_latencies = result["all_o"]
    if not len(full_oracle_latencies):
        raise ValueError("oracle request trace contains no reads")
    mean_oracle_latency = float(full_oracle_latencies.mean())
    if not math.isfinite(mean_oracle_latency) or mean_oracle_latency <= 0:
        raise ValueError("oracle request trace has invalid mean read latency")
    payload = {
        "request_metric_schema_version": C.REQUEST_METRIC_SCHEMA_VERSION,
        "matcher_schema_version": expected_matcher_schema,
        "match_mode": result.get("match_mode", "unspecified"),
        "stable_eligible_o": int(result.get("stable_eligible_o", 0)),
        "stable_eligible_m": int(result.get("stable_eligible_m", 0)),
        "stable_cov_o": float(result.get("stable_cov_o", 0.0)),
        "stable_cov_m": float(result.get("stable_cov_m", 0.0)),
        "stable_eligibility_fraction_o": (
            int(result.get("stable_eligible_o", 0)) / int(result["n_o"])
        ),
        "stable_eligibility_fraction_m": (
            int(result.get("stable_eligible_m", 0)) / int(result["n_m"])
        ),
        "stable_logical_pairs": result.get("stable_logical_pairs"),
        "address_mismatch_pairs": result.get("address_mismatch_pairs"),
        "physical_consistent_pairs": result.get("physical_consistent_pairs"),
        "physical_consistency_rate": result.get("physical_consistency_rate"),
        "trusted_pair_policy": result.get("trusted_pair_policy"),
        **paired_error_statistics(dv, mean_oracle_latency),
        "cov_o": float(result["cov_o"]),
        "cov": float(result["cov_o"]),
        "cov_m": float(result["cov_m"]),
        "matched": int(len(dv)),
        "n_oracle": int(result["n_o"]),
        "n_model": int(result["n_m"]),
        "oracle_read_mean_latency": mean_oracle_latency,
        "normalization": "full_oracle_read_mean_latency",
        "request_trace_scope": request_trace_scope,
        "request_trace_timebase": request_trace_timebase,
        "_trace_inputs": signatures,
    }
    if "legacy_fallback_reason" in result:
        payload["legacy_fallback_reason"] = result["legacy_fallback_reason"]
    validate_request_payload(
        payload,
        signatures=signatures,
        request_trace_scope=request_trace_scope,
        request_trace_timebase=request_trace_timebase,
        expected_matcher_schema=expected_matcher_schema,
    )
    return payload


def aggregate_request_metrics(requests):
    """Aggregate complete per-workload request metrics without scalar ranking."""
    if not requests:
        raise ValueError("cannot aggregate an empty request-metric set")
    required = {
        "mae",
        "sgn",
        "tail",
        "cov_o",
        "cov_m",
        "match_mode",
        "stable_cov_o",
        "stable_cov_m",
    }
    for workload, metric in requests.items():
        missing = required - set(metric)
        if missing:
            raise ValueError(f"{workload}: request metrics lack {sorted(missing)}")
        if metric["match_mode"] not in ("stable_id", "legacy_occurrence"):
            raise ValueError(f"{workload}.match_mode is invalid")
        if finite_float(metric["mae"], f"{workload}.mae") < 0:
            raise ValueError(f"{workload}.mae must be non-negative")
        if finite_float(metric["tail"], f"{workload}.tail") < 0:
            raise ValueError(f"{workload}.tail must be non-negative")
        finite_float(metric["sgn"], f"{workload}.sgn")
        for key in ("cov_o", "cov_m", "stable_cov_o", "stable_cov_m"):
            request_coverage(metric[key], f"{workload}.{key}")

    maes = {key: float(value["mae"]) for key, value in requests.items()}
    sgns = {key: float(value["sgn"]) for key, value in requests.items()}
    tails = {key: float(value["tail"]) for key, value in requests.items()}
    cov_o = {key: float(value["cov_o"]) for key, value in requests.items()}
    cov_m = {key: float(value["cov_m"]) for key, value in requests.items()}
    stable_o = {key: float(value["stable_cov_o"]) for key, value in requests.items()}
    stable_m = {key: float(value["stable_cov_m"]) for key, value in requests.items()}
    worst_mae = max(maes, key=maes.get)
    worst_sgn = max(sgns, key=lambda key: abs(sgns[key]))
    worst_tail = max(tails, key=tails.get)
    min_cov_o = min(cov_o, key=cov_o.get)
    min_cov_m = min(cov_m, key=cov_m.get)
    min_stable_o = min(stable_o, key=stable_o.get)
    min_stable_m = min(stable_m, key=stable_m.get)
    return {
        "mean_mae": sum(maes.values()) / len(maes),
        "worst_mae": maes[worst_mae],
        "worst_mae_wl": worst_mae,
        "mean_abs_sgn": sum(abs(value) for value in sgns.values()) / len(sgns),
        "worst_abs_sgn": abs(sgns[worst_sgn]),
        "worst_sgn": sgns[worst_sgn],
        "worst_sgn_wl": worst_sgn,
        "worst_tail": tails[worst_tail],
        "worst_tail_wl": worst_tail,
        "min_cov_o": cov_o[min_cov_o],
        "min_cov_o_wl": min_cov_o,
        "min_cov_m": cov_m[min_cov_m],
        "min_cov_m_wl": min_cov_m,
        "min_stable_cov_o": stable_o[min_stable_o],
        "min_stable_cov_o_wl": min_stable_o,
        "min_stable_cov_m": stable_m[min_stable_m],
        "min_stable_cov_m_wl": min_stable_m,
        "match_modes": sorted({value["match_mode"] for value in requests.values()}),
        "match_mode_by_workload": {key: value["match_mode"] for key, value in requests.items()},
    }
