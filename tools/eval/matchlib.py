"""Stable-ID per-request matching with an explicit legacy fallback.

New traces carry ``(source, frontend_id, frontend_sub_id)`` from the
frontend to the controller. Those fields identify the same logical memory
transaction across self-paced closed-loop runs even when request ordering or
population differs. Old traces (or frontends that emit no IDs) retain the
historical ``(source, address, occurrence)`` matcher, and the result labels
that mode explicitly so it cannot be mistaken for identity-based matching.
"""

import numpy as np
import pandas as pd

from . import artifacts as A

MATCHER_SCHEMA_VERSION = 4

_BASE_COLUMNS = ("arrive", "depart", "type", "source", "addr")
_STABLE_COLUMNS = ("frontend_id", "frontend_sub_id", "admission_ordinal")
_IDENTITY_COLUMNS = ("src", "frontend_id", "frontend_sub_id")


class RequestIdentityError(ValueError):
    """Well-formed observations do not establish valid cross-run request IDs."""


def _load_frame(path):
    stored_path = A.resolve(path)
    header = pd.read_csv(stored_path, nrows=0, on_bad_lines="error")
    missing = set(_BASE_COLUMNS) - set(header.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    present_stable = set(_STABLE_COLUMNS) & set(header.columns)
    if present_stable and present_stable != set(_STABLE_COLUMNS):
        missing_stable = set(_STABLE_COLUMNS) - present_stable
        raise ValueError(
            f"{path}: incomplete stable-ID schema; missing {sorted(missing_stable)}"
        )

    columns = list(_BASE_COLUMNS)
    if present_stable:
        columns.extend(_STABLE_COLUMNS)
    df = pd.read_csv(
        stored_path,
        usecols=columns,
        dtype={column: np.int64 for column in columns},
        on_bad_lines="error",
    ).rename(columns={"source": "src"})

    if not df["type"].isin((0, 1)).all():
        raise ValueError(f"{path}: request type must be 0 (read) or 1 (write)")
    if (df[["arrive", "addr"]] < 0).any().any() or (df["src"] < -1).any():
        raise ValueError(
            f"{path}: arrive/addr must be non-negative and source must be at least -1"
        )
    reads = df["type"] == 0
    if (df.loc[reads, "depart"] < df.loc[reads, "arrive"]).any():
        raise ValueError(f"{path}: read departure precedes arrival")

    if present_stable:
        if (df["frontend_id"] < -1).any():
            raise ValueError(f"{path}: frontend_id must be at least -1")
        if (df["frontend_sub_id"] < 0).any():
            raise ValueError(f"{path}: frontend_sub_id must be non-negative")
        if (df["admission_ordinal"] < -1).any():
            raise ValueError(f"{path}: admission_ordinal must be at least -1")
        stable_eligible = df["frontend_id"] >= 0
        if (df.loc[stable_eligible, "admission_ordinal"] < 0).any():
            raise ValueError(
                f"{path}: stable-ID-eligible rows require a non-negative "
                "admission_ordinal"
            )
        admitted = df["admission_ordinal"] >= 0
        if df.loc[admitted, "admission_ordinal"].duplicated().any():
            raise ValueError(f"{path}: duplicate admission_ordinal in channel trace")

    df["lat"] = df["depart"] - df["arrive"]
    return df.loc[reads].copy(), bool(present_stable)


def parse(path):
    """Return legacy per-source ``(address, latency)`` read arrays.

    This compatibility API still validates the complete trace, including the
    stable-ID columns when present. New matching code should call :func:`match`.
    """
    df, _ = _load_frame(path)
    out = {}
    for src, group in df.groupby("src", sort=False):
        out[src] = (group["addr"].to_numpy(), group["lat"].to_numpy())
    return out


def occ_index(a):
    """Return each address's zero-based occurrence in original order."""
    order = np.argsort(a, kind="stable")
    sa = a[order]
    n = len(sa)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.r_[True, sa[1:] != sa[:-1]]
    base = np.where(starts, np.arange(n), 0)
    np.maximum.accumulate(base, out=base)
    occ_sorted = np.arange(n) - base
    occ = np.empty(n, dtype=np.int64)
    occ[order] = occ_sorted
    return occ


def _legacy_pairs(oracle, model):
    oracle = oracle.copy()
    model = model.copy()
    oracle["occurrence"] = oracle.groupby(["src", "addr"], sort=False).cumcount()
    model["occurrence"] = model.groupby(["src", "addr"], sort=False).cumcount()
    keys = ["src", "addr", "occurrence"]
    return oracle.merge(model, on=keys, how="inner", sort=False, suffixes=("_o", "_m"))


def _stable_pairs(oracle, model, oracle_path, model_path):
    eligible_o = oracle[oracle["frontend_id"] >= 0].copy()
    eligible_m = model[model["frontend_id"] >= 0].copy()
    for frame, path in ((eligible_o, oracle_path), (eligible_m, model_path)):
        if frame.duplicated(list(_IDENTITY_COLUMNS)).any():
            raise RequestIdentityError(f"{path}: duplicate stable request identity")

    pairs = eligible_o.merge(
        eligible_m,
        on=list(_IDENTITY_COLUMNS),
        how="inner",
        sort=False,
        suffixes=("_o", "_m"),
        validate="one_to_one",
    )
    return pairs, len(eligible_o), len(eligible_m)


def _raise_address_mismatch(pairs):
    mismatch = pairs.loc[pairs["addr_o"] != pairs["addr_m"]].iloc[0]
    identity = tuple(int(mismatch[column]) for column in _IDENTITY_COLUMNS)
    raise RequestIdentityError(
        "stable request identity maps to different addresses: "
        f"identity={identity}, oracle={int(mismatch['addr_o'])}, "
        f"model={int(mismatch['addr_m'])}"
    )


def match(oracle_path, model_path, *, champsim_filter_physical_mismatches=False):
    """Return paired read errors, populations, coverage, and match provenance.

    ``match_mode == 'stable_id'`` is the trustworthy closed-loop metric. A
    ``legacy_occurrence`` result is retained only for old/uninstrumented traces
    and includes ``legacy_fallback_reason``.

    By default, a stable identity associated with different addresses fails
    closed. ChampSim is the sole exception because its timing-dependent
    first-touch mapper can expose the same virtual-line identity at different
    physical addresses in independently paced runs. The explicit
    ``champsim_filter_physical_mismatches`` mode first measures logical stable
    pairing, reports physical consistency, and removes mismatched-address pairs
    before constructing latency errors. It never scores a physically
    inconsistent pair.
    """
    oracle, oracle_has_schema = _load_frame(oracle_path)
    model, model_has_schema = _load_frame(model_path)
    n_o, n_m = len(oracle), len(model)

    eligible_o = int((oracle["frontend_id"] >= 0).sum()) if oracle_has_schema else 0
    eligible_m = int((model["frontend_id"] >= 0).sum()) if model_has_schema else 0
    fallback_reason = None
    logical_pairs = None
    address_mismatch_pairs = None
    if oracle_has_schema and model_has_schema and (eligible_o > 0 or eligible_m > 0):
        logical_pairs, eligible_o, eligible_m = _stable_pairs(
            oracle, model, oracle_path, model_path
        )
        match_mode = "stable_id"
        address_mismatch = logical_pairs["addr_o"] != logical_pairs["addr_m"]
        address_mismatch_pairs = int(address_mismatch.sum())
        if address_mismatch_pairs and not champsim_filter_physical_mismatches:
            _raise_address_mismatch(logical_pairs)
        pairs = (
            logical_pairs.loc[~address_mismatch].copy()
            if champsim_filter_physical_mismatches
            else logical_pairs
        )
    else:
        if champsim_filter_physical_mismatches:
            raise RequestIdentityError(
                "ChampSim physical-mismatch filtering requires eligible stable IDs"
            )
        pairs = _legacy_pairs(oracle, model)
        match_mode = "legacy_occurrence"
        if not oracle_has_schema or not model_has_schema:
            fallback_reason = "stable_id_columns_missing"
        else:
            fallback_reason = "no_eligible_stable_ids"

    dv = (pairs["lat_m"] - pairs["lat_o"]).to_numpy(dtype=np.int64)
    olat = pairs["lat_o"].to_numpy(dtype=np.int64)
    all_o = oracle["lat"].to_numpy(dtype=np.int64)
    all_m = model["lat"].to_numpy(dtype=np.int64)
    logical_pair_count = len(logical_pairs) if logical_pairs is not None else None
    physical_consistent_pairs = (
        logical_pair_count - address_mismatch_pairs
        if logical_pair_count is not None
        else None
    )
    result = {
        "dv": dv,
        "olat": olat,
        "n_o": n_o,
        "n_m": n_m,
        "cov_o": len(dv) / max(n_o, 1),
        "cov_m": len(dv) / max(n_m, 1),
        "all_o": all_o,
        "all_m": all_m,
        "match_mode": match_mode,
        "matcher_schema_version": MATCHER_SCHEMA_VERSION,
        "stable_eligible_o": eligible_o,
        "stable_eligible_m": eligible_m,
        "stable_cov_o": (
            len(dv) / eligible_o
            if match_mode == "stable_id" and eligible_o > 0
            else 0.0
        ),
        "stable_cov_m": (
            len(dv) / eligible_m
            if match_mode == "stable_id" and eligible_m > 0
            else 0.0
        ),
        "stable_eligibility_fraction_o": eligible_o / max(n_o, 1),
        "stable_eligibility_fraction_m": eligible_m / max(n_m, 1),
        "stable_logical_pairs": logical_pair_count,
        "logical_cov_o": (
            logical_pair_count / max(n_o, 1)
            if logical_pair_count is not None
            else None
        ),
        "logical_cov_m": (
            logical_pair_count / max(n_m, 1)
            if logical_pair_count is not None
            else None
        ),
        "logical_stable_cov_o": (
            logical_pair_count / max(eligible_o, 1)
            if logical_pair_count is not None
            else None
        ),
        "logical_stable_cov_m": (
            logical_pair_count / max(eligible_m, 1)
            if logical_pair_count is not None
            else None
        ),
        "address_mismatch_pairs": address_mismatch_pairs,
        "physical_consistent_pairs": physical_consistent_pairs,
        "physical_consistency_rate": (
            physical_consistent_pairs / logical_pair_count
            if logical_pair_count
            else None
        ),
        "trusted_pair_policy": (
            "champsim_exact_physical_filter"
            if champsim_filter_physical_mismatches
            else "strict_address_equality"
            if logical_pair_count is not None
            else "legacy_occurrence"
        ),
    }
    if fallback_reason is not None:
        result["legacy_fallback_reason"] = fallback_reason
    return result
