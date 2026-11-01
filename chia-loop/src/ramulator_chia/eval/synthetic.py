"""Generic SyntheticPattern observation pairing, independent of run control.

These calculations are shared with the earlier diagnostic adapter. They contain
no scenario catalog, learned model rules or campaign lookup.
"""

import math

import pandas as pd

from . import artifacts
from .metrics import paired_error_statistics


def read_population(path, params, stats):
    """Recover read ordinals from unique per-stream admission times, not returns."""
    with artifacts.open_text(path) as stream:
        frame = pd.read_csv(
            stream,
            usecols=["arrive", "depart", "type", "source", "addr"],
            dtype={
                "arrive": "int64",
                "depart": "int64",
                "type": "int64",
                "source": "int64",
                "addr": "uint64",
            },
        )
    if (
        not frame["type"].isin([0, 1]).all()
        or (frame["arrive"] < 0).any()
        or (frame["depart"] < frame["arrive"]).any()
        or ((frame["type"] == 0) & (frame["depart"] == frame["arrive"])).any()
    ):
        raise RuntimeError("invalid synthetic trace type/latency")
    reads = frame[frame["type"] == 0].sort_values(["source", "arrive"]).copy()
    counts = reads.groupby("source").size().to_dict()
    if (
        counts != {i: params["num_requests"] for i in range(params["streams"])}
        or reads.duplicated(["source", "arrive"]).any()
    ):
        raise RuntimeError("synthetic read population or admission order is ambiguous")
    reads["ordinal"] = reads.groupby("source").cumcount()
    reads["latency"] = reads["depart"] - reads["arrive"]
    if (
        len(reads) != stats["reads_sent"]
        or len(frame) - len(reads) != stats["writes_sent"]
        or int(reads["latency"].sum()) != stats["total_read_latency"]
    ):
        raise RuntimeError("synthetic trace/callback statistics disagree")
    return reads


def summarize(oracle, candidate, records, limit):
    paired = oracle.merge(
        candidate,
        on=["source", "ordinal"],
        suffixes=("_oracle", "_candidate"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if (
        paired.empty
        or not (paired["_merge"] == "both").all()
        or not (paired["addr_oracle"] == paired["addr_candidate"]).all()
    ):
        raise RuntimeError("synthetic exact read pairing failed; no partial score is returned")
    delta = paired["latency_candidate"] - paired["latency_oracle"]
    scale = float(paired["latency_oracle"].mean())
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("synthetic oracle mean latency is not positive and finite")
    paired["error"] = delta
    errors = paired_error_statistics(delta, scale)
    samples = paired.drop(columns="_merge")
    oracle_cycles = records["oracle"]["frontend_stats"]["cycles"]
    candidate_cycles = records["candidate"]["frontend_stats"]["cycles"]
    if any(type(v) is not int or v <= 0 for v in (oracle_cycles, candidate_cycles)):
        raise RuntimeError("invalid synthetic elapsed cycle counts")
    return {
        "matched_reads": len(paired),
        "read_coverage_oracle": 1.0,
        "read_coverage_candidate": 1.0,
        "L": scale,
        "L_definition": "mean of all oracle synthetic read latencies in DRAM cycles",
        "request_mae_over_L": errors["mae"],
        "signed_drift_over_L": errors["sgn"],
        "abs_signed_drift_over_L": abs(errors["sgn"]),
        "paired_p99_over_L": errors["tail"],
        "signed_min_cycles": errors["extreme_min_cycles"],
        "signed_max_cycles": errors["extreme_max_cycles"],
        "paired_errors": errors,
        "synthetic_elapsed_error_pct": 100 * (candidate_cycles - oracle_cycles) / oracle_cycles,
        "oracle": records["oracle"]["frontend_stats"],
        "candidate": records["candidate"]["frontend_stats"],
        "slices": {
            "first_reads": samples.head(limit).to_dict("records"),
            "most_negative": samples.nsmallest(limit, "error").to_dict("records"),
            "most_positive": samples.nlargest(limit, "error").to_dict("records"),
        },
    }
