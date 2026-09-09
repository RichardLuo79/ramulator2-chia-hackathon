"""Pair fixed-arrival controller replay with its recorded oracle population.

This is an open-loop diagnostic, not a CPU evaluation. Backpressure delays
admission, so response latency is callback/departure minus the *offered* arrival.
Write callbacks acknowledge writes; they need not mark physical bus retirement.
"""

import numpy as np
import pandas as pd

from . import artifacts
from .metrics import paired_error_statistics

ARRIVAL_COLUMNS = ("arrive", "addr", "type", "source")


def read_frame(path, columns):
    with artifacts.open_text(path) as stream:
        return pd.read_csv(stream, usecols=list(columns), dtype={key: "int64" for key in columns})


def oracle_population(path, controller_stats):
    """Require recorded admission order; never substitute completion/file order."""
    columns = (*ARRIVAL_COLUMNS, "depart", "admission_ordinal")
    frame = read_frame(path, columns)
    if (
        frame.empty
        or frame.isna().any().any()
        or not frame["type"].isin([0, 1]).all()
        or (frame[["arrive", "addr", "admission_ordinal"]] < 0).any().any()
        # SimpleO3 cache writebacks legitimately have no originating core.
        # Preserve -1; assigning a fabricated core would change the stream.
        or (frame["source"] < -1).any()
        or frame["admission_ordinal"].duplicated().any()
        or (frame["depart"] < frame["arrive"]).any()
    ):
        raise ValueError("oracle controller input lacks a valid exact admission population")
    reads = frame["type"] == 0
    if not reads.any() or (frame.loc[reads, "depart"] <= frame.loc[reads, "arrive"]).any():
        raise ValueError("oracle controller read latency must be positive")
    if any(
        int((frame["type"] == kind).sum()) != controller_stats[f"num_{name}_reqs"]
        for kind, name in ((0, "read"), (1, "write"))
    ):
        raise ValueError("oracle controller trace does not cover the admitted population")
    return frame.sort_values(["arrive", "admission_ordinal"]).reset_index(drop=True)


def summarize(oracle, replay_path, batch_stats):
    columns = (*ARRIVAL_COLUMNS, "admit", "depart", "frontend_id", "frontend_sub_id")
    frame = read_frame(replay_path, columns)
    if len(frame) != len(oracle) or frame.empty:
        raise ValueError("replay changed the full offered population")
    if (
        not np.array_equal(frame["frontend_id"], np.arange(len(frame)))
        or (frame["frontend_sub_id"] != 0).any()
        or not frame[list(ARRIVAL_COLUMNS)].equals(oracle[list(ARRIVAL_COLUMNS)])
        or (frame["admit"] < frame["arrive"]).any()
        or (frame["depart"] < frame["admit"]).any()
        or (frame["depart"] > batch_stats["cycles"]).any()
    ):
        raise ValueError("replay identity, arrival, admission or callback order changed")
    reads = frame["type"] == 0
    if (
        batch_stats["reads_completed"] != int(reads.sum())
        or batch_stats["writes_completed"] != int((~reads).sum())
        or (frame.loc[reads, "depart"] <= frame.loc[reads, "admit"]).any()
    ):
        raise ValueError("replay read/write callbacks did not drain correctly")
    oracle_latency = oracle.loc[reads, "depart"] - oracle.loc[reads, "arrive"]
    replay_latency = frame.loc[reads, "depart"] - frame.loc[reads, "arrive"]
    scale = float(oracle_latency.mean())
    wait = frame.loc[reads, "admit"] - frame.loc[reads, "arrive"]
    return {
        "diagnostic_only": True,
        "eligible_for_promotion": False,
        "matched_reads": int(reads.sum()),
        "offered_writes": int((~reads).sum()),
        "read_coverage_oracle": 1.0,
        "read_coverage_candidate": 1.0,
        "L": scale,
        "L_definition": (
            "mean of all recorded training oracle controller read latencies in DRAM cycles"
        ),
        "latency_definition": "departure minus offered arrival, including replay admission waiting",
        "reference": "recorded oracle controller completions",
        "paired_errors": paired_error_statistics(replay_latency - oracle_latency, scale),
        "read_admission_wait_cycles": {"mean": float(wait.mean()), "maximum": int(wait.max())},
        "callback_drain_cycles": batch_stats["cycles"],
        "write_semantics": "all write acknowledgments drained; not physical write-burst retirement",
    }
