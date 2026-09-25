"""Per-request metrics for a --record-requests ChampSim run.

Matches each <trace>_oracle_req.csv.ch0 / <trace>_<label>_req.csv.ch0 pair in
the run's output directory and prints the same per-request table as
postprocess.py (bulk MAE/L, signed drift/L, paired tail, coverage), writing
request_summary.json alongside the traces.

The checked-in ChampSim bridge encodes virtual transaction line plus demand
kind in ``frontend_sub_id``. This command uses matchlib's explicit
ChampSim-only filter: it pairs logical stable identities first, reports both
logical coverage and physical-address consistency, and computes latency error
only for exact-physical pairs. Generic matchlib callers remain fail-closed on
any stable-identity address mismatch.

Usage: python tools/eval/match_champsim.py eval_out/champsim/DDR5_champreq [--label eval]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = __import__("ramulator_chia.layout", fromlist=["RAMULATOR"]).RAMULATOR
sys.path.insert(0, str(REPO_ROOT))
from resources.champsim_bridge.cs_gentest import LABEL_PATTERN, TRACES  # noqa: E402
from ramulator_chia.eval import artifacts as A  # noqa: E402
from ramulator_chia.eval import matchlib  # noqa: E402


def _indexed_paths(rundir, suffix):
    return {
        path.name[: -len(suffix)]: path
        for path in A.glob_logical(rundir / f"*{suffix}")
    }


def _require_complete_matrix(rundir, label):
    if not LABEL_PATTERN.fullmatch(label):
        raise ValueError(f"invalid label {label!r}")

    expected = set(TRACES)
    oracle_suffix = "_oracle_req.csv.ch0"
    model_suffix = f"_{label}_req.csv.ch0"
    oracle_paths = _indexed_paths(rundir, oracle_suffix)
    model_paths = _indexed_paths(rundir, model_suffix)
    for kind, paths in (("oracle", oracle_paths), (label, model_paths)):
        found = set(paths)
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        if missing or unexpected:
            raise RuntimeError(
                f"incomplete {kind} request-trace matrix: "
                f"missing={missing}, unexpected={unexpected}"
            )
    return oracle_paths, model_paths


def summarize(rundir, label):
    oracle_paths, model_paths = _require_complete_matrix(rundir, label)
    rows = {}
    for trace in TRACES:
        op = oracle_paths[trace]
        mp = model_paths[trace]
        r = matchlib.match(
            op, mp, champsim_filter_physical_mismatches=True
        )
        dv = r["dv"]
        if not len(dv):
            raise RuntimeError(f"{trace}: request matcher produced zero pairs")
        L = float(r["all_o"].mean())
        if not np.isfinite(L) or L <= 0:
            raise RuntimeError(f"{trace}: invalid full-oracle mean read latency {L}")
        stable_mode = r["match_mode"] == "stable_id"
        rows[trace] = {
            "L": round(L, 1),
            "mae": round(float(np.abs(dv).mean()) / L, 5),
            "sgn": round(float(dv.mean()) / L, 5),
            "tail": round(float(np.percentile(np.abs(dv), 99)) / L, 4),
            "cov_o": round(r["cov_o"], 5),
            "cov_m": round(r["cov_m"], 5),
            "n": int(len(dv)),
            "n_o": r["n_o"],
            "n_m": r["n_m"],
            "match_mode": r["match_mode"],
            "matcher_schema_version": r["matcher_schema_version"],
            "stable_eligible_o": r["stable_eligible_o"],
            "stable_eligible_m": r["stable_eligible_m"],
            "stable_cov_o": round(r["stable_cov_o"], 5),
            "stable_cov_m": round(r["stable_cov_m"], 5),
            "stable_eligibility_fraction_o": round(
                r["stable_eligibility_fraction_o"], 5
            ),
            "stable_eligibility_fraction_m": round(
                r["stable_eligibility_fraction_m"], 5
            ),
            "stable_logical_pairs": r["stable_logical_pairs"],
            "logical_cov_o": round(r["logical_cov_o"], 5),
            "logical_cov_m": round(r["logical_cov_m"], 5),
            "logical_stable_cov_o": round(r["logical_stable_cov_o"], 5),
            "logical_stable_cov_m": round(r["logical_stable_cov_m"], 5),
            "address_mismatch_pairs": r["address_mismatch_pairs"],
            "physical_consistent_pairs": r["physical_consistent_pairs"],
            "physical_consistency_rate": round(
                r["physical_consistency_rate"], 5
            ),
            "trusted_pair_policy": r["trusted_pair_policy"],
            "stable_identity_address_space": (
                "virtual_transaction" if stable_mode else None
            ),
            "exact_physical_stable_cov_o": (
                round(r["stable_cov_o"], 5) if stable_mode else None
            ),
            "exact_physical_stable_cov_m": (
                round(r["stable_cov_m"], 5) if stable_mode else None
            ),
            "legacy_fallback_reason": r.get("legacy_fallback_reason"),
            "normalization": "full_oracle_read_mean_latency",
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", type=pathlib.Path)
    ap.add_argument("--label", default="eval")
    a = ap.parse_args()
    rows = summarize(a.rundir, a.label)
    out = a.rundir / "request_summary.json"
    with out.open("w") as stream:
        json.dump(rows, stream, indent=1)
    header = (
        f"{'trace':24s} {'L':>8s} {'mae/L':>7s} {'sgn/L':>7s} "
        f"{'tail':>6s} {'logical':>7s} {'phys':>7s} {'trusted':>7s} {'n':>9s}"
    )
    print(header)
    for t, r in rows.items():
        print(f"{t:24s} {r['L']:8.1f} {r['mae']:7.3f} {r['sgn']:+7.3f} "
              f"{r['tail']:6.2f} {r['logical_cov_o']:7.4f} "
              f"{r['physical_consistency_rate']:7.4f} {r['cov_o']:7.4f} {r['n']:9d}")
    print(f"mean mae/L {np.mean([r['mae'] for r in rows.values()]):.3f}  "
          f"mean|sgn|/L {np.mean([abs(r['sgn']) for r in rows.values()]):.3f}")
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
