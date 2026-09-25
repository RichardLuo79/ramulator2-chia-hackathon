"""Compute all metrics from raw run manifests and traces.

Everything derived lives here; runners only record raw data. Per workload:

Run level (from RAW per-core cycles, all aggregations reported):
  per-core signed deviations; mean |per-core dev| (headline for mixes);
  makespan deviation (max core cycles); signed-mean deviation
  (a cancellation-sensitive diagnostic).

Per request (occurrence-matched, coverage both ways):
  bulk MAE / L, signed mean / L, paired-error tail p99(|d|)/L, extremes.

Distribution level (full populations, no matching):
  p50/p99/p99.9 per side, direct p99 deviation, Wasserstein-1 / L,
  tail mass at the oracle q99.

Usage: python tools/eval/postprocess.py --frontend simpleo3 --std DDR5
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

from ramulator_chia.eval import artifacts as A
from ramulator_chia.eval import config as C
from ramulator_chia.eval import matchlib
from ramulator_chia.eval import metrics

SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def cycles_metrics(om, mm):
    raw = metrics.cycles_metrics(om, mm)
    return {key: [round(item, 4) for item in value] if isinstance(value, list) else round(value, 4)
            for key, value in raw.items()}


def request_metrics(oracle_trace, model_trace):
    r = matchlib.match(oracle_trace, model_trace)
    dv = r["dv"]
    if len(dv) == 0:
        raise ValueError("request traces contain no trusted pairs")
    if r["match_mode"] != "stable_id":
        raise ValueError(
            "closed-loop request scoring requires stable-ID pairing; got "
            f"{r['match_mode']}"
        )
    coverage_counts = (
        len(dv), r["n_o"], r["n_m"], r["stable_eligible_o"],
        r["stable_eligible_m"],
    )
    if len(set(coverage_counts)) != 1:
        raise ValueError(
            "closed-loop request scoring requires exact bidirectional eligible "
            f"coverage; counts={coverage_counts}"
        )
    ao, am = r["all_o"], r["all_m"]
    L = float(ao.mean())
    if not np.isfinite(L) or L <= 0:
        raise ValueError("full oracle mean read latency L must be finite and positive")
    grid = np.linspace(0.0001, 0.9999, 10_000)
    w1 = float(np.mean(np.abs(np.quantile(am, grid) - np.quantile(ao, grid))))
    q99o = float(np.percentile(ao, 99))
    return {
        "matched": int(len(dv)), "n_oracle": r["n_o"], "n_model": r["n_m"],
        "coverage_oracle": round(r["cov_o"], 6), "coverage_model": round(r["cov_m"], 6),
        "match_mode": r["match_mode"],
        "matcher_schema_version": r["matcher_schema_version"],
        "legacy_fallback_reason": r.get("legacy_fallback_reason"),
        "stable_eligible_oracle": r["stable_eligible_o"],
        "stable_eligible_model": r["stable_eligible_m"],
        "stable_coverage_oracle": round(r["stable_cov_o"], 6),
        "stable_coverage_model": round(r["stable_cov_m"], 6),
        "L_oracle_mean_lat": round(L, 2),
        "normalization": "full_oracle_read_mean_latency",
        "bulk_mae_over_L": round(float(np.abs(dv).mean()) / L, 5),
        "signed_mean_over_L": round(float(dv.mean()) / L, 5),
        "paired_tail_p99_over_L": round(float(np.percentile(np.abs(dv), 99)) / L, 4),
        "extreme_min_cyc": int(dv.min()), "extreme_max_cyc": int(dv.max()),
        "extreme_min_over_L": round(float(dv.min()) / L, 5),
        "extreme_max_over_L": round(float(dv.max()) / L, 5),
        "dist": {
            "p50_o": float(np.percentile(ao, 50)), "p50_m": float(np.percentile(am, 50)),
            "p99_o": q99o, "p99_m": float(np.percentile(am, 99)),
            "p999_o": float(np.percentile(ao, 99.9)),
            "p999_m": float(np.percentile(am, 99.9)),
            "direct_p99_dev_pct": round(100 * (float(np.percentile(am, 99)) - q99o) / q99o, 3),
            "w1_over_L": round(w1 / L, 5),
            "tail_mass_at_oracle_q99": round(float((am > q99o).mean()), 6),
            "pop_dev_pct": round(100 * (r["n_m"] - r["n_o"]) / r["n_o"], 4),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="simpleo3")
    ap.add_argument("--std", default="DDR5", choices=list(C.STD))
    ap.add_argument(
        "--models", default=",".join(C.MODEL_ORDER),
        help="comma-separated model output labels to compare",
    )
    ap.add_argument("--workloads", nargs="*", default=None)
    ap.add_argument(
        "--output", type=pathlib.Path, default=None,
        help="summary destination (default: <frontend>/<standard>/summary.json)",
    )
    ap.add_argument("--no-requests", action="store_true",
                    help="skip per-request/distribution metrics (no traces)")
    a = ap.parse_args()
    models = [model.strip() for model in a.models.split(",") if model.strip()]
    if (not models or len(set(models)) != len(models) or
            any(not SAFE_LABEL.fullmatch(model) or model == "oracle"
                for model in models)):
        ap.error("--models must contain unique safe non-oracle output labels")
    root = C.OUT / a.frontend / a.std
    wanted = set(a.workloads) if a.workloads is not None else None
    by_model = {model: {} for model in models}
    workload_dirs = sorted(root.iterdir() if root.exists() else [])
    for wldir in workload_dirs:
        if wanted is not None and wldir.name not in wanted:
            continue
        om_p = wldir / "oracle" / "manifest.json"
        if not om_p.exists():
            continue
        with om_p.open() as stream:
            om = json.load(stream)
        for model in models:
            mm_p = wldir / model / "manifest.json"
            if not mm_p.exists():
                continue
            with mm_p.open() as stream:
                mm = json.load(stream)
            row = {
                "cycles": cycles_metrics(om, mm),
                "wall_s": {"oracle": om.get("wall_s"), "model": mm.get("wall_s")},
            }
            ot = wldir / "oracle" / "trace.csv.ch0"
            mt = wldir / model / "trace.csv.ch0"
            if not a.no_requests:
                if not A.exists(ot) or not A.exists(mt):
                    raise RuntimeError(
                        f"missing request trace for {wldir.name}/{model}"
                    )
                row["requests"] = request_metrics(ot, mt)
            by_model[model][wldir.name] = row

    if wanted is not None:
        for model in models:
            missing = sorted(wanted - set(by_model[model]))
            if missing:
                raise RuntimeError(
                    f"missing completed oracle/{model} runs: {', '.join(missing)}"
                )

    model_summaries = {}
    for model, rows in by_model.items():
        if not rows:
            continue
        cycle_mae = [r["cycles"]["mean_abs_per_core_pct"] for r in rows.values()]
        core_extremes = [
            abs(value)
            for row in rows.values()
            for value in row["cycles"]["per_core_dev_pct"]
        ]
        request_rows = [r["requests"] for r in rows.values() if "requests" in r]
        oracle_wall = sum(
            r["wall_s"]["oracle"] for r in rows.values()
            if isinstance(r["wall_s"]["oracle"], (int, float))
        )
        model_wall = sum(
            r["wall_s"]["model"] for r in rows.values()
            if isinstance(r["wall_s"]["model"], (int, float))
        )
        aggregate = {
            "n_workloads": len(rows),
            "cycle_macro_mae_pct": round(float(np.mean(cycle_mae)), 4),
            "cycle_worst_core_abs_pct": round(float(np.max(core_extremes)), 4),
            "wall_s": round(model_wall, 3),
            "oracle_wall_s": round(oracle_wall, 3),
            "speedup_vs_oracle": (
                round(oracle_wall / model_wall, 3) if model_wall > 0 else None
            ),
        }
        if request_rows:
            aggregate.update({
                "request_macro_mae_over_L": round(float(np.mean([
                    row["bulk_mae_over_L"] for row in request_rows
                ])), 5),
                "request_macro_abs_signed_drift_over_L": round(float(np.mean([
                    abs(row["signed_mean_over_L"]) for row in request_rows
                ])), 5),
                "request_worst_paired_p99_over_L": round(float(np.max([
                    row["paired_tail_p99_over_L"] for row in request_rows
                ])), 5),
                "request_most_negative_over_L": round(float(np.min([
                    row["extreme_min_over_L"] for row in request_rows
                ])), 5),
                "request_most_positive_over_L": round(float(np.max([
                    row["extreme_max_over_L"] for row in request_rows
                ])), 5),
                "request_coverage": "exact_bidirectional_stable_id",
            })
        model_summaries[model] = {"aggregate": aggregate, "per_workload": rows}

    summary = {
        "summary_schema_version": 1,
        "rev": C.git_rev(),
        "frontend": a.frontend,
        "std": a.std,
        "reference_config": C.REFERENCE,
        "metric_definitions": {
            "cycle_macro_mae_pct": (
                "mean across workloads of mean absolute per-core cycle error (%)"),
            "request_macro_mae_over_L": (
                "mean across workloads of mean absolute paired request-latency "
                "error divided by L"),
            "request_macro_abs_signed_drift_over_L": (
                "mean across workloads of absolute signed mean paired error "
                "divided by L"),
            "request_worst_paired_p99_over_L": (
                "maximum workload p99 of absolute paired request error divided by L"),
            "L": "mean latency of all oracle reads in that workload",
        },
        "models": model_summaries,
    }
    out = a.output or (root / "summary.json")
    C.atomic_write_json(out, summary)

    print(f"{a.frontend} {a.std}")
    print(
        f"{'model':12s} {'n':>3s} {'cycle MAE%':>11s} {'request MAE/L':>14s} "
        f"{'|drift|/L':>11s} {'worst p99/L':>12s} {'speedup':>9s}"
    )
    for model, payload in model_summaries.items():
        agg = payload["aggregate"]
        print(
            f"{model:12s} {agg['n_workloads']:3d} "
            f"{agg['cycle_macro_mae_pct']:11.3f} "
            f"{agg.get('request_macro_mae_over_L', float('nan')):14.4f} "
            f"{agg.get('request_macro_abs_signed_drift_over_L', float('nan')):11.4f} "
            f"{agg.get('request_worst_paired_p99_over_L', float('nan')):12.4f} "
            f"{agg.get('speedup_vs_oracle') or float('nan'):9.2f}"
        )
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
