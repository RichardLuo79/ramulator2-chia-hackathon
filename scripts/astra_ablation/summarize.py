#!/usr/bin/env python3
"""Tables for the Astra single-core ablation from the run's comparison rows (frozen matcher output).

Per-case metrics are taken verbatim from the compare_transfer scores in comparisons.json:
core error = cycles.per_core_dev_pct[0] (|.| = mean_abs_per_core_pct), MAE/L = request.mae,
signed drift/L = request.sgn, P99/L = request.tail, P99.9/L = request.p999_over_L,
coverage = request.cov_o / request.cov_m. Group values are means over the 24 cases (as the published
headline). The control's means must equal the published headline exactly.

  summarize.py --run DIR (results/foundation/astra-ablation/run) --variants variants.json --expected expected_control.json --out DIR
"""
import argparse, csv, json, statistics
from pathlib import Path

SUMMARY = ["variant", "description", "lines_changed", "core_err_pct", "mae_over_L", "signed_drift_over_L",
           "p99_over_L", "coverage", "delta_core_vs_control", "delta_mae_vs_control",
           "core_signed_pct", "model_coverage", "p999_over_L", "worst_case", "worst_case_core_abs_pct",
           "cases", "candidate_id", "model_cpp_sha256"]
PER_CASE = ["variant", "case", "core_signed_pct", "core_abs_pct", "mae_over_L", "signed_drift_over_L", "p99_over_L",
            "p999_over_L", "L_cycles", "oracle_coverage", "model_coverage", "pairs", "address_mismatch_pairs",
            "delta_core_abs_vs_control", "delta_core_signed_vs_control", "delta_mae_vs_control",
            "model_receipt_sha256", "oracle_receipt_sha256", "comparison_id", "candidate_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--variants", required=True, type=Path)
    ap.add_argument("--expected", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    plan = json.load(open(a.run / "plan.json"))
    comps = json.load(open(a.run / "comparisons.json"))
    meta = {f"{v['id']}-{v['name']}": v for v in json.load(open(a.variants))}
    exp = json.load(open(a.expected))
    cid = {v: r["candidate"]["candidate_id"] for v, r in plan["variants"].items()}
    rows = {}
    for j in plan["jobs"]:
        if j["arm"] == "oracle":
            continue
        c = comps.get(f"{j['arm']}-{j['id']}")
        if not c or "score" not in c:
            raise SystemExit(f"missing comparison: {j['variant']} {j['case']}")
        s, q = c["score"], c["score"]["request"]
        rows[j["variant"], j["case"]] = dict(
            variant=j["variant"], case=j["case"], core_signed_pct=s["cycles"]["per_core_dev_pct"][0],
            core_abs_pct=s["cycles"]["mean_abs_per_core_pct"], mae_over_L=q["mae"], signed_drift_over_L=q["sgn"],
            p99_over_L=q["tail"], p999_over_L=q["p999_over_L"], L_cycles=q["oracle_read_mean_latency"],
            oracle_coverage=q["cov_o"], model_coverage=q["cov_m"], pairs=q["matched"],
            address_mismatch_pairs=q["address_mismatch_pairs"], model_receipt_sha256=c["model_receipt_sha256"],
            oracle_receipt_sha256=c["oracle_receipt_sha256"], comparison_id=c["id"], candidate_id=cid[j["variant"]])
    variants = sorted({v for v, _ in rows})
    cases = sorted({k for _, k in rows})
    ctrl = "00-control"
    for (v, k), r in rows.items():
        b = rows[ctrl, k]
        r["delta_core_abs_vs_control"] = r["core_abs_pct"] - b["core_abs_pct"]
        r["delta_core_signed_vs_control"] = r["core_signed_pct"] - b["core_signed_pct"]
        r["delta_mae_vs_control"] = r["mae_over_L"] - b["mae_over_L"]
    mean = lambda v, f: statistics.fmean(rows[v, k][f] for k in cases)
    summary, movers = [], {}
    for v in variants:
        worst = max(cases, key=lambda k: rows[v, k]["core_abs_pct"])
        summary.append(dict(
            variant=v, description=meta[v]["description"], lines_changed=meta[v]["lines_changed"],
            core_err_pct=mean(v, "core_abs_pct"), mae_over_L=mean(v, "mae_over_L"),
            signed_drift_over_L=mean(v, "signed_drift_over_L"), p99_over_L=mean(v, "p99_over_L"),
            coverage=mean(v, "oracle_coverage"),
            delta_core_vs_control=mean(v, "core_abs_pct") - mean(ctrl, "core_abs_pct"),
            delta_mae_vs_control=mean(v, "mae_over_L") - mean(ctrl, "mae_over_L"),
            core_signed_pct=mean(v, "core_signed_pct"), model_coverage=mean(v, "model_coverage"),
            p999_over_L=mean(v, "p999_over_L"), worst_case=worst, worst_case_core_abs_pct=rows[v, worst]["core_abs_pct"],
            cases=len(cases), candidate_id=cid[v], model_cpp_sha256=meta[v]["model_cpp_sha256"]))
        movers[v] = dict(
            core=[(k, rows[v, k]["delta_core_abs_vs_control"], rows[ctrl, k]["core_signed_pct"], rows[v, k]["core_signed_pct"])
                  for k in sorted(cases, key=lambda k: -abs(rows[v, k]["delta_core_abs_vs_control"]))[:5]],
            mae=[(k, rows[v, k]["delta_mae_vs_control"], rows[ctrl, k]["mae_over_L"], rows[v, k]["mae_over_L"])
                 for k in sorted(cases, key=lambda k: -abs(rows[v, k]["delta_mae_vs_control"]))[:5]],
            improved_core=[k for k in cases if rows[v, k]["core_abs_pct"] < rows[ctrl, k]["core_abs_pct"]],
            improved_mae=[k for k in cases if rows[v, k]["mae_over_L"] < rows[ctrl, k]["mae_over_L"]],
            identical_cases=[k for k in cases if rows[v, k]["model_receipt_sha256"] and
                             rows[v, k]["core_signed_pct"] == rows[ctrl, k]["core_signed_pct"] and
                             rows[v, k]["mae_over_L"] == rows[ctrl, k]["mae_over_L"]])
    # control must reproduce the published headline exactly
    h = exp["headline"]
    check = dict(core_abs_pct=(mean(ctrl, "core_abs_pct"), h["core_abs_pct"]), mae_L=(mean(ctrl, "mae_over_L"), h["mae_L"]),
                 drift_L=(mean(ctrl, "signed_drift_over_L"), h["drift_L"]), p99_L=(mean(ctrl, "p99_over_L"), h["p99_L"]),
                 core_signed_pct=(mean(ctrl, "core_signed_pct"), h["core_signed_pct"]),
                 oracle_coverage=(mean(ctrl, "oracle_coverage"), h["oracle_coverage"]),
                 model_coverage=(mean(ctrl, "model_coverage"), h["model_coverage"]))
    headline_ok = all(x == y for x, y in check.values())
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "ablation_astra_sc_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, SUMMARY); w.writeheader(); w.writerows(summary)
    with open(a.out / "ablation_astra_sc_per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, PER_CASE); w.writeheader()
        w.writerows(rows[v, k] for v in variants for k in cases)
    (a.out / "ablation_astra_sc_movers.json").write_text(json.dumps(dict(
        control_headline_check={k: dict(reproduced=x, published=y, equal=x == y) for k, (x, y) in check.items()},
        control_headline_bit_identical=headline_ok, movers=movers), indent=1) + "\n")
    print(json.dumps(dict(control_headline_bit_identical=headline_ok)))
    for r in summary:
        print(f"{r['variant']:24s} core {r['core_err_pct']:8.4f} (d {r['delta_core_vs_control']:+8.4f})  "
              f"mae/L {r['mae_over_L']:.5f} (d {r['delta_mae_vs_control']:+.5f})  drift {r['signed_drift_over_L']:+.5f}  "
              f"p99 {r['p99_over_L']:.4f}  cov {r['coverage']:.6f}")


if __name__ == "__main__":
    main()
