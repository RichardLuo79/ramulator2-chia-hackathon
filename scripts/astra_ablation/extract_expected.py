#!/usr/bin/env python3
"""Extract the published Astra single-core control expectations (local, read-only).

Writes one JSON with, per single-core test case at DDR5_16Gb_x8_q64: the published per-case score
row (results/foundation/campaigns.json), the hardware-study job id of the same endpoint at the
default point (results/extensions/transfer.json; equals digest_json(measurement identity)), and the
frozen endpoint candidate identity (results/manifests/endpoints.json).

  extract_expected.py --artifact <artifact root> --out expected_control.json
"""
import argparse, json, statistics
from pathlib import Path

FLOATS = ("core_abs_pct", "core_signed_pct", "mae_L", "drift_L", "abs_drift_L", "p99_L", "p999_L", "L",
          "oracle_coverage", "model_coverage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    scores = Path("results/foundation/campaigns.json")
    transfer = Path("results/extensions/transfer.json")
    pub = [r for r in json.load(open(a.artifact / scores))["rows"]
           if r["arm"] == "astra_single_core" and r["cohort"] == "test" and r["cores"] == 1]
    hw = {r["case"]: r for r in json.load(open(a.artifact / transfer))["rows"]
          if r["study"] == "hardware" and r["arm"] == "astra_single_core" and r["point"] == "DDR5_16Gb_x8_q64"
          and r["cores"] == 1}
    endpoint = json.load(open(a.artifact / "results/manifests/endpoints.json"))["astra_single_core"]["candidate"]
    cases = {}
    for r in sorted(pub, key=lambda r: r["case"]):
        h = hw[r["case"]]
        if float(h["signed_pct"]) != float(r["core_signed_pct"]) or float(h["request_mae_over_L"]) != float(r["mae_L"]):
            raise SystemExit("hardware default point differs from postrun: " + r["case"])
        cases[r["case"]] = dict({k: float(r[k]) for k in FLOATS}, pairs=int(r["pairs"]),
                                model_receipt=r["model_receipt"], oracle_receipt=r["oracle_receipt"],
                                hardware_job_id=h["id"])
    if len(cases) != 24:
        raise SystemExit("expected 24 single-core test cases")
    mean = lambda k: statistics.fmean(c[k] for c in cases.values())
    out = dict(schema="astra-sc-ablation-expected-v1", point="DDR5_16Gb_x8_q64", endpoint_candidate=endpoint,
               headline={k: mean(k) for k in ("core_abs_pct", "core_signed_pct", "mae_L", "drift_L",
                                               "p99_L", "oracle_coverage", "model_coverage")},
               cases=cases,
               sources=dict(per_case=str(scores), hardware=str(transfer),
                            endpoints="results/manifests/endpoints.json"))
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["headline"]))


if __name__ == "__main__":
    main()
