"""Operator-only figures/tables for completed initial Gemini comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tools.chia_loop.run_records import reporting_view

NAMES = {"seed": "Seed", "fixedlat": "FixedLat", "md1": "MD1", "wmg1": "WMG1",
         "mess": "MESS", "pro": "Gemini Pro", "flash": "Gemini Flash"}
COLORS = {"seed": "#7c8798", "fixedlat": "#bac3ce", "md1": "#dfb77d", "wmg1": "#c3a6c5",
          "mess": "#95b9a4", "pro": "#2677b5", "flash": "#da6a35"}


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = reporting_view(root)
    if manifest["status"] != "completed":
        raise RuntimeError("analysis requires completed/frozen campaigns")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    insts = manifest["instructions_per_core"]
    maximum_iterations = manifest["policy"]["maximum_iterations"]
    repaired_protocol = "iteration_unit" in manifest["policy"]
    for arm in manifest["models"]:
        NAMES[arm] = (manifest["models"][arm].replace("gemini-", "Gemini ")
                     .replace("-preview", "").replace("-", " ")
                     .replace(" pro", " Pro").replace(" flash", " Flash"))
        if manifest["arms"][arm]["incumbent"] == "seed":
            NAMES[arm] += " (seed)"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "savefig.dpi": 160})
    data = {}
    for split in ("training", "test"):
        reports = root / split / "reports"
        data[split] = {"seed": read(reports / "seed.json")["models"]["seed"],
                       **read(reports / "comparisons.json")["models"]}
        for arm in manifest["models"]:
            state = manifest["arms"][arm]
            data[split][arm] = state["selected"]["metrics"] if split == "training" else manifest["final_test_metrics"][arm]
    order = ["seed", *manifest["comparisons"], *manifest["models"]]
    rows, workloads = [], []
    for split, models in data.items():
        for model in order:
            rows.append({"split": split, "model": model, **models[model]["aggregate"]})
            for workload, values in models[model]["per_workload"].items():
                workloads.append({"split": split, "model": model, "workload": workload,
                    "core_signed_error_pct": values["cycles"]["per_core_dev_pct"][0],
                    **{k:v for k,v in values["requests"].items() if not isinstance(v, dict)}})
    for filename, records in (("headline.csv", rows), ("per_workload.csv", workloads)):
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (out / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7), constrained_layout=True)
    for column, split in enumerate(("training", "test")):
        for row, (key, ylabel) in enumerate((("cycle_macro_mae_pct", "Core-cycle MAE (%)"),
                                           ("request_macro_mae_over_L", "Request MAE / L"))):
            ax = axes[row, column]
            values = [data[split][model]["aggregate"][key] for model in order]
            ax.bar(range(len(order)), values, color=[COLORS[m] for m in order], width=.7)
            ax.set_xticks(range(len(order)), [NAMES[m].replace("Gemini ", "") for m in order], rotation=20)
            for i, value in enumerate(values):
                ax.annotate(f"{value:.2f}" if row == 0 else f"{value:.3f}", (i,value),
                            xytext=(0,4), textcoords="offset points", ha="center", fontsize=9)
            ax.set_ylim(0, max(values) * 1.23 + 1e-4); ax.grid(axis="y", alpha=.2)
            ax.set_ylabel(ylabel)
            if row == 0: ax.set_title("Training" if split == "training" else "Frozen final test")
    fig.suptitle(f"SimpleO3 + DDR5 · {insts:,} instructions/core · lower error is better", fontsize=13)
    fig.savefig(out / "headline.png"); fig.savefig(out / "headline.svg"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7), constrained_layout=True)
    for col, split in enumerate(("training", "test")):
        wls = list(data[split]["seed"]["per_workload"])
        for row in range(2):
            ax = axes[row,col]
            for i, model in enumerate(order):
                values = []
                for w in wls:
                    v = data[split][model]["per_workload"][w]
                    values.append(v["cycles"]["per_core_dev_pct"][0] if row == 0 else v["requests"]["bulk_mae_over_L"])
                width = .8 / len(order)
                ax.bar(np.arange(len(wls)) + (i-(len(order)-1)/2)*width, values, width=width*.96, color=COLORS[model], label=NAMES[model])
            ax.set_xticks(range(len(wls)), wls, rotation=15 if len(wls) > 2 else 0); ax.axhline(0, color="#555", lw=.7)
            ax.grid(axis="y", alpha=.2)
            ax.set_ylabel("Signed core-cycle error (%)" if row == 0 else "Request MAE / L")
            if row == 0: ax.set_title("Training" if split == "training" else "Frozen final test")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=7, frameon=False)
    fig.savefig(out / "per_workload.png"); fig.savefig(out / "per_workload.svg"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm in manifest["models"]:
        state = manifest["arms"][arm]
        for ax, key in zip(axes, ("cycle_macro_mae_pct", "request_macro_mae_over_L")):
            x, y = [0], [data["training"]["seed"]["aggregate"][key]]
            for attempt in state["history"]:
                x.append(attempt["iteration"])
                y.append(attempt["metrics"][key] if attempt.get("promoted") else y[-1])
                if attempt["status"] == "valid":
                    ax.scatter(attempt["iteration"], attempt["metrics"][key], color=COLORS[arm], marker="o", s=35)
                else:
                    ax.scatter(attempt["iteration"], y[-1], color=COLORS[arm], marker="x", s=60)
            ax.step(x,y, where="post", color=COLORS[arm], label=NAMES[arm])
            ax.set_xticks(range(maximum_iterations + 1)); ax.set_xlabel(
                "Design iteration (0 = seed)" if repaired_protocol else "Proposal attempt (0 = unmodified seed)")
            ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Training core-cycle MAE (%)"); axes[1].set_ylabel("Training request MAE / L")
    axes[0].legend(frameon=False)
    fig.suptitle("Incumbent trajectory · circles = valid trials; crosses = rejected attempts")
    fig.savefig(out / "evolution.png"); fig.savefig(out / "evolution.svg"); plt.close(fig)

    text = ["# Gemini CHIA results", "", "Independent run IDs: " +
        ", ".join(f"`{identifier}`" for identifier in manifest["run_ids"].values()) + ".", "",
        f"HIGH thinking, at most {maximum_iterations} " +
        ("evaluated designs (with draft repair)" if repaired_protocol else "proposals") +
        f" and USD {manifest['policy']['usd_cap']:g} per run. " +
        "Each starts from the same fixed-delay seed. Core cycles and read latencies come from closed-loop SimpleO3; "
        "the final test is evaluated only after selection is frozen. " +
        ("The historical shared execution waited for both selections; its raw records are unchanged."
         if len(manifest["models"]) > 1 else "This record contains exactly one model run."), "",
        f"Training: {', '.join(manifest['training'])}. Test: {', '.join(manifest['final_test'])}. "
        f"All are single-core, {insts:,} issued instructions with a fully drained ROI; DDR5_16Gb_x8 / DDR5_4800AN, "
        "refresh disabled. Cold-start full-prefix evaluation; no separate unscored warmup. "
        + ("The short ROI is only a pipeline check, not a generalization or model-ranking study."
           if insts < 20_000_000 else "One trial per backend is not evidence of a general model-capability ranking."), "",
        f"Generation ceiling: {manifest['policy']['maximum_output_tokens']:,} tokens per call. "
        "HIGH is a dynamic reasoning-effort setting, not an equal token allocation across models. "
        + manifest.get("meta_reviewer_test_exposure", "").capitalize() + ".", "",
        "## Outcomes and estimated spending", "",
        "| Run model | Selected source | Attempts | Valid | Promotions | API calls | Known-usage standard estimate (USD) | Unknown-usage calls | Conservative cap accounted (USD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in manifest["models"]:
        state = manifest["arms"][arm]; budget = state["budget"]
        text.append(f"| {state['model']} | {state['incumbent']} | {len(state['history'])} | "
            f"{sum(h['status']=='valid' for h in state['history'])} | {sum(bool(h.get('promoted')) for h in state['history'])} | "
            f"{budget['api_attempts']} | {budget['estimated_standard_usd']:.4f} | {budget['unknown_usage_calls']} | {budget['cap_charge_usd']:.4f} |")
    text.extend(["", "Estimates include thinking tokens, conservatively charge cached input at the full standard rate, "
        "and exclude credit/discount effects. They are not Cloud Billing invoices. The separate cap ledger uses "
        "higher tariffs and retains pessimistic reservations for unknown calls. "
        "Reported call counts cover each individual run; standard estimates and cap charges also include any explicitly "
        "recorded infrastructure-attempt carryover, available separately in the budget fields. "
        "[Pricing source](https://cloud.google.com/vertex-ai/generative-ai/pricing), checked 2026-09-05.", "",
        "## Accuracy", "", "![Headline comparison](headline.png)", ""])
    for split in ("training", "test"):
        text.extend([f"### {split.title()}", "", "| Model | Core MAE % | Request MAE/L | Abs drift/L | Worst paired P99/L | Most negative / positive d/L |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for model in order:
            a = data[split][model]["aggregate"]
            text.append(f"| {NAMES[model]} | {a['cycle_macro_mae_pct']:.3f} | {a['request_macro_mae_over_L']:.4f} | "
                f"{a['request_macro_abs_signed_drift_over_L']:.4f} | {a['request_worst_paired_p99_over_L']:.4f} | "
                f"{a['request_most_negative_over_L']:.3f} / {a['request_most_positive_over_L']:.3f} |")
        text.append("")
    text.extend(["![Workload breakdown](per_workload.png)", "", "![Evolution](evolution.png)", "",
                 "## Proposal record", "",
                 "The mechanism descriptions below are the proposing models' own **unverified claims**, "
                 "not conclusions of this report. Rejected proposals have no measured candidate accuracy. "
                 "A selected `seed` means no promoted generated design; its scores are the unchanged skeleton's scores.", ""])
    for arm in manifest["models"]:
        text.extend([f"### {NAMES[arm]}", ""])
        for h in manifest["arms"][arm]["history"]:
            desc = h["explanation"].get("mechanism", h["explanation"].get("reason", "No mechanism returned"))
            if not isinstance(desc, str): desc = json.dumps(desc)
            text.append(f"{h['iteration']}. `{h['id']}` from `{h['parent']}`: {h['status']}" +
                        (", promoted." if h.get("promoted") else ", not promoted.") + " " + desc)
            if h.get("reason"): text.extend(["", "   Rejection: " + h["reason"][:1000].replace("\n", " ")])
            text.append("")
    text.extend(["## Definitions and caveats", "",
        "For each workload, L is the mean latency of **all oracle logical reads**, in frontend cycles. "
        "Reads are paired one-to-one by stable request identity, with exact coverage and address/type agreement. "
        "For each pair, d = candidate latency − oracle latency. Request MAE/L = mean(|d|)/L; "
        "abs drift/L = |mean(d)|/L; paired P99/L = percentile99(|d|)/L. Headline MAE and drift average "
        "workloads equally; the headline P99 is the worst workload P99. Extremes are the smallest/largest d/L "
        "over workloads. Core MAE averages absolute relative per-core cycle errors within workload, then across workloads.", "",
        "The comparison models use their existing fixed harness settings, without retraining them. "
        "The MESS curve provenance remains in the run configuration. Historical feasibility-study scores are not "
        "used here because they are not this fresh branch's matched experiment.", "",
        "The no-command-scheduler/immutable-departure contract is enforced by frozen lifecycle code, static "
        "source restrictions, isolated builds, runtime file/network denial, and recorded orchestrating-agent "
        "semantic reviews. No human modeling hint was supplied. These reviews mean this is a Gemini proposal "
        "loop with an external compliance reviewer, not a single-model search with no external supervision.", "",
        "The protected driver copies the original simulation interleave exactly. Preflight found identical "
        "raw traces, core cycles, and integer stats. Floating diagnostic-stat serialization has lower precision "
        "than the Python dictionary; scored metrics are calculated from unchanged raw traces. "
        "Simulation-only wall times are recorded separately from construction and trace loading. They are "
        "instrumented and were not repeated under controlled CPU affinity, so no reliable speedup "
        "ranking is claimed. No transfer study, multi-seed replication, or rule ablation "
        "was performed by the runs reported here.", "",
        "## Artifacts", "",
        "Each new run has one `run_manifest.json`, `state.json`, `ledger.json`, and `interactions/` directory. "
        "Historical paired execution directories retain their original manifest and `arms/<backend>/` evidence; "
        "compact exports give each of those model runs its own record and link them only in a comparison. "
        "Candidate folders contain source, submitted-region/explanation records, build evidence and compliance reviews. "
        "Full requests, model responses and inspection results remain in their originating run with verified gzip "
        "compression for larger files. Raw traces have per-file checksum-bearing gzip archives; compact manifests "
        "and CSV tables remain directly readable. `headline.csv` and `per_workload.csv` retain numerical results.", ""])
    (out / "report.md").write_text("\n".join(text))
    print(out / "report.md")


if __name__ == "__main__":
    main()
