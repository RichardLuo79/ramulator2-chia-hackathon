"""Operator-only transfer tables/plots; never proposal or promotion input."""
import csv
import io
import pathlib


def write(root, report):
    frontend = report["frontend"]
    out = pathlib.Path(root) / "transfer/reports"
    lines = [f"# {frontend} DDR5 frozen-controller transfer", "",
        "The selected source and parameters are unchanged from SimpleO3 training. "
        "Seed and all four comparison models use the same frontend setup.", "",
        "Request latency is measured at the DRAM controller, not SimpleO3's logical LLC boundary. "
        "L is this workload's full-oracle mean controller read latency. "
        "Do not pool these values with SimpleO3 scores. Missing/partial pairing is not zero error.", "",
        "| Model | Core error (%) | Request MAE/L (diagnostic if partial) | Worst paired P99/L | Minimum O/M coverage | Exact request headline eligible? |",
        "| --- | ---: | ---: | ---: | --- | --- |"]
    csv_text = io.StringIO()
    writer = csv.writer(csv_text)
    writer.writerow(["frontend", "model", "workload", "family_group", "core_signed_error_pct", "core_abs_error_pct",
        "request_mae_over_L", "request_signed_drift_over_L", "paired_p99_over_L", "most_negative_over_L", "most_positive_over_L",
        "matched_reads", "oracle_reads", "model_reads", "coverage_oracle", "coverage_model", "request_headline_eligible"])
    for model, data in report["models"].items():
        a, d = data["aggregate"], data["aggregate"]["request_diagnostic"]
        lines.append(f"| {model} | {a['cycle_macro_mae_pct']:.4f} | {d['request_macro_mae_over_L']:.5f} | "
            f"{d['request_worst_paired_p99_over_L']:.4f} | {a['minimum_coverage_oracle']:.5f}/{a['minimum_coverage_model']:.5f} | "
            f"{'yes' if a['request_metric_eligible'] else 'no — partial pairing'} |")
        for workload, row in data["per_workload"].items():
            r = row["request"]
            writer.writerow([frontend, model, workload, row["family_group"], row["core_signed_error_pct"], row["core_abs_error_pct"],
                r["mae"], r["sgn"], r["tail"], r["most_negative_over_L"], r["most_positive_over_L"],
                r["matched"], r["n_oracle"], r["n_model"], r["cov_o"], r["cov_m"], r["request_metric_eligible"]])
    lines += ["", "Core cycles: " + report["core_metric"] + ". "
        "Request traces cover the simulator lifecycle, not necessarily exactly the scored core ROI. "
        "The CSV and JSON retain every workload, signed drift, paired tails, extremes, and coverage.", ""]
    (out / (frontend + ".md")).write_text("\n".join(lines))
    (out / (frontend + "_per_workload.csv")).write_text(csv_text.getvalue())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(report["models"])
    exact = all(report["models"][name]["aggregate"]["request_metric_eligible"] for name in names)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, key, title in zip(axes, ["cycle_macro_mae_pct", "request_macro_mae_over_L" if exact else "coverage"],
            ["Core-cycle error (%)", "Exact paired-request MAE/L" if exact else
             "Trusted read-pair coverage (%)\n(request-error headline ineligible)"]):
        values = []
        for name in names:
            a = report["models"][name]["aggregate"]
            values.append(100 * min(a["minimum_coverage_oracle"], a["minimum_coverage_model"]) if key == "coverage"
                          else a[key] if key in a else a["request_diagnostic"][key])
        bars = ax.barh(names, values, color=["#297A59" if n == "selected" else "#667C99" for n in names])
        ax.bar_label(bars, fmt="%.6f" if key == "coverage" else "%.3f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_xlim(0, max(values, default=1) * 1.25 or 1)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(frontend + " · DDR5 · frozen-source transfer")
    fig.savefig(out / (frontend + ".svg"))
    plt.close(fig)
