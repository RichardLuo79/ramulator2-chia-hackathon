#!/usr/bin/env python3
"""Render the paper's data figures (Figs. 2-9) from the CSVs in results/paper/."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "paper"
OUTPUT = ROOT / "results" / "figures"

COLORS = {"astra_single_core": "#0072B2", "astra_multicore": "#56B4E9", "deepseek_single_core": "#D55E00",
          "gemini_single_core": "#CC79A7", "opus_single_core": "#009E73", "fixedlat": "#9C9C9C",
          "md1": "#454545", "wmg1": "#7E7061", "mess": "#666666"}
NAMES = {"fixedlat": "Queued FixedLat", "md1": "M/D/1", "wmg1": "WMG1", "mess": "Mess",
         "astra_single_core": "GPT-6 Astra", "deepseek_single_core": "DeepSeek V4.1 Flash",
         "gemini_single_core": "Gemini 3.8 Flash", "opus_single_core": "Opus 5.5"}
WRAPPED_NAMES = {**NAMES, "fixedlat": "Queued\nFixedLat", "astra_single_core": "GPT-6\nAstra",
                 "deepseek_single_core": "DeepSeek\nV4.1 Flash", "gemini_single_core": "Gemini\n3.8 Flash"}
BASELINES = ("fixedlat", "md1", "wmg1", "mess")
MODELS = ("astra_single_core", "deepseek_single_core", "gemini_single_core", "opus_single_core")
ARMS = BASELINES + MODELS
COMPACT_WIDTH = 3.5 * .95
MAE_REQ = r"$\mathrm{MAE}_{\mathrm{Req}}$"


def read(name: str) -> list[dict]:
    """Rows of results/paper/<name>.csv, with numeric fields converted."""
    def value(text: str):
        for kind in (int, float):
            try:
                return kind(text)
            except ValueError:
                pass
        return {"True": True, "False": False}.get(text, text)
    with (DATA / f"{name}.csv").open(newline="") as f:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": .65,
        "lines.linewidth": 1.25, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "chia-four-page-paper", "savefig.dpi": 200})


def save(fig, output: Path, name: str, *, strip_svg: bool = True) -> None:
    """Write PDF/SVG/PNG, trimming blank top/bottom margins without changing the width."""
    fig.canvas.draw()
    content = fig.get_tightbbox(fig.canvas.get_renderer())
    pad = 1.5 / 72
    crop = Bbox.from_extents(0, max(0, content.y0 - pad), fig.get_figwidth(),
                             min(fig.get_figheight(), content.y1 + pad))
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else ({"Date": None} if ext == "svg" else {})
        path = output / f"{name}.{ext}"
        fig.savefig(path, metadata=metadata, bbox_inches=crop, pad_inches=0)
        if ext == "svg" and strip_svg:
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def check_figure_text(fig) -> None:
    """Fail before export if a visible label would be clipped by the canvas."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if (box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5
                or box.y1 > fig.bbox.height + .5):
            raise ValueError(f"Clipped figure label: {text.get_text()!r}")


# Fig. 2 ---------------------------------------------------------------------------------------

WORKLOAD_LABELS = {"bark": "Bark", "biogpt": "BioGPT", "cam4": "cam4", "gap_bc": "GAP bc", "gcc": "gcc",
                   "ligra_cf": "Ligra CF", "ligra_radii": "Ligra radii", "pop2": "pop2", "xalancbmk": "xalancbmk"}


def workload_label(case: str) -> str:
    key = case.removeprefix("test-c1-")
    if key.startswith("sierra_a_"):
        _, _, group, region = key.split("_")
        return f"S{group}/{region}"
    return WORKLOAD_LABELS.get(key) or key.replace("tahoe_", "T/")


def label_cut_bars(fig, ax, cuts: list[tuple[float, str, str]], size: float) -> None:
    """Write each cut bar's value above the panel in one row, spreading overlapping values sideways."""
    renderer = fig.canvas.get_renderer()
    points = 72 / fig.dpi
    notes = [ax.annotate(text, (xi, 1), xycoords=("data", "axes fraction"), xytext=(0, 1.5),
                         textcoords="offset points", ha="center", va="bottom", fontsize=size, color=color)
             for xi, text, color in sorted(cuts)]
    anchors = [ax.transData.transform((xi, 0))[0] * points for xi, _, _ in sorted(cuts)]
    widths = [note.get_window_extent(renderer).width * points for note in notes]
    gap = 1.5
    # Merge overlapping neighbours into clusters, each laid out side by side around the mean
    # of its bars, until no two clusters overlap.
    clusters = [[i] for i in range(len(notes))]

    def layout(cluster: list[int]) -> list[float]:
        total = sum(widths[i] for i in cluster) + gap * (len(cluster) - 1)
        left = sum(anchors[i] for i in cluster) / len(cluster) - total / 2
        centers = []
        for i in cluster:
            centers.append(left + widths[i] / 2)
            left += widths[i] + gap
        return centers

    merged = True
    while merged:
        merged = False
        for k in range(len(clusters) - 1):
            a, b = clusters[k], clusters[k + 1]
            right = layout(a)[-1] + widths[a[-1]] / 2
            left = layout(b)[0] - widths[b[0]] / 2
            if right + gap > left:
                clusters[k:k + 2] = [a + b]
                merged = True
                break
    for cluster in clusters:
        for i, center in zip(cluster, layout(cluster)):
            notes[i].set_position((center - anchors[i], 1.5))


def fig2_per_workload(output: Path = OUTPUT) -> None:
    arms = ("mess",) + MODELS
    panels = (("core_abs_pct", "Core-cycle\nerror (%)", 10, (0, 5, 10),
               lambda v: f"{v:.0f}%" if v >= 10 else f"{v:.1f}%"),
              ("mae_L", MAE_REQ, .8, (0, .4, .8), lambda v: f"{v:.2f}"))
    rows = {(r["model"], r["case"]): r for r in read("single-core-test")}
    cases = sorted({case for _, case in rows})
    x = np.arange(len(cases))
    width = .8 / len(arms)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 2.3), sharex=True)
    fig.subplots_adjust(left=.065, right=.995, top=.845, bottom=.22, hspace=.3)
    cuts = {ax: [] for ax in axes}
    for ax, (metric, ylabel, ymax, ticks, fmt) in zip(axes, panels):
        for i, arm in enumerate(arms):
            values = np.array([float(rows[arm, case][metric]) for case in cases])
            pos = x + (i - (len(arms) - 1) / 2) * width
            ax.bar(pos, values, width, color=COLORS[arm], label=NAMES[arm], lw=0)
            for xi, value in zip(pos, values):
                if value <= ymax:
                    continue
                # Break marks on the cut bar; its value goes above the panel.
                for y in (.88 * ymax, .93 * ymax):
                    ax.plot([xi - .7 * width, xi + .7 * width], [y - .025 * ymax, y + .025 * ymax],
                            color="white", lw=1.1, clip_on=False, zorder=3)
                cuts[ax].append((xi, fmt(value), COLORS[arm]))
        ax.set(ylim=(0, ymax), yticks=ticks)
        ax.set_ylabel(ylabel, fontsize=7, labelpad=1.5)
        ax.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        ax.margins(x=.01)
    fig.align_ylabels(axes)
    for ax in axes:
        label_cut_bars(fig, ax, cuts[ax], 6)
    axes[1].set_xticks(x, [workload_label(case) for case in cases], rotation=45, fontsize=6.5,
                       ha="right", rotation_mode="anchor")
    # Matplotlib fills legend columns first; reorder so the legend reads in model order.
    handles, names = axes[0].get_legend_handles_labels()
    ncol = 5
    order = [i for col in range(ncol) for i in range(col, len(handles), ncol)]
    fig.legend([handles[i] for i in order], [names[i] for i in order], loc="upper center",
               bbox_to_anchor=(.53, 1.0), ncol=ncol, frameon=False, columnspacing=1.2,
               handlelength=1.2, fontsize=7)
    save(fig, output, "fig2-per-workload")


# Fig. 3 ---------------------------------------------------------------------------------------

def fig3_accuracy(output: Path = OUTPUT) -> None:
    rows = {r["model"]: r for r in read("single-core-test-means")}
    fig, axes = plt.subplots(2, 1, figsize=(COMPACT_WIDTH, 1.88), sharex=True)
    fig.subplots_adjust(left=.155, right=.985, bottom=.35, top=.97, hspace=.42)
    x = np.arange(len(ARMS))
    for ax, metric, title, limit in zip(axes, ("core_abs_pct", "mae_L"),
            ("Core-cycle\nerror (%)", MAE_REQ), (29, .66)):
        vals = [float(rows[a][metric]) for a in ARMS]
        ax.bar(x, vals, width=.70, color=[COLORS[a] for a in ARMS], zorder=3)
        for i, val in enumerate(vals):
            ax.text(i, val + limit * .025, f"{val:.2f}" if metric == "core_abs_pct" else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set(ylim=(0, limit), ylabel=title)
        ax.set_yticks([0, 10, 20] if metric == "core_abs_pct" else [0, .2, .4, .6])
        ax.grid(axis="y", alpha=.18, zorder=0)
        ax.axvline(3.5, color="#B2182B", lw=1.05, ls=(0, (3, 2)), zorder=4)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=7, length=2)
        ax.yaxis.label.set_size(7.5)
    axes[-1].set_xticks(x, [WRAPPED_NAMES[a] for a in ARMS], rotation=50, ha="right",
                       rotation_mode="anchor", fontsize=7, linespacing=1.0)
    check_figure_text(fig)
    save(fig, output, "fig3-accuracy")


# Fig. 4 ---------------------------------------------------------------------------------------

def fig4_convergence(output: Path = OUTPUT) -> None:
    rows = read("convergence")
    fig, axes = plt.subplots(2, 2, figsize=(COMPACT_WIDTH, 2.16), sharex=True, sharey="row")
    fig.subplots_adjust(left=.19, right=.975, bottom=.17, top=.705, wspace=.16, hspace=.25)
    for i, metric in enumerate(("core_abs_pct", "mae_L")):
        for j, split in enumerate(("training", "validation")):
            ax = axes[i, j]
            for c in ("astra", "deepseek", "gemini", "opus"):
                color = COLORS[c + "_single_core"]
                chosen = sorted((r for r in rows if r["model"] == c and r["split"] == split
                                 and r["kind"] == "selected"), key=lambda r: r["iteration"])
                ax.plot([r["iteration"] for r in chosen], [r[metric] for r in chosen],
                        ls="-", lw=.9, color=color, label=NAMES[c + "_single_core"],
                        gid=f"{c}:chosen", zorder=2)
                for accepted, marker in ((True, "o"), (False, "x")):
                    proposed = sorted((r for r in rows if r["model"] == c and r["split"] == split
                                       and r["kind"] == "candidate" and r["accepted"] == accepted),
                                      key=lambda r: r["iteration"])
                    ax.plot([r["iteration"] for r in proposed], [r[metric] for r in proposed],
                            ls="none", marker=marker, ms=2.5 if accepted else 3.5,
                            mew=.6 if accepted else .8, color=color,
                            gid=f"{c}:{'accepted' if accepted else 'rejected'}", zorder=3 if accepted else 4)
            ax.set(xlim=(-.25, 10.3))
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.tick_params(labelsize=7, length=2, pad=2)
            ax.grid(axis="y", alpha=.18)
            if i == 0:
                ax.set(yscale="log", ylim=(.035, 35))
                ax.set_yticks([.1, 1, 10], ["0.1", "1", "10"])
                ax.minorticks_off()
                ax.set_title(split.capitalize(), fontsize=7.5, pad=3)
            else:
                ax.set(ylim=(0, .57))
                ax.set_yticks([0, .2, .4], ["0", "0.2", "0.4"])
    axes[0, 0].set_ylabel("Core-cycle\nerror (%)", fontsize=7.5, labelpad=2)
    axes[1, 0].set_ylabel(MAE_REQ, fontsize=7.5, labelpad=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.02), ncol=2,
               fontsize=7, frameon=False, handlelength=1.3, columnspacing=.9, handletextpad=.35, labelspacing=.2)
    decision_handles = [Line2D([], [], color="#444444", ls="none", marker="o", ms=2.5,
                               mew=.6, label="Accepted"),
                        Line2D([], [], color="#444444", ls="none", marker="x", ms=3.5,
                               mew=.8, label="Rejected")]
    fig.legend(handles=decision_handles, loc="upper center", bbox_to_anchor=(.53, .885), ncol=2,
               fontsize=7, frameon=False, handlelength=1.1, columnspacing=.9, handletextpad=.35)
    fig.text(.58, .028, "Iteration (0 = seed)", ha="center", fontsize=7.5)
    check_figure_text(fig)
    save(fig, output, "fig4-convergence")


# Fig. 5 ---------------------------------------------------------------------------------------

def fig5_speed(output: Path = OUTPUT) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(3.5, 2.58), sharex=True, sharey=True)
    fig.subplots_adjust(left=.22, right=.965, top=.68, bottom=.16, hspace=.22, wspace=.16)
    intervals = [1, 4, 16, 64, 256]
    styles = {"fixedlat": ("--", ""), "md1": (":", ""), "wmg1": ((0, (5, 2, 1, 2)), ""),
              "mess": ("-.", ""), "astra_single_core": ("-", "o"),
              "deepseek_single_core": ("-", "^"), "gemini_single_core": ("-", "D"),
              "opus_single_core": ("-", "s")}
    points = {(r["model"], r["pattern"], r["read_percent"], r["interval"]): r
              for r in read("speed") if r["model"] != "oracle"}
    for j, reads in enumerate((100, 75, 50)):
        for k, pattern in enumerate(("streaming", "random")):
            ax = axes[j, k]
            ax.axhline(1, color="#aaa", lw=.7, zorder=1)
            for arm in ARMS:
                vals = [points[arm, pattern, reads, i]["oracle_relative_speedup"] for i in intervals]
                ls, marker = styles[arm]
                ax.plot(intervals, vals, label=NAMES[arm], color=COLORS[arm],
                        ls=ls, marker=marker, ms=2.0, lw=.7 if arm in BASELINES else .9)
            ax.set(xscale="log", yscale="log", ylim=(.85, 190), xlim=(.8, 320))
            ax.set_xticks(intervals, [str(i) for i in intervals], fontsize=7)
            ax.set_yticks([1, 10, 100], ["1", "10", "100"], fontsize=7)
            ax.minorticks_off()
            ax.tick_params(length=2, pad=2)
            ax.grid(axis="y", alpha=.18)
            if j == 0:
                ax.set_title("Streaming" if pattern == "streaming" else "Random", fontsize=7.5, pad=3)
            if k == 0:
                ax.set_ylabel(f"{reads}%\nreads", fontsize=7, rotation=0, labelpad=9,
                              ha="right", va="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.012), ncol=2,
               frameon=False, fontsize=7, columnspacing=.8, handlelength=1.5, labelspacing=.2, handletextpad=.4)
    fig.text(.57, .025, "Offered interval (DRAM cycles)", ha="center", fontsize=7.5)
    fig.text(.023, .42, "Speedup (×)", rotation=90, va="center", ha="center", fontsize=7.5)
    check_figure_text(fig)
    save(fig, output, "fig5-speed")


# Fig. 6 ---------------------------------------------------------------------------------------

def cost_label(low: float, high: float) -> str:
    """A cost, or a cost interval marked with a dagger."""
    if high - low > .01:
        return f"{low:.0f}–{high:.0f}†"
    return f"{high:.0f}" if high >= 100 else f"{high:.2f}"


def fig6_usage(output: Path = OUTPUT) -> None:
    rows = {r["model"]: r for r in read("usage")}
    campaigns = ["astra", "deepseek", "gemini", "opus"]
    colors = [COLORS[c + "_single_core"] for c in campaigns]
    fig, axes = plt.subplots(1, 2, figsize=(COMPACT_WIDTH, 1.40), sharey=True)
    fig.subplots_adjust(left=.30, right=.98, top=.77, bottom=.17, wspace=.25)
    y = np.arange(4)
    axes[0].barh(y - .16, [rows[c]["input_tokens"] / 1e6 for c in campaigns], height=.29,
                 color=colors, label="Input", zorder=3)
    axes[0].barh(y + .16, [rows[c]["output_tokens"] / 1e6 for c in campaigns], height=.29,
                 color=colors, hatch="////", edgecolor="white", lw=.5, label="Output", zorder=3)
    axes[0].set(xscale="log", xlim=(.1, 2000), ylim=(3.5, -.5))
    axes[0].set_xticks([.1, 10, 1000], ["0.1", "10", "1,000"])
    axes[0].set_title("Tokens\n(millions)", fontsize=7.5, pad=3)
    axes[1].barh(y, [rows[c]["cost_usd_low"] for c in campaigns], color=colors, height=.60, zorder=3)
    for i, c in enumerate(campaigns):
        lo, hi = rows[c]["cost_usd_low"], rows[c]["cost_usd_high"]
        if hi - lo > .01:
            axes[1].errorbar(lo, i, xerr=[[0], [hi - lo]], capsize=2, color="#222", lw=.8, zorder=5)
        axes[1].text(hi * 1.3, i, cost_label(lo, hi), ha="left", va="center", fontsize=7)
    axes[1].set(xscale="log", xlim=(1, 20000))
    axes[1].set_xticks([1, 100, 10000], ["1", "100", r"$10^4$"])
    axes[1].set_title("API-equivalent\ncost ($)", fontsize=7.5, pad=3)
    names = ["GPT-6 Astra", "DeepSeek V4.1\nFlash", "Gemini 3.8\nFlash", "Opus 5.5"]
    axes[0].set_yticks(y, names, fontsize=7, linespacing=1.0)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7, length=2, pad=2)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.minorticks_off()
        ax.grid(axis="x", alpha=.18, zorder=0)
    check_figure_text(fig)
    save(fig, output, "fig6-usage")


# Figs. 7-9 ------------------------------------------------------------------------------------

EXTENSION_STYLE = {"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                   "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}


def gem5_mode_means(rows: list[dict]) -> list[dict]:
    """Weight individual workloads equally, keeping SE and FS separate."""
    return [{"model": arm, "mode": mode, "cases": len(group),
             "abs_error_pct": statistics.mean(r["abs_error_pct"] for r in group)}
            for arm in sorted({r["model"] for r in rows}) for mode in ("SE", "FS")
            if (group := [r for r in rows if r["model"] == arm and r["mode"] == mode])]


def fig7_gem5(output: Path = OUTPUT) -> None:
    groups = (("SE", "o", "SE (6 workloads)"), ("FS", "s", "FS (9 workloads)"))
    table = gem5_mode_means(read("gem5"))
    with plt.rc_context(EXTENSION_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 1.95))
        fig.subplots_adjust(left=.32, right=.985, bottom=.222, top=.825)
        for i, arm in enumerate(ARMS):
            for j, (mode, marker, _) in enumerate(groups):
                row = next(r for r in table if r["model"] == arm and r["mode"] == mode)
                ax.plot(row["abs_error_pct"], i + (j - .5) * .28, marker=marker, ls="none",
                        ms=3.4, mew=.7, color=COLORS[arm],
                        mfc="white" if j == 1 else COLORS[arm], zorder=3)
        ax.set_yticks(range(len(ARMS)), [NAMES[a] for a in ARMS], fontsize=7)
        ax.set_ylim(len(ARMS) - .48, -.52)
        ax.set_xscale("log"); ax.set_xlim(.1, 24)
        ax.set_xticks([.1, 1, 10], ["0.1", "1", "10"])
        ax.set_xlabel("Mean absolute elapsed-time error (%)", fontsize=7.5, labelpad=2)
        ax.tick_params(axis="both", labelsize=7, pad=2)
        ax.grid(axis="x", which="major", color=".90", lw=.5)
        ax.axhline(3.5, color=".75", lw=.6)
        handles = [Line2D([], [], marker=m, ls="none", color=".25", ms=3.4,
                          mfc="white" if j == 1 else ".25", label=label)
                   for j, (_, m, label) in enumerate(groups)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995), ncol=2,
                   frameon=False, fontsize=7, handletextpad=.35, columnspacing=.75)
        check_figure_text(fig); save(fig, output, "fig7-gem5", strip_svg=False)


def fig8_hardware(output: Path = OUTPUT) -> None:
    arms = (("astra_single_core", "o"), ("deepseek_single_core", "s"),
            ("gemini_single_core", "^"), ("opus_single_core", "D"))
    organizations = (("DDR5_16Gb_x8", "8 × 4 banks\n8 GiB, ×8"),
                     ("DDR5_8Gb_x8", "8 × 2 banks\n4 GiB, ×8"),
                     ("DDR5_16Gb_x16", "4 × 4 banks\n4 GiB, ×16"))
    table = read("hardware")
    with plt.rc_context(EXTENSION_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.09), sharex=True)
        fig.subplots_adjust(left=.17, right=.985, bottom=.2224, top=.697, hspace=.30)
        for ax, metric, label, limit in zip(axes, ("core_abs_pct", "mae_L"),
                ("Core-cycle\nerror (%)", MAE_REQ), (2.6, .25)):
            for j, (organization, _) in enumerate(organizations):
                for arm, marker in arms:
                    values = [next(r[metric] for r in table if r["model"] == arm
                                   and r["organization"] == organization and r["queue_entries"] == q)
                              for q in (32, 64, 128)]
                    ax.plot(np.arange(3) + 3*j, values, marker=marker, ms=3.2, lw=.9,
                            color=COLORS[arm], label=NAMES[arm] if j == 0 else None, zorder=3)
            ax.set_ylim(0, limit); ax.set_xlim(-.45, 8.45)
            ax.set_ylabel(label, fontsize=7.5, labelpad=3)
            ax.grid(axis="y", color=".90", lw=.5)
            ax.tick_params(labelsize=7, pad=2)
            for x in (2.5, 5.5): ax.axvline(x, color=".75", lw=.6, ls=":")
        axes[0].set_yticks([0, 1, 2]); axes[1].set_yticks([0, .1, .2])
        for j, (_, label) in enumerate(organizations):
            axes[0].text(j*3+1, 1.10, label, transform=axes[0].get_xaxis_transform(),
                         ha="center", va="bottom", fontsize=7, linespacing=1.1)
        axes[1].set_xticks(range(9), ["32", "64", "128"] * 3)
        axes[1].set_xlabel("Read/write queue depth (entries)", fontsize=7.5, labelpad=3)
        # Matplotlib fills legend columns first; reorder so the two rows read in plot order.
        handles, names = axes[0].get_legend_handles_labels()
        order = [0, 2, 1, 3]
        fig.legend([handles[i] for i in order], [names[i] for i in order], loc="upper center",
                   bbox_to_anchor=(.5, 1), ncol=2, frameon=False, fontsize=7, handlelength=1.2,
                   columnspacing=1)
        check_figure_text(fig); save(fig, output, "fig8-hardware", strip_svg=False)


def fig9_multicore(output: Path = OUTPUT) -> None:
    models = ("astra", "deepseek", "gemini", "opus")
    arms = BASELINES + models
    table = read("multicore")
    with plt.rc_context(EXTENSION_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.18), sharey=True)
        fig.subplots_adjust(left=.32, right=.985, bottom=.232, top=.819, wspace=.28)
        for ax, metric, label, limits, ticks in zip(axes, ("core_abs_pct", "mae_L"),
                ("Core-cycle\nerror (%)", MAE_REQ), ((0, 32), (.38, .72)),
                ([0, 15, 30], [.4, .5, .6, .7])):
            for i, arm in enumerate(arms):
                is_model = arm in models
                color = COLORS[arm + "_single_core" if is_model else arm]
                for cores, marker in ((4, "o"), (8, "D")):
                    stages = ("single_core", "multicore") if is_model else (None,)
                    for stage in stages:
                        name = arm + "_" + stage if stage else arm
                        row = next(r for r in table if r["model"] == name and r["cores"] == cores)
                        offset = (-.25 if cores == 4 else .25) + ((-.11 if stage == "single_core" else .11) if stage else 0)
                        ax.plot(row[metric], i + offset, marker=marker, ls="none", ms=2.8, mew=.75,
                                color=color, mfc="white" if stage == "single_core" else color, zorder=3)
            ax.set(xlim=limits, xticks=ticks, ylim=(len(arms)-.3, -.5))
            ax.set_xlabel(label, fontsize=7.5, labelpad=3)
            ax.tick_params(labelsize=7, pad=2)
            ax.grid(axis="x", color=".90", lw=.5)
            ax.axhline(3.5, color=".75", lw=.6)
        names = [NAMES[a + "_single_core" if a in models else a] for a in arms]
        axes[0].set_yticks(range(len(arms)), names, fontsize=7)
        # Shape encodes the core count and fill the synthesis stage.
        handles = [Line2D([], [], marker=m, ls="none", color=".25", ms=3.5, mfc=fill, label=label)
                   if m else Line2D([], [], ls="none", label=label)
                   for m, fill, label in ((None, None, "Cores:"), ("o", ".25", "4"), ("D", ".25", "8"),
                                          (None, None, "Iterations:"), ("o", "white", "10"), ("o", ".25", "15"))]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995), ncol=6,
                   frameon=False, fontsize=7, columnspacing=.6, handletextpad=.3, handlelength=.8)
        check_figure_text(fig); save(fig, output, "fig9-multicore", strip_svg=False)


FIGURES = (fig2_per_workload, fig3_accuracy, fig4_convergence, fig5_speed, fig6_usage,
           fig7_gem5, fig8_hardware, fig9_multicore)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    style()
    for render in FIGURES:
        render(args.output)
    print(f"{len(FIGURES)} figures (PDF/SVG/PNG) in {args.output}")


if __name__ == "__main__":
    main()
