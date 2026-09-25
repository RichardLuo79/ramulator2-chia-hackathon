"""Generate the eight-figure paper notebook from visible, reusable plotting code."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import textwrap

import nbformat
from nbclient import NotebookClient

import paper_plots
import paper_workflow
from paper_evidence import SNAPSHOT, load_bundle, sha, verify_exports

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''# Load only the compact, checksummed paper snapshot. No network or simulations.
from pathlib import Path
import sys
import csv
import hashlib
import json
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
import numpy as np
from IPython.display import SVG, display

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "results/paper/paper-v1/manifest.json").is_file())
sys.path.insert(0, str(ROOT / "analyses"))
import paper_plots
from paper_plots import (ARMS, NAMES, ADDITION_COLORS, COMPACT_WIDTH, WRAPPED_NAMES,
                         style, save, csv_table, check_figure_text, gem5_mode_means,
                         PLOTTING_SOURCE)
from paper_workflow import (WIDTH, HEIGHT, LAYOUT_HEIGHT, INK, TEAL, MUTED, LINE,
                            PALE, BORDER, CAPTION, elbow_path, validate_layout)
from paper_evidence import load_bundle, verify_exports

expected_manifest_sha256 = "MANIFEST_SHA256"
manifest_file = ROOT / "results/paper/paper-v1/manifest.json"
if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != expected_manifest_sha256:
    raise ValueError("Notebook belongs to a different paper snapshot")
manifest, evidence = load_bundle(ROOT)
BUNDLE_ROOT = ROOT / "results/paper" / manifest["snapshot"]
data, additions, progression, extensions = (evidence[k] for k in
    ("evidence", "additions", "convergence", "extensions"))
OUTPUT = ROOT / "analyses/figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
paper_plots.DISPLAY_FIGURES = True
# L includes unpaired eligible oracle reads; workload means have equal weights.
# Speed uses each timing batch's own oracle. Cost values are API-equivalent
# estimates; unknown usage remains unknown rather than being replaced by zero.
'''

WORKFLOW_EXPORT = '''with plt.rc_context():
    fig, fitted = draw()
    validate_layout(fig, fitted)
    for ext in ("svg", "pdf", "png"):
        if ext == "pdf":
            metadata = {"Title": "CHIA: development and independent promotion review",
                        "Subject": CAPTION, "CreationDate": None, "ModDate": None}
        elif ext == "svg":
            metadata = {"Title": "CHIA loop workflow", "Description": CAPTION, "Date": None}
        else:
            metadata = {"Title": "CHIA loop workflow", "Description": CAPTION}
        fig.savefig(OUTPUT / f"chia-loop-workflow.{ext}", dpi=400,
                    metadata=metadata, facecolor="white")
    display(SVG(filename=str(OUTPUT / "chia-loop-workflow.svg")))
    plt.close(fig)
'''


def figure_source(item):
    if item["number"] == 1:
        return inspect.getsource(paper_workflow.draw) + "\n" + WORKFLOW_EXPORT
    function = item["function"]
    params = {2: "data", 3: "data, progression", 4: "additions", 5: "additions",
              6: "extensions", 7: "extensions", 8: "extensions"}[item["number"]]
    invocation = f"style()\n{function}({params}, OUTPUT)\n"
    if item["number"] >= 6:
        invocation = '''style()
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
''' + f"{function}({params}, OUTPUT)\n"
    source = inspect.getsource(getattr(paper_plots, function))
    source += "\nwith plt.rc_context():\n" + textwrap.indent(invocation, "    ")
    if item["number"] == 8:
        source += "\n# Reconcile all exported source tables with the paper snapshot.\n"
        source += "export_report = verify_exports(ROOT, OUTPUT, manifest)\n"
    return source


def build(output=None, *, execute=True, figure_output=None):
    output = Path(output or ROOT / "analyses/chia_campaign_results.ipynb").resolve()
    figure_output = Path(figure_output or ROOT / "analyses/figures").resolve()
    if not output.is_relative_to(ROOT) or not figure_output.is_relative_to(ROOT):
        raise ValueError("Keep notebook and exports inside the checkout for portable relative paths")
    manifest, _ = load_bundle(ROOT)
    setup = SETUP.replace('ROOT / "analyses/figures"', "ROOT / " + repr(figure_output.relative_to(ROOT).as_posix()))
    setup = setup.replace("MANIFEST_SHA256", sha(ROOT / "results/paper" / SNAPSHOT / "manifest.json"))
    cells = [nbformat.v4.new_code_cell(setup, id="paper-setup")]
    for item in manifest["figures"]:
        cells += [nbformat.v4.new_markdown_cell(
            f'## Fig. {item["number"]}. {item["caption_markdown"]}', id=f'caption-{item["number"]}'),
            nbformat.v4.new_code_cell(figure_source(item), id=f'figure-{item["number"]}')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "paper_snapshot": manifest["snapshot"],
        "paper_manifest_sha256": sha(ROOT / "results/paper" / SNAPSHOT / "manifest.json"),
        "figure_count": 8})
    output.parent.mkdir(parents=True, exist_ok=True)
    if execute:
        NotebookClient(notebook, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        report = verify_exports(ROOT, figure_output, manifest)
        if not all(row["png_matches_paper"] for row in report["figures"]):
            changed = [row["name"] for row in report["figures"] if not row["png_matches_paper"]]
            raise ValueError("Rendered pixels differ from the paper; check pinned dependencies: " + ", ".join(changed))
    nbformat.write(notebook, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analyses/chia_campaign_results.ipynb")
    parser.add_argument("--figure-output", type=Path, default=ROOT / "analyses/figures")
    args = parser.parse_args()
    print(build(args.output, figure_output=args.figure_output))


if __name__ == "__main__":
    main()
