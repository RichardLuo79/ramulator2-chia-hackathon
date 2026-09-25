"""Operator-only import of an existing paper snapshot; never runs an experiment.

The normal notebook uses only the imported bundle. This command needs an explicit
paper checkout, writes a NEW version, and refuses to replace an existing bundle.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    ("chia-loop-workflow", "fig:workflow", [], "draw"),
    ("foundation-accuracy", "fig:foundation-accuracy", ["evidence"], "foundation_accuracy"),
    ("foundation-convergence", "fig:foundation-convergence", ["evidence", "convergence"], "foundation_convergence"),
    ("foundation-speed", "fig:foundation-speed", ["additions"], "foundation_speed"),
    ("foundation-usage", "fig:foundation-usage", ["additions"], "foundation_usage"),
    ("extension-gem5", "fig:extension-gem5", ["extensions"], "extension_gem5"),
    ("extension-hardware", "fig:extension-hardware", ["extensions"], "extension_hardware"),
    ("extension-multicore", "fig:extension-multicore", ["extensions"], "extension_multicore"),
)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def active_tex(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def braced(text, start):
    if text[start] != "{":
        raise ValueError("Expected a LaTeX group")
    depth = 1
    for i in range(start + 1, len(text)):
        if text[i - 1] != "\\":
            depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start + 1:i]
    raise ValueError("Unclosed LaTeX group")


def caption_markdown(text):
    text = text.replace(r"\MAEReq", r"\mathrm{MAE}_{\mathrm{Req}}")
    text = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", text)
    text = text.replace(r"\%", "%").replace("~", " ")
    return " ".join(text.split())


def source_nodes(path):
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    nodes = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if isinstance(node.targets[0], ast.Tuple):
                names += [t.id for t in node.targets[0].elts]
        else:
            continue
        chunk = "".join(lines[node.lineno - 1:node.end_lineno])
        for name in names:
            nodes[name] = chunk
    return nodes


def extract_module(path, names, header, changes=None):
    nodes = source_nodes(path)
    chunks, seen, provenance = [], set(), {}
    for name in names:
        original = nodes[name]
        updated = original
        for before, after in (changes or {}).items():
            updated = updated.replace(before, after)
        provenance[name] = {"original_sha256": sha(original.encode()),
                            "adapted_sha256": sha(updated.encode())}
        if updated not in seen:
            chunks.append(updated.rstrip())
            seen.add(updated)
    return header + "\n\n".join(chunks) + "\n", provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.version):
        parser.error("Unsafe snapshot name")
    paper = args.paper.resolve()
    target = ROOT / "results/paper" / args.version
    if target.exists():
        raise FileExistsError("Paper snapshots are immutable: " + str(target))
    main_text = active_tex((paper / "main.tex").read_text())
    sections = re.findall(r"\\input\{(sections/[^}]+)\}", main_text)
    files = {"main.tex": (paper / "main.tex").read_bytes()}
    captions = {}
    for relative in sections:
        relative += ".tex"
        files[relative] = (paper / relative).read_bytes()
        text = active_tex(files[relative].decode())
        for match in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", text, re.S):
            block = match.group(1)
            caption = braced(block, block.index("{", block.index(r"\caption")))
            label = braced(block, block.index("{", block.index(r"\label")))
            image = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block).group(1)
            captions[label] = {"caption_latex": caption, "caption_markdown": caption_markdown(caption),
                               "section": relative, "paper_asset": image}
    if list(captions) != [row[1] for row in FIGURES]:
        raise ValueError("The paper's active figure order changed; review the importer mapping")

    source_paths = ("scripts/plot_paper.py", "scripts/plot_chia_workflow.py", "scripts/import_paper_evidence.py")
    files.update({p: (paper / p).read_bytes() for p in source_paths})
    manifest = {"schema": "chia-paper-figures-v1", "snapshot": args.version,
                "paper_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paper, text=True).strip(),
                "uses_working_tree": True, "paper_sources": {k: sha(v) for k, v in files.items()},
                "files": {}, "figures": [], "renderer_adaptations": [
                    "Repository-relative bundle paths and validator imports.",
                    "SVG display after export for notebook output; original plotting calculations unchanged.",
                    "Workflow export separated from its original command-line main function."]}
    target.mkdir(parents=True)

    def preserve(relative, payload):
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        manifest["files"][relative] = {"sha256": sha(payload), "bytes": len(payload)}

    for suffix in ("evidence", "additions", "convergence", "extensions"):
        relative = f"paper-{suffix}.json"
        preserve(relative, (paper / "data" / relative).read_bytes())
    for relative in ("simulation-configuration.tex", "model-mechanisms.tex"):
        preserve("tables/" + relative, (paper / "tables" / relative).read_bytes())
    for number, (name, label, data, function) in enumerate(FIGURES, 1):
        record = dict(number=number, name=name, label=label, **captions[label],
                      data=[f"paper-{suffix}.json" for suffix in data], function=function,
                      module="paper_workflow" if number == 1 else "paper_plots",
                      paper_exports={ext: sha((paper / "figures" / (name + "." + ext)).read_bytes())
                                     for ext in ("pdf", "svg", "png")}, reference_tables=[])
        csv_names = ([name + ".csv"] if number > 1 else [])
        if number == 4:
            csv_names.append("foundation-speed-summary.csv")
        for csv_name in csv_names:
            preserve("reference-tables/" + csv_name, (paper / "figures" / csv_name).read_bytes())
            record["reference_tables"].append(csv_name)
        manifest["figures"].append(record)

    plot_names = ["COLORS", "NAMES", "ARMS", "ADDITION_COLORS", "COMPACT_WIDTH", "WRAPPED_NAMES", "style",
                  "vertical_export_bbox", "save", "csv_table", "check_figure_text", "foundation_accuracy",
                  "foundation_speed", "foundation_usage", "foundation_convergence", "gem5_mode_means",
                  "extension_gem5", "extension_hardware", "extension_multicore"]
    header = '''"""Paper renderers, mechanically extracted; scientific calculations unchanged."""
from __future__ import annotations
import csv
import hashlib
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
from IPython.display import SVG, display

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "results/paper" / "VERSION"
PLOTTING_SOURCE = Path(__file__)
DISPLAY_FIGURES = False

'''.replace('"VERSION"', repr(args.version))
    changes = {'from import_paper_evidence import check_convergence': 'from paper_checks import check_convergence',
               'ROOT / "data/paper-evidence.json"': 'BUNDLE_ROOT / "paper-evidence.json"',
               'Path(__file__).read_bytes()': 'PLOTTING_SOURCE.read_bytes()',
               '    plt.close(fig)': '    if DISPLAY_FIGURES:\n        display(SVG(filename=str(output / f"{name}.svg")))\n    plt.close(fig)'}
    plots, mapping = extract_module(paper / source_paths[0], plot_names, header, changes)
    (ROOT / "analyses/paper_plots.py").write_text(plots)
    manifest["plotting_functions"] = mapping
    check_names = ["close", "require", "check_speed_study", "check_convergence", "extension_headlines", "check_extensions"]
    checks, mapping = extract_module(paper / source_paths[2], check_names,
                                    '"""Unchanged semantic checks from the verified paper importer."""\nimport math\nimport statistics\n\n')
    (ROOT / "analyses/paper_checks.py").write_text(checks)
    manifest["validation_functions"] = mapping
    workflow_names = ["WIDTH", "LAYOUT_HEIGHT", "INK", "TEAL", "MUTED", "LINE", "PALE", "BORDER", "CAPTION",
                      "elbow_path", "draw", "validate_layout"]
    workflow, mapping = extract_module(paper / source_paths[1], workflow_names, '''"""The paper's current vector workflow; original geometry and labels."""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

''')
    (ROOT / "analyses/paper_workflow.py").write_text(workflow)
    manifest["workflow_functions"] = mapping
    manifest["renderers"] = {f"analyses/{name}.py": sha((ROOT / "analyses" / (name + ".py")).read_bytes())
                             for name in ("paper_plots", "paper_checks", "paper_workflow")}
    manifest["dependency_files"] = {f"analyses/{name}": sha((ROOT / "analyses" / name).read_bytes())
                                    for name in ("requirements.txt", "requirements.lock.txt")}
    manifest["execution_dependencies"] = {
        "python": "3.12", "external_services": [], "raw_archives": [],
        "common_helpers": ["analyses/paper_evidence.py", "analyses/paper_notebook.py"],
        "note": "All figures use the pinned analysis dependencies and shared helpers; code hashes are also in the release inventory."}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"snapshot": args.version, "figures": len(captions), "files": len(manifest["files"]),
                      "input_bytes": sum(v["bytes"] for v in manifest["files"].values())}))


if __name__ == "__main__":
    main()
