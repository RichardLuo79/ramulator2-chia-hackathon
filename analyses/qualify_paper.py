"""Execute a relocated notebook with old workspaces and new sockets denied.

Uses the artifact's existing Landlock/seccomp boundary. A temporary qualification
cell is not included in the delivered notebook. No simulations are run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deny-path", action="append", type=Path, default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in args.deny_path:
        if not path.is_file():
            raise FileNotFoundError("A denial probe must name an existing file")
    destination = Path(tempfile.mkdtemp(prefix="chia-paper-offline-"))
    checkout = destination / "checkout"
    checkout.mkdir()
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                    cwd=ROOT).decode().split("\0")
    copied = []
    for name in sorted(set(filter(None, names))):
        path = ROOT / name
        if path.is_symlink():
            raise ValueError("Unexpected symlink in release")
        if path.is_file():
            target = checkout / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(name)
    # These directories are new, isolated analysis caches, not credential stores.
    for name in (".ipython", ".jupyter", ".jupyter-runtime", ".mpl"):
        (destination / name).mkdir()
    env = {**os.environ, "IPYTHONDIR": str(destination / ".ipython"),
           "JUPYTER_CONFIG_DIR": str(destination / ".jupyter"),
           "JUPYTER_RUNTIME_DIR": str(destination / ".jupyter-runtime"),
           "MPLCONFIGDIR": str(destination / ".mpl"), "PYTHONDONTWRITEBYTECODE": "1"}
    read = [str(destination), str(Path(sys.prefix).resolve()), "/usr", "/etc", "/proc", "/dev"]
    read += [p for p in ("/lib", "/lib64", "/bin") if Path(p).exists()]
    boundary = f'''from pathlib import Path
import json
import socket
import sys
sys.path.insert(0, {str(checkout / "chia-loop/src")!r})
from ramulator_chia.sandbox import restrict
restrict({read!r}, {[str(destination), '/dev/null']!r})
denied = 0
for path in {list(map(str, args.deny_path))!r}:
    try:
        Path(path).read_bytes()
    except PermissionError:
        denied += 1
    else:
        raise AssertionError("Old workspace remained readable")
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except PermissionError:
    network_denied = True
else:
    raise AssertionError("Network socket remained available")
Path({str(destination / 'boundary.json')!r}).write_text(json.dumps(
    dict(denied_paths=denied, network_denied=network_denied)))
'''
    notebook = nbformat.read(checkout / "analyses/chia_campaign_results.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.insert(0, nbformat.v4.new_code_cell(boundary, id="temporary-boundary-check"))
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(checkout)}}).execute(env=env)
    qualification = json.loads((destination / "boundary.json").read_bytes())
    rendered = json.loads((checkout / "analyses/figures/manifest.json").read_bytes())
    if not all(row["png_matches_paper"] for row in rendered["figures"]):
        raise AssertionError("Relocated rendering differs from paper")
    for relative in ("results/paper/paper-v1/manifest.json", "analyses/paper_plots.py"):
        if (checkout / relative).read_bytes() != (ROOT / relative).read_bytes():
            raise AssertionError("Relocated evidence changed")
    result = {"schema": "chia-paper-offline-qualification-v1", "passed": True,
              "copied_files": len(copied), "cells_executed": sum(c.cell_type == "code" for c in notebook.cells) - 1,
              "figures": len(rendered["figures"]), "boundary": qualification,
              "paper_pngs_identical": True, "csv_tables_identical": True,
              "simulations": 0, "provider_calls": 0,
              "python": sys.version.split()[0],
              "notebook_sha256": hashlib.sha256((ROOT / "analyses/chia_campaign_results.ipynb").read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("Temporary qualification copy:", destination)


if __name__ == "__main__":
    main()
