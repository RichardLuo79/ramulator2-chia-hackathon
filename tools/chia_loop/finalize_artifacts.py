"""Retry post-run archival only; never generate, select, or evaluate a model."""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from tools.chia_loop import artifacts, real_eval
from tools.chia_loop.core import atomic_write_json


def finalize(root):
    root = pathlib.Path(root).resolve()
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "completed" or not (root / "both_frozen.json").exists():
        raise ValueError("artifact finalization requires completed evaluation and frozen selections")
    if any(state.get("status") != "frozen" for state in manifest["arms"].values()):
        raise ValueError("cannot finalize an active optimization arm")
    for failure in root.glob("training/simpleo3/DDR5/*/*/failure.json"):
        real_eval.archive_failed_run(failure.parent)
    count = real_eval.verify_run_archives(root)
    artifacts.compress(root, root / "aux_archive_manifest.json", min_bytes=16_384)
    artifacts.verify(root / "aux_archive_manifest.json")
    recoveries = []
    for record in root.glob("*/simpleo3/DDR5/*/*/*.recovery.json"):
        recovery = json.loads(record.read_text())
        if recovery["status"] != "restored":
            raise ValueError("an archive recovery is incomplete")
        recoveries.append(str(record.relative_to(root)))
    manifest.update(archives_verified=True, post_run_artifact_finalization={
        "time": time.time(), "verified_traces": count,
        "archive_recovery_records": recoveries,
        "evaluations_repeated": False, "additional_generation_calls": 0})
    atomic_write_json(path, manifest)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path)
    args = parser.parse_args()
    print("Verified traces:", finalize(args.root))


if __name__ == "__main__":
    main()
