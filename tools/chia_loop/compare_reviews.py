"""Compare immutable single-run review exports without pooling runs or budgets."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib

from tools.chia_loop.export_review import digest, encoded


def compare_reviews(summaries, destination):
    destination = pathlib.Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("comparison refuses to replace an existing directory")
    if len(summaries) < 2:
        raise ValueError("comparison requires at least two individual runs")
    records, references, seen = [], [], set()
    for filename in summaries:
        path = pathlib.Path(filename).resolve()
        data = path.read_bytes()
        run = json.loads(data)
        if run.get("record_type") != "optimization_run_summary" or run.get("status") != "completed":
            raise ValueError("comparison inputs must be completed individual run summaries")
        if run["run_id"] in seen:
            raise ValueError("duplicate run ID; iterations are not independent repetitions")
        seen.add(run["run_id"])
        for name, expected in run["artifact_sha256"].items():
            artifact = (path.parent / name).resolve()
            if not artifact.is_relative_to(path.parent) or digest(artifact.read_bytes()) != expected:
                raise ValueError("run artifact escaped its directory or failed its checksum")
        if run["artifact_sha256"].get(run["source_file"]) != run["source_sha256"]:
            raise ValueError("selected source identity mismatch")
        if records:
            for field in ("policy", "instructions_per_core", "training", "final_test", "seed_sha256", "evaluation_protocol_sha256"):
                if run[field] != records[0][field]:
                    raise ValueError("unmatched comparison setup: " + field)
        records.append(run)
        references.append({"run_id": run["run_id"], "model": run["model"],
            "summary": os.path.relpath(path, destination), "summary_sha256": digest(data)})
    rows = [{"run_id": run["run_id"], "model": run["model"], "split": split,
             **run[f"{split}_metrics"]} for run in records for split in ("training", "test")]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)
    table = stream.getvalue().encode()
    comparison = {"schema_version": 2, "record_type": "run_comparison", "comparison_id": destination.name,
        "runs": references, "experimental_unit": "one independent optimization run; no pooled budget or lineage",
        "matched_setup_verified": True, "artifact_sha256": {"headline.csv": digest(table)}}
    destination.mkdir(parents=True)
    (destination / "headline.csv").write_bytes(table)
    (destination / "summary.json").write_bytes(encoded(comparison))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=pathlib.Path)
    parser.add_argument("--destination", required=True, type=pathlib.Path)
    args = parser.parse_args()
    compare_reviews(args.summaries, args.destination)
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
