"""Export compact, audited results without copying interaction/setup records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib

from tools.chia_loop.run_records import reporting_view


FIGURES_AND_TABLES = (
    "headline.csv", "per_workload.csv", "headline.svg", "per_workload.svg", "evolution.svg",
)
AUDIT_FIELDS = (
    "status", "runs", "candidate_callback_checks", "paired_logical_reads_across_reports",
    "traces", "raw_bytes", "gzip_bytes", "space_reduction_percent",
    "failed_run_archives_verified", "protocol_and_visible_snapshots_verified",
    "archive_recoveries_verified", "scored_trace_archives_recovered",
    "runtime_and_candidate_hashes_verified", "test_after_both_freezes",
    "heldout_ids_absent_from_paid_requests", "actual_api_prompt_matches_saved_prompt",
    "caps_respected", "all_requests_use_configured_HIGH_and_output_maximum",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def model_table(data, backend):
    """Retain this run and the fixed baselines, never another optimization run."""
    reader = csv.DictReader(io.StringIO(data.decode()))
    if not reader.fieldnames or "model" not in reader.fieldnames:
        raise ValueError("result table requires a model column")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(row for row in reader if row["model"] in {backend, "seed", "fixedlat", "md1", "wmg1", "mess"})
    return stream.getvalue().encode()


def export_review(root, destination):
    root, destination = pathlib.Path(root).resolve(), pathlib.Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("review export refuses to replace an existing directory")
    manifest_bytes = (root / "run_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "completed" or not manifest.get("archives_verified"):
        raise ValueError("only completed, frozen, archived runs can be published")
    audit_bytes = (root / "final_integrity_audit.json").read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("status") != "pass":
        raise ValueError("a passing final integrity audit is required")

    view = reporting_view(root)
    legacy = manifest.get("record_type") != "optimization_run"
    if not view["arms"] or set(view["arms"]) != set(view["models"]):
        raise ValueError("completed run lacks a matching model state")
    if not legacy and (audit.get("run_id") != manifest["run_id"] or audit.get("model") != manifest["model"]):
        raise ValueError("integrity audit belongs to a different model run")
    # Fixed allowlist: never recursively copy a run or serialize its full state.
    figures = {}
    for name in FIGURES_AND_TABLES:
        path = (root / "analysis" / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("analysis artifact resolves outside campaign")
        figures[name] = path.read_bytes()
    if not legacy:
        for name in ("headline.csv", "per_workload.csv"):
            rows = list(csv.DictReader(io.StringIO(figures[name].decode())))
            allowed = {manifest["backend"], "seed", "fixedlat", "md1", "wmg1", "mess"}
            if not rows or any(row.get("model") not in allowed for row in rows):
                raise ValueError("individual run table contains a different model or no results")
    artifacts = dict(figures) if legacy else {}
    references = []
    for arm, state in view["arms"].items():
        if state["model"] != view["models"][arm]:
            raise ValueError("state belongs to a different model run")
        if state["status"] != "frozen":
            raise ValueError("each run's selection must be frozen")
        selected = state["selected"]
        source = pathlib.Path(selected["source_path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("selected source resolves outside campaign")
        data = source.read_bytes()
        if digest(data) != selected["sha256"]:
            raise ValueError("selected source hash mismatch")
        source_name = "selected.cpp"
        run_id = view["run_ids"][arm]
        prefix = f"runs/{run_id}/" if legacy else ""
        run_artifacts = {source_name: data}
        if legacy:
            run_artifacts.update({name: model_table(figures[name], arm)
                                  for name in ("headline.csv", "per_workload.csv")})
        else:
            run_artifacts.update(figures)
        history = state["history"]
        # Only numeric accounting fields, not provider requests, errors, or paths.
        budget = {key: value for key, value in state["budget"].items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)}
        generation = (audit.get("generation", {}) if not legacy else
            audit.get("runs_by_id", {}).get(run_id, audit.get("arms", {}).get(arm, {})))
        summary = {
            "schema_version": 2, "record_type": "optimization_run_summary", "run_id": run_id,
            "status": "completed", "backend": arm, "frozen_at": state.get("frozen_at"),
            "policy": view["policy"],
            "instructions_per_core": manifest["instructions_per_core"],
            "training": manifest["training"], "final_test": manifest["final_test"],
            "seed_sha256": manifest["preparation"]["seed_sha256"],
            "evaluation_protocol_sha256": digest(encoded({
                "protocol_hashes": manifest.get("protocol_hashes", {}),
                "visible_hashes": manifest.get("visible_hashes", {}),
                "input_inventory_sha256": digest((root / "input_inventory.json").read_bytes())
                    if (root / "input_inventory.json").exists() else None})),
            "provenance": {"run_manifest_sha256": digest(manifest_bytes),
                "integrity_audit_sha256": digest(audit_bytes),
                "exporter_sha256": digest(pathlib.Path(__file__).read_bytes()),
                "historical_shared_execution": root.name if legacy else None,
                "raw_records_rewritten": False,
                "test_access": manifest.get("test_access", "after_both_incumbents_frozen" if legacy else None)},
            "integrity": {"status": "pass", "source_sha256_verified_on_export": True,
                "scope": "This model's results; shared-store counts are not per-run sample counts.",
                "test_after_selection_freeze": audit.get("test_after_selection_freeze", audit.get("test_after_both_freezes"))},
            "model": state["model"], "selected_id": state["incumbent"],
            "source_file": source_name, "source_sha256": selected["sha256"],
            "evaluated_designs": sum(h["status"] == "valid" for h in history),
            "submitted_drafts": sum(len(h.get("drafts", [])) for h in history),
            "promotions": sum(bool(h.get("promoted")) for h in history),
            "generation_finish_reasons": generation.get("finish_reasons", {}),
            "generation_attempts_by_role": generation.get("generation_attempts_by_role", {}),
            "generation_finish_reasons_by_role": generation.get("finish_reasons_by_role", {}),
            "known_standard_usd_by_role": generation.get("known_standard_usd_by_role", {}),
            "compliance_review": {key: manifest.get("review_service", {})[key]
                for key in ("model", "rubric_sha256", "cost_in_same_run_cap", "cross_run_information")
                if key in manifest.get("review_service", {})},
            "cache_adjusted_estimate_current_calls_usd": generation.get("cache_adjusted_estimate_known_calls_usd"),
            "budget": budget,
            "training_metrics": selected["metrics"]["aggregate"],
            "test_metrics": view["final_test_metrics"][arm]["aggregate"],
            "trajectory": [{key: h[key] for key in
                ("id", "parent", "iteration", "status", "promoted", "metrics") if key in h}
                for h in history],
        }
        summary["artifact_sha256"] = {name: digest(data) for name, data in run_artifacts.items()}
        run_artifacts["summary.json"] = encoded(summary)
        references.append({"run_id": run_id, "model": state["model"],
            "summary": prefix + "summary.json", "summary_sha256": digest(run_artifacts["summary.json"])})
        artifacts.update({prefix + name: data for name, data in run_artifacts.items()})
    if legacy:
        summary = {"schema_version": 2, "record_type": "run_comparison", "comparison_id": root.name,
            "runs": references, "experimental_unit": "one independent optimization run per model",
            "provenance": {"legacy_manifest_sha256": digest(manifest_bytes),
                "legacy_integrity_audit_sha256": digest(audit_bytes), "raw_records_rewritten": False},
            "shared_artifact_integrity": {key: audit[key] for key in AUDIT_FIELDS if key in audit},
            "shared_artifact_count_scope": "Historical shared execution store, not totals for each model run.",
            "artifact_sha256": {name: digest(data) for name, data in figures.items()}}
        artifacts["summary.json"] = encoded(summary)
    destination.mkdir(parents=True)
    for name, data in artifacts.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    args = parser.parse_args()
    export_review(args.root, args.destination)
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
