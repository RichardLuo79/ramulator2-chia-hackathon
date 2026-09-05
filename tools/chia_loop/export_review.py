"""Export compact, audited results without copying interaction/setup records."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


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


def export_review(root, destination):
    root, destination = pathlib.Path(root).resolve(), pathlib.Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("review export refuses to replace an existing directory")
    manifest_bytes = (root / "run_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "completed" or not manifest.get("archives_verified"):
        raise ValueError("only completed, frozen, archived campaigns can be published")
    audit_bytes = (root / "final_integrity_audit.json").read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("status") != "pass":
        raise ValueError("a passing final integrity audit is required")

    # Fixed allowlist: never recursively copy a campaign or serialize its full state.
    artifacts = {}
    for name in FIGURES_AND_TABLES:
        path = (root / "analysis" / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("analysis artifact resolves outside campaign")
        artifacts[name] = path.read_bytes()
    summary = {
        "schema_version": 1, "campaign": root.name,
        "run_manifest_sha256": digest(manifest_bytes),
        "integrity_audit_sha256": digest(audit_bytes),
        "exporter_sha256": digest(pathlib.Path(__file__).read_bytes()),
        "policy": manifest["policy"],
        "instructions_per_core": manifest["instructions_per_core"],
        "training": manifest["training"], "final_test": manifest["final_test"],
        "seed_sha256": manifest["preparation"]["seed_sha256"],
        "integrity": {key: audit[key] for key in AUDIT_FIELDS},
        "arms": {},
    }
    for arm in ("pro", "flash"):
        state = manifest["arms"][arm]
        if state["status"] != "frozen":
            raise ValueError("both selections must be frozen")
        selected = state["selected"]
        source = pathlib.Path(selected["source_path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("selected source resolves outside campaign")
        data = source.read_bytes()
        if digest(data) != selected["sha256"]:
            raise ValueError("selected source hash mismatch")
        source_name = f"{arm}_selected.cpp"
        artifacts[source_name] = data
        history = state["history"]
        # Only numeric accounting fields, not provider requests, errors, or paths.
        budget = {key: value for key, value in state["budget"].items()
                  if isinstance(value, (int, float)) and not isinstance(value, bool)}
        summary["arms"][arm] = {
            "model": state["model"], "selected_id": state["incumbent"],
            "source_file": source_name, "source_sha256": selected["sha256"],
            "evaluated_designs": sum(h["status"] == "valid" for h in history),
            "submitted_drafts": sum(len(h.get("drafts", [])) for h in history),
            "promotions": sum(bool(h.get("promoted")) for h in history),
            "generation_finish_reasons": audit.get("arms", {}).get(arm, {}).get("finish_reasons", {}),
            "cache_adjusted_estimate_current_calls_usd": audit.get("arms", {}).get(arm, {}).get(
                "cache_adjusted_estimate_known_calls_usd"),
            "budget": budget,
            "training_metrics": selected["metrics"]["aggregate"],
            "test_metrics": manifest["final_test_metrics"][arm]["aggregate"],
            "trajectory": [{key: h[key] for key in
                ("id", "parent", "iteration", "status", "promoted", "metrics") if key in h}
                for h in history],
        }
    summary["artifact_sha256"] = {name: digest(data) for name, data in artifacts.items()}
    artifacts["summary.json"] = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    destination.mkdir(parents=True)
    for name, data in artifacts.items():
        (destination / name).write_bytes(data)
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
