#!/usr/bin/env python3
"""Post-freeze integrity and provider-usage report, never optimization feedback.

Unlike the historical first-smoke recovery script, this does not repair prompts
or assume a particular selected model, ROI, iteration count, or output cap.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import pathlib
import time

from tools.chia_loop.core import atomic_write_json


def load(path):
    path = pathlib.Path(path)
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = load(root / "run_manifest.json")
    assert manifest["status"] == "completed" and manifest["archives_verified"]
    policy = manifest["policy"]
    for relative, expected in manifest["protocol_hashes"].items():
        assert digest(root / "protocol" / relative) == expected, relative
    for relative, expected in manifest["visible_hashes"].items():
        assert digest(root / "visible" / relative) == expected, relative
    runtime = load(root / "runtime_manifest.json")
    for filename, key in (("libramulator.so", "library_sha256"), ("isolated_sim", "executable_sha256")):
        assert digest(root / "runtime" / filename) == runtime[key]
    assert runtime["optimization"] == "-O3"
    frozen = load(root / "both_frozen.json")
    freeze_time = max(s["frozen_at"] for s in frozen.values())
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    test_start = next(e["time"] for e in events if e["event"] == "both_frozen_test_started")
    assert test_start >= freeze_time
    seed_hash = digest(root / "seed/atomic_controller.cpp")
    assert seed_hash == manifest["preparation"]["seed_sha256"]
    assert all(not row["wraps"] for row in load(root / "input_inventory.json").values())
    usage, arm_summaries = [], {}
    for arm, state in manifest["arms"].items():
        assert 1 <= len(state["history"]) <= policy["iterations_per_arm"]
        assert state["selected"]["sha256"] == frozen[arm]["source_sha256"]
        assert state["status"] == "frozen"
        drafts = [d for h in state["history"] for d in h.get("drafts", [])]
        for draft in drafts:
            outcome = load(pathlib.Path(draft["directory"]) / "outcome.json")
            assert outcome == draft
        for candidate_id, candidate in state["candidates"].items():
            assert digest(candidate["source_path"]) == candidate["sha256"]
            build = load(pathlib.Path(candidate["plugin"]).parent / "build.json")
            assert build["source_sha256"] == candidate["sha256"]
            assert digest(candidate["plugin"]) == build["plugin_sha256"]
            assert build["optimization"] == "-O3"
            if candidate_id != "seed":
                review = load(pathlib.Path(candidate["source_path"]).parent / "review.json")
                assert review["approved"] is True and review["source_sha256"] == candidate["sha256"]
        ledger = load(root / "arms" / arm / "ledger.json")
        assert sum(c["cap_charge_usd"] for c in ledger["calls"]) <= policy["usd_cap_per_arm"]
        assert len(ledger["calls"]) == state["budget"]["api_attempts"]
        adjusted_cost = 0.0
        for call in ledger["calls"]:
            base = root / "arms" / arm / "interactions" / f"iter_{call['iteration']:03d}" / f"call_{call['id']:03d}"
            request = load(base.with_suffix(".request.json"))
            assert request["model"] == manifest["models"][arm]
            assert request["config"]["thinking_config"]["thinking_level"] == policy["thinking_level"]
            assert request["config"]["max_output_tokens"] == policy["maximum_output_tokens"]
            if "maximum_input_tokens" in policy:
                assert 0 <= call["counted_input_tokens"] <= policy["maximum_input_tokens"]
                assert call["turn"] <= policy["model_turns_per_proposal"]
            for workload in manifest["final_test"]:
                assert workload not in json.dumps(request), "held-out identity in paid request"
            sent_prompt = request["contents"][0]["parts"][0]["text"]
            prompt = root / "arms" / arm / "candidates" / f"{arm}_{call['iteration']:03d}/prompt.md"
            assert sent_prompt == prompt.read_text()
            row = {"arm": arm, "iteration": call["iteration"], "call_id": call["id"],
                "turn": call["turn"], "state": call["state"], "finish_reason": "NO_RESPONSE",
                "prompt_tokens": 0, "cached_tokens": 0, "thinking_tokens": 0, "answer_tokens": 0,
                "billed_output_tokens": 0, "standard_usd": call["estimated_standard_usd"],
                "cap_charge_usd": call["cap_charge_usd"], "wall_seconds": None,
                "model_version": None}
            if call["state"] == "usage_recorded":
                raw = load(base.with_suffix(".response.json"))
                evidence = load(base.with_suffix(".call.json"))
                assert raw["usage_metadata"] == call["usage"]
                assert hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest() == evidence["response_sha256"]
                tokens = raw["usage_metadata"]
                row.update({"finish_reason": (raw.get("candidates") or [{}])[0].get("finish_reason"),
                    "prompt_tokens": tokens.get("prompt_token_count") or 0,
                    "cached_tokens": tokens.get("cached_content_token_count") or 0,
                    "thinking_tokens": tokens.get("thoughts_token_count") or 0,
                    "answer_tokens": tokens.get("candidates_token_count") or 0,
                    "billed_output_tokens": call["billed_output_including_thinking"],
                    "wall_seconds": evidence["wall_s"], "model_version": raw.get("model_version")})
                rate = manifest["pricing"][arm]
                input_rate = rate["long_input" if row["prompt_tokens"] > 200_000 else "input"]
                output_rate = rate["long_output" if row["prompt_tokens"] > 200_000 else "output"]
                adjusted_cost += ((row["prompt_tokens"] - .9 * row["cached_tokens"]) * input_rate
                    + row["billed_output_tokens"] * output_rate) / 1e6
            usage.append(row)
        this_arm = [r for r in usage if r["arm"] == arm]
        arm_summaries[arm] = {"selected": state["incumbent"], "attempts": len(state["history"]),
            "drafts_submitted": len(drafts), "draft_rejections": sum(d["status"] == "rejected" for d in drafts),
            "valid": sum(h["status"] == "valid" for h in state["history"]),
            "promotions": sum(bool(h.get("promoted")) for h in state["history"]),
            "stop_reason": state.get("stop_reason"), "budget": state["budget"],
            "finish_reasons": dict(collections.Counter(r["finish_reason"] for r in this_arm)),
            "cache_adjusted_estimate_known_calls_usd": adjusted_cost,
            "peak_billed_output_tokens": max((r["billed_output_tokens"] for r in this_arm), default=0),
            "maximum_answer_tokens": max((r["answer_tokens"] for r in this_arm), default=0)}
    runs = callback_runs = pairings = raw_bytes = gzip_bytes = traces = 0
    for split in ("training", "test", "parity", "preflight"):
        for path in (root / split).glob("simpleo3/DDR5/*/*/manifest.json"):
            run = load(path)
            assert run["insts_per_core"] == manifest["instructions_per_core"]
            assert run["frontend_stats"]["logical_requests_live"] == 0
            assert run["frontend_stats"]["internal_writebacks_live"] == 0
            if split == "test":
                assert path.stat().st_mtime >= test_start
            if split != "parity":
                assert run["optimization"] == "-O3"
            if run["model"] == "candidate":
                ctrl = run["controller_stats"]
                for kind in ("read", "write"):
                    assert ctrl[f"num_{kind}_reqs"] == ctrl[f"num_{kind}_reqs_served"]
                    assert ctrl[f"peak_inflight_{kind}s"] <= 64
                callback_runs += 1
            archive = load(path.parent / "archive_manifest.json")
            assert len(archive["artifacts"]) == 2
            for name, entry in archive["artifacts"].items():
                stored = path.parent / entry["archive_path"]
                assert stored.stat().st_size == entry["archive_size"]
                assert digest(stored) == entry["archive_sha256"]
                key = "controller_trace" if name.startswith("controller") else "raw_trace"
                assert entry["raw_sha256"] == run[key]["sha256"]
                assert entry["raw_size"] == run[key]["size"]
                assert not (path.parent / name).exists()
                traces += 1
                raw_bytes += entry["raw_size"]
                gzip_bytes += entry["archive_size"]
            runs += 1
        for report in (root / split / "reports").glob("*.json"):
            for scores in load(report)["models"].values():
                for values in scores["per_workload"].values():
                    r = values["requests"]
                    assert r["match_mode"] == "stable_id"
                    assert r["matched"] == r["n_oracle"] == r["n_model"] == r["stable_eligible_oracle"] == r["stable_eligible_model"]
                    assert r["coverage_oracle"] == r["coverage_model"] == 1
                    pairings += r["matched"]
    failed_archives = 0
    from tools.eval import archive_results as AR
    for archive in root.glob("training/simpleo3/DDR5/*/*/failed_archive_manifest.json"):
        failure = load(archive.parent / "failure.json")
        assert failure["complete"] is False and failure["eligible_for_metrics"] is False
        assert not (archive.parent / "manifest.json").exists()
        AR.verify(archive)
        failed_archives += 1
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    with (out / "provider_calls.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(usage[0]) if usage else ["arm", "call_id"])
        writer.writeheader(); writer.writerows(usage)
    audit = {"status": "pass", "time": time.time(), "runs": runs, "candidate_callback_checks": callback_runs,
        "paired_logical_reads_across_reports": pairings, "traces": traces, "raw_bytes": raw_bytes,
        "gzip_bytes": gzip_bytes, "space_reduction_percent": 100 * (1 - gzip_bytes / raw_bytes),
        "failed_run_archives_verified": failed_archives,
        "protocol_and_visible_snapshots_verified": True, "runtime_and_candidate_hashes_verified": True,
        "test_after_both_freezes": True, "heldout_ids_absent_from_paid_requests": True,
        "actual_api_prompt_matches_saved_prompt": True, "caps_respected": True,
        "all_requests_use_configured_HIGH_and_output_maximum": True, "arms": arm_summaries,
        "analysis_source_sha256": digest(__file__),
        "figure_report_source_sha256": digest(pathlib.Path(__file__).with_name("analyze_gemini.py")),
        "post_run_working_copy_changes": {p: digest(pathlib.Path(__file__).resolve().parents[2] / p)
            for p, expected in manifest["protocol_hashes"].items()
            if digest(pathlib.Path(__file__).resolve().parents[2] / p) != expected}}
    atomic_write_json(root / "final_integrity_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
