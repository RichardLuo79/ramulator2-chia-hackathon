"""Read-only completion watcher: write a comparison report, never steer a run.

This produces local artifacts. It does NOT wake a Codex conversation or send
an external notification. Neither the watcher nor its report is agent input.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from tools.chia_loop import recovery as R
from tools.chia_loop.core import atomic_write_json


def snapshot(root):
    manifest = R.read_json(root / "run_manifest.json")
    supervisor = R.read_json(root / "supervisor_state.json") if (root / "supervisor_state.json").exists() else {}
    state = R.read_json(root / "state.json") if (root / "state.json").exists() else manifest.get("state", {})
    ledger = R.read_json(root / "ledger.json") if (root / "ledger.json").exists() else {"calls": []}
    terminal = supervisor.get("status") in {"completed", "needs_attention", "stopped"}
    completed = (terminal and supervisor.get("status") == "completed" and manifest.get("status") == "completed"
                 and manifest.get("archives_verified") is True)
    calls, carry = ledger["calls"], ledger.get("carryover", {})
    return {"run_id": root.name, "root": str(root), "model": manifest.get("model"),
        "status": "completed" if completed else "stopped_incomplete" if terminal else "running",
        "supervisor_status": supervisor.get("status"), "terminal": terminal,
        "evaluated_designs": sum(h.get("status") in {"valid", "evaluated"} for h in state.get("history", [])),
        "promotions": sum(bool(h.get("promoted")) for h in state.get("history", [])),
        "incumbent": state.get("incumbent"), "termination": state.get("termination"),
        "stop_reason": state.get("stop_reason", supervisor.get("reason")),
        "human_intervention": manifest.get("human_intervention"),
        "known_standard_usd": carry.get("estimated_standard_usd", 0) + sum(c.get("estimated_standard_usd") or 0 for c in calls),
        "conservative_cap_charge_usd": carry.get("cap_charge_usd", 0) + sum(c["cap_charge_usd"] for c in calls),
        "generation_attempts": len(calls),
        "review_attempts": sum(c.get("purpose") == "review" for c in calls),
        "unknown_usage_calls": carry.get("unknown_usage_calls", 0) + sum(c.get("estimated_standard_usd") is None for c in calls),
        "training_metrics": state.get("candidates", {}).get(state.get("incumbent"), {}).get("metrics"),
        "test_metrics": manifest.get("final_test_metrics") if completed else None,
        "history": [{k: h[k] for k in ("iteration", "status", "promoted", "metrics") if k in h}
                    for h in state.get("history", [])]}


def report(roots, output, wait=False, interval=30):
    output.mkdir(parents=True, exist_ok=True)
    with R.exclusive_lock(output / ".watcher.lock"):
        while True:
            runs = [snapshot(root) for root in roots]
            done = all(run["terminal"] for run in runs)
            atomic_write_json(output / "status.json", {"record_type": "read_only_completion_watch",
                "checked_at": time.time(), "all_terminal": done, "chat_wakeup_configured": False,
                "runs": [{k: r[k] for k in ("run_id", "status", "evaluated_designs", "generation_attempts")} for r in runs]})
            if done or not wait:
                payload = {"record_type": "post_run_comparison", "generated_at": time.time(),
                           "all_terminal": done, "optimization_feedback": False, "runs": runs}
                atomic_write_json(output / "results.json", payload)
                lines = ["# Independent Gemini run comparison", "",
                    "This is a read-only report, not a combined optimization run or agent feedback.", "",
                    "| Model | Status | Evaluated designs | Selected | Test core MAE (%) | Test request MAE / L | Known cost ($) |",
                    "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
                for run in runs:
                    metrics = (run["test_metrics"] or {}).get("aggregate", {})
                    core = metrics.get("cycle_macro_mae_pct")
                    request = metrics.get("request_macro_mae_over_L")
                    lines.append(f"| {run['model']} | {run['status']} | {run['evaluated_designs']} | {run['incumbent']} | "
                        + (f"{core:.4f}" if core is not None else "not available") + " | "
                        + (f"{request:.6f}" if request is not None else "not available")
                        + f" | {run['known_standard_usd']:.4f} |")
                lines += ["", "Known costs include carried financial charges, proposal calls and automatic reviews. "
                    "Unknown-response reservations remain in the conservative guard; these estimates are not billing invoices.",
                    "Full per-workload metrics, evolution history and accounting are in `results.json`.", "",
                    "A stopped/incomplete run has no eligible final-test score in this report. "
                    "The watcher never resumes, approves, repairs, deletes, or changes a run.", "",
                    "This local watcher does not wake a chat session or send notifications.", ""]
                (output / "summary.md").write_text("\n".join(lines))
                return payload
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    roots = [r.resolve(strict=True) for r in args.roots]
    output = args.output.resolve()
    if any(output == r or r in output.parents or output in r.parents for r in roots):
        parser.error("comparison output must be outside each optimization run")
    report(roots, output, args.wait)


if __name__ == "__main__":
    main()
