#!/usr/bin/env python3
"""Unattended single-run supervisor. Never changes a protocol, budget or source."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.chia_loop import artifacts, real_core as P, recovery as R
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.gemini_loop import configured_policy


def supervise(root, backend, *, resume=False):
    root = pathlib.Path(root).resolve()
    if not (root / "preflight_pass.json").exists():
        raise RuntimeError("prepare and pass the full-window preflight before supervision")
    if (root / "run_manifest.json").exists() and not resume:
        raise RuntimeError("an existing run requires --resume; old protocols cannot be upgraded in place")
    policy = configured_policy(root)
    with R.exclusive_lock(root / ".supervisor.lock"):
        path = root / "supervisor_state.json"
        state = R.read_json(path) if path.exists() else {
            "run_id": root.name, "model": P.MODELS[backend], "started_at": time.time(),
            "launches": 0, "crash_restarts": 0, "human_intervention": False}
        if state["run_id"] != root.name or state["model"] != P.MODELS[backend]:
            raise RuntimeError("supervisor state belongs to a different run/model")
        atomic_write_json(path, state)
        while True:
            R.check_stop(root)
            if time.time() - state["started_at"] >= policy["maximum_run_wall_seconds"]:
                state.update(status="needs_attention", reason="supervisor wall-time guard reached")
                atomic_write_json(path, state)
                return 78
            command = [sys.executable, "-m", "tools.chia_loop.gemini_loop", "--root", str(root), "--model", backend]
            if resume or state["launches"]:
                command.append("--resume")
            state["launches"] += 1
            state.update(status="running", last_launch_at=time.time())
            atomic_write_json(path, state)
            # Outside the worker finalizer's default artifact classes: this
            # stream stays open until the child has completely exited.
            log = root / "supervisor_logs" / f"launch_{state['launches']:03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"run_id": root.name, "event": "runner_launch", "launch": state["launches"], "log": str(log)}), flush=True)
            with log.open("x") as stream:
                process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except KeyboardInterrupt:
                    # Request a safe stop; do not silently leave the paid child running.
                    (root / "STOP").touch(exist_ok=True)
                    process.wait()
                    raise
            artifacts.compress(root, root / "aux_archive_manifest.json",
                               min_bytes=16_384, parts={"supervisor_logs"})
            artifacts.verify(root / "aux_archive_manifest.json")
            recorded = R.read_json(root / "run_manifest.json") if (root / "run_manifest.json").exists() else {}
            if code == 0 and recorded.get("status") == "completed" and recorded.get("archives_verified"):
                state.update(status="completed", finished_at=time.time())
                atomic_write_json(path, state)
                return 0
            pause = recorded.get("pause", {})
            launch_pause = root / "launch_pause.json"
            if launch_pause.exists() and launch_pause.stat().st_mtime >= state["last_launch_at"]:
                pause = R.read_json(launch_pause)
            if code == 75 and pause.get("retryable"):
                retry_at = max(time.time() + 1, pause.get("retry_at", 0))
                state.update(status="cooldown", reason=pause.get("reason"), retry_at=retry_at)
                atomic_write_json(path, state)
                R.wait_until(root, min(retry_at, state["started_at"] + policy["maximum_run_wall_seconds"]))
                continue
            # Only an abrupt process death with an unfinished journal is restarted.
            # Auth, protocol, storage, budget, review and explicit STOP decisions
            # must never be reinterpreted as permission to retry indefinitely.
            unfinished = recorded.get("status") == "running" or (
                recorded.get("status") == "completed" and not recorded.get("archives_verified"))
            if (unfinished and code not in (0, 2, 75, 78, 130, -2)
                    and state["crash_restarts"] < policy["maximum_process_restarts"]):
                state["crash_restarts"] += 1
                state.update(status="crash_recovery", last_exit_code=code)
                atomic_write_json(path, state)
                R.wait_until(root, time.time() + 10 * state["crash_restarts"])
                continue
            state.update(status="needs_attention", last_exit_code=code,
                         reason=pause.get("reason", recorded.get("failure", "runner stopped without a recoverable outcome")))
            atomic_write_json(path, state)
            print(json.dumps({"run_id": root.name, "event": "needs_attention", "reason": state["reason"]}), flush=True)
            return code or 78


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--model", choices=P.MODELS, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        return supervise(args.root, args.model, resume=args.resume)
    except R.OperatorStop:
        path = args.root / "supervisor_state.json"
        state = R.read_json(path) if path.exists() else {"run_id": args.root.name, "model": P.MODELS[args.model]}
        state.update(status="stopped", external_stop_observed=True, human_intervention=None,
                     reason="external STOP requested; actor not inferred")
        atomic_write_json(path, state)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
