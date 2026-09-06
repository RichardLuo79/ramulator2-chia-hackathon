"""One authorized Astra run after prerequisite supervisors have terminated.

Dependency checks read only predecessor supervisor status and locks. An explicit
--continue-training-from additionally imports that effort's training checkpoint
during non-generating preparation, never its held-out results. Each effort has
its own queue record.
Preparation and generation use the existing guarded runner without callbacks
to this chat. No automatic repair, protocol upgrade, or authorization increase.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import shutil
import sys
import time

from tools.chia_loop import recovery as R
from tools.chia_loop.core import atomic_write_json
from . import runner as B, transport as T

TERMINAL = {"completed", "needs_attention", "stopped"}


def dependencies(roots):
    observations = []
    for root in roots:
        path = root / "supervisor_state.json"
        state = R.read_json(path) if path.is_file() else {}
        if state and state.get("run_id") != root.name:
            raise RuntimeError("predecessor supervisor owner mismatch")
        idle = True
        for name in (".supervisor.lock", ".runner.lock"):
            try:
                # Read-only existing file: never create or modify a predecessor.
                with (root / name).open("r") as lock:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except (FileNotFoundError, BlockingIOError):
                idle = False
        observations.append({"run_id": root.name, "supervisor_status": state.get("status"),
                             "worker_locks_released": idle,
                             "ready": idle and state.get("status") in TERMINAL})
    return observations


def fingerprint(binary):
    """Bind the queued implementation/build inputs before delayed execution."""
    files = {B.REPO / "CMakeLists.txt"}
    if (B.REPO / "tests/utils.py").is_file():
        files.add(B.REPO / "tests/utils.py")
    for directory in ("src", "python", "tools/chia_loop", "tools/eval", "tests/unit_tests",
                      "ext/fmt/include", "ext/yaml-cpp/include"):
        files.update(p for p in (B.REPO / directory).rglob("*") if p.is_file() and
                     (p.suffix in {".py", ".cpp", ".h", ".hpp", ".cmake"} or p.name == "CMakeLists.txt"))
    files.update((B.REPO / "tools/chia_loop/prompts").glob("*.md"))
    files.update((B.REPO / "tools/chia_loop/configs").glob("*.json"))
    build = B.REPO / "build-bench"
    files.update(p for p in build.rglob("*") if p.is_file() and
                 (p.suffix in {".o", ".a", ".so"} or p.name in {
                     "CMakeCache.txt", "compile_commands.json", "link.txt", "flags.make"}))
    return {"files": {str(p.relative_to(B.REPO)): T.P.sha(p.read_bytes()) for p in sorted(files)},
            "codex_binary_sha256": T.P.sha(pathlib.Path(binary).read_bytes()),
            "python_version": sys.version, "policy": B.POLICY}


def execute(args):
    if not args.authorize_paid:
        raise ValueError("queueing generation requires explicit authorization")
    root = args.root.resolve()
    args.root = root
    T.check_stop(root)
    if root.exists():
        raise RuntimeError("dependency queue requires a fresh run root; no implicit resume")
    T.F.limits(args.max_iterations, args.usd_cap, args.cpus,
               iteration_guard=getattr(args, "iteration_guard", False))
    after = [p.resolve(strict=True) for p in args.after]
    if not after or any(root == p or root in p.parents or p in root.parents for p in after):
        raise ValueError("predecessor directories must be outside this independent run")
    binary = pathlib.Path(args.codex_binary or shutil.which("codex") or "").resolve(strict=True)
    if not binary.is_file():
        raise ValueError("Codex executable is required")
    args.codex_binary = str(binary)
    args.auth_file = args.auth_file.resolve(strict=True)
    args.auth_mode = "chatgpt"
    args.wait_for_cpus = True
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.with_name(root.name + ".queue.json")
    with R.exclusive_lock(root.with_name(root.name + ".queue.lock")):
        if path.exists():
            raise RuntimeError("existing queue receipt must not be replaced or automatically relaunched")
        record = {"record_type": "single_run_dependency_queue", "run_id": root.name, "pid": os.getpid(),
            "model": T.MODEL, "effort": args.effort, "maximum_iterations": args.max_iterations,
            "usd_equivalent_guard": args.usd_cap, "cpu_budget": args.cpus, "auth_mode": "chatgpt",
            "guard_mode": "iterations" if getattr(args, "iteration_guard", False) else "usd",
            "generation_authorized": True, "authorization": "operator requested Astra pipeline after current Gemini jobs",
            "started_at": time.time(), "status": "waiting_for_predecessors",
            "after": [str(p) for p in after], "dependencies": dependencies(after),
            "pinned_implementation": fingerprint(binary), "model_generation_started": False}
        evaluation_path = pathlib.Path(getattr(args, "evaluation_config", None) or B.W.DEFAULT).resolve(strict=True)
        record["evaluation_profile"] = {"path": str(evaluation_path), "sha256": T.P.sha(evaluation_path.read_bytes())}
        loop_path = pathlib.Path(getattr(args, "loop_config", None) or B.L.DEFAULT).resolve(strict=True)
        record["loop_profile"] = {"path": str(loop_path), "sha256": T.P.sha(loop_path.read_bytes()),
                                  "configuration": B.L.read_source(loop_path)}
        if getattr(args, "carry_budget_from", None) is not None:
            record["financial_predecessor"] = str(args.carry_budget_from.resolve(strict=True))
        if getattr(args, "continue_training_from", None) is not None:
            record["training_predecessor"] = str(args.continue_training_from.resolve(strict=True))
            record["authorization"] = (f"operator requested same-effort continuation to {args.max_iterations} total designs; "
                                       f"guard mode: {record['guard_mode']}")
        atomic_write_json(path, record)
        try:
            while True:
                T.check_stop(root)
                record.update(dependencies=dependencies(after), checked_at=time.time())
                atomic_write_json(path, record)
                if all(p["ready"] for p in record["dependencies"]):
                    break
                if time.time() - record["started_at"] >= 86400:
                    raise R.OperationalPause("24-hour predecessor wait guard", retryable=False)
                time.sleep(30)
            if fingerprint(binary) != record["pinned_implementation"]:
                raise RuntimeError("queued implementation or build inputs changed; do not launch")
            if T.P.sha(evaluation_path.read_bytes()) != record["evaluation_profile"]["sha256"]:
                raise RuntimeError("queued evaluation profile changed; do not launch")
            if (T.P.sha(loop_path.read_bytes()) != record["loop_profile"]["sha256"]
                    or B.L.read_source(loop_path) != record["loop_profile"]["configuration"]):
                raise RuntimeError("queued loop profile changed; do not launch")
            record.update(status="preparing", preparation_started_at=time.time())
            atomic_write_json(path, record)
            B.prepare_with_wait(args)
            T.check_stop(root)
            if fingerprint(binary) != record["pinned_implementation"]:
                raise RuntimeError("implementation changed during preparation; do not launch")
            if B.W.load(root) != B.W.validate(json.loads(evaluation_path.read_text())):
                raise RuntimeError("prepared evaluation differs from queued profile")
            if B.L.load(root) != record["loop_profile"]["configuration"]:
                raise RuntimeError("prepared loop configuration differs from queued profile")
            B.verify(root)
            record.update(status="supervising", preparation_passed=True, launch_authorized_at=time.time(),
                          model_generation_started=None, live_status="see this run's supervisor_state.json and ledger.json")
            atomic_write_json(path, record)
            # The supervisor owns all model calls, reservations, bounded retries,
            # CPU leases, checkpoints, frozen testing and compressed artifacts.
            B.supervise(root)
            supervisor = R.read_json(root / "supervisor_state.json")
            record.update(status=supervisor["status"], finished_at=time.time())
        except BaseException as exc:
            record.update(status="stopped" if isinstance(exc, (R.OperatorStop, KeyboardInterrupt)) else "needs_attention",
                          error_kind=type(exc).__name__, finished_at=time.time())
            raise
        finally:
            # Generation truth comes from this run's ledger, not a phase name.
            ledger = R.read_json(root / "ledger.json") if (root / "ledger.json").exists() else {"calls": []}
            record["model_generation_started"] = bool(ledger["calls"])
            atomic_write_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--after", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--effort", choices=T.EFFORTS, required=True)
    parser.add_argument("--auth-file", type=pathlib.Path, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    guard = parser.add_mutually_exclusive_group(required=True)
    guard.add_argument("--usd-cap", type=float)
    guard.add_argument("--iteration-guard", action="store_true")
    parser.add_argument("--cpus", type=int, default=6)
    parser.add_argument("--evaluation-config", type=pathlib.Path, default=B.W.DEFAULT)
    parser.add_argument("--loop-config", type=pathlib.Path, default=B.L.DEFAULT)
    parser.add_argument("--codex-binary")
    parser.add_argument("--carry-budget-from", type=pathlib.Path,
                        help="retain the same-effort predecessor's usage under the existing cap, without scientific history")
    parser.add_argument("--continue-training-from", type=pathlib.Path,
                        help="import completed own-run training checkpoints, not held-out results; preserve predecessor")
    parser.add_argument("--authorize-paid", action="store_true", required=True)
    args = parser.parse_args()
    try:
        execute(args)
    except Exception as exc:
        print(json.dumps({"status": "needs_attention", "error_kind": type(exc).__name__}), flush=True)
        raise SystemExit(78)


if __name__ == "__main__":
    main()
