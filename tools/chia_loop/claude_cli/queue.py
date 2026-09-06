"""One independent Fable job, with separate preparation and paid-launch gates."""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

from tools.chia_loop import recovery as R
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.codex_cli.queue import fingerprint as shared_fingerprint
from . import runner as B, transport as T


def fingerprint(binary):
    # The current checkout/build inputs are shared with the existing protected
    # evaluator. This reads code, never another experiment's scientific data.
    result = shared_fingerprint(binary)
    result["claude_binary_sha256"] = result.pop("codex_binary_sha256")
    result["policy"] = B.POLICY
    return result


def execute(args):
    if args.authorize_paid == args.prepare_only:
        raise ValueError("choose preparation only OR explicitly authorize generation")
    root = args.root.resolve()
    if root.exists():
        raise RuntimeError("a new queue requires a fresh run root")
    args.root, args.auth_mode, args.usd_cap = root, "claude_subscription", None
    args.iteration_guard, args.wait_for_cpus = True, True
    T.limits(args.max_iterations, None, args.cpus, iteration_guard=True)
    T.check_stop(root)
    binary = pathlib.Path(args.claude_binary or shutil.which("claude") or "").resolve(strict=True)
    args.claude_binary = str(binary)
    args.auth_file = args.auth_file.resolve(strict=True)
    T.auth.check(args.auth_file)
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.with_name(root.name + ".queue.json")
    with R.exclusive_lock(root.with_name(root.name + ".queue.lock")):
        if path.exists():
            raise RuntimeError("an existing queue receipt cannot be overwritten")
        state = {"run_id": root.name, "model": T.MODEL, "effort": args.effort,
                 "maximum_iterations": args.max_iterations, "usd_cap": None, "guard_mode": "iterations",
                 "cpu_budget": args.cpus, "generation_authorized": args.authorize_paid,
                 "status": "preparing", "started_at": time.time(), "pinned_implementation": fingerprint(binary)}
        evaluation = pathlib.Path(args.evaluation_config).resolve(strict=True)
        loop = pathlib.Path(args.loop_config).resolve(strict=True)
        state["evaluation_profile"] = {"path": str(evaluation), "sha256": T.P.sha(evaluation.read_bytes())}
        state["loop_profile"] = {"path": str(loop), "sha256": T.P.sha(loop.read_bytes())}
        atomic_write_json(path, state)
        try:
            B.prepare_with_wait(args)
            T.check_stop(root)
            if fingerprint(binary) != state["pinned_implementation"]:
                raise RuntimeError("queued implementation changed; no generation launch")
            if (T.P.sha(evaluation.read_bytes()) != state["evaluation_profile"]["sha256"]
                    or T.P.sha(loop.read_bytes()) != state["loop_profile"]["sha256"]):
                raise RuntimeError("queued profile changed; no generation launch")
            B.verify(root)
            state.update(status="prepared", preparation_passed=True)
            atomic_write_json(path, state)
            if args.authorize_paid:
                state["status"] = "supervising"
                atomic_write_json(path, state)
                B.supervise(root)
                state["status"] = R.read_json(root / "supervisor_state.json")["status"]
        except BaseException as exc:
            state.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, R.OperatorStop)) else "needs_attention",
                         error_kind=type(exc).__name__)
            raise
        finally:
            state["finished_at"] = time.time()
            state["model_generation_started"] = bool(R.read_json(root / "ledger.json")["calls"]) if (root / "ledger.json").exists() else False
            atomic_write_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--effort", required=True, choices=T.EFFORTS)
    parser.add_argument("--auth-file", required=True, type=pathlib.Path)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--cpus", type=int, default=3)
    parser.add_argument("--claude-binary")
    parser.add_argument("--evaluation-config", type=pathlib.Path, default=B.REPO / "tools/chia_loop/configs/ddr5_frontend_transfer_v1.json")
    parser.add_argument("--loop-config", type=pathlib.Path, default=B.L.DEFAULT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--authorize-paid", action="store_true")
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
