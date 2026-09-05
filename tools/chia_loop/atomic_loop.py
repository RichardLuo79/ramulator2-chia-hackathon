#!/usr/bin/env python3
"""Run the SimpleO3 + DDR5 proof-of-concept as a native CHIA graph.

The dummy backend replaces only the LLM/coding agent. Builds, simulations,
comparators, metrics, validation, provenance checks, and archival are real.
Passing the smoke run demonstrates orchestration and contracts, not model
accuracy or generalization.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

import ray
from chia.base.ChiaFunction import ChiaFunction, get
from chia.trace.profiler import get_collector, get_profiler, start_collector, stop_collector

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.chia_loop import artifacts as aux_artifacts  # noqa: E402
from tools.chia_loop import core  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_env(repo: pathlib.Path, eval_root: pathlib.Path | None = None) -> dict:
    env = os.environ.copy()
    python_paths = [str(repo / "python"), str(repo / "tools"), str(repo)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["RAMULATOR_OPT_BUILD"] = str(repo / "build-bench")
    if eval_root is not None:
        env["EVAL_OUT"] = str(eval_root)
    return env


def _run_checked(command: list[str], *, repo: pathlib.Path,
                 log_path: pathlib.Path, eval_root: pathlib.Path | None = None) -> dict:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=repo,
        env=_command_env(repo, eval_root),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + shlex.join(command) + "\n\n[stdout]\n" + completed.stdout
        + "\n[stderr]\n" + completed.stderr
        + f"\n[returncode] {completed.returncode}\n[wall_seconds] {elapsed:.6f}\n",
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with status {completed.returncode}; see {log_path}")
    return {
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
    }


@ChiaFunction(num_cpus=1, max_retries=0)
def dummy_agent(iteration: int, current: dict, prior_result: dict | None,
                max_step_fraction: float) -> dict:
    proposal = core.dummy_proposal(
        iteration, current, prior_result, max_step_fraction)
    get_profiler().add_info({
        "agent_backend": "dummy",
        "iteration": iteration,
        "candidate": proposal["candidate"],
        "change_kind": proposal["change_kind"],
        "human_intervened_during_call": False,
    })
    return proposal


@ChiaFunction(num_cpus=12, max_retries=0)
def build_candidate(repo_text: str, proposal: dict, log_dir_text: str) -> dict:
    repo = pathlib.Path(repo_text)
    log_dir = pathlib.Path(log_dir_text)
    configure = _run_checked([
        "cmake", "-S", str(repo), "-B", str(repo / "build-bench"),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG",
    ], repo=repo, log_path=log_dir / "configure.log")
    build = _run_checked([
        "cmake", "--build", str(repo / "build-bench"), "--parallel", "12",
    ], repo=repo, log_path=log_dir / "build.log")
    flags = repo / "build-bench" / "CMakeFiles" / "ramulator.dir" / "flags.make"
    flag_text = flags.read_text(errors="replace")
    if "-O3" not in flag_text:
        raise RuntimeError(f"optimized build lacks -O3: {flags}")
    result = {
        "proposal": proposal,
        "configure": configure,
        "build": build,
        "optimization": "-O3",
        "parallel_build_jobs": 12,
        "flags_path": str(flags),
        "flags_sha256": _sha256(flags),
    }
    get_profiler().add_info({
        "optimization": "-O3",
        "build_wall_seconds": build["wall_seconds"],
        "candidate": proposal["candidate"],
    })
    return result


def _simpleo3_command(repo: pathlib.Path, workloads: list[str], models: list[str],
                      workers: int, insts_per_core: int,
                      candidate_label: str | None = None,
                      candidate: dict | None = None) -> list[str]:
    command = [
        sys.executable, str(repo / "tools/eval/run_simpleo3.py"),
        "--std", "DDR5", "--workloads", *workloads,
        "--models", ",".join(models),
        "--workers", str(workers),
        "--insts-per-core", str(insts_per_core),
    ]
    if "candidate" in models:
        if candidate_label is None or candidate is None:
            raise ValueError("candidate evaluation requires a label and parameters")
        command.extend([
            "--candidate-label", candidate_label,
            "--candidate-kw", json.dumps(candidate, sort_keys=True),
        ])
    return command


def _postprocess_command(repo: pathlib.Path, workloads: list[str], labels: list[str],
                         output: pathlib.Path) -> list[str]:
    return [
        sys.executable, str(repo / "tools/eval/postprocess.py"),
        "--frontend", "simpleo3", "--std", "DDR5",
        "--workloads", *workloads,
        "--models", ",".join(labels),
        "--output", str(output),
    ]


@ChiaFunction(num_cpus=12, max_retries=0)
def evaluate_baselines(repo_text: str, eval_root_text: str,
                       training: list[str], validation: list[str],
                       comparisons: list[str], workers: int,
                       insts_per_core: int, build_evidence: dict,
                       report_dir_text: str, log_dir_text: str) -> dict:
    repo = pathlib.Path(repo_text)
    eval_root = pathlib.Path(eval_root_text)
    report_dir = pathlib.Path(report_dir_text)
    log_dir = pathlib.Path(log_dir_text)
    all_workloads = training + validation
    run = _run_checked(
        _simpleo3_command(
            repo, all_workloads, ["oracle", *comparisons], workers,
            insts_per_core),
        repo=repo, eval_root=eval_root,
        log_path=log_dir / "run.log",
    )
    reports = {}
    for split, workloads in (("training", training), ("validation", validation)):
        report = report_dir / f"baseline_{split}.json"
        command = _postprocess_command(repo, workloads, comparisons, report)
        post = _run_checked(
            command, repo=repo, eval_root=eval_root,
            log_path=log_dir / f"postprocess_{split}.log")
        reports[split] = {
            "path": str(report),
            "sha256": _sha256(report),
            "summary": json.loads(report.read_text()),
            "postprocess": post,
        }
    result = {
        "models": ["oracle", *comparisons],
        "workloads": all_workloads,
        "run": run,
        "reports": reports,
        "build_flags_sha256": build_evidence["flags_sha256"],
    }
    get_profiler().add_info({
        "frontend": "SimpleO3", "standard": "DDR5",
        "workload_count": len(all_workloads),
        "models": result["models"],
        "wall_seconds": run["wall_seconds"],
    })
    return result


@ChiaFunction(num_cpus=12, max_retries=0)
def evaluate_candidate(repo_text: str, eval_root_text: str,
                       workloads: list[str], comparisons: list[str],
                       workers: int, insts_per_core: int, proposal: dict,
                       candidate_label: str, build_evidence: dict,
                       report_path_text: str, log_dir_text: str,
                       split: str) -> dict:
    repo = pathlib.Path(repo_text)
    eval_root = pathlib.Path(eval_root_text)
    report = pathlib.Path(report_path_text)
    log_dir = pathlib.Path(log_dir_text)
    run = _run_checked(
        _simpleo3_command(
            repo, workloads, ["candidate"], workers, insts_per_core,
            candidate_label, proposal["candidate"]),
        repo=repo, eval_root=eval_root,
        log_path=log_dir / "run.log",
    )
    post = _run_checked(
        _postprocess_command(
            repo, workloads, [candidate_label, *comparisons], report),
        repo=repo, eval_root=eval_root,
        log_path=log_dir / "postprocess.log",
    )
    summary = json.loads(report.read_text())
    result = {
        "split": split,
        "candidate_label": candidate_label,
        "proposal": proposal,
        "workloads": workloads,
        "summary": summary,
        "report_path": str(report),
        "report_sha256": _sha256(report),
        "run": run,
        "postprocess": post,
        "build_flags_sha256": build_evidence["flags_sha256"],
    }
    objective = core.objectives(result)
    get_profiler().add_info({
        "frontend": "SimpleO3", "standard": "DDR5", "split": split,
        "candidate_label": candidate_label,
        "candidate": proposal["candidate"],
        **objective,
    })
    return result


@ChiaFunction(num_cpus=1, max_retries=0)
def choose_candidate(training_results: list[dict]) -> dict:
    selection = core.select_candidate(training_results)
    get_profiler().add_info(selection["selected_objectives"] | {
        "selection_policy": selection["policy"],
        "selected_label": selection["selected_label"],
    })
    return selection


@ChiaFunction(num_cpus=1, max_retries=0)
def archive_traces(repo_text: str, eval_root_text: str,
                   manifest_text: str, log_path_text: str) -> dict:
    repo = pathlib.Path(repo_text)
    eval_root = pathlib.Path(eval_root_text)
    manifest = pathlib.Path(manifest_text)
    command = [
        sys.executable, str(repo / "tools/eval/archive_results.py"),
        "compress", str(eval_root), "--manifest", str(manifest),
    ]
    compress = _run_checked(
        command, repo=repo, eval_root=eval_root,
        log_path=pathlib.Path(log_path_text))
    verify_log = pathlib.Path(log_path_text).with_name("verify.log")
    verify_result = _run_checked([
        sys.executable, str(repo / "tools/eval/archive_results.py"),
        "verify", str(manifest),
    ], repo=repo, eval_root=eval_root, log_path=verify_log)
    payload = json.loads(manifest.read_text())
    result = {
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "trace_count": len(payload["artifacts"]),
        "compress": compress,
        "verify": verify_result,
    }
    get_profiler().add_info({
        "trace_count": result["trace_count"],
        "codec": "gzip",
        "verified": True,
    })
    return result


def _append_event(path: pathlib.Path, event: str, **fields) -> None:
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_text(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True,
        capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=pathlib.Path,
        default=REPO / "tools/chia_loop/smoke.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--human-note", action="append", default=[])
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = core.load_config(config_path)
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime(
        "smoke-%Y%m%dT%H%M%SZ")
    if not run_id or any(character not in
                         "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                         for character in run_id):
        parser.error("--run-id must be a safe path component")
    campaign = REPO / "eval_out" / "chia" / run_id
    try:
        campaign.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite campaign {campaign}") from exc

    branch = _git_text("branch", "--show-current")
    if branch != "atomic-chia-loop":
        raise RuntimeError(
            f"CHIA smoke must run from local atomic-chia-loop, not {branch!r}")
    if _git_text("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("refusing to run from a tracked-dirty worktree")

    eval_root = campaign / "evaluation"
    reports = campaign / "reports"
    interactions = campaign / "interactions"
    logs = campaign / "logs"
    profiles = campaign / "profiles"
    artifacts_dir = campaign / "artifacts"
    for path in (eval_root, reports, interactions, logs, profiles, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)

    protected = core.protected_snapshot(REPO)
    core.atomic_write_json(campaign / "protected_inputs.json", protected)
    config_identity = {
        "path": str(config_path),
        "sha256": _sha256(config_path),
        "content": config,
    }
    human_interventions = [
        {
            "phase": "pre_run",
            "actor": "human",
            "decision": "Use SimpleO3 + DDR5 for the first smoke run.",
        },
        {
            "phase": "pre_run",
            "actor": "human",
            "decision": "Use a dummy agent before configuring a real LLM backend.",
        },
        {
            "phase": "pre_run",
            "actor": "human",
            "decision": "Require -O3 and at most 12 evaluation workers.",
        },
        {
            "phase": "pre_run",
            "actor": "human",
            "decision": (
                "Exclude the under-review bank-level queueing model from this "
                "branch while preserving it on atomic-prototype."),
        },
        *[
            {"phase": "pre_run", "actor": "human", "decision": note}
            for note in args.human_note
        ],
    ]
    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": {
            "branch": branch,
            "revision": _git_text("rev-parse", "HEAD"),
            "remote_mutations_permitted": False,
            "push_performed": False,
        },
        "chia": {
            "distribution": "chialoops",
            "version": importlib.metadata.version("chialoops"),
            "native_chia_functions": True,
            "profile_enabled": True,
        },
        "agent": {
            "backend": "dummy",
            "llm_api_key_required": False,
            "llm_api_key_used": False,
            "source_edit_performed": False,
            "candidate_change_channel": "validated_runtime_parameter",
        },
        "config": config_identity,
        "human_interventions": human_interventions,
        "metrics": {
            "co_equal_objectives": [
                "cycle_macro_mae_pct", "request_macro_mae_over_L"],
            "selection": "incumbent_retention_unless_pareto_dominated",
            "request_coverage_gate": "exact_bidirectional_stable_id",
        },
    }
    core.atomic_write_json(campaign / "run_manifest.json", run_manifest)
    event_log = campaign / "audit.jsonl"
    _append_event(event_log, "campaign_started", run_id=run_id, branch=branch)

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": os.pathsep.join([
                str(REPO / "python"), str(REPO / "tools"), str(REPO)]),
        }
    }
    ray.init(
        num_cpus=config["workers"], include_dashboard=False,
        ignore_reinit_error=True, runtime_env=runtime_env,
    )
    start_collector(log_dir=str(profiles))
    profiler = get_profiler()
    profiler.log_event(
        "human_constraints", interventions=human_interventions,
        human_intervened_during_run=False)

    baseline_result = None
    training_results = []
    selection = None
    validation_result = None
    trace_archive = None
    profile_path = None
    try:
        current = dict(config["initial_candidate"])
        for iteration in range(config["iterations"]):
            prior = training_results[-1] if training_results else None
            request = {
                "backend": "dummy",
                "iteration": iteration,
                "current_candidate": current,
                "feedback": prior,
                "contract": (
                    "Propose one explainable change to the fixed-delay skeleton; "
                    "do not alter evaluators, traces, comparators, or metrics."),
                "human_intervened": False,
            }
            proposal = get(dummy_agent.chia_remote(
                iteration, current, prior,
                config["dummy_max_latency_step_fraction"]))
            core.atomic_write_json(
                interactions / f"iteration_{iteration:03d}.json",
                {"request": request, "response": proposal})
            _append_event(
                event_log, "agent_response", iteration=iteration,
                backend="dummy", candidate=proposal["candidate"],
                human_intervened=False)

            build = get(build_candidate.chia_remote(
                str(REPO), proposal, str(logs / f"iteration_{iteration:03d}")))
            core.assert_protected_unchanged(REPO, protected)
            _append_event(
                event_log, "optimized_build_complete", iteration=iteration,
                optimization=build["optimization"], workers=12,
                flags_sha256=build["flags_sha256"])

            if iteration == 0:
                baseline_result = get(evaluate_baselines.chia_remote(
                    str(REPO), str(eval_root),
                    config["training_workloads"], config["validation_workloads"],
                    config["comparison_models"], config["workers"],
                    config["insts_per_core"], build, str(reports),
                    str(logs / "baselines")))
                core.assert_protected_unchanged(REPO, protected)
                _append_event(
                    event_log, "baseline_matrix_complete",
                    models=baseline_result["models"],
                    workloads=baseline_result["workloads"])

            label = f"candidate_iter_{iteration:03d}"
            report = reports / f"training_{label}.json"
            training = get(evaluate_candidate.chia_remote(
                str(REPO), str(eval_root), config["training_workloads"],
                config["comparison_models"], config["workers"],
                config["insts_per_core"], proposal, label, build,
                str(report), str(logs / label), "training"))
            core.assert_protected_unchanged(REPO, protected)
            training_results.append(training)
            _append_event(
                event_log, "training_evaluation_complete", iteration=iteration,
                candidate_label=label, objectives=core.objectives(training),
                exact_request_coverage=True)
            current = dict(proposal["candidate"])

        selection = get(choose_candidate.chia_remote(training_results))
        core.atomic_write_json(reports / "selection.json", selection)
        _append_event(
            event_log, "candidate_selected",
            candidate_label=selection["selected_label"],
            objectives=selection["selected_objectives"],
            policy=selection["policy"])
        selected_training = next(
            result for result in training_results
            if result["candidate_label"] == selection["selected_label"])
        selected_proposal = selected_training["proposal"]
        selected_build = get(build_candidate.chia_remote(
            str(REPO), selected_proposal, str(logs / "selected_build")))
        core.assert_protected_unchanged(REPO, protected)
        validation_result = get(evaluate_candidate.chia_remote(
            str(REPO), str(eval_root), config["validation_workloads"],
            config["comparison_models"], config["workers"],
            config["insts_per_core"], selected_proposal,
            selection["selected_label"], selected_build,
            str(reports / "validation_selected.json"),
            str(logs / "validation_selected"), "validation"))
        core.assert_protected_unchanged(REPO, protected)
        _append_event(
            event_log, "validation_complete",
            candidate_label=selection["selected_label"],
            objectives=core.objectives(validation_result),
            exact_request_coverage=True)

        trace_archive = get(archive_traces.chia_remote(
            str(REPO), str(eval_root),
            str(artifacts_dir / "trace_archive_manifest.json"),
            str(logs / "archive" / "compress.log")))
        _append_event(
            event_log, "trace_archive_verified",
            trace_count=trace_archive["trace_count"], codec="gzip")
        core.assert_protected_unchanged(REPO, protected)

        collector = get_collector()
        if collector is not None:
            ray.get(collector.get_events.remote())
            profile_path = ray.get(collector.get_log_path.remote())
        run_manifest.update({
            "status": "smoke_passed",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "smoke_claim": (
                "The CHIA graph, optimized build, real simulator evaluations, "
                "comparators, exact request matching, validation split, and "
                "verified trace archive completed. This is not an accuracy claim."),
            "baseline_result": baseline_result,
            "training_results": training_results,
            "selection": selection,
            "validation_result": validation_result,
            "trace_archive": trace_archive,
            "chia_profile_path": profile_path,
            "human_intervened_during_run": False,
        })
        core.atomic_write_json(campaign / "run_manifest.json", run_manifest)
        _append_event(event_log, "campaign_smoke_passed", run_id=run_id)
    except BaseException as exc:
        run_manifest.update({
            "status": "failed",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "failure_type": type(exc).__name__,
            "failure": str(exc),
        })
        core.atomic_write_json(campaign / "run_manifest.json", run_manifest)
        _append_event(
            event_log, "campaign_failed", failure_type=type(exc).__name__,
            failure=str(exc))
        raise
    finally:
        try:
            collector = get_collector()
            if collector is not None:
                ray.get(collector.get_events.remote())
                if profile_path is None:
                    profile_path = ray.get(collector.get_log_path.remote())
        finally:
            stop_collector()
            ray.shutdown()

    auxiliary_manifest = artifacts_dir / "aux_archive_manifest.json"
    aux_payload = aux_artifacts.compress(
        campaign, auxiliary_manifest,
        min_bytes=config["archive_aux_min_bytes"])
    aux_artifacts.verify(auxiliary_manifest)
    run_manifest["auxiliary_archive"] = {
        "manifest_path": str(auxiliary_manifest),
        "manifest_sha256": _sha256(auxiliary_manifest),
        "artifact_count": len(aux_payload["artifacts"]),
        "minimum_raw_bytes": config["archive_aux_min_bytes"],
        "verified": True,
    }
    run_manifest["chia_profile_path"] = profile_path
    core.atomic_write_json(campaign / "run_manifest.json", run_manifest)
    core.assert_protected_unchanged(REPO, protected)
    print(json.dumps({
        "status": run_manifest["status"],
        "campaign": str(campaign),
        "selected": selection,
        "validation_objectives": core.objectives(validation_result),
        "trace_archive_count": trace_archive["trace_count"],
        "aux_archive_count": len(aux_payload["artifacts"]),
        "chia_profile": profile_path,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
