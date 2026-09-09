#!/usr/bin/env python3
"""Non-billable preparation of one fresh full-window Gemini run.

Never imports sources, candidates, metrics, or ledgers from an earlier run.
The isolated and ordinary Python runners must agree before any paid call.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from tools.chia_loop import real_core as P, real_eval as E
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.traffic import traffic_population
from tools.chia_loop.run_records import validate_limits
from tools.chia_loop import evaluation_config as W, transfer, recovery as R
from tools.chia_loop import loop_config as L
from tools.chia_loop import prompt_cache as K


def progress(stage, **extra):
    print(json.dumps({"time": time.time(), "stage": stage, **extra}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--model", choices=P.MODELS, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--evaluation-config", type=pathlib.Path, default=W.DEFAULT,
                    help="operator-owned DDR5 workload/transfer profile; frozen before generation")
    ap.add_argument("--loop-config", type=pathlib.Path, default=L.DEFAULT,
                    help="operator-owned feature/feedback/search ablation profile")
    ap.add_argument("--max-iterations", type=int, default=P.MAX_ITERATIONS)
    ap.add_argument("--prompt-cache", choices=K.MODES, default=K.DEFAULT,
                    help="frozen cache-layout ablation; no explicit billed cache resources")
    guard = ap.add_mutually_exclusive_group()
    guard.add_argument("--usd-cap", type=float,
        help="explicit per-run authorization; never increases an existing run's budget")
    guard.add_argument("--iteration-guard", action="store_true",
        help="explicit paid authorization without a USD cap; retain all usage accounting")
    ap.add_argument("--cpus", type=int, default=12, help="CPU budget for this run; concurrent run budgets must total <=12")
    ap.add_argument("--carry-budget-from", type=pathlib.Path,
        help="retain charges from an aborted infrastructure attempt; never imports model feedback")
    args = ap.parse_args()
    if not args.iteration_guard and args.usd_cap is None:
        args.usd_cap = P.CAP_USD
    try:
        limits = validate_limits(args.max_iterations, args.usd_cap, args.cpus, args.iteration_guard)
    except ValueError as exc:
        ap.error(str(exc))
    if not 1 <= args.workers <= args.cpus:
        ap.error("workers must be in [1, cpus]")
    with R.cpu_lease(REPO / "eval_out/chia/.cpu_leases", args.cpus):
        return prepare(args, limits)


def prepare(args, limits):
    root = args.root.resolve()
    arm, model = args.model, P.MODELS[args.model]
    budget_description = "no USD stopping cap (iteration guard)" if limits.get("iteration_guard") else f"USD {args.usd_cap:g}"
    root.mkdir(parents=True, exist_ok=False)
    evaluation = W.install(root, args.evaluation_config)
    loop = L.install(root, getattr(args, "loop_config", None))
    cache_policy = K.install(root, getattr(args, "prompt_cache", K.DEFAULT))
    train = E.workloads(root, "training")
    if args.carry_budget_from:
        origin = args.carry_budget_from.resolve()
        previous = json.loads((origin / "run_manifest.json").read_text())
        previous_model = previous.get("model", previous.get("models", {}).get(arm))
        if previous_model != model or previous.get("no_candidate_evaluated") is not True:
            raise RuntimeError("carryover requires a same-backend infrastructure abort before candidate evaluation")
        path = (origin if previous.get("record_type") == "optimization_run" else origin / "arms" / arm) / "ledger.json"
        totals = P.Ledger(path, arm).totals()
        if totals.get("unknown_usage_calls", 0):
            raise RuntimeError("settle or audit unknown generation usage before restarting")
        carried = {**{k: totals.get(k, 0) for k in ("cap_charge_usd", "estimated_standard_usd", "unknown_usage_calls")},
            "api_attempts": totals["api_attempts"] + totals.get("carryover_api_attempts", 0),
            "prior_ledger": str(path), "prior_ledger_sha256": P.sha(path.read_bytes()),
            "reason": "infrastructure attempt charges retained within the original per-run authorization"}
        atomic_write_json(root / "budget_carryover.json", carried)
    atomic_write_json(root / "window_policy.json", {
        "instructions_per_core": evaluation["simpleo3"]["instructions_per_core"],
        "initialization": "cold caches and DRAM; full prefix with drain; no unscored warmup",
        "roi_selection": "established single-core 20M instruction default; not selected by candidate scores",
        "minimum_oracle_owner_reads": evaluation["simpleo3"]["minimum_oracle_owner_reads"],
        "sustained_traffic": "report time-decile counts; legitimate idle phases are not excluded"})
    seed = (REPO / P.MUTABLE).read_text()
    _, regions = P.regions(seed)
    expected_seed = "voidinit_model(){}Clk_tpredict_departure(constRequest&req){returnm_clk+m_latency;}"
    if re.sub(r"\s+", "", regions["CODE"]) != expected_seed or regions["INCLUDES"].strip():
        raise RuntimeError("fresh campaign requires the unchanged fixed-delay skeleton")
    prep = {"status": "preparing", "started_at": time.time(), "run_id": root.name,
        "backend": arm, "model": model, "limits": limits,
        "loop_configuration": loop, "loop_configuration_sha256": L.identity(loop),
        "prompt_cache": cache_policy,
        "seed_sha256": P.sha(seed), "seed_source": P.MUTABLE,
        "maximum_output_tokens": P.MAX_OUTPUT, "output_limit_source":
            "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/" +
            ("3-1-pro" if arm == "pro" else "3-8-flash"),
        "authorization": f"At most {args.max_iterations} evaluated designs and {budget_description}; paid execution requires user authorization",
        "previous_campaign_inputs_imported": False, "human_modeling_hints": False,
        "workers": args.workers, "optimization": "-O3",
        "changes": [f"{model}, HIGH and 65,536 output tokens; independent single-model run",
            "complete editable-region bodies, with build/compliance/runtime repair feedback",
            "editable model parameters and read-only resolved controller behavior",
            "frozen, operator-configured inspection, feedback and search limits",
            f"token-counted context and model-specific conservative usage accounting; {budget_description}",
            "fresh full 20M instruction evolution; immediate verified trace compression"],
        "live_generation_preflight": "first counted proposal in this run; no separate paid probe",
        "review_mode": "automatic isolated API reviewer, same frozen rubric/model across runs; cost shares this run's cap",
        "recovery": "journaled proposals and API attempts; bounded operational recovery never opens final test",
        "heldout_exposure": "operator has seen these families in earlier evaluation; fresh agents see training only"}
    atomic_write_json(root / "preparation_manifest.json", prep)
    progress("optimized_runtime")
    E.prepare_runtime(root)
    transfer.prepare(root)
    plugin = E.compile_candidate(root, seed, root / "seed")
    W.inventory(root, E.C.trace_path, E.C.file_provenance)
    progress("training_oracle_and_four_comparisons")
    E.evaluate(root, ["oracle", *E.COMPARISONS], split="training", workers=args.workers)
    progress("training_seed")
    E.evaluate(root, ["candidate"], plugin=plugin, label="seed", split="training", workers=min(2, args.workers))
    populations = [traffic_population(root / "training/simpleo3/DDR5" / w / "oracle",
        evaluation["simpleo3"]["minimum_oracle_owner_reads"]) for w in train]
    atomic_write_json(root / "training_traffic_coverage.json", populations)
    if not all(p["owner_count_pass"] for p in populations):
        raise RuntimeError("training lacks sufficient oracle DRAM traffic")
    progress("standard_runner_parity", instruction_window=E.evaluation_insts(root))
    env = os.environ.copy()
    env["EVAL_OUT"] = str(root / "parity")
    env["PYTHONPATH"] = ":".join(str(REPO / p) for p in ("python", "tools", "."))
    E.command([sys.executable, REPO / "tools/eval/run_simpleo3.py", "--std", "DDR5",
        "--workloads", *train, "--models", "oracle,candidate", "--candidate-label", "seed",
        "--insts-per-core", str(E.evaluation_insts(root)), "--workers", str(min(4, args.workers))],
        root / "logs/standard_parity.log", timeout=1800, env=env)
    progress("unit_tests_parity_and_loaded_isolation")
    E.command([sys.executable, REPO / "tools/chia_loop/preflight_real.py", root],
        root / "logs/preflight.log", timeout=1800, env=env)
    E.command([sys.executable, "-m", "pytest", "-q", "tests/unit_tests/test_chia_prompt_cache.py",
               "tests/unit_tests/test_chia_launch_reliability.py"],
        root / "logs/prompt_cache_preflight.log", timeout=240, env=env)
    progress("archive_verification")
    E.verify_run_archives(root)
    import google.auth
    from google.auth.transport.requests import Request
    credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError("application default credentials did not refresh")
    from google import genai
    from google.genai import types
    with genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT", "ramulator-chia"),
            location="global", http_options=types.HttpOptions(timeout=60_000,
                retry_options=types.HttpRetryOptions(attempts=1))) as client:
        counts = client.models.count_tokens(model=model, contents="Configuration preflight.",
            config=types.CountTokensConfig(system_instruction="Return only a proposal JSON object.")).total_tokens
    prep.update({"status": "ready", "finished_at": time.time(), "adc_refreshed": True,
                 "adc_project": project, "api_calls": 0, "nonbillable_count_tokens_preflight": counts})
    atomic_write_json(root / "preparation_manifest.json", prep)
    progress("ready_for_individual_run", root=str(root), model=model, api_calls=0)


if __name__ == "__main__":
    main()
