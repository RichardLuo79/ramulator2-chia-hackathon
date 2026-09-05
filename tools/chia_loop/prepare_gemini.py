#!/usr/bin/env python3
"""Non-billable preparation of a fresh full-window Gemini comparison.

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


def progress(stage, **extra):
    print(json.dumps({"time": time.time(), "stage": stage, **extra}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not 1 <= args.workers <= 12:
        ap.error("workers must be in [1, 12]")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(root / "window_policy.json", {
        "instructions_per_core": E.C.INSTS_SINGLE,
        "initialization": "cold caches and DRAM; full prefix with drain; no unscored warmup",
        "roi_selection": "established single-core 20M instruction default; not selected by candidate scores",
        "minimum_oracle_owner_reads": 10_000,
        "sustained_traffic": "report time-decile counts; legitimate idle phases are not excluded"})
    seed = (REPO / P.MUTABLE).read_text()
    _, regions = P.regions(seed)
    expected_seed = "voidinit_model(){}Clk_tpredict_departure(constRequest&req){returnm_clk+m_latency;}"
    if re.sub(r"\s+", "", regions["CODE"]) != expected_seed or regions["INCLUDES"].strip():
        raise RuntimeError("fresh campaign requires the unchanged fixed-delay skeleton")
    prep = {"status": "preparing", "started_at": time.time(),
        "seed_sha256": P.sha(seed), "seed_source": P.MUTABLE,
        "maximum_output_tokens": P.MAX_OUTPUT, "output_limit_source": [
            "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro",
            "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash"],
        "authorization": "User requested Gemini 3.8 Flash, relaxed limits, and a fresh comparison; five evaluated designs and USD 50 per arm",
        "previous_campaign_inputs_imported": False, "human_modeling_hints": False,
        "workers": args.workers, "optimization": "-O3",
        "changes": ["Gemini 3.8 Flash versus Gemini 3.1 Pro, HIGH and 65,536 output tokens",
            "complete editable-region bodies, with build/compliance/runtime repair feedback",
            "editable model parameters and read-only resolved controller behavior",
            "48 model turns, 192 inspections, and 12 drafts per evaluated iteration",
            "token-counted context and model-specific conservative USD 50 ledger",
            "fresh full 20M instruction evolution; immediate verified trace compression"],
        "live_generation_preflight": "first counted proposal in each arm; no separate paid probe",
        "heldout_exposure": "operator has seen these families in earlier evaluation; fresh agents see training only"}
    atomic_write_json(root / "preparation_manifest.json", prep)
    progress("optimized_runtime")
    E.prepare_runtime(root)
    plugin = E.compile_candidate(root, seed, root / "seed")
    inputs = {}
    for workload in E.TRAIN + E.TEST:
        path = E.C.trace_path(workload)
        count = subprocess.run(["awk", '{n++; inst += $1 + ($2 != -1)} END {printf "%d %.0f\\n", n, inst}', path],
            capture_output=True, text=True, check=True)
        records, instructions = map(int, count.stdout.split())
        if instructions < E.evaluation_insts(root):
            raise RuntimeError("ROI would wrap input trace: " + workload)
        inputs[workload] = {**E.C.file_provenance(path), "records": records,
            "available_instructions": instructions, "wraps": False}
    atomic_write_json(root / "input_inventory.json", inputs)
    progress("training_oracle_and_four_comparisons")
    E.evaluate(root, ["oracle", *E.COMPARISONS], split="training", workers=args.workers)
    progress("training_seed")
    E.evaluate(root, ["candidate"], plugin=plugin, label="seed", split="training", workers=min(2, args.workers))
    populations = [traffic_population(root / "training/simpleo3/DDR5" / w / "oracle", 10_000) for w in E.TRAIN]
    atomic_write_json(root / "training_traffic_coverage.json", populations)
    if not all(p["owner_count_pass"] for p in populations):
        raise RuntimeError("training lacks sufficient oracle DRAM traffic")
    progress("standard_runner_parity", instruction_window=E.evaluation_insts(root))
    env = os.environ.copy()
    env["EVAL_OUT"] = str(root / "parity")
    env["PYTHONPATH"] = ":".join(str(REPO / p) for p in ("python", "tools", "."))
    E.command([sys.executable, REPO / "tools/eval/run_simpleo3.py", "--std", "DDR5",
        "--workloads", *E.TRAIN, "--models", "oracle,candidate", "--candidate-label", "seed",
        "--insts-per-core", str(E.evaluation_insts(root)), "--workers", str(min(4, args.workers))],
        root / "logs/standard_parity.log", timeout=1800, env=env)
    progress("unit_tests_parity_and_loaded_isolation")
    E.command([sys.executable, REPO / "tools/chia_loop/preflight_real.py", root],
        root / "logs/preflight.log", timeout=1800, env=env)
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
        counts = {arm: client.models.count_tokens(model=model, contents="Configuration preflight.",
            config=types.CountTokensConfig(system_instruction="Return only a proposal JSON object.")).total_tokens
            for arm, model in P.MODELS.items()}
    prep.update({"status": "ready", "finished_at": time.time(), "adc_refreshed": True,
                 "adc_project": project, "api_calls": 0, "nonbillable_count_tokens_preflight": counts})
    atomic_write_json(root / "preparation_manifest.json", prep)
    progress("ready_for_paid_campaign", root=str(root), api_calls=0)


if __name__ == "__main__":
    main()
