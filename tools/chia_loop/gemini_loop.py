#!/usr/bin/env python3
"""One budgeted, single-model Gemini source-evolution run as a native CHIA graph.

An operator-side *compliance* review gate is intentional: the orchestrating
coding agent checks boundedness/atomicity without supplying modeling fixes.
No held-out result is evaluated until this run's incumbent is frozen. The model
backend does not inherit CHIA's default tools/retries: every paid request has
an explicit HIGH setting, persisted reservation, and raw response record.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import pathlib
import shutil
import sys
import time
import traceback

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ray
from chia.base.ChiaFunction import ChiaFunction, get
from chia.trace.profiler import start_collector, stop_collector, get_collector, get_profiler
from tools.chia_loop import artifacts, real_core as P, real_eval as E
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.run_records import model_pricing, execution_limits

POLICY = {
    "version": "gemini_individual_run_v6", "maximum_iterations": 5, "usd_cap": 50,
    "iteration_unit": "successfully evaluated design; draft repairs stay inside the iteration",
    "thinking_level": "HIGH", "temperature": 1.0,
    "maximum_output_tokens": P.MAX_OUTPUT, "maximum_context_utf8_bytes": P.MAX_CONTEXT_BYTES,
    "maximum_input_tokens": P.MAX_INPUT_TOKENS,
    "model_turns_per_proposal": 48, "diagnostic_calls_per_proposal": 192,
    "drafts_per_iteration": 12, "finalization_turn_reserve": 2,
    "api_attempts_per_proposal": 60, "transient_retries_per_turn": 1,
    "provider_timeout_seconds": 1800, "simulator_cpu_seconds": E.SIM_CPU_SECONDS,
    "consecutive_truncation_stop": 2,
    "live_completion_preflight": "first counted proposal in this run; no extra paid preflight attempts",
    "simulator_file_bytes": E.SIM_FILE_BYTES,
    "compiler_cpu_seconds": 180, "process_memory_bytes": 4 * 1024**3,
    "promotion": "strict two-objective Pareto dominance of incumbent",
    "parent_selection": "round_robin_sorted_nondominated_archive",
    "accuracy_tail_or_speed_rejection_threshold": None,
    "validity": ["protected lifecycle unchanged", "static source boundary", "semantic compliance review",
                 "bounded causal state", "successful O3 build and drained simulation",
                 "finite errors", "exact bidirectional stable ID coverage", "committed callback latency audit"],
    "reviewer": "orchestrating coding agent; compliance only; no human modeling hints",
    "tool_protocol": "JSON inspections and region-body submissions with build/review/runtime repair feedback",
    "context_policy": "count input tokens before dispatch; retain conversation/signatures; no silent truncation",
    "operator_stop_file": "STOP: settle an in-flight call, then stop before another generation/evaluation",
}

VISIBLE = [
    "src/ramulator/controller/i_controller.h", "src/ramulator/base/request.h",
    "src/ramulator/controller/controller_base.h", "src/ramulator/controller/controller_base.cpp",
    "src/ramulator/base/config_node.h",
    "src/ramulator/base/type.h", "src/ramulator/dram/dram_spec.h", "src/ramulator/dram/device.h",
    "src/ramulator/controller/impl/generic_ddr_controller.cpp",
    "src/ramulator/controller/impl/fixed_lat_controller.cpp",
    "src/ramulator/controller/impl/md1_controller.cpp",
    "src/ramulator/controller/impl/wmg1_controller.cpp",
    "src/ramulator/controller/impl/mess_controller.cpp",
    "src/ramulator/controller/impl/zoo_probe.h", "src/ramulator/base/param.h",
    "src/ramulator/controller/addr_mapper/impl/ro_ba_ra_co_ch.cpp",
    "src/ramulator/controller/scheduler/impl/frfcfs_rowhit.cpp",
    "src/ramulator/dram/impl/DDR5.cpp", "python/ramulator/dram/ddr5.py",
]
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "ramulator-chia")


def configured_policy(root):
    return {**POLICY, **execution_limits(root)}


def run_ledger(root, backend):
    root = pathlib.Path(root)
    return P.Ledger(root / "ledger.json", backend, run_id=root.name,
                    cap_usd=configured_policy(root)["usd_cap"])


def event(root, kind, **data):
    record = {"time": time.time(), "event": kind, "run_id": pathlib.Path(root).name, **data}
    with (pathlib.Path(root) / "events.jsonl").open("a") as f:
        # Workers may emit events too. A file lock is process-safe and does not
        # capture an unpickleable thread lock in CHIA's dispatched function.
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())
    print(json.dumps(record, sort_keys=True), flush=True)


def inspect_tool(root, parent, request):
    tool = request.get("tool")
    if tool in {"read_file", "search_file"}:
        path = request.get("path")
        if path == P.MUTABLE:
            text = pathlib.Path(parent["source_path"]).read_text()
        elif path in VISIBLE:
            text = (pathlib.Path(root) / "visible" / path).read_text()
        else:
            raise ValueError("path is not in the frozen readable manifest")
        lines = text.splitlines()
        if tool == "search_file":
            query = request.get("query")
            if not isinstance(query, str) or not query or len(query) > 256:
                raise ValueError("search_file requires a nonempty literal query of at most 256 characters")
            matches = [i for i, line in enumerate(lines) if query.casefold() in line.casefold()]
            selected = sorted({i for match in matches[:40]
                for i in range(max(0, match - 3), min(len(lines), match + 4))})
            return {"path": path, "sha256": P.sha(text), "total_matches": len(matches),
                "content": "\n".join(f"{i+1}: {lines[i]}" for i in selected)}
        start = max(1, int(request.get("start_line", 1)))
        limit = min(1200, max(1, int(request.get("max_lines", 400))))
        return {"path": path, "sha256": P.sha(text), "total_lines": len(lines),
                "content": "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines) if start-1 <= i < start-1+limit)}
    if tool == "training_diagnostics":
        return E.training_diagnostics(root, parent["label"], request.get("workload"),
            request.get("kind", "extremes"), request.get("limit", 40),
            start_row=request.get("start_row", 0), arrival_min=request.get("arrival_min"),
            arrival_max=request.get("arrival_max"), request_type=request.get("request_type"))
    raise ValueError("unknown inspection tool")


@ChiaFunction(num_cpus=1, max_retries=0)
def propose(root_text, arm, iteration, system, user_prompt, parent):
    from google import genai
    from google.genai import types
    root = pathlib.Path(root_text)
    interactions = root / "interactions" / f"iter_{iteration:03d}"
    interactions.mkdir(parents=True, exist_ok=True)
    ledger = run_ledger(root, arm)
    client = genai.Client(vertexai=True, project=PROJECT, location="global",
        http_options=types.HttpOptions(timeout=POLICY["provider_timeout_seconds"] * 1000,
            retry_options=types.HttpRetryOptions(attempts=1)))
    config = types.GenerateContentConfig(system_instruction=system, temperature=1.0,
        max_output_tokens=P.MAX_OUTPUT, thinking_config=types.ThinkingConfig(thinking_level="HIGH", include_thoughts=True),
        response_mime_type="application/json")
    contents = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]
    directory = root / "candidates" / f"{arm}_{iteration:03d}"
    seed_source = (root / "seed/atomic_controller.cpp").read_text()
    diagnostics, attempts, drafts, consecutive_truncations = 0, 0, [], 0
    maximum_turns = POLICY["model_turns_per_proposal"]
    try:
        for turn in range(1, maximum_turns + 1):
            if (root / "STOP").exists():
                return {"status": "stopped", "reason": "operator stop requested", "drafts": drafts}
            response = None
            for retry in range(2):
                if attempts >= POLICY["api_attempts_per_proposal"]:
                    return {"status": "failed", "reason": "iteration API-attempt safety limit", "drafts": drafts}
                payload = {"model": P.MODELS[arm], "config": config.model_dump(mode="json", exclude_none=True),
                           "contents": [c.model_dump(mode="json", exclude_none=True) for c in contents]}
                encoded = json.dumps(payload, ensure_ascii=False)
                # countTokens is not a generation and incurs no model-token charge.
                # Failure is fail-closed; never silently guess away the context cap.
                counted = client.models.count_tokens(model=P.MODELS[arm], contents=contents,
                    config=types.CountTokensConfig(system_instruction=system))
                call_id = ledger.reserve(encoded, iteration, turn, input_tokens=counted.total_tokens)
                attempts += 1
                prefix = interactions / f"call_{call_id:03d}"
                atomic_write_json(prefix.with_suffix(".request.json"), payload)
                started = time.time()
                try:
                    response = client.models.generate_content(model=P.MODELS[arm], contents=contents, config=config)
                    raw = response.model_dump(mode="json", exclude_none=True)
                    atomic_write_json(prefix.with_suffix(".response.json"), raw)
                    usage = raw.get("usage_metadata")
                    ledger.settle(call_id, usage)
                    atomic_write_json(prefix.with_suffix(".call.json"), {"wall_s": time.time()-started,
                        "iteration": iteration, "turn": turn, "retry": retry,
                        "response_sha256": P.sha(json.dumps(raw, sort_keys=True)), "usage": usage})
                    break
                except Exception as exc:
                    # No content/credential-bearing request headers are logged.
                    detail = f"{type(exc).__name__}: {str(exc)[:2000]}"
                    status = getattr(exc, "code", None)
                    ledger.settle(call_id, None, detail, http_status=status)
                    atomic_write_json(prefix.with_suffix(".error.json"), {"error": detail, "retry": retry})
                    if status not in (408, 429, 500, 502, 503, 504) or retry == 1:
                        return {"status": "failed", "reason": detail, "drafts": drafts}
                    time.sleep(2)
            proposal = P.parse_provider_response(raw)
            if (root / "STOP").exists():
                return {"status": "stopped", "reason": "operator stop requested after settling in-flight call", "drafts": drafts}
            consecutive_truncations = (consecutive_truncations + 1
                if proposal.get("failure_kind") == "generation_truncated" else 0)
            if consecutive_truncations >= POLICY["consecutive_truncation_stop"]:
                return {"status": "failed", "reason": "repeated generation truncation at provider maximum", "drafts": drafts}
            if proposal.get("status") == "no_change":
                return {**proposal, "drafts": drafts}
            feedback = {}
            if proposal.get("status") == "inspect":
                requests = proposal.get("requests", [])
                if turn > maximum_turns - POLICY["finalization_turn_reserve"]:
                    feedback = {"error": "Finalization reserve: submit region bodies or no_change now."}
                elif not isinstance(requests, list) or not requests or len(requests) > 16:
                    feedback = {"error": "Return 1 to 16 inspection requests per turn."}
                elif len(requests) + diagnostics > POLICY["diagnostic_calls_per_proposal"]:
                    feedback = {"error": "Inspection safety limit reached. Submit a model or no_change; repair feedback remains available."}
                else:
                    results = []
                    for request in requests:
                        diagnostics += 1
                        try:
                            result = inspect_tool(root, parent, request)
                        except Exception as exc:
                            result = {"error": str(exc)}
                        results.append({"request": request, "result": result})
                    atomic_write_json(interactions / f"tools_turn_{turn}.json", results)
                    feedback = {"tool_results": results}
            elif proposal.get("status") == "proposal":
                if len(drafts) >= POLICY["drafts_per_iteration"]:
                    return {"status": "failed", "reason": "draft safety limit reached", "drafts": drafts}
                draft_number = len(drafts) + 1
                draft_dir = directory / f"draft_{draft_number:03d}"
                draft_dir.mkdir(parents=True)
                atomic_write_json(draft_dir / "proposal.json", proposal)
                draft = {"number": draft_number, "turn": turn, "directory": str(draft_dir)}
                try:
                    result = evaluate_draft(root, parent, seed_source, proposal, draft_dir,
                        f"{arm}_{iteration:03d}_d{draft_number:02d}", arm=arm, iteration=iteration)
                    draft.update({"status": "valid", "source_sha256": result["sha256"]})
                    drafts.append(draft)
                    atomic_write_json(draft_dir / "outcome.json", draft)
                    return {"status": "evaluated", "proposal": proposal, "candidate": result, "drafts": drafts}
                except Exception as exc:
                    draft.update({"status": "rejected", "reason": str(exc)[-8000:]})
                    drafts.append(draft)
                    atomic_write_json(draft_dir / "outcome.json", draft)
                    event(root, "draft_rejected", arm=arm, iteration=iteration, draft=draft_number, reason=draft["reason"])
                    feedback = {"draft_feedback": draft, "instruction": "Repair your own draft and resubmit complete region bodies. No evaluated iteration has been consumed."}
            else:
                feedback = {"error": proposal, "instruction": "Return valid proposal/inspect/no_change JSON. No source from an incomplete response was applied."}
            atomic_write_json(interactions / f"feedback_turn_{turn}.json", feedback)
            # Preserve the actual returned content, including thought signatures,
            # as recommended for multi-turn reasoning. Never synthesize them.
            if (response.candidates and response.candidates[0].content
                    and response.candidates[0].content.parts):
                contents.append(response.candidates[0].content)
            contents.append(types.Content(role="user", parts=[types.Part(text=json.dumps({
                **feedback, "remaining_diagnostics": POLICY["diagnostic_calls_per_proposal"]-diagnostics,
                "remaining_model_turns": maximum_turns-turn,
                "remaining_drafts": POLICY["drafts_per_iteration"]-len(drafts),
                "budget": ledger.totals()}))]))
        return {"status": "failed", "reason": "iteration model-turn safety limit", "drafts": drafts}
    except P.BudgetExhausted as exc:
        return {"status": "budget_stop", "reason": str(exc), "drafts": drafts}
    finally:
        client.close()


def evaluate_draft(root, parent, seed_source, proposal, directory, label, *, arm, iteration):
    """Trust boundary for a submission. Rejection is feedback, never a silent fix."""
    if proposal.get("parent_source_sha256") != parent["sha256"] or proposal.get("parent_id") != parent["id"]:
        raise ValueError("proposal parent ID/hash mismatch")
    for field in ("evidence", "hypothesis", "mechanism", "genericity_and_atomicity", "complexity", "expected_effects", "limitations"):
        if not proposal.get(field):
            raise ValueError(f"proposal omits {field}")
    source = P.assemble_regions(pathlib.Path(parent["source_path"]).read_text(), proposal.get("regions"))
    checks = P.validate_source(seed_source, source)
    (directory / "atomic_controller.cpp").write_text(source)
    atomic_write_json(directory / "static_checks.json", checks)
    plugin = get(build.chia_remote(str(root), source, str(directory / "build")))
    atomic_write_json(directory / "review_needed.json", {"source_sha256": P.sha(source), "policy": configured_policy(root)})
    event(root, "compliance_review_needed", arm=arm, iteration=iteration, directory=str(directory))
    review_deadline = time.monotonic() + 7200
    while not (directory / "review.json").exists():
        if (root / "STOP").exists():
            raise InterruptedError("operator stop requested before evaluation")
        if time.monotonic() >= review_deadline:
            raise TimeoutError("compliance review unavailable for two hours")
        time.sleep(1)
    review = json.loads((directory / "review.json").read_text())
    if review.get("source_sha256") != P.sha(source) or review.get("approved") is not True:
        raise ValueError("semantic compliance review: " + review.get("reason", "not approved"))
    if (root / "STOP").exists():
        raise InterruptedError("operator stop requested before evaluation")
    metrics = get(score.chia_remote(str(root), plugin, label, "training"))
    return {"source_path": str(directory / "atomic_controller.cpp"), "sha256": P.sha(source),
            "plugin": plugin, "metrics": metrics, "label": label}


@ChiaFunction(num_cpus=1, max_retries=0)
def build(root, source, directory):
    return E.compile_candidate(root, source, directory)


@ChiaFunction(num_cpus=2, max_retries=0)
def score(root, plugin, label, split):
    result = E.evaluate(root, ["candidate"], plugin=plugin, label=label, split=split)[label]
    get_profiler().add_info({"label": label, "split": split, "aggregate": result["aggregate"]})
    return result


def make_prompt(root, arm, iteration, candidates, incumbent, parent_id, history):
    parent = candidates[parent_id]
    comparisons = json.loads((root / "training/reports/comparisons.json").read_text())["models"]
    template = (root / "prompts/iteration_v1.md").read_text()
    fields = {
        "iteration_number": iteration, "max_proposal_iterations": configured_policy(root)["maximum_iterations"],
        "parent_id": parent_id, "parent_source_sha256": parent["sha256"], "incumbent_id": incumbent,
        "parent_source": pathlib.Path(parent["source_path"]).read_text(),
        "parent_and_incumbent_training_scores": json.dumps({"parent": parent["metrics"]["aggregate"], "incumbent": candidates[incumbent]["metrics"]["aggregate"]}),
        "training_configuration_and_trace_clock_contract": json.dumps({
            "training": E.TRAIN, "instructions_per_core": E.evaluation_insts(root), "frontend": "SimpleO3",
            "std": "DDR5", "org": "DDR5_16Gb_x8", "timing": "DDR5_4800AN", "channels": 1,
            "clock_ratio_frontend": 8, "clock_ratio_memory": 3,
            "logical_trace_clock": "frontend cycles", "controller_trace_clock": "DRAM cycles",
            "logical_boundary": "LLC access including hit/merge/owner; all logical reads scored",
            "reference": E.C.REFERENCE, "refresh": "none", "llc_mshr_per_core": 16,
            "llc_latency_frontend_cycles": 47, "fixed_issued_ROI_fully_drained": True}),
        "per_workload_training_metrics_and_coverage": json.dumps(parent["metrics"]["per_workload"]),
        "training_diagnostic_slices": json.dumps({w: E.training_diagnostics(root, parent["label"], w, limit=3) for w in E.TRAIN}),
        "published_comparison_training_metrics": json.dumps(comparisons),
        "accepted_and_rejected_proposals_with_reasons": json.dumps(history),
        "last_build_or_evaluation_diagnostics": json.dumps(history[-1:] if history else []),
        "this_run_pareto_archive": json.dumps({k: candidates[k]["metrics"]["aggregate"] for k in P.pareto(candidates)}),
        "promotion_and_guardrail_configuration": json.dumps(configured_policy(root)),
        "allowed_tools_and_readable_file_manifest": json.dumps({"files": [P.MUTABLE, *VISIBLE],
            "tools": ["read_file", "search_file", "training_diagnostics"], "submission": "complete includes/code region bodies, no diff or boundary markers"}),
        "remaining_model_call_diagnostic_and_token_limits": json.dumps({"model_turns": POLICY["model_turns_per_proposal"],
            "diagnostics": POLICY["diagnostic_calls_per_proposal"], "drafts": POLICY["drafts_per_iteration"],
            "output_tokens_per_turn": P.MAX_OUTPUT, "budget": run_ledger(root, arm).totals(), "cap_usd": configured_policy(root)["usd_cap"]}),
    }
    return P.render(template, fields)


def run_model(root, arm):
    root = pathlib.Path(root)
    arm_dir = root
    arm_dir.mkdir(parents=True, exist_ok=True)
    if (arm_dir / "state.json").exists():
        raise RuntimeError("refusing implicit paid campaign resume; audit existing state first")
    carryover = root / "budget_carryover.json"
    if carryover.exists():
        run_ledger(root, arm).initialize_carryover(json.loads(carryover.read_text()))
    seed_source = (root / "seed/atomic_controller.cpp").read_text()
    seed_metrics = json.loads((root / "training/reports/seed.json").read_text())["models"]["seed"]
    candidates = {"seed": {"source_path": str(root / "seed/atomic_controller.cpp"), "sha256": P.sha(seed_source),
        "plugin": str(root / "seed/candidate.so"), "metrics": seed_metrics, "label": "seed"}}
    incumbent, history = "seed", []
    system = (root / "prompts/system_v1.md").read_text()
    event(root, "run_started", backend=arm, model=P.MODELS[arm])
    state = {}
    for iteration in range(1, configured_policy(root)["maximum_iterations"] + 1):
        archive = P.pareto(candidates)
        parent_id = archive[(iteration - 1) % len(archive)]
        parent = {**candidates[parent_id], "id": parent_id}
        label = f"{arm}_{iteration:03d}"
        directory = arm_dir / "candidates" / label
        directory.mkdir(parents=True, exist_ok=True)
        prompt = make_prompt(root, arm, iteration, candidates, incumbent, parent_id, history)
        (directory / "prompt.md").write_text(prompt)
        event(root, "proposal_started", arm=arm, iteration=iteration, parent=parent_id)
        try:
            proposal = get(propose.chia_remote(str(root), arm, iteration, system, prompt, parent))
        except Exception as exc:
            # A Ray dispatch/infrastructure exception is not an agent result.
            # Fail the campaign before test evaluation; keep all local evidence.
            event(root, "infrastructure_failure", arm=arm, iteration=iteration, reason=str(exc)[-8000:])
            raise
        atomic_write_json(directory / "proposal.json", proposal)
        event(root, "proposal_response_received", arm=arm, iteration=iteration,
            proposal_status=proposal.get("status"), failure_kind=proposal.get("failure_kind"))
        record = {"id": label, "parent": parent_id, "iteration": iteration,
                  "drafts": proposal.get("drafts", []),
                  "explanation": {k: v for k, v in proposal.get("proposal", {}).items() if k != "regions"}}
        if proposal.get("status") == "evaluated":
            candidates[label] = proposal["candidate"]
            metrics = candidates[label]["metrics"]
            promoted = P.dominates(P.objectives(metrics), P.objectives(candidates[incumbent]["metrics"]))
            if promoted:
                incumbent = label
            record.update({"status": "valid", "promoted": promoted, "metrics": metrics["aggregate"]})
            event(root, "candidate_evaluated", arm=arm, iteration=iteration, promoted=promoted, metrics=metrics["aggregate"])
        else:
            record.update({"status": "stopped", "reason": proposal.get("reason", "no_change")})
            event(root, "proposal_rejected", arm=arm, iteration=iteration, reason=record["reason"])
        history.append(record)
        state = {"run_id": root.name, "backend": arm, "model": P.MODELS[arm], "status": "running", "history": history,
                 "candidates": candidates, "incumbent": incumbent, "pareto_archive": P.pareto(candidates),
                 "budget": run_ledger(root, arm).totals()}
        atomic_write_json(arm_dir / "state.json", state)
        if proposal.get("status") != "evaluated":
            state["stop_reason"] = record["reason"]
            break
    state.update({"status": "frozen", "frozen_at": time.time(), "selected": candidates[incumbent]})
    atomic_write_json(arm_dir / "state.json", state)
    event(root, "selection_frozen", backend=arm, incumbent=incumbent, budget=state["budget"])
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--model", choices=P.MODELS, required=True, help="exactly one backend per run")
    args = ap.parse_args()
    root = args.root.resolve()
    arm = args.model
    if not (root / "preflight_pass.json").exists():
        raise RuntimeError("paid calls require a completed isolation/evaluation/budget preflight")
    # A leftover infrastructure-smoke result must never become full-ROI
    # training feedback merely because the evaluator's default changed.
    insts = E.evaluation_insts(root)
    for workload in E.TRAIN:
        for label in ("oracle", "seed", *E.COMPARISONS):
            prepared = json.loads((root / "training/simpleo3/DDR5" / workload / label / "manifest.json").read_text())
            if prepared["insts_per_core"] != insts:
                raise RuntimeError("prepared training ROI differs from campaign window policy")
    if (root / "run_manifest.json").exists():
        raise RuntimeError("run already has a manifest; no automatic paid resume")
    preparation = json.loads((root / "preparation_manifest.json").read_text())
    if preparation.get("model") != P.MODELS[arm] or preparation.get("run_id") != root.name:
        raise RuntimeError("preparation belongs to a different model or run; prepare an individual run first")
    (root / "visible").mkdir()
    for relative in VISIBLE:
        source = REPO / relative
        if not source.is_file():
            raise RuntimeError(f"missing public source allowlist path: {relative}")
        dest = root / "visible" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    shutil.copytree(REPO / "tools/chia_loop/prompts", root / "prompts")
    protocol_paths = [REPO / "tools/chia_loop" / name for name in (
        "artifacts.py", "core.py", "gemini_loop.py", "real_core.py", "real_eval.py",
        "preflight_real.py", "prepare_gemini.py", "sandbox.py", "traffic.py",
        "review_candidate.py", "run_records.py", "analyze_gemini.py", "audit_gemini_campaign.py")]
    protocol_paths += list((REPO / "tools/chia_loop/prompts").glob("*.md"))
    protocol_paths += list((REPO / "tools/eval").glob("*.py"))
    for path in protocol_paths:
        destination = root / "protocol" / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    manifest = {"schema_version": 2, "record_type": "optimization_run", "run_id": root.name,
        "backend": arm, "model": P.MODELS[arm], "status": "running", "started_at": time.time(), "policy": configured_policy(root),
        "project": PROJECT, "location": "global", "pricing": model_pricing(P.PRICING, arm),
        "budget_carryover": json.loads((root / "budget_carryover.json").read_text()) if (root / "budget_carryover.json").exists() else {},
        "training": E.TRAIN, "final_test": E.TEST, "instructions_per_core": E.evaluation_insts(root),
        "comparisons": E.COMPARISONS, "prior_design_exposed": False,
        "human_modeling_hints": False, "test_access": "after_this_run_selection_frozen",
        "preparation": preparation,
        "meta_reviewer_test_exposure": "these workload families were evaluated before this fresh campaign; agents see training only",
        "protocol_hashes": {str(p.relative_to(REPO)): P.sha(p.read_bytes()) for p in protocol_paths},
        "visible_hashes": {p: P.sha((root / "visible" / p).read_bytes()) for p in VISIBLE},
        "versions": {p: importlib.metadata.version(p) for p in ("chialoops", "ray", "google-genai", "numpy", "pandas")}}
    atomic_write_json(root / "run_manifest.json", manifest)
    ray.init(address="local", num_cpus=manifest["policy"]["cpu_budget"], include_dashboard=False, log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(REPO) + ":" + str(REPO / "tools"),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}})
    start_collector(log_dir=str(root / "profiles"))
    try:
        state = run_model(root, arm)
        manifest["state"] = state
        if (root / "STOP").exists():
            raise InterruptedError("operator stop requested; held-out evaluation not started")
        atomic_write_json(root / "selection_frozen.json", {"run_id": root.name,
            "source_sha256": state["selected"]["sha256"], "incumbent": state["incumbent"], "frozen_at": state["frozen_at"]})
        event(root, "frozen_test_started")
        E.evaluate(root, ["oracle", *E.COMPARISONS], split="test", workers=min(6, manifest["policy"]["cpu_budget"]))
        E.evaluate(root, ["candidate"], split="test", plugin=str(root / "seed/candidate.so"), label="seed")
        tests = get(score.chia_remote(str(root), state["selected"]["plugin"], arm + "_final", "test"))
        for relative, expected in manifest["protocol_hashes"].items():
            if P.sha((REPO / relative).read_bytes()) != expected:
                raise RuntimeError("protocol mutated during paid campaign: " + relative)
        manifest.update({"status": "completed", "finished_at": time.time(),
                         "state": state, "final_test_metrics": tests})
        event(root, "run_completed", test_metrics=tests["aggregate"])
    except BaseException:
        manifest.update({"status": "failed", "failure": traceback.format_exc(), "finished_at": time.time()})
        raise
    finally:
        atomic_write_json(root / "run_manifest.json", manifest)
        collector = get_collector()
        if collector is not None:
            ray.get(collector.get_events.remote())
        stop_collector(); ray.shutdown()
    E.verify_run_archives(root)
    artifacts.compress(root, root / "aux_archive_manifest.json", min_bytes=16_384)
    artifacts.verify(root / "aux_archive_manifest.json")
    manifest["archives_verified"] = True
    atomic_write_json(root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
