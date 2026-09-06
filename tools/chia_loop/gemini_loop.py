#!/usr/bin/env python3
"""One budgeted, single-model Gemini source-evolution run as a native CHIA graph.

An isolated API reviewer checks boundedness/atomicity without supplying fixes.
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
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import ray
from chia.base.ChiaFunction import ChiaFunction, get
from chia.trace.profiler import start_collector, stop_collector, get_collector, get_profiler
from tools.chia_loop import artifacts, real_core as P, real_eval as E
from tools.chia_loop import compliance, generation, recovery as R
from tools.chia_loop import loop_config as L
from tools.chia_loop.generation import transient_generation_error
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.run_records import model_pricing, execution_limits

POLICY = {
    "version": "gemini_individual_run_v11", "maximum_iterations": 5, "usd_cap": 50,
    "evaluation": "run-local DDR5 profile; frozen-source ChampSim/gem5 transfer excluded from feedback",
    "iteration_unit": "successfully evaluated design; draft repairs stay inside the iteration",
    "thinking_level": "HIGH", "temperature": 1.0,
    "maximum_output_tokens": P.MAX_OUTPUT, "maximum_context_utf8_bytes": P.MAX_CONTEXT_BYTES,
    "maximum_input_tokens": P.MAX_INPUT_TOKENS,
    "model_turns_per_proposal": 48, "diagnostic_calls_per_proposal": 192,
    "drafts_per_iteration": 12, "finalization_turn_reserve": 2,
    "api_attempts_per_proposal": 60, "transient_retries_per_turn": 2,
    "transport_retry_policy": "network, timeout, remote-protocol and unexpected HTTP 499 errors; retain unknown-usage reservation and reserve every retry",
    "provider_timeout_seconds": 1800, "simulator_cpu_seconds": E.SIM_CPU_SECONDS,
    "maximum_transport_failures_per_operation": 8, "maximum_outage_seconds": 3600,
    "recovery_cooldown_seconds": 60, "recovery_cooldown_max_seconds": 900,
    "maximum_run_wall_seconds": 86400, "maximum_process_restarts": 3,
    "minimum_free_disk_bytes": 8 * 1024**3,
    "automatic_operational_recovery": True,
    "terminal_search_statuses": ["no_change", "budget_stop", "limit_stop", "iteration_limit"],
    "infrastructure_failure_action": "checkpoint; recover boundedly or stop without held-out testing",
    "reviewer_backend": "pro", "review_response_attempts": 3,
    "reviewer_information": "isolated source/explanation only; no tools, history, or scores",
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
    "reviewer": "automatic isolated Gemini API reviewer; same rubric/model for all runs; compliance only",
    "review_cost": "included in the proposing run's unchanged USD cap and per-iteration API attempt limit",
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
    return {**L.policy(root, POLICY), **execution_limits(root)}


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
    R.assert_training_open(root)
    tool = request.get("tool")
    if tool not in L.tools(root):
        raise ValueError("inspection tool is disabled or unknown in this run")
    if tool in {"read_file", "search_file"}:
        path = request.get("path")
        if path == P.MUTABLE:
            text = pathlib.Path(parent["source_path"]).read_text()
        elif path in L.visible_files(root, VISIBLE):
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
        if request.get("kind", "extremes") not in L.diagnostic_kinds(root):
            raise ValueError("training diagnostic kind is disabled in this run")
        return E.training_diagnostics(root, parent["label"], request.get("workload"),
            request.get("kind", "extremes"), request.get("limit", 40),
            start_row=request.get("start_row", 0), arrival_min=request.get("arrival_min"),
            arrival_max=request.get("arrival_max"), request_type=request.get("request_type"))
    if tool == "synthetic_diagnostics":
        from tools.chia_loop import synthetic_diagnostics as D
        return D.inspect(root, parent, request)
    raise ValueError("unknown inspection tool")


@ChiaFunction(num_cpus=1, max_retries=0)
def propose(root_text, arm, iteration, system, user_prompt, parent):
    root = pathlib.Path(root_text)
    try:
        R.assert_training_open(root)
        with R.exclusive_lock(root / "candidates" / f"{arm}_{iteration:03d}" / ".proposal.lock"):
            return propose_checkpointed(root, arm, iteration, system, user_prompt, parent)
    except R.OperatorStop as exc:
        return {"status": "stopped", "reason": str(exc), "retryable": False}
    except R.OperationalPause as exc:
        return {"status": "operational_pause", "reason": str(exc),
                "retryable": exc.retryable, "retry_at": exc.retry_at}
    except Exception as exc:
        # A broken journal, disk, SDK or local program is not a modeling result.
        return {"status": "infrastructure_error", "reason": f"{type(exc).__name__}: {str(exc)[-8000:]}",
                "retryable": False}


def propose_checkpointed(root, arm, iteration, system, user_prompt, parent):
    from google.genai import types
    policy = configured_policy(root)
    interactions = root / "interactions" / f"iter_{iteration:03d}"
    interactions.mkdir(parents=True, exist_ok=True)
    ledger = run_ledger(root, arm)
    directory = root / "candidates" / f"{arm}_{iteration:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "proposal_state.json"
    identity = {"run_id": root.name, "backend": arm, "iteration": iteration,
        "system_sha256": P.sha(system), "prompt_sha256": P.sha(user_prompt),
        "parent_id": parent.get("id"), "parent_sha256": parent.get("sha256")}
    saved = R.read_json(checkpoint) if checkpoint.exists() else {
        "identity": identity, "turn": 1, "diagnostics": 0, "drafts": [], "consecutive_truncations": 0,
        "contents": [types.Content(role="user", parts=[types.Part(text=user_prompt)]).model_dump(mode="json", exclude_none=True)]}
    if saved["identity"] != identity:
        raise RuntimeError("proposal checkpoint parent/prompt identity changed")
    if "result" in saved:
        return saved["result"]
    atomic_write_json(checkpoint, saved)
    contents = [types.Content.model_validate(content) for content in saved["contents"]]
    seed_source = (root / "seed/atomic_controller.cpp").read_text()
    diagnostics, drafts = saved["diagnostics"], saved["drafts"]
    consecutive_truncations = saved["consecutive_truncations"]
    maximum_turns = policy["model_turns_per_proposal"]

    def finish(result):
        pending = saved.pop("pending_draft", None)
        if pending and result["status"] != "evaluated":
            pending.update(status="unevaluated", reason=result.get("reason", "search stopped"))
            drafts.append(pending)
            atomic_write_json(pathlib.Path(pending["directory"]) / "outcome.json", pending)
        saved["result"] = {**result, "drafts": drafts}
        atomic_write_json(checkpoint, saved)
        return saved["result"]

    try:
        for turn in range(saved["turn"], maximum_turns + 1):
            R.check_stop(root)
            raw = generation.generate(root, ledger, policy, project=PROJECT, backend=arm,
                purpose="proposal", iteration=iteration, turn=turn,
                operation_key=f"proposal_{iteration:03d}_{turn:03d}", system=system, contents=contents)
            proposal = P.parse_provider_response(raw)
            R.check_stop(root)
            consecutive_truncations = (consecutive_truncations + 1
                if proposal.get("failure_kind") == "generation_truncated" else 0)
            if consecutive_truncations >= policy["consecutive_truncation_stop"]:
                return finish({"status": "limit_stop", "reason": "repeated generation truncation at provider maximum"})
            if proposal.get("status") == "no_change":
                return finish(proposal)
            feedback = {}
            if proposal.get("status") == "inspect":
                requests = proposal.get("requests", [])
                if turn > maximum_turns - policy["finalization_turn_reserve"]:
                    feedback = {"error": "Finalization reserve: submit region bodies or no_change now."}
                elif not isinstance(requests, list) or not requests or len(requests) > 16:
                    feedback = {"error": "Return 1 to 16 inspection requests per turn."}
                elif len(requests) + diagnostics > policy["diagnostic_calls_per_proposal"]:
                    feedback = {"error": "Inspection safety limit reached. Submit a model or no_change; repair feedback remains available."}
                else:
                    results = []
                    for request in requests:
                        diagnostics += 1
                        try:
                            result = inspect_tool(root, parent, request)
                        except (R.OperationalPause, OSError):
                            raise
                        except Exception as exc:
                            result = {"error": str(exc)}
                        results.append({"request": request, "result": result})
                    atomic_write_json(interactions / f"tools_turn_{turn}.json", results)
                    feedback = {"tool_results": results}
            elif proposal.get("status") == "proposal":
                if len(drafts) >= policy["drafts_per_iteration"]:
                    return finish({"status": "limit_stop", "reason": "draft safety limit reached"})
                draft_number = len(drafts) + 1
                draft_dir = directory / f"draft_{draft_number:03d}"
                draft_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(draft_dir / "proposal.json", proposal)
                draft = {"number": draft_number, "turn": turn, "directory": str(draft_dir)}
                saved["pending_draft"] = dict(draft)
                atomic_write_json(checkpoint, saved)
                try:
                    outcome = draft_dir / "outcome.json"
                    if outcome.exists():
                        previous = R.read_json(outcome)
                        if previous["status"] == "rejected":
                            raise ValueError(previous["reason"])
                        result = R.read_json(draft_dir / "result.json")
                    else:
                        result = evaluate_draft(root, parent, seed_source, proposal, draft_dir,
                            f"{arm}_{iteration:03d}_d{draft_number:02d}", arm=arm, iteration=iteration, turn=turn)
                        atomic_write_json(draft_dir / "result.json", result)
                    draft.update({"status": "valid", "source_sha256": result["sha256"]})
                    drafts.append(draft)
                    atomic_write_json(draft_dir / "outcome.json", draft)
                    return finish({"status": "evaluated", "proposal": proposal, "candidate": result})
                except (R.OperationalPause, R.SearchLimit, P.BudgetExhausted, OSError):
                    raise
                except Exception as exc:
                    draft.update({"status": "rejected", "reason": str(exc)[-8000:]})
                    drafts.append(draft)
                    atomic_write_json(draft_dir / "outcome.json", draft)
                    event(root, "draft_rejected", arm=arm, iteration=iteration, draft=draft_number, reason=draft["reason"])
                    feedback = {"draft_feedback": draft, "instruction": "Repair your own draft and resubmit complete region bodies. No evaluated iteration has been consumed."}
                saved.pop("pending_draft", None)
            else:
                feedback = {"error": proposal, "instruction": "Return valid proposal/inspect/no_change JSON. No source from an incomplete response was applied."}
            atomic_write_json(interactions / f"feedback_turn_{turn}.json", feedback)
            # Preserve the actual returned content, including thought signatures,
            # as recommended for multi-turn reasoning. Never synthesize them.
            content = (raw.get("candidates") or [{}])[0].get("content")
            if content and content.get("parts"):
                contents.append(types.Content.model_validate(content))
            contents.append(types.Content(role="user", parts=[types.Part(text=json.dumps({
                **feedback, "remaining_diagnostics": policy["diagnostic_calls_per_proposal"]-diagnostics,
                "remaining_model_turns": maximum_turns-turn,
                "remaining_drafts": policy["drafts_per_iteration"]-len(drafts),
                "budget": ledger.totals()}))]))
            saved.update(turn=turn + 1, diagnostics=diagnostics, drafts=drafts,
                consecutive_truncations=consecutive_truncations,
                contents=[c.model_dump(mode="json", exclude_none=True) for c in contents])
            atomic_write_json(checkpoint, saved)
        return finish({"status": "limit_stop", "reason": "iteration model-turn safety limit"})
    except P.BudgetExhausted as exc:
        return finish({"status": "budget_stop", "reason": str(exc)})
    except R.SearchLimit as exc:
        return finish({"status": "limit_stop", "reason": str(exc)})


def evaluate_draft(root, parent, seed_source, proposal, directory, label, *, arm, iteration, turn=1):
    """Trust boundary for a submission. Rejection is feedback, never a silent fix."""
    R.check_stop(root)
    R.check_storage(root, configured_policy(root)["minimum_free_disk_bytes"])
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
    review = compliance.automatic_review(root, source, proposal, directory,
        ledger=run_ledger(root, arm), policy=configured_policy(root), project=PROJECT,
        iteration=iteration, turn=turn)
    event(root, "compliance_review_completed", arm=arm, iteration=iteration,
        approved=review["approved"], reviewer=review["reviewer"], human_intervention=False)
    if review.get("source_sha256") != P.sha(source) or review.get("approved") is not True:
        raise ValueError("semantic compliance review: " + review.get("reason", "not approved"))
    R.check_stop(root)
    R.check_storage(root, configured_policy(root)["minimum_free_disk_bytes"])
    metrics = get(score.chia_remote(str(root), plugin, label, "training"))
    return {"source_path": str(directory / "atomic_controller.cpp"), "sha256": P.sha(source),
            "plugin": plugin, "metrics": metrics, "label": label}


@ChiaFunction(num_cpus=1, max_retries=0)
def build(root, source, directory):
    return E.compile_candidate(root, source, directory, resume=True)


@ChiaFunction(num_cpus=2, max_retries=0)
def score(root, plugin, label, split):
    result = E.evaluate(root, ["candidate"], plugin=plugin, label=label, split=split, resume=True)[label]
    get_profiler().add_info({"label": label, "split": split, "aggregate": result["aggregate"]})
    return result


def make_prompt(root, arm, iteration, candidates, incumbent, parent_id, history):
    parent = candidates[parent_id]
    comparisons = L.comparisons(root)
    history = history if L.enabled(root, "evolution_history") else []
    policy = configured_policy(root)
    template = (root / "prompts/iteration_v1.md").read_text()
    fields = {
        "iteration_number": iteration, "max_proposal_iterations": configured_policy(root)["maximum_iterations"],
        "parent_id": parent_id, "parent_source_sha256": parent["sha256"], "incumbent_id": incumbent,
        "parent_source": pathlib.Path(parent["source_path"]).read_text(),
        "parent_and_incumbent_training_scores": json.dumps({"parent": parent["metrics"]["aggregate"], "incumbent": candidates[incumbent]["metrics"]["aggregate"]}),
        "training_configuration_and_trace_clock_contract": json.dumps({
            "training": E.workloads(root, "training"), "instructions_per_core": E.evaluation_insts(root), "frontend": "SimpleO3",
            "std": "DDR5", "org": "DDR5_16Gb_x8", "timing": "DDR5_4800AN", "channels": 1,
            "clock_ratio_frontend": 8, "clock_ratio_memory": 3,
            "logical_trace_clock": "frontend cycles", "controller_trace_clock": "DRAM cycles",
            "logical_boundary": "LLC access including hit/merge/owner; all logical reads scored",
            "reference": E.C.REFERENCE, "refresh": "none", "llc_mshr_per_core": 16,
            "llc_latency_frontend_cycles": 47, "fixed_issued_ROI_fully_drained": True}),
        "per_workload_training_metrics_and_coverage": json.dumps(L.feedback_metrics(root, parent["metrics"]).get("per_workload", {})),
        "training_diagnostic_slices": json.dumps(L.initial_diagnostics(root, parent)),
        "published_comparison_training_metrics": json.dumps(comparisons),
        "accepted_and_rejected_proposals_with_reasons": json.dumps(history),
        "last_build_or_evaluation_diagnostics": json.dumps(history[-1:] if history else []),
        "this_run_pareto_archive": json.dumps({k: candidates[k]["metrics"]["aggregate"] for k in P.pareto(candidates)} if L.enabled(root, "evolution_history") else {}),
        "promotion_and_guardrail_configuration": json.dumps(policy),
        "allowed_tools_and_readable_file_manifest": json.dumps(L.tool_manifest(root, [P.MUTABLE, *VISIBLE])),
        "remaining_model_call_diagnostic_and_token_limits": json.dumps({"model_turns": policy["model_turns_per_proposal"],
            "diagnostics": policy["diagnostic_calls_per_proposal"], "drafts": policy["drafts_per_iteration"],
            "output_tokens_per_turn": P.MAX_OUTPUT, "budget": run_ledger(root, arm).totals(), "cap_usd": configured_policy(root)["usd_cap"]}),
    }
    return P.render(template, fields)


def run_model(root, arm):
    root = pathlib.Path(root)
    R.assert_training_open(root)
    carryover = root / "budget_carryover.json"
    if carryover.exists() and not (root / "ledger.json").exists():
        run_ledger(root, arm).initialize_carryover(json.loads(carryover.read_text()))
    else:
        run_ledger(root, arm).initialize()
    seed_source = (root / "seed/atomic_controller.cpp").read_text()
    seed_metrics = json.loads((root / "training/reports/seed.json").read_text())["models"]["seed"]
    candidates = {"seed": {"source_path": str(root / "seed/atomic_controller.cpp"), "sha256": P.sha(seed_source),
        "plugin": str(root / "seed/candidate.so"), "metrics": seed_metrics, "label": "seed"}}
    state_path = root / "state.json"
    if state_path.exists():
        state = R.read_json(state_path)
        if state.get("run_id") != root.name or state.get("model") != P.MODELS[arm]:
            raise RuntimeError("search checkpoint belongs to a different run/model")
        candidates, incumbent, history = state["candidates"], state["incumbent"], state["history"]
    else:
        incumbent, history = "seed", []
        state = {"run_id": root.name, "backend": arm, "model": P.MODELS[arm],
                 "status": "running", "history": history, "candidates": candidates,
                 "incumbent": incumbent, "budget": run_ledger(root, arm).totals()}
        atomic_write_json(state_path, state)
    system = (root / "prompts/system_v1.md").read_text()
    event(root, "search_entered", backend=arm, model=P.MODELS[arm], completed_iterations=len(history))
    last_iteration = len(history) if state.get("termination") else configured_policy(root)["maximum_iterations"]
    for iteration in range(len(history) + 1, last_iteration + 1):
        R.check_stop(root)
        if (root / "run_manifest.json").exists():
            verify_pinned_run(root, R.read_json(root / "run_manifest.json"))
        label = f"{arm}_{iteration:03d}"
        directory = root / "candidates" / label
        directory.mkdir(parents=True, exist_ok=True)
        if state.get("active"):
            active = state["active"]
            if active["iteration"] != iteration:
                raise RuntimeError("active proposal disagrees with committed search history")
            parent_id = active["parent_id"]
            prompt = (directory / "prompt.md").read_text()
            if P.sha(prompt) != active["prompt_sha256"]:
                raise RuntimeError("active proposal prompt changed")
        else:
            parent_id = L.parent(root, candidates, incumbent, iteration)
            prompt = make_prompt(root, arm, iteration, candidates, incumbent, parent_id, history)
            (directory / "prompt.md").write_text(prompt)
            state["active"] = {"iteration": iteration, "parent_id": parent_id, "prompt_sha256": P.sha(prompt)}
            atomic_write_json(state_path, state)
        parent = {**candidates[parent_id], "id": parent_id}
        event(root, "proposal_started", arm=arm, iteration=iteration, parent=parent_id)
        try:
            proposal = get(propose.chia_remote(str(root), arm, iteration, system, prompt, parent))
        except Exception as exc:
            event(root, "infrastructure_failure", arm=arm, iteration=iteration, reason=str(exc)[-8000:])
            proposal = {"status": "infrastructure_error", "reason": str(exc)[-8000:], "retryable": False}
        status = proposal.get("status")
        if status not in {"evaluated", "no_change", "budget_stop", "limit_stop"}:
            state.update(status="paused" if proposal.get("retryable") else "needs_attention",
                pause=proposal, budget=run_ledger(root, arm).totals())
            atomic_write_json(state_path, state)
            event(root, "search_paused", **proposal)
            return state
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
        if proposal.get("status") != "evaluated":
            # Persist the terminal decision in the SAME transaction as history;
            # a restart must never begin a new proposal after no_change/budget.
            state["termination"] = proposal["status"]
            state["stop_reason"] = record["reason"]
        atomic_write_json(state_path, state)
        # Proposing worker is finished: these interactions are closed. Keep
        # live profiler logs and active checkpoint files out of compression.
        artifacts.compress(root, root / "aux_archive_manifest.json", min_bytes=16_384, parts={"interactions"})
        if proposal.get("status") != "evaluated":
            break
    state.update({"status": "frozen", "frozen_at": time.time(), "selected": candidates[incumbent],
                  "termination": state.get("termination", "iteration_limit")})
    atomic_write_json(state_path, state)
    event(root, "selection_frozen", backend=arm, incumbent=incumbent, budget=state["budget"])
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--model", choices=P.MODELS, required=True, help="exactly one backend per run")
    ap.add_argument("--resume", action="store_true", help="resume only this run's pinned protocol/checkpoints; never optimize after final testing")
    args = ap.parse_args()
    root = args.root.resolve()
    try:
        with R.exclusive_lock(root / ".runner.lock"):
            with R.cpu_lease(REPO / "eval_out/chia/.cpu_leases", execution_limits(root)["cpu_budget"]):
                return execute(args)
    except R.OperationalPause as exc:
        atomic_write_json(root / "launch_pause.json", {"reason": str(exc),
            "retryable": exc.retryable, "retry_at": exc.retry_at, "time": time.time()})
        return 75 if exc.retryable else 78
    except Exception as exc:
        atomic_write_json(root / "launch_pause.json", {
            "reason": f"{type(exc).__name__}: {str(exc)[-8000:]}", "retryable": False, "time": time.time()})
        print(json.dumps({"run_id": root.name, "event": "needs_attention",
                          "reason": f"{type(exc).__name__}: {str(exc)[-8000:]}"}), flush=True)
        return 78


def verify_pinned_run(root, manifest):
    """Verify actual bytes, not git ancestry, before any resumed work."""
    if manifest["policy"] != configured_policy(root):
        raise RuntimeError("cannot resume under changed execution/reviewer/recovery policy")
    if manifest.get("project") != PROJECT:
        raise RuntimeError("cannot resume with a different billing project")
    for package, version in manifest.get("versions", {}).items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("cannot resume with a changed dependency: " + package)
    for relative, expected in manifest["protocol_hashes"].items():
        if P.sha((REPO / relative).read_bytes()) != expected or P.sha((root / "protocol" / relative).read_bytes()) != expected:
            raise RuntimeError("frozen protocol changed: " + relative)
    for relative, expected in manifest["visible_hashes"].items():
        if P.sha((root / "visible" / relative).read_bytes()) != expected:
            raise RuntimeError("frozen visible source changed: " + relative)
    for relative, expected in manifest.get("input_hashes", {}).items():
        if P.sha((root / relative).read_bytes()) != expected:
            raise RuntimeError("frozen input/configuration changed: " + relative)
    for path in (root / "protocol/tools/chia_loop/prompts").glob("*.md"):
        if P.sha((root / "prompts" / path.name).read_bytes()) != P.sha(path.read_bytes()):
            raise RuntimeError("frozen prompt changed: " + path.name)
    runtime = R.read_json(root / "runtime_manifest.json")
    for name, key in (("libramulator.so", "library_sha256"), ("isolated_sim", "executable_sha256")):
        if P.sha((root / "runtime" / name).read_bytes()) != runtime[key]:
            raise RuntimeError("frozen runtime changed: " + name)
    for relative, digest in runtime["export_hashes"].items():
        if P.sha((root / "export" / relative).read_bytes()) != digest:
            raise RuntimeError("frozen build header changed: " + relative)
    state_path = root / "state.json"
    if state_path.exists():
        for candidate in R.read_json(state_path)["candidates"].values():
            if P.sha(pathlib.Path(candidate["source_path"]).read_bytes()) != candidate["sha256"]:
                raise RuntimeError("candidate source changed during recovery")
            build_record = R.read_json(pathlib.Path(candidate["plugin"]).parent / "build.json")
            if P.sha(pathlib.Path(candidate["plugin"]).read_bytes()) != build_record["plugin_sha256"]:
                raise RuntimeError("candidate binary changed during recovery")


def execute(args):
    root = args.root.resolve()
    arm = args.model
    if not (root / "preflight_pass.json").exists():
        raise RuntimeError("paid calls require a completed isolation/evaluation/budget preflight")
    # A leftover infrastructure-smoke result must never become full-ROI
    # training feedback merely because the evaluator's default changed.
    insts = E.evaluation_insts(root)
    for workload in E.workloads(root, "training"):
        for label in ("oracle", "seed", *E.COMPARISONS):
            prepared = json.loads((root / "training/simpleo3/DDR5" / workload / label / "manifest.json").read_text())
            if prepared["insts_per_core"] != insts:
                raise RuntimeError("prepared training ROI differs from campaign window policy")
    existing = (root / "run_manifest.json").exists()
    if existing:
        if not args.resume:
            raise RuntimeError("existing run requires explicit --resume or the supervisor")
        manifest = R.read_json(root / "run_manifest.json")
        if manifest.get("run_id") != root.name or manifest.get("model") != P.MODELS[arm]:
            raise RuntimeError("manifest belongs to a different run/model")
        verify_pinned_run(root, manifest)
        if manifest["status"] == "completed":
            if not manifest.get("archives_verified"):
                from tools.chia_loop.finalize_artifacts import finalize
                finalize(root)
            return 0
        return execute_pinned(root, arm, manifest)
    preparation = json.loads((root / "preparation_manifest.json").read_text())
    if (preparation.get("loop_configuration") != L.load(root)
            or preparation.get("loop_configuration_sha256") != L.identity(L.load(root))):
        raise RuntimeError("loop profile differs from preparation; prepare a fresh run")
    if preparation.get("model") != P.MODELS[arm] or preparation.get("run_id") != root.name:
        raise RuntimeError("preparation belongs to a different model or run; prepare an individual run first")
    if args.resume and (root / "state.json").exists():
        raise RuntimeError("cannot resume a search without its frozen run manifest")
    (root / "visible").mkdir(exist_ok=True)
    for relative in L.visible_files(root, VISIBLE):
        source = REPO / relative
        if not source.is_file():
            raise RuntimeError(f"missing public source allowlist path: {relative}")
        dest = root / "visible" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    shutil.copytree(REPO / "tools/chia_loop/prompts", root / "prompts", dirs_exist_ok=True)
    protocol_paths = [REPO / "tools/chia_loop" / name for name in (
        "artifacts.py", "core.py", "gemini_loop.py", "real_core.py", "real_eval.py",
        "preflight_real.py", "prepare_gemini.py", "sandbox.py", "traffic.py",
        "review_candidate.py", "run_records.py", "analyze_gemini.py", "audit_gemini_campaign.py",
        "generation.py", "recovery.py", "compliance.py", "supervise_gemini.py",
        "evaluation_config.py", "loop_config.py", "synthetic_diagnostics.py", "transfer.py", "transfer_sandbox.py", "transfer_gem5_board.py", "transfer_report.py")]
    protocol_paths += list((REPO / "tools/chia_loop/prompts").glob("*.md"))
    protocol_paths += list((REPO / "tools/eval").glob("*.py"))
    protocol_paths += list((REPO / "tools/eval/gem5").glob("*.py"))
    protocol_paths += list((REPO / "resources/champsim_bridge").glob("*.py"))
    protocol_paths.append(REPO / "tools/chia_loop/champsim_trace_format.cpp")
    protocol_paths += [REPO / "tests/utils.py", REPO / "src/ramulator/frontend/impl/memory_trace/synthetic_pattern.cpp"]
    for path in protocol_paths:
        destination = root / "protocol" / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    input_paths = [p for p in (root / "preparation_manifest.json", root / "window_policy.json",
        root / "input_inventory.json", root / "runtime_manifest.json", root / "preflight_pass.json", root / "budget_carryover.json",
        root / "seed/atomic_controller.cpp", root / "seed/build.json",
        root / "evaluation_config.json", root / "transfer_inputs.json",
        root / "loop_config.json", root / "loop_config_identity.json") if p.exists()]
    input_paths += [p for p in (root / "transfer/runtime").rglob("*") if p.is_file()]
    input_paths += list((root / "training").glob("simpleo3/DDR5/*/*/manifest.json"))
    input_paths += list((root / "training/reports").glob("*.json"))
    input_paths += list((root / "training/report_receipts").glob("*.json"))
    manifest = {"schema_version": 3, "record_type": "optimization_run", "run_id": root.name,
        "backend": arm, "model": P.MODELS[arm], "status": "running", "started_at": time.time(), "policy": configured_policy(root),
        "project": PROJECT, "location": "global", "pricing": model_pricing(P.PRICING, arm),
        "budget_carryover": json.loads((root / "budget_carryover.json").read_text()) if (root / "budget_carryover.json").exists() else {},
        "training": E.workloads(root, "training"), "final_test": E.workloads(root, "test"), "instructions_per_core": E.evaluation_insts(root),
        "comparisons": E.COMPARISONS, "prior_design_exposed": False,
        "human_modeling_hints": False, "test_access": "after_this_run_selection_frozen",
        "human_intervention": False, "phase": "training",
        "review_service": {"model": P.MODELS[POLICY["reviewer_backend"]],
            "pricing": model_pricing(P.PRICING, POLICY["reviewer_backend"]),
            "rubric_sha256": P.sha((root / "prompts/compliance_v1.md").read_bytes()),
            "cost_in_same_run_cap": True, "cross_run_information": False},
        "preparation": preparation,
        "meta_reviewer_test_exposure": "these workload families were evaluated before this fresh campaign; agents see training only",
        "protocol_hashes": {str(p.relative_to(REPO)): P.sha(p.read_bytes()) for p in protocol_paths},
        "visible_hashes": {p: P.sha((root / "visible" / p).read_bytes()) for p in L.visible_files(root, VISIBLE)},
        "input_hashes": {str(p.relative_to(root)): P.sha(p.read_bytes()) for p in input_paths},
        "versions": {p: importlib.metadata.version(p) for p in ("chialoops", "ray", "google-genai", "numpy", "pandas")}}
    atomic_write_json(root / "run_manifest.json", manifest)
    return execute_pinned(root, arm, manifest)


def execute_pinned(root, arm, manifest):
    R.check_stop(root)
    verify_pinned_run(root, manifest)
    ray.init(address="local", num_cpus=manifest["policy"]["cpu_budget"], include_dashboard=False, log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(REPO) + ":" + str(REPO / "tools"),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}})
    start_collector(log_dir=str(root / "profiles" / f"session_{time.time_ns()}"))
    result_code = 0
    try:
        manifest.update(status="running", archives_verified=False)
        atomic_write_json(root / "run_manifest.json", manifest)
        state = R.read_json(root / "state.json") if (root / "state.json").exists() else {}
        if state.get("status") != "frozen":
            if time.time() - manifest["started_at"] >= manifest["policy"]["maximum_run_wall_seconds"]:
                raise R.OperationalPause("run wall-time recovery/search guard reached", retryable=False)
            state = run_model(root, arm)
        manifest["state"] = state
        R.check_stop(root)
        if state.get("status") != "frozen":
            manifest.update(status=state["status"], pause=state.get("pause", {}))
            result_code = 75 if state.get("pause", {}).get("retryable") else 78
        else:
            verify_pinned_run(root, manifest)
            tests = final_test(root, arm, manifest, state)
            for relative, expected in manifest["protocol_hashes"].items():
                if P.sha((REPO / relative).read_bytes()) != expected:
                    raise RuntimeError("protocol mutated during paid campaign: " + relative)
            manifest.update({"status": "completed", "phase": "completed", "finished_at": time.time(),
                             "state": state, "final_test_metrics": tests})
            event(root, "run_completed", test_metrics=tests["aggregate"])
    except R.OperationalPause as exc:
        manifest.update(status="paused" if exc.retryable else "needs_attention",
            pause={"reason": str(exc), "retryable": exc.retryable, "retry_at": exc.retry_at})
        result_code = 75 if exc.retryable else 78
    except BaseException:
        manifest.update({"status": "failed", "failure": traceback.format_exc(), "finished_at": time.time()})
        raise
    finally:
        if (root / "STOP").exists():
            manifest.update(external_stop_observed=True, human_intervention=None)
            event(root, "external_stop_observed", actor="unspecified external operator",
                  note="Operational intervention recorded; no modeling feedback inferred")
        atomic_write_json(root / "run_manifest.json", manifest)
        try:
            collector = get_collector()
            if collector is not None:
                ray.get(collector.get_events.remote())
        finally:
            try:
                stop_collector()
            finally:
                ray.shutdown()
    E.verify_run_archives(root)
    artifacts.compress(root, root / "aux_archive_manifest.json", min_bytes=16_384)
    artifacts.verify(root / "aux_archive_manifest.json")
    manifest["archives_verified"] = True
    atomic_write_json(root / "run_manifest.json", manifest)
    return result_code


def final_test(root, arm, manifest, state):
    if state.get("termination") not in manifest["policy"]["terminal_search_statuses"]:
        raise RuntimeError("held-out testing requires an explicit scientific search termination")
    frozen = {"run_id": root.name, "source_sha256": state["selected"]["sha256"],
              "incumbent": state["incumbent"], "frozen_at": state["frozen_at"]}
    path = root / "selection_frozen.json"
    if path.exists() and R.read_json(path) != frozen:
        raise RuntimeError("cannot change a frozen selection")
    atomic_write_json(path, frozen)
    manifest["phase"] = "final_test"
    atomic_write_json(root / "run_manifest.json", manifest)
    if not (root / "test_started.json").exists():
        atomic_write_json(root / "test_started.json", {**frozen, "started_at": time.time()})
    events = root / "events.jsonl"
    if not events.exists() or not any(json.loads(line).get("event") == "frozen_test_started"
                                    for line in events.read_text().splitlines() if line.strip()):
        # Also recover a crash between writing the marker and appending its event.
        event(root, "frozen_test_started", time=R.read_json(root / "test_started.json")["started_at"])
    E.evaluate(root, ["oracle", *E.COMPARISONS], split="test",
        workers=min(6, manifest["policy"]["cpu_budget"]), resume=True)
    E.evaluate(root, ["candidate"], split="test", plugin=str(root / "seed/candidate.so"), label="seed", resume=True)
    result = get(score.chia_remote(str(root), state["selected"]["plugin"], arm + "_final", "test"))
    manifest.update(final_test_metrics=result, phase="frozen_transfer")
    atomic_write_json(root / "run_manifest.json", manifest)
    from tools.chia_loop import transfer
    manifest["transfer_metrics"] = transfer.run(root, state, workers=min(6, manifest["policy"]["cpu_budget"]))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
