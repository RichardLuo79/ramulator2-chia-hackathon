"""Independent Astra-effort CHIA runs. Preparation and tests are nonbillable."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import ray
from chia.base.ChiaFunction import ChiaFunction, get
from chia.trace.profiler import start_collector, stop_collector
from tools.chia_loop import real_core as P, real_eval as E, recovery as R, artifacts
from tools.chia_loop import compliance as C, gemini_loop as G
from tools.chia_loop import evaluation_config as W, transfer
from tools.chia_loop import loop_config as L
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.traffic import traffic_population
from . import transport as T
from . import usage as U
from . import retrospective as V

REPO = pathlib.Path(__file__).resolve().parents[3]
POLICY = {"version": "codex_cli_individual_run_v7", "model": T.MODEL, "usage_schema": U.SCHEMA,
          "evaluation": "run-local DDR5 profile; frozen-source ChampSim/gem5 transfer excluded from feedback",
          "stream_output": "finalized output items reconciled with terminal response; no partial-answer fallback",
          "financial_continuity": "all prior usage retained; explicit iteration-only guard available for ChatGPT",
          "scientific_continuation": "same-effort completed training checkpoints only; frozen predecessor stays intact",
          "reasoning_summary": T.REASONING_SUMMARY,
          "retrospective": "provider-exposed summaries and observable actions; no hidden chain of thought",
          "model_turns_per_proposal": 48, "diagnostic_calls_per_proposal": 192,
          "drafts_per_iteration": 12, "review_attempts": 3, "maximum_run_wall_seconds": 86400,
          "reviewer_effort": "xhigh", "maximum_output_tokens": T.MAX_OUTPUT,
          "promotion": "strict_two_objective_Pareto", "native_codex_tools": False,
          "context": "fresh ephemeral CLI; explicit same-run conversation only",
          "human_intervention": False, "unknown_response_reservation": "retained_in_full"}


def event(root, kind, **fields):
    G.event(root, kind, **fields)


def load_config(root):
    return R.read_json(pathlib.Path(root) / "codex_config.json")


def configured_policy(root):
    return L.policy(root, POLICY)


def prepare(args):
    """No LLM calls; scientific history requires explicit same-effort continuation."""
    iteration_guard = getattr(args, "iteration_guard", False)
    limits = T.F.limits(args.max_iterations, args.usd_cap, args.cpus,
                       iteration_guard=iteration_guard, auth_mode=getattr(args, "auth_mode", "chatgpt"))
    root = args.root.resolve()
    T.check_stop(root)
    with R.cpu_lease(REPO / "eval_out/chia/.cpu_leases", limits["cpu_budget"]):
        continuation = getattr(args, "continue_training_from", None)
        if continuation is not None:
            from . import continuation as H
            H.predecessor(continuation, args.effort, args.max_iterations)
        root.mkdir(parents=True, exist_ok=False)
        evaluation = W.install(root, getattr(args, "evaluation_config", None))
        loop = L.install(root, getattr(args, "loop_config", None))
        policy = configured_policy(root)
        if continuation is not None and (L.load(continuation) != loop or W.load(continuation) != evaluation):
            raise ValueError("continuation requires identical loop/evaluation profiles; ablations must start fresh")
        train = E.workloads(root, "training")
        source = (REPO / P.MUTABLE).read_text()
        _, regions = P.regions(source)
        expected = "voidinit_model(){}Clk_tpredict_departure(constRequest&req){returnm_clk+m_latency;}"
        if re.sub(r"\s+", "", regions["CODE"]) != expected or regions["INCLUDES"].strip():
            raise RuntimeError("fresh runs require the unchanged fixed-delay skeleton")
        config = {**limits, "run_id": root.name, "model": T.MODEL, "effort": args.effort,
                  "auth_mode": args.auth_mode, "policy": policy,
                  "guard_mode": "iterations" if iteration_guard else "usd",
                  "auth_file": str(args.auth_file.resolve()) if args.auth_file else None,
                  "billing": "API usage" if args.auth_mode == "api" else "ChatGPT quotas; USD guard is API-equivalent, not an invoice",
                  "prior_design_exposed": False, "human_modeling_hints": False}
        predecessor = getattr(args, "carry_budget_from", None)
        if continuation is not None:
            if predecessor is not None:
                raise ValueError("choose scientific continuation OR financial-only restart")
            predecessor = continuation
            config["continuation_requested"] = True
        if predecessor is not None:
            config["budget_carryover_sha256"] = T.F.prepare(root, predecessor,
                model=T.MODEL, effort=args.effort, cap=limits["usd_cap"], iteration_guard=iteration_guard)
        atomic_write_json(root / "codex_config.json", config)
        T.Ledger(root, limits["usd_cap"]).transaction(lambda data: None)
        atomic_write_json(root / "preparation_manifest.json", {"status": "preparing", "limits": limits,
            "run_id": root.name, "model": T.MODEL, "effort": args.effort, "optimization": "-O3",
            "loop_configuration": loop, "loop_configuration_sha256": L.identity(loop),
            "seed_sha256": P.sha(source), "paid_generation_calls": 0})
        atomic_write_json(root / "window_policy.json", {"instructions_per_core": evaluation["simpleo3"]["instructions_per_core"],
            "initialization": "cold prefix, no wrap, complete drain"})
        E.prepare_runtime(root)
        transfer.prepare(root)
        binary = pathlib.Path(args.codex_binary or shutil.which("codex") or "").resolve(strict=True)
        shutil.copyfile(binary, root / "runtime/codex")
        (root / "runtime/codex").chmod(0o755)
        version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True).stdout.strip()
        catalog = json.loads(subprocess.run([str(binary), "debug", "models", "--bundled"],
                                           capture_output=True, text=True, check=True).stdout)
        models = catalog if isinstance(catalog, list) else catalog["models"]
        selected = next(m for m in models if m["slug"] == T.MODEL)
        if not set(T.EFFORTS) <= {r["effort"] for r in selected["supported_reasoning_levels"]}:
            raise RuntimeError("installed CLI does not advertise both exact Astra efforts")
        atomic_write_json(root / "cli_identity.json", {"version": version,
            "binary_sha256": P.sha((root / "runtime/codex").read_bytes()),
            "model": T.MODEL, "supported_reasoning_levels": selected["supported_reasoning_levels"]})
        plugin = E.compile_candidate(root, source, root / "seed")
        W.inventory(root, E.C.trace_path, E.C.file_provenance)
        for relative in L.visible_files(root, G.VISIBLE):
            dest = root / "visible" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, dest)
        shutil.copytree(REPO / "tools/chia_loop/prompts", root / "prompts")
        E.evaluate(root, ["oracle", *E.COMPARISONS], split="training", workers=args.cpus)
        E.evaluate(root, ["candidate"], plugin=plugin, label="seed", split="training", workers=2)
        coverage = [traffic_population(root / "training/simpleo3/DDR5" / w / "oracle",
            evaluation["simpleo3"]["minimum_oracle_owner_reads"]) for w in train]
        if not all(v["owner_count_pass"] for v in coverage):
            raise RuntimeError("insufficient training oracle traffic")
        atomic_write_json(root / "training_traffic_coverage.json", coverage)
        env = {**os.environ, "EVAL_OUT": str(root / "parity"),
               "PYTHONPATH": ":".join(str(REPO / p) for p in ("python", "tools", "."))}
        E.command([sys.executable, REPO / "tools/eval/run_simpleo3.py", "--std", "DDR5", "--workloads", *train,
            "--models", "oracle,candidate", "--candidate-label", "seed", "--insts-per-core", str(E.evaluation_insts(root)), "--workers", "2"],
            root / "logs/parity.log", timeout=1800, env=env)
        E.command([sys.executable, REPO / "tools/chia_loop/preflight_real.py", root],
                  root / "logs/preflight.log", timeout=1800, env=env)
        E.command([sys.executable, "-m", "pytest", "-q", "tests/unit_tests/test_chia_codex_cli.py",
                   "tests/unit_tests/test_chia_codex_usage.py", "tests/unit_tests/test_chia_codex_retrospective.py",
                   "tests/unit_tests/test_chia_codex_queue.py", "tests/unit_tests/test_chia_codex_stream.py",
                   "tests/unit_tests/test_chia_codex_budget.py", "tests/unit_tests/test_chia_codex_continuation.py"],
                  root / "logs/codex_preflight.log", timeout=240,
                  env={**env, "CHIA_CODEX_PREFLIGHT_BINARY": str(root / "runtime/codex")})
        if continuation is not None:
            H.import_training(root, continuation, args.effort, args.max_iterations, args.cpus)
            config["continuation_sha256"] = P.sha((root / "continuation.json").read_bytes())
            atomic_write_json(root / "codex_config.json", config)
        E.verify_run_archives(root)
        preparation = R.read_json(root / "preparation_manifest.json")
        preparation.update(status="ready", finished_at=time.time())
        atomic_write_json(root / "preparation_manifest.json", preparation)
        files = list((REPO / "tools/chia_loop").glob("*.py")) + list(T.HERE.glob("*.py"))
        files += list((REPO / "tools/eval").glob("*.py")) + list((REPO / "tools/chia_loop/prompts").glob("*.md"))
        files += list((REPO / "tools/eval/gem5").glob("*.py"))
        files += list((REPO / "resources/champsim_bridge").glob("*.py"))
        files.append(REPO / "tools/chia_loop/champsim_trace_format.cpp")
        files += [REPO / "tests/utils.py", REPO / "src/ramulator/frontend/impl/memory_trace/synthetic_pattern.cpp"]
        protocol = {}
        for file in files:
            relative = str(file.relative_to(REPO))
            dest = root / "protocol" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest)
            protocol[relative] = P.sha(file.read_bytes())
        pinned = [root / name for name in ("codex_config.json", "cli_identity.json", "preparation_manifest.json",
            "window_policy.json", "input_inventory.json", "runtime_manifest.json", "preflight_pass.json", "evaluation_config.json",
            "loop_config.json", "loop_config_identity.json")]
        if (root / "transfer_inputs.json").exists():
            pinned.append(root / "transfer_inputs.json")
            pinned += [p for p in (root / "transfer/runtime").rglob("*") if p.is_file()]
        pinned += list((root / "prompts").glob("*.md")) + list((root / "training/reports").glob("*.json"))
        pinned += [p for p in (root / "visible").rglob("*") if p.is_file()]
        pinned += [p for p in (root / "export").rglob("*") if p.is_file()]
        pinned += list((root / "training").glob("simpleo3/DDR5/*/*/manifest.json"))
        pinned += list((root / "training/report_receipts").glob("*.json"))
        pinned += [root / "seed/atomic_controller.cpp", root / "seed/candidate.so", root / "runtime/codex",
                   root / "runtime/libramulator.so", root / "runtime/isolated_sim"]
        if (root / "budget_carryover.json").exists():
            pinned.append(root / "budget_carryover.json")
        if continuation is not None:
            pinned.append(root / "continuation.json")
        atomic_write_json(root / "run_manifest.json", {"run_id": root.name, "record_type": "codex_optimization_run",
            "status": "prepared", "configuration": config, "policy": policy, "protocol_hashes": protocol,
            "input_hashes": {str(p.relative_to(root)): P.sha(p.read_bytes()) for p in pinned},
            "versions": {p: importlib.metadata.version(p) for p in ("chialoops", "ray", "numpy", "pandas", "httpx")},
            "python_version": sys.version, "usage_tariff": U.TARIFF,
            "training": train, "final_test": E.workloads(root, "test"), "comparisons": E.COMPARISONS,
            "human_intervention": continuation is not None, "human_operational_intervention": continuation is not None,
            "human_modeling_hints": False, "prior_design_exposed": False, "archives_verified": True,
            "scientific_history_imported": continuation is not None,
            "evaluation_interpretation": "exploratory extension after operator test exposure" if continuation is not None else "initial trial",
            "meta_reviewer_test_exposure": "these test families have been evaluated previously; agents see training only"})
        event(root, "prepared", paid_calls=0, effort=args.effort)
        U.write_report(root)
        V.write_report(root)


def prepare_with_wait(args):
    """A non-generating preparation queue; only resource contention is retried."""
    root = args.root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    status_path = root.with_name(root.name + ".preparation_status.json")
    stop_path = root.with_name(root.name + ".preparation_STOP")
    with R.exclusive_lock(root.with_name(root.name + ".preparation.lock")):
        status = {"run_id": root.name, "model": T.MODEL, "effort": args.effort,
                  "maximum_iterations": args.max_iterations, "cpu_budget": args.cpus,
                  "generation_calls": 0, "auto_launch_generation": False,
                  "started_at": time.time(), "pid": os.getpid()}
        try:
            while True:
                if stop_path.exists():
                    raise R.OperatorStop("preparation stop requested")
                if time.time() - status["started_at"] >= POLICY["maximum_run_wall_seconds"]:
                    raise R.OperationalPause("preparation wait deadline", retryable=False)
                status.update(status="preparing", checked_at=time.time())
                atomic_write_json(status_path, status)
                try:
                    prepare(args)
                    status.update(status="prepared", finished_at=time.time())
                    atomic_write_json(status_path, status)
                    return
                except R.OperationalPause as exc:
                    if not args.wait_for_cpus or not exc.retryable or root.exists():
                        raise
                    status.update(status="waiting_for_cpu_slots", checked_at=time.time())
                    atomic_write_json(status_path, status)
                    R.wait_until(root, time.time() + 30)
        except BaseException as exc:
            status.update(status="stopped" if isinstance(exc, R.OperatorStop) else "needs_attention",
                          error_kind=type(exc).__name__, finished_at=time.time())
            atomic_write_json(status_path, status)
            raise


def verify(root):
    manifest = R.read_json(root / "run_manifest.json")
    if (manifest["configuration"] != load_config(root) or manifest["configuration"]["policy"] != configured_policy(root)
            or manifest["policy"] != configured_policy(root)):
        raise RuntimeError("run configuration/policy changed")
    if manifest.get("python_version") != sys.version or manifest.get("usage_tariff") != U.TARIFF:
        raise RuntimeError("Python runtime or usage tariff changed")
    config = manifest["configuration"]
    T.F.limits(config["maximum_iterations"], config["usd_cap"], config["cpu_budget"],
               iteration_guard=config.get("guard_mode") == "iterations", auth_mode=config["auth_mode"])
    if config.get("continuation_requested"):
        from . import continuation as H
        H.verify_import(root, config)
    for relative, digest in manifest["protocol_hashes"].items():
        if P.sha((REPO / relative).read_bytes()) != digest or P.sha((root / "protocol" / relative).read_bytes()) != digest:
            raise RuntimeError("frozen protocol changed: " + relative)
    for relative, digest in manifest["input_hashes"].items():
        if P.sha((root / relative).read_bytes()) != digest:
            raise RuntimeError("frozen input changed: " + relative)
    for name, version in manifest["versions"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("dependency changed: " + name)
    if (root / "state.json").exists():
        for candidate in R.read_json(root / "state.json")["candidates"].values():
            if P.sha(pathlib.Path(candidate["source_path"]).read_bytes()) != candidate["sha256"]:
                raise RuntimeError("candidate changed outside the run")
            build = R.read_json(pathlib.Path(candidate["plugin"]).parent / "build.json")
            if P.sha(pathlib.Path(candidate["plugin"]).read_bytes()) != build["plugin_sha256"]:
                raise RuntimeError("candidate binary changed outside the run")
    return manifest


@ChiaFunction(num_cpus=1, max_retries=0)
def generate(root, operation, system, conversation, role):
    root = pathlib.Path(root)
    config = load_config(root)
    effort = config["effort"] if role == "proposal" else POLICY["reviewer_effort"]
    return T.invoke(root, operation, system, conversation, effort=effort, role=role,
        cap=config["usd_cap"], upstream=T.Upstream(config["auth_mode"], config["auth_file"], root / "runtime/codex"),
        binary=root / "runtime/codex")


def call(root, operation, system, conversation, role="proposal"):
    # Explicit retries own their ledger entries. CHIA/Ray never retries a paid
    # function implicitly. Saved Responses bytes are replayed without payment.
    for attempt in range(3):
        verify(root)
        T.check_stop(root)
        R.check_run_deadline(root)
        try:
            return get(generate.chia_remote(str(root), operation, system, conversation, role))
        except Exception as exc:
            cause = exc.as_instanceof_cause() if hasattr(exc, "as_instanceof_cause") else exc
            if isinstance(cause, P.BudgetExhausted):
                raise P.BudgetExhausted(str(cause))
            if not isinstance(cause, R.OperationalPause) or not cause.retryable or attempt == 2:
                raise R.OperationalPause("CLI generation stopped; see the trusted journal", retryable=False) from exc
            event(root, "transport_cooldown", operation=operation, attempt=attempt + 1)
            R.wait_until(root, max(time.time() + 1, cause.retry_at))


def prompt(root, state, parent_id, iteration):
    config, parent = load_config(root), state["candidates"][parent_id]
    fields = {"iteration": iteration, "maximum_iterations": config["maximum_iterations"],
        "parent_id": parent_id, "parent_source_sha256": parent["sha256"],
        "parent_source": pathlib.Path(parent["source_path"]).read_text(),
        "incumbent_id": state["incumbent"], "parent_metrics": L.feedback_metrics(root, parent["metrics"]),
        "training_configuration": {"workloads": E.workloads(root, "training"), "instructions_per_core": E.evaluation_insts(root),
            "frontend": "SimpleO3", "standard": "DDR5", "org": "DDR5_16Gb_x8", "timing": "DDR5_4800AN",
            "channels": 1, "frontend_clock_ratio": 8, "memory_clock_ratio": 3,
            "reference": E.C.REFERENCE, "logical_trace_clock": "frontend cycles", "controller_trace_clock": "DRAM cycles",
            "logical_boundary": "LLC access, including hits/merges/DRAM owners; all logical reads scored"},
        "training_diagnostics": L.initial_diagnostics(root, parent),
        "comparisons": L.comparisons(root),
        "history": state["history"] if L.enabled(root, "evolution_history") else [],
        "policy": configured_policy(root), "readable_files": L.visible_files(root, [P.MUTABLE, *G.VISIBLE]),
        "tool_manifest": L.tool_manifest(root, [P.MUTABLE, *G.VISIBLE]),
        "budget": T.Ledger(root, config["usd_cap"]).totals(), "human_modeling_hint": None}
    return {"role": "user", "content": json.dumps(fields, sort_keys=True)}


def review(root, source, proposal, directory, operation):
    submitted = C.review_input(source, proposal)
    rubric = (root / "prompts/compliance_v1.md").read_text()
    binding = {"source_sha256": P.sha(source), "review_input_sha256": P.sha(json.dumps(submitted, sort_keys=True)),
               "rubric_sha256": P.sha(rubric), "reviewer": T.MODEL, "effort": POLICY["reviewer_effort"]}
    if (directory / "review.json").exists():
        prior = R.read_json(directory / "review.json")
        if any(prior[k] != v for k, v in binding.items()) or prior["human_intervention"] is not False:
            raise RuntimeError("review binding changed")
        if prior["approved"] != (C.validate_decision(prior, P.sha(source)) == "pass"):
            raise RuntimeError("review approval contradicts its checks")
        return prior
    atomic_write_json(directory / "review_input.json", submitted)
    conversation = [{"role": "user", "content": json.dumps(submitted, sort_keys=True)}]
    for index in range(POLICY["review_attempts"]):
        decision = call(root, operation + f"_{index:02d}", rubric, conversation, "review")
        try:
            verdict = C.validate_decision(decision, P.sha(source))
            break
        except ValueError as exc:
            conversation += [{"role": "assistant", "content": json.dumps(decision)},
                             {"role": "user", "content": "Schema error: " + str(exc)}]
    else:
        verdict = "uncertain"
        decision = {"source_sha256": P.sha(source), "verdict": verdict,
                    "checks": {k: {"verdict": verdict, "reason": "No valid complete reviewer response"} for k in C.CHECKS}}
    result = {**decision, **binding, "approved": verdict == "pass", "human_intervention": False,
              "reviewer_kind": "isolated_codex_cli", "time": time.time()}
    atomic_write_json(directory / "review.json", result)
    return result


def draft(root, parent, proposal, directory, label):
    if proposal.get("parent_id") != parent["id"] or proposal.get("parent_source_sha256") != parent["sha256"]:
        raise ValueError("parent ID/hash mismatch")
    if any(not proposal.get(field) for field in ("evidence", *C.EXPLANATION_FIELDS)):
        raise ValueError("complete technical explanation required")
    source = P.assemble_regions(pathlib.Path(parent["source_path"]).read_text(), proposal.get("regions"))
    checks = P.validate_source((root / "seed/atomic_controller.cpp").read_text(), source)
    (directory / "atomic_controller.cpp").write_text(source)
    atomic_write_json(directory / "static_checks.json", checks)
    plugin = get(G.build.chia_remote(str(root), source, str(directory / "build")))
    decision = review(root, source, proposal, directory, "review_" + label)
    if not decision["approved"]:
        raise ValueError("Compliance feedback: " + json.dumps(decision["checks"]))
    metrics = get(G.score.chia_remote(str(root), plugin, label, "training"))
    return {"source_path": str(directory / "atomic_controller.cpp"), "sha256": P.sha(source),
            "plugin": plugin, "label": label, "metrics": metrics}


def evolve_one(root, state, parent_id, iteration):
    policy = configured_policy(root)
    directory = root / "candidates" / f"astra_{iteration:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "proposal_state.json"
    parent = {**state["candidates"][parent_id], "id": parent_id}
    if path.exists():
        saved = R.read_json(path)
        if saved["parent_id"] != parent_id or saved["parent_sha256"] != parent["sha256"]:
            raise RuntimeError("active parent changed")
    else:
        saved = {"parent_id": parent_id, "parent_sha256": parent["sha256"], "turn": 1, "diagnostics": 0,
                 "drafts": [], "conversation": [prompt(root, state, parent_id, iteration)]}
        atomic_write_json(path, saved)
    if "result" in saved:
        return saved["result"]
    system = (root / "prompts/system_v1.md").read_text()
    try:
        for turn in range(saved["turn"], policy["model_turns_per_proposal"] + 1):
            proposal = call(root, f"proposal_{iteration:03d}_{turn:03d}", system, saved["conversation"])
            status = proposal.get("status")
            if status == "no_change":
                result = {"status": "no_change", "reason": proposal.get("limitations", "model declared no_change")}
                break
            feedback = {}
            if status == "inspect":
                requests = proposal.get("requests", [])
                if (not isinstance(requests, list) or not 1 <= len(requests) <= 16
                        or saved["diagnostics"] + len(requests) > policy["diagnostic_calls_per_proposal"]
                        or turn > policy["model_turns_per_proposal"] - 2):
                    feedback = {"error": "Inspection limit; submit complete region bodies or no_change."}
                else:
                    feedback["tool_results"] = []
                    for request in requests:
                        saved["diagnostics"] += 1
                        try:
                            value = G.inspect_tool(root, parent, request)
                        except (R.OperationalPause, OSError):
                            raise
                        except Exception as exc:
                            value = {"error": str(exc)}
                        feedback["tool_results"].append({"request": request, "result": value})
            elif status == "proposal":
                number = len(saved["drafts"]) + 1
                if number > policy["drafts_per_iteration"]:
                    result = {"status": "limit_stop", "reason": "draft limit"}
                    break
                label = f"astra_{iteration:03d}_d{number:02d}"
                target = directory / f"draft_{number:03d}"
                target.mkdir(exist_ok=True)
                atomic_write_json(target / "proposal.json", proposal)
                try:
                    if (target / "result.json").exists():
                        candidate = R.read_json(target / "result.json")
                    elif (target / "rejection.json").exists():
                        raise ValueError(R.read_json(target / "rejection.json")["reason"])
                    else:
                        candidate = draft(root, parent, proposal, target, label)
                        atomic_write_json(target / "result.json", candidate)
                    saved["drafts"].append({"directory": str(target), "status": "valid", "source_sha256": candidate["sha256"]})
                    result = {"status": "evaluated", "candidate": candidate, "proposal": proposal}
                    break
                except (P.BudgetExhausted, R.OperationalPause, OSError):
                    raise
                except Exception as exc:
                    # Compilation/runtime/contract failures are model feedback;
                    # provider, disk, provenance and budget faults are not.
                    feedback = {"reason": str(exc)[-8000:], "instruction": "Repair your own draft; no human source changes."}
                    atomic_write_json(target / "rejection.json", feedback)
                    saved["drafts"].append({"directory": str(target), "status": "rejected", **feedback})
            else:
                feedback = {"error": "Return valid proposal, inspect, or no_change JSON."}
            saved["conversation"] += [{"role": "assistant", "content": json.dumps(proposal)},
                                      {"role": "user", "content": json.dumps(feedback)}]
            saved["turn"] = turn + 1
            atomic_write_json(path, saved)
        else:
            result = {"status": "limit_stop", "reason": "model turn limit"}
    except P.BudgetExhausted as exc:
        result = {"status": "budget_stop", "reason": str(exc)}
    saved["result"] = {**result, "drafts": saved["drafts"]}
    atomic_write_json(path, saved)
    return saved["result"]


def search(root):
    config = load_config(root)
    R.assert_training_open(root)
    path = root / "state.json"
    if path.exists():
        state = R.read_json(path)
    else:
        seed = {"source_path": str(root / "seed/atomic_controller.cpp"), "plugin": str(root / "seed/candidate.so"),
                "sha256": P.sha((root / "seed/atomic_controller.cpp").read_bytes()), "label": "seed",
                "metrics": R.read_json(root / "training/reports/seed.json")["models"]["seed"]}
        state = {"run_id": root.name, "status": "running", "incumbent": "seed", "candidates": {"seed": seed}, "history": []}
        atomic_write_json(path, state)
    stop = len(state["history"]) if state.get("termination") else config["maximum_iterations"]
    for iteration in range(len(state["history"]) + 1, stop + 1):
        verify(root)
        T.check_stop(root)
        manifest = R.read_json(root / "run_manifest.json")
        if time.time() - manifest["started_at"] > POLICY["maximum_run_wall_seconds"]:
            raise R.OperationalPause("run wall-time guard", retryable=False)
        parent = state.get("active_parent") or L.parent(root, state["candidates"], state["incumbent"], iteration)
        state.update(active_parent=parent, status="running")
        atomic_write_json(path, state)
        event(root, "proposal_started", iteration=iteration, parent=parent)
        result = evolve_one(root, state, parent, iteration)
        record = {"iteration": iteration, "parent": parent, "status": result["status"], "drafts": result["drafts"]}
        if result["status"] == "evaluated":
            candidate_id = f"astra_{iteration:03d}"
            candidate = result["candidate"]
            state["candidates"][candidate_id] = candidate
            promoted = P.dominates(P.objectives(candidate["metrics"]), P.objectives(state["candidates"][state["incumbent"]]["metrics"]))
            if promoted:
                state["incumbent"] = candidate_id
            record.update(promoted=promoted, metrics=candidate["metrics"]["aggregate"],
                          explanation={k: v for k, v in result["proposal"].items() if k != "regions"})
            event(root, "candidate_evaluated", iteration=iteration, promoted=promoted, metrics=record["metrics"])
        else:
            state.update(termination=result["status"], stop_reason=result["reason"])
        state["history"].append(record)
        state.pop("active_parent", None)
        atomic_write_json(path, state)
        U.write_report(root)
        V.write_report(root)
        if state.get("termination"):
            break
    state.update(status="frozen", termination=state.get("termination", "iteration_limit"), frozen_at=time.time(),
                 selected=state["candidates"][state["incumbent"]])
    atomic_write_json(path, state)
    return state


def finish(root, state):
    if state.get("termination") not in {"iteration_limit", "no_change", "limit_stop", "budget_stop"}:
        raise RuntimeError("held-out test requires scientific termination")
    frozen = {"run_id": root.name, "source_sha256": state["selected"]["sha256"],
              "incumbent": state["incumbent"], "frozen_at": state["frozen_at"]}
    path = root / "selection_frozen.json"
    if path.exists() and R.read_json(path) != frozen:
        raise RuntimeError("cannot change selection after freeze")
    atomic_write_json(path, frozen)
    if not (root / "test_started.json").exists():
        atomic_write_json(root / "test_started.json", {**frozen, "started_at": time.time()})
        event(root, "frozen_test_started")
    E.evaluate(root, ["oracle", *E.COMPARISONS], split="test", workers=load_config(root)["cpu_budget"], resume=True)
    label = "selected_" + load_config(root)["effort"]
    return E.evaluate(root, ["candidate"], plugin=state["selected"]["plugin"],
                      label=label, split="test", workers=2, resume=True)[label]


def run(root, resume=False):
    root = root.resolve()
    T.check_stop(root)
    config = load_config(root)
    with R.exclusive_lock(root / ".runner.lock"), R.cpu_lease(REPO / "eval_out/chia/.cpu_leases", config["cpu_budget"]):
        manifest = verify(root)
        if manifest["status"] != "prepared" and not resume:
            raise RuntimeError("existing run requires --resume")
        if manifest["status"] == "completed":
            if not manifest.get("aux_archives_verified"):
                artifacts.compress(root, root / "aux_archive_manifest.json", min_bytes=16384)
                artifacts.verify(root / "aux_archive_manifest.json")
                manifest["aux_archives_verified"] = True
                atomic_write_json(root / "run_manifest.json", manifest)
            U.write_report(root)
            V.write_report(root)
            return
        manifest.update(status="running", started_at=manifest.get("started_at", time.time()), archives_verified=False)
        atomic_write_json(root / "run_manifest.json", manifest)
        event(root, "runner_resumed" if resume else "runner_started", actor="trusted_launcher",
              human_modeling_hint=False, auth_mode=config["auth_mode"], effort=config["effort"])
        ray.init(address="local", num_cpus=config["cpu_budget"], include_dashboard=False, log_to_driver=False,
                 runtime_env={"env_vars": {"PYTHONPATH": str(REPO) + ":" + str(REPO / "tools"),
                     "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}})
        start_collector(log_dir=str(root / "profiles" / f"session_{time.time_ns()}"))
        try:
            state = R.read_json(root / "state.json") if (root / "state.json").exists() else {}
            if state.get("status") != "frozen":
                state = search(root)
            verify(root)
            tests = finish(root, state)
            manifest.update(state=state, final_test_metrics=tests, phase="frozen_transfer")
            atomic_write_json(root / "run_manifest.json", manifest)
            transfers = transfer.run(root, state, workers=config["cpu_budget"])
            manifest["transfer_metrics"] = transfers
            atomic_write_json(root / "run_manifest.json", manifest)
            E.verify_run_archives(root)
            manifest.update(status="completed", phase="completed", state=state, final_test_metrics=tests, transfer_metrics=transfers, finished_at=time.time(),
                            budget=T.Ledger(root, config["usd_cap"]).totals(), archives_verified=True)
        except BaseException as exc:
            manifest.update(status="needs_attention", error_kind=type(exc).__name__,
                            reason="Stopped without advancing search or opening held-out testing for an infrastructure fault")
            event(root, "runner_stopped", error_kind=type(exc).__name__,
                  operator_stop_observed=isinstance(exc, (R.OperatorStop, KeyboardInterrupt)))
            raise
        finally:
            atomic_write_json(root / "run_manifest.json", manifest)
            stop_collector()
            ray.shutdown()
            U.write_report(root)
            V.write_report(root)
        artifacts.compress(root, root / "aux_archive_manifest.json", min_bytes=16384)
        artifacts.verify(root / "aux_archive_manifest.json")
        manifest["aux_archives_verified"] = True
        atomic_write_json(root / "run_manifest.json", manifest)
        U.write_report(root)
        V.write_report(root)


def supervise(root, resume=False):
    """Restart abrupt process loss only; never repair a model or waive a gate."""
    root = root.resolve()
    T.check_stop(root)
    with R.exclusive_lock(root / ".supervisor.lock"):
        path = root / "supervisor_state.json"
        state = R.read_json(path) if path.exists() else {"run_id": root.name, "status": "starting",
            "started_at": time.time(), "launches": 0, "crash_restarts": 0,
            "human_intervention": load_config(root).get("continuation_requested", False), "human_modeling_hints": False}
        if state["run_id"] != root.name:
            raise RuntimeError("supervisor owner changed")
        if state["launches"] and not resume:
            raise RuntimeError("existing supervisor requires --resume")
        while time.time() - state["started_at"] < POLICY["maximum_run_wall_seconds"]:
            T.check_stop(root)
            verify(root)
            args = [sys.executable, "-m", "tools.chia_loop.codex_cli", "run", "--root", str(root), "--authorize-paid"]
            if resume or state["launches"]:
                args.append("--resume")
            state.update(status="running", launches=state["launches"] + 1)
            atomic_write_json(path, state)
            directory = root / "supervisor_logs"
            directory.mkdir(exist_ok=True)
            with (directory / f"launch_{state['launches']:03d}.log").open("x") as log:
                process = subprocess.Popen(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except KeyboardInterrupt:
                    (root / "STOP").touch(exist_ok=True)
                    process.wait()
                    raise
            manifest = R.read_json(root / "run_manifest.json")
            if code == 0 and manifest["status"] == "completed" and manifest.get("aux_archives_verified"):
                state.update(status="completed", finished_at=time.time())
                atomic_write_json(path, state)
                return
            if code == 75:  # CPU slots are leased to other experiments.
                state.update(status="resource_wait")
                atomic_write_json(path, state)
                R.wait_until(root, time.time() + 30)
                continue
            if manifest["status"] == "running" and code not in (0, 2, 78, 130, -2) and state["crash_restarts"] < 3:
                state["crash_restarts"] += 1
                atomic_write_json(path, state)
                R.wait_until(root, time.time() + 10)
                continue
            state.update(status="needs_attention", last_exit_code=code, reason=manifest.get("error_kind", "worker stopped"))
            atomic_write_json(path, state)
            return
        state.update(status="needs_attention", reason="24-hour supervisor guard")
        atomic_write_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="nonbillable full-window preparation; fresh root only")
    prep.add_argument("--root", required=True, type=pathlib.Path)
    prep.add_argument("--effort", required=True, choices=T.EFFORTS)
    prep.add_argument("--auth-mode", default="chatgpt", choices=("api", "chatgpt"),
                      help="default: reuse existing ChatGPT/Codex login in the trusted broker")
    prep.add_argument("--auth-file", type=pathlib.Path)
    prep.add_argument("--max-iterations", required=True, type=int)
    guard = prep.add_mutually_exclusive_group(required=True)
    guard.add_argument("--usd-cap", type=float)
    guard.add_argument("--iteration-guard", action="store_true", help="ChatGPT only: track usage without a USD stopping guard")
    prep.add_argument("--cpus", type=int, default=6)
    prep.add_argument("--evaluation-config", type=pathlib.Path, default=W.DEFAULT,
                      help="operator-owned DDR5 cohorts and frozen-source transfer profile")
    prep.add_argument("--loop-config", type=pathlib.Path, default=L.DEFAULT,
                      help="operator-owned feature/feedback/search ablation profile")
    prep.add_argument("--codex-binary")
    prep.add_argument("--carry-budget-from", type=pathlib.Path,
                      help="financial-only continuity from one stopped/completed same-effort run; never import its designs")
    prep.add_argument("--continue-training-from", type=pathlib.Path,
                      help="explicit same-effort continuation; preserve old frozen results and import training checkpoints only")
    prep.add_argument("--wait-for-cpus", action="store_true", help="queue non-generating preparation for shared CPU slots")
    usage = commands.add_parser("usage", help="read this run's usage; no model or account calls")
    usage.add_argument("--root", required=True, type=pathlib.Path)
    usage.add_argument("--write", action="store_true", help="also write private-run JSON/CSV/Markdown summaries")
    retrospective = commands.add_parser("retrospective", help="read this run's actions and exposed summaries; no LLM calls")
    retrospective.add_argument("--root", required=True, type=pathlib.Path)
    retrospective.add_argument("--write", action="store_true", help="also write private-run JSON/Markdown iteration reports")
    execute = commands.add_parser("run", help="explicitly authorized paid execution; never implicitly launched by prepare")
    execute.add_argument("--root", required=True, type=pathlib.Path)
    execute.add_argument("--authorize-paid", required=True, action="store_true")
    execute.add_argument("--resume", action="store_true")
    background = commands.add_parser("supervise", help="bounded unattended process recovery")
    background.add_argument("--root", required=True, type=pathlib.Path)
    background.add_argument("--authorize-paid", required=True, action="store_true")
    background.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if args.auth_mode == "chatgpt" and not args.auth_file:
                parser.error("ChatGPT mode requires --auth-file; no implicit host-home discovery")
            prepare_with_wait(args)
        elif args.command == "usage":
            payload = U.write_report(args.root) if args.write else U.report(args.root)
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "retrospective":
            payload = V.write_report(args.root) if args.write else V.report(args.root)
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "supervise":
            supervise(args.root, args.resume)
        else:
            run(args.root, args.resume)
    except R.OperationalPause as exc:
        print(json.dumps({"status": "paused" if exc.retryable else "needs_attention", "reason": str(exc)}))
        raise SystemExit(75 if exc.retryable else 78)


if __name__ == "__main__":
    main()
