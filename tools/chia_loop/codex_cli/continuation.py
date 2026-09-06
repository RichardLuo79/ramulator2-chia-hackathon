"""Explicit training-only continuation of a completed, same-effort Astra run.

The predecessor remains frozen. Its test results never become agent input.
Rebuild and replay every inherited design against fresh, identical training
inputs before allowing generation. Earlier interactions stay in their original
archive; financial carryover includes their known and unknown usage exactly once.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import fcntl
import importlib.metadata
import json
import pathlib
import shutil
import sys
import time

from tools.chia_loop import compliance as C, real_core as P, real_eval as E, recovery as R
from tools.chia_loop.core import atomic_write_json
from . import transport as T, usage as U

# Only authorization/accounting/continuation plumbing may differ. In particular,
# the evaluator, prompts, metrics, source visibility and compliance rubric cannot.
UPGRADE_FILES = {"tools/chia_loop/codex_cli/" + name for name in (
    "runner.py", "transport.py", "budget.py", "usage.py", "retrospective.py", "queue.py")}


def own_file(root, path):
    path = pathlib.Path(path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError("predecessor evidence escapes its own run")
    return path


@contextlib.contextmanager
def quiescent(root):
    """Read existing lock files; never clear a predecessor's lifecycle guards."""
    with contextlib.ExitStack() as stack:
        for name in (".supervisor.lock", ".runner.lock"):
            stream = stack.enter_context((root / name).open("r"))
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def completed_prefix(state, maximum_iterations):
    history = state["history"]
    completed = []
    for row in history:
        if row["status"] != "evaluated":
            break
        if row["iteration"] != len(completed) + 1:
            raise RuntimeError("nonconsecutive predecessor iteration history")
        completed.append(copy.deepcopy(row))
    tail = history[len(completed):]
    if tail and (len(tail) != 1 or tail[0]["status"] != "budget_stop"
                 or tail[0]["iteration"] != len(completed) + 1):
        raise RuntimeError("only a trailing unevaluated budget stop may be retired")
    if not completed or type(maximum_iterations) is not int or maximum_iterations <= len(completed):
        raise ValueError("continuation ceiling must exceed the completed design count")
    expected = {"seed", *(f"astra_{h['iteration']:03d}" for h in completed)}
    if set(state["candidates"]) != expected or state["incumbent"] not in expected:
        raise RuntimeError("candidate archive disagrees with completed training history")
    return completed, copy.deepcopy(tail)


def accuracy(metrics):
    """Exclude measured host runtime, never accuracy, coverage, or populations."""
    value = copy.deepcopy(metrics)
    for key in ("wall_s", "oracle_wall_s", "speedup_vs_oracle"):
        value["aggregate"].pop(key, None)
    for workload in value.get("per_workload", {}).values():
        workload.pop("wall_s", None)
    return value


def predecessor(source, effort, maximum_iterations):
    from . import runner as B
    root = pathlib.Path(source).resolve(strict=True)
    with quiescent(root):
        manifest = R.read_json(root / "run_manifest.json")
        config = R.read_json(root / "codex_config.json")
        supervisor = R.read_json(root / "supervisor_state.json")
        state = R.read_json(root / "state.json")
        if (manifest.get("run_id") != root.name or config.get("run_id") != root.name
                or state.get("run_id") != root.name or supervisor.get("run_id") != root.name
                or manifest.get("configuration") != config or config.get("model") != T.MODEL
                or config.get("effort") != effort or config.get("auth_mode") != "chatgpt"):
            raise RuntimeError("continuation requires its own same-model, same-effort ChatGPT predecessor")
        if (manifest.get("status") != "completed" or supervisor.get("status") != "completed"
                or not manifest.get("archives_verified") or not manifest.get("aux_archives_verified")
                or state.get("status") != "frozen"
                or state.get("termination") not in {"iteration_limit", "budget_stop"}):
            raise RuntimeError("continuation requires a completed and archived predecessor")
        if manifest.get("prior_design_exposed") is not False or config.get("human_modeling_hints") is not False:
            raise RuntimeError("cannot import a predecessor exposed to prior design or modeling hints")
        if manifest.get("python_version") != sys.version or manifest.get("usage_tariff") != U.TARIFF:
            raise RuntimeError("predecessor runtime or tariff differs")
        for name, version in manifest["versions"].items():
            if importlib.metadata.version(name) != version:
                raise RuntimeError("predecessor dependency differs: " + name)
        for name, digest in manifest["protocol_hashes"].items():
            if P.sha(own_file(root, root / "protocol" / name).read_bytes()) != digest:
                raise RuntimeError("predecessor frozen protocol changed: " + name)
            if name not in UPGRADE_FILES and P.sha((B.REPO / name).read_bytes()) != digest:
                raise RuntimeError("scientific protocol changed: " + name)
        for name, digest in manifest["input_hashes"].items():
            if P.sha(own_file(root, root / name).read_bytes()) != digest:
                raise RuntimeError("predecessor frozen input changed: " + name)
        history, retired = completed_prefix(state, maximum_iterations)
        for identity, candidate in state["candidates"].items():
            source_path = own_file(root, candidate["source_path"])
            plugin = own_file(root, candidate["plugin"])
            build = R.read_json(plugin.parent / "build.json")
            if (P.sha(source_path.read_bytes()) != candidate["sha256"]
                    or build["source_sha256"] != candidate["sha256"]
                    or P.sha(plugin.read_bytes()) != build["plugin_sha256"] or build["optimization"] != "-O3"):
                raise RuntimeError("predecessor candidate evidence changed: " + identity)
            report = R.read_json(root / "training/reports" / (candidate["label"] + ".json"))
            receipt = R.read_json(root / "training/report_receipts" / (candidate["label"] + ".json"))
            if (report["models"][candidate["label"]] != candidate["metrics"]
                    or receipt["report_sha256"] != P.sha((root / "training/reports" / (candidate["label"] + ".json")).read_bytes())):
                raise RuntimeError("predecessor training report changed: " + identity)
            for name, digest in receipt["identity"]["manifests"].items():
                if not pathlib.PurePosixPath(name).is_relative_to("training"):
                    raise RuntimeError("inherited score is not training-only")
                if P.sha(own_file(root, root / name).read_bytes()) != digest:
                    raise RuntimeError("predecessor training evidence changed")
        usage = U.report(root)
        if usage["audit_issues"]:
            raise RuntimeError("predecessor usage audit failed")
        return {"source_run_id": root.name, "source_root": str(root),
                "source_manifest_sha256": P.sha((root / "run_manifest.json").read_bytes()),
                "source_state_sha256": P.sha((root / "state.json").read_bytes()),
                "source_ledger_sha256": P.sha((root / "ledger.json").read_bytes()),
                "history": history, "retired_unevaluated_stop": retired,
                "candidates": state["candidates"], "incumbent": state["incumbent"]}


def import_training(root, source, effort, maximum_iterations, cpus):
    root, source = pathlib.Path(root).resolve(), pathlib.Path(source).resolve(strict=True)
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("a continuation needs its own separate directory")
    R.assert_training_open(root)
    if (root / "state.json").exists() or (root / "continuation.json").exists():
        raise RuntimeError("cannot replace an existing continuation checkpoint")
    inherited = predecessor(source, effort, maximum_iterations)
    # The new prepare pass checks full-window parity and process isolation. This
    # pass also requires exact scientific inputs, runtime library and seed bytes.
    for name in ("window_policy.json", "seed/atomic_controller.cpp", "runtime/codex"):
        if P.sha((root / name).read_bytes()) != P.sha((source / name).read_bytes()):
            raise RuntimeError("continuation preparation differs: " + name)
    old_runtime, new_runtime = (R.read_json(p / "runtime_manifest.json") for p in (source, root))
    for key in ("object_sha256", "binding_sha256", "interleave_body_sha256", "driver_sha256",
                "library_sha256", "optimization", "export_hashes"):
        if old_runtime[key] != new_runtime[key]:
            raise RuntimeError("continuation trusted runtime differs: " + key)
    old_inputs, new_inputs = (R.read_json(p / "input_inventory.json") for p in (source, root))
    if old_inputs != new_inputs:
        raise RuntimeError("continuation trace inventory differs")
    seed = inherited["candidates"]["seed"]
    new_seed = R.read_json(root / "training/reports/seed.json")["models"]["seed"]
    if accuracy(seed["metrics"]) != accuracy(new_seed):
        raise RuntimeError("continuation seed accuracy differs")
    seed = {**seed, "source_path": str(root / "seed/atomic_controller.cpp"), "plugin": str(root / "seed/candidate.so")}

    def restore(item):
        identity, old = item
        R.check_stop(root)
        R.check_storage(root)
        directory = root / "candidates" / identity / "inherited"
        original = own_file(source, old["source_path"])
        text = original.read_text()
        P.validate_source((root / "seed/atomic_controller.cpp").read_text(), text)
        approved = R.read_json(original.parent / "review.json")
        proposal = R.read_json(original.parent / "proposal.json")
        review_input = C.review_input(text, proposal)
        if (approved.get("approved") is not True or approved.get("human_intervention") is not False
                or C.validate_decision(approved, old["sha256"]) != "pass"
                or approved.get("review_input_sha256") != P.sha(json.dumps(review_input, sort_keys=True))
                or R.read_json(original.parent / "review_input.json") != review_input
                or approved.get("rubric_sha256") != P.sha((root / "prompts/compliance_v1.md").read_bytes())):
            raise RuntimeError("inherited source lacks its bound automatic compliance approval")
        directory.mkdir(parents=True, exist_ok=False)
        # Byte copies are evidence, not a new proposal/review/model intervention.
        for name in ("atomic_controller.cpp", "proposal.json", "review_input.json", "review.json", "static_checks.json"):
            shutil.copyfile(own_file(source, original.parent / name), directory / name)
        plugin = E.compile_candidate(root, text, directory / "build")
        measured = E.evaluate(root, ["candidate"], plugin=plugin, label=old["label"], split="training", workers=2)[old["label"]]
        if accuracy(measured) != accuracy(old["metrics"]):
            raise RuntimeError("inherited design did not reproduce exact training accuracy: " + identity)
        candidate = {**old, "source_path": str(directory / "atomic_controller.cpp"), "plugin": plugin}
        atomic_write_json(directory / "reproduction.json", {"source_run_id": source.name,
            "source_sha256": old["sha256"], "training_accuracy_identical": True, "metrics": measured,
            "new_model_calls": 0, "source_modified": False, "prior_review_reused": True})
        return identity, candidate

    items = [(key, value) for key, value in inherited["candidates"].items() if key != "seed"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, cpus // 2)) as pool:
        candidates = {"seed": seed, **dict(pool.map(restore, items))}
    # Recheck the frozen predecessor after the potentially lengthy replay.
    for name, key in (("run_manifest.json", "source_manifest_sha256"),
                      ("state.json", "source_state_sha256"), ("ledger.json", "source_ledger_sha256")):
        if P.sha((source / name).read_bytes()) != inherited[key]:
            raise RuntimeError("predecessor changed during import")
    state = {"run_id": root.name, "status": "running", "incumbent": inherited["incumbent"],
             "candidates": candidates, "history": inherited["history"],
             "inherited_evaluated_designs": len(inherited["history"])}
    receipt = {**inherited, "candidates": candidates, "schema_version": 1, "run_id": root.name,
        "model": T.MODEL, "effort": effort, "created_at": time.time(),
        "maximum_iterations": maximum_iterations, "completed_designs": len(inherited["history"]),
        "remaining_designs": maximum_iterations - len(inherited["history"]),
        "all_inherited_training_accuracy_reproduced": True,
        "held_out_results_imported": False, "other_effort_history_imported": False,
        "feasibility_design_imported": False, "human_operational_intervention": True,
        "human_modeling_hints": False, "original_interactions_preserved_in_source_run": True,
        "evaluation_interpretation": "exploratory extension after operator test exposure",
        "checkpoint_policy": "resume completed training prefix; incomplete budget-stopped draft stays archived in predecessor"}
    atomic_write_json(root / "continuation.json", receipt)
    atomic_write_json(root / "state.json", state)
    return receipt


def verify_import(root, config):
    path = root / "continuation.json"
    if not path.exists() or P.sha(path.read_bytes()) != config.get("continuation_sha256"):
        raise RuntimeError("continuation checkpoint is missing or changed")
    receipt, state = R.read_json(path), R.read_json(root / "state.json")
    if (receipt["run_id"] != root.name or receipt["model"] != config["model"]
            or receipt["effort"] != config["effort"]
            or receipt["maximum_iterations"] != config["maximum_iterations"]
            or receipt.get("held_out_results_imported") is not False
            or receipt.get("all_inherited_training_accuracy_reproduced") is not True
            or state["history"][:receipt["completed_designs"]] != receipt["history"]
            or any(state["candidates"].get(k) != v for k, v in receipt["candidates"].items())):
        raise RuntimeError("continuation provenance or inherited training state changed")
