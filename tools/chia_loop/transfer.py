"""Post-freeze DDR5 frontend transfer; no LLM calls or candidate selection.

Evaluate the *same* selected DSO, seed, oracle and all four comparison models.
Each case owns its verified compressed traces and identity-bound completion
receipt. Results live outside training and never feed the proposer/reviewer.
"""
from __future__ import annotations

import concurrent.futures
import functools
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from tools.chia_loop import evaluation_config as W, real_eval as E, real_core as P, recovery as R
from tools.chia_loop.core import atomic_write_json
from tools.eval import config as C, artifacts as A, archive_results as AR, matchlib, metrics
from resources.champsim_bridge import cs_gentest as CS

REPO = pathlib.Path(__file__).resolve().parents[2]
TIMEOUTS = {"champsim": 3600, "gem5": 7200}
MODELS = ["oracle", "seed", "selected", *E.COMPARISONS]


def _publish_immutable(path, value):
    if path.exists():
        if R.read_json(path) != value:
            raise RuntimeError("immutable transfer identity changed: " + str(path))
    else:
        atomic_write_json(path, value)


def prepare(root):
    """Bind external inputs and runtime assets before training, without running them."""
    root = pathlib.Path(root).resolve()
    config = W.load(root)
    if not config["transfer"]:
        return None
    path = root / "transfer_inputs.json"
    if path.exists():
        return verify_inputs(root)
    runtime = root / "transfer/runtime"
    runtime.mkdir(parents=True, exist_ok=False)
    for name in ("transfer_gem5_board.py",):
        shutil.copyfile(REPO / "tools/chia_loop" / name, runtime / name)
    # Only configuration Python, not source history or a native simulation DSO.
    shutil.copytree(REPO / "python/ramulator", runtime / "python/ramulator",
                    ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__"))
    shutil.copyfile(C.MESS_CURVES["DDR5"], runtime / "mess_DDR5.txt")
    external = {}
    settings = config["transfer"].get("champsim")
    if settings:
        if set(settings["workloads"]) - set(C.CHAMPSIM_WORKLOADS):
            raise ValueError("unconfigured ChampSim workload")
        bridge = CS.bridge_source_provenance(C.CHAMPSIM_DIR)
        frontend_config = C.CHAMPSIM_DIR / "champsim_config.json"
        generated_config = C.CHAMPSIM_DIR / ".csconfig/core_inst.cc.inc"
        frontend_settings = R.read_json(frontend_config)
        clock = re.search(r"DRAM\{\s*champsim::chrono::picoseconds\{\d+\},\s*champsim::chrono::picoseconds\{(\d+)\}",
                          generated_config.read_text())
        if (frontend_settings.get("num_cores") != 1 or frontend_settings.get("block_size") != 64 or
                clock is None or int(clock[1]) != 625):
            raise RuntimeError("ChampSim DDR5 adapter requires one core, 64B lines and a 625ps MC tick")
        options = C.CHAMPSIM_DIR / "global.options"
        flags = re.findall(r"-O(?:[0-3sgz]|fast)\b", options.read_text())
        if not flags or flags[-1] != "-O3":
            raise RuntimeError("ChampSim build recipe must use -O3")
        format_source = REPO / "tools/chia_loop/champsim_trace_format.cpp"
        probe = runtime / "trace_format_probe"
        E.command(["/usr/bin/g++", "-O3", "-DNDEBUG", "-std=c++17", "-I" + str(C.CHAMPSIM_DIR / "inc"),
                   format_source, "-o", probe], root / "logs/transfer_trace_format_build.log")
        record_bytes = int(subprocess.check_output([str(probe)], text=True))
        if record_bytes <= 0:
            raise RuntimeError("invalid ChampSim instruction format size")
        inventory = {}
        for workload in settings["workloads"]:
            trace = C.CHAMPSIM_TRACES / (workload + ".champsimtrace.xz")
            listing = subprocess.check_output(["xz", "--robot", "--list", str(trace)], text=True)
            totals = next(line.split("\t") for line in listing.splitlines() if line.startswith("totals\t"))
            uncompressed = int(totals[4])
            if uncompressed % record_bytes:
                raise RuntimeError("ChampSim input is not an integral instruction trace: " + workload)
            instructions = uncompressed // record_bytes
            if instructions < settings["warmup_instructions"] + settings["roi_instructions"]:
                raise RuntimeError("ChampSim window would wrap its input: " + workload)
            inventory[workload] = {"instructions": instructions, "record_bytes": record_bytes,
                                   "uncompressed_bytes": uncompressed, "wraps": False}
        external["champsim"] = {
            "binary": C.file_provenance(C.CHAMPSIM_BIN),
            "build_options": C.file_provenance(options),
            "frontend_config": C.file_provenance(frontend_config),
            "generated_config": C.file_provenance(generated_config),
            "clocking": {"mc_period_ps": 625, "ramulator_ticks_per_8": 12,
                         "mean_controller_period_ps": 625 * 8 / 12},
            "instruction_format": C.file_provenance(C.CHAMPSIM_DIR / "inc/trace_instruction.h"),
            "instruction_inventory": inventory,
            "bridge": {name: C.file_provenance(row["installed_path"]) for name, row in bridge.items()},
            "workloads": {w: C.file_provenance(C.CHAMPSIM_TRACES / (w + ".champsimtrace.xz"))
                          for w in settings["workloads"]},
        }
    settings = config["transfer"].get("gem5")
    if settings:
        from tools.eval.gem5.run_gem5 import _gem5_source_inputs
        if set(settings["workloads"]) - set(C.GEM5_BENCH):
            raise ValueError("unconfigured gem5 benchmark")
        binary = pathlib.Path(C.GEM5_BIN).resolve()
        if binary.name not in {"gem5.opt", "gem5.fast"}:
            raise RuntimeError("gem5 transfer requires an optimized .opt/.fast binary")
        sources = _gem5_source_inputs(binary)
        for name, fingerprint in sources.items():
            canonical = REPO / "resources/gem5_wrappers" / pathlib.Path(fingerprint["path"]).name
            # The installed SConscript additionally records local build paths.
            if canonical.is_file() and canonical.name != "SConscript":
                if P.sha(canonical.read_bytes()) != fingerprint["sha256"]:
                    raise RuntimeError("installed gem5 bridge differs: " + name)
        recipe = next(p / "src/SConscript" for p in binary.parents if (p / "SConstruct").exists())
        if "['-O3']" not in recipe.read_text() and "['-O3'," not in recipe.read_text():
            raise RuntimeError("cannot verify gem5 optimized build recipe")
        external["gem5"] = {
            "binary": C.file_provenance(binary), "build_recipe": C.file_provenance(recipe),
            "bridge": sources,
            "workloads": {w: C.file_provenance(C.GEM5_BENCH_DIR / w) for w in settings["workloads"]},
            "arguments": {w: shlex.split(C.GEM5_BENCH[w]) for w in settings["workloads"]},
        }
    files = {str(p.relative_to(root)): C.file_provenance(p) for p in runtime.rglob("*") if p.is_file()}
    payload = {"schema_version": 1, "standard": "DDR5", "profile": config,
               "profile_sha256": P.sha((root / "evaluation_config.json").read_bytes()),
               "assets": files, "external": external,
               "external_build_attestation": "optimized build recipe and existing executable hashes; not a fresh frontend rebuild"}
    atomic_write_json(path, payload)
    return payload


def verify_inputs(root):
    root = pathlib.Path(root)
    inputs = R.read_json(root / "transfer_inputs.json")
    if inputs["profile"] != W.load(root) or inputs["profile_sha256"] != P.sha((root / "evaluation_config.json").read_bytes()):
        raise RuntimeError("transfer profile changed after preparation")
    def verify(value):
        if isinstance(value, dict):
            if {"path", "size", "sha256"} <= set(value):
                if not C.file_provenance_matches(value["path"], value):
                    raise RuntimeError("transfer input changed: " + value["path"])
            else:
                for child in value.values():
                    verify(child)
    verify(inputs["assets"])
    verify(inputs["external"])
    return inputs


def selection(root, state):
    """Never infer permission to test from an incumbent or a source path alone."""
    root = pathlib.Path(root).resolve()
    if state.get("status") != "frozen" or state.get("termination") not in {
            "iteration_limit", "no_change", "limit_stop", "budget_stop"}:
        raise RuntimeError("transfer requires scientifically terminated frozen selection")
    frozen = R.read_json(root / "selection_frozen.json")
    selected = state["selected"]
    if (frozen["run_id"] != root.name or frozen["source_sha256"] != selected["sha256"] or
            frozen["incumbent"] != state["incumbent"] or frozen["frozen_at"] != state["frozen_at"]):
        raise RuntimeError("transfer selection does not match freeze receipt")
    for label, source, plugin, digest in (
            ("selected", pathlib.Path(selected["source_path"]), pathlib.Path(selected["plugin"]), selected["sha256"]),
            ("seed", root / "seed/atomic_controller.cpp", root / "seed/candidate.so", None)):
        if not source.resolve().is_relative_to(root) or not plugin.resolve().is_relative_to(root):
            raise RuntimeError("transfer candidate must belong to this run")
        build = R.read_json(plugin.parent / "build.json")
        actual = P.sha(source.read_bytes())
        if ((digest is not None and actual != digest) or build["source_sha256"] != actual or
                build["plugin_sha256"] != P.sha(plugin.read_bytes()) or build["optimization"] != "-O3"):
            raise RuntimeError("transfer source/plugin/build binding failed: " + label)
    runtime = R.read_json(root / "runtime_manifest.json")
    if runtime["optimization"] != "-O3" or runtime["library_sha256"] != P.sha((root / "runtime/libramulator.so").read_bytes()):
        raise RuntimeError("transfer runtime differs from training runtime")
    return {**frozen, "plugin_sha256": P.sha(pathlib.Path(selected["plugin"]).read_bytes()),
            "runtime_library_sha256": runtime["library_sha256"]}


def memory_config(root, model, trace, frontend):
    import ramulator
    dram = ramulator.dram.DDR5(org_preset=C.STD["DDR5"]["org"], timing_preset=C.STD["DDR5"]["timing"])
    controller = E.S._build_controller(ramulator, "candidate" if model in {"seed", "selected"} else model,
                                       dram, "DDR5", trace, {})
    memory = ramulator.memory_system.GenericDRAM(clock_ratio=1 if frontend == "champsim" else 3,
        controllers=[controller], channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    config = {"frontend": {"impl": "External", "clock_ratio": 1}, "memory_system": memory.to_config()}
    if model == "mess":
        config["memory_system"]["controllers"][0]["curve_path"] = str(pathlib.Path(root) / "transfer/runtime/mess_DDR5.txt")
    return config


def run_case(root, inputs, state, frontend, workload, model, frozen):
    root = pathlib.Path(root)
    out = root / "transfer" / frontend / "DDR5" / workload / model
    out.mkdir(parents=True, exist_ok=True)
    runtime = root / "runtime"
    plugin = pathlib.Path(state["selected"]["plugin"]) if model == "selected" else root / "seed/candidate.so" if model == "seed" else None
    external = inputs["external"][frontend]
    identity = {"frontend": frontend, "std": "DDR5", "workload": workload, "model": model,
        "selection": frozen, "transfer_inputs_sha256": P.sha((root / "transfer_inputs.json").read_bytes()),
        "plugin_sha256": P.sha(plugin.read_bytes()) if plugin else None,
        "driver_sha256": P.sha(pathlib.Path(__file__).read_bytes()),
        "launcher_sha256": P.sha((REPO / "tools/chia_loop/transfer_sandbox.py").read_bytes())}
    with R.exclusive_lock(out / ".evaluation.lock") as lease:
        R.check_stop(root)
        R.check_storage(root)
        manifest = out / "manifest.json"
        if manifest.exists():
            previous = R.read_json(manifest)
            if previous["identity"] != identity or not A.raw_provenance_matches(out / "trace.csv.ch0", previous["raw_trace"]):
                raise RuntimeError("cached transfer case changed: " + str(out))
            E.archive_completed_run(out)
            AR.verify(out / "archive_manifest.json")
            return previous
        # Preserve unfinished attempts and never overwrite their raw evidence.
        (out / "logs").mkdir(exist_ok=True)
        staging = pathlib.Path(tempfile.mkdtemp(prefix="attempt-", dir=out / "logs"))
        config_path = staging / "memory.json"
        config = memory_config(root, model, staging / "trace.csv", frontend)
        atomic_write_json(config_path, config)
        binary, workload_path = external["binary"]["path"], external["workloads"][workload]["path"]
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "TMPDIR": str(staging),
               "LD_LIBRARY_PATH": str(runtime), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(staging)}
        read = [*E.SYSTEM_READ, binary, workload_path, staging, runtime / "libramulator.so", root / "transfer/runtime"]
        if plugin:
            env["LD_PRELOAD"] = str(plugin)
            read.append(plugin)
        if frontend == "champsim":
            settings = inputs["profile"]["transfer"][frontend]
            command = [binary, "-w", str(settings["warmup_instructions"]), "-i", str(settings["roi_instructions"]), workload_path]
            env.update(RAMULATOR_CONFIG=str(config_path), RAMULATOR_TICKS_PER_8="12")
        else:
            command = [binary, "--outdir=" + str(staging), str(root / "transfer/runtime/transfer_gem5_board.py")]
            env.update(CHIA_MEMORY_CONFIG=str(config_path), CHIA_BENCHMARK=workload_path,
                CHIA_BENCHMARK_ARGS=json.dumps(external["arguments"][workload]),
                CHIA_PYTHON=str(root / "transfer/runtime/python"), CHIA_EXIT_RECEIPT=str(staging / "exit.json"))
            # gem5's Python support reads local executable metadata, not host credentials.
            read += [p for p in ("/proc/self/maps", "/proc/meminfo", "/proc/cpuinfo", "/dev/urandom") if pathlib.Path(p).exists()]
        # Check dynamic-loader resolution with no candidate preload or inherited environment.
        resolved, _ = CS.verify_library_resolution(pathlib.Path(binary), runtime, config_path, 12)
        if resolved != (runtime / "libramulator.so").resolve():
            raise RuntimeError("wrong transfer library")
        policy_path = staging / "launch.json"
        atomic_write_json(policy_path, {"cwd": str(staging), "command": command, "environment": env,
            "read": [str(p) for p in read], "write": [str(staging)], "cpu_seconds": TIMEOUTS[frontend]})
        start = time.time()
        try:
            evidence = E.command([sys.executable, "-m", "tools.chia_loop.transfer_sandbox", policy_path],
                      staging / "simulation.log", env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO)},
                      timeout=TIMEOUTS[frontend] + 120, pass_fds=(lease.fileno(),))
            log = (staging / "simulation.log").read_text()
            if frontend == "champsim":
                matches = re.findall(r"Simulation finished CPU 0 instructions: (\d+) cycles: (\d+)", log)
                if len(matches) != 1:
                    raise RuntimeError("ChampSim did not complete one measured ROI")
                instructions, cycles = map(int, matches[0])
                if instructions < settings["roi_instructions"]:
                    raise RuntimeError("ChampSim ROI was truncated")
            else:
                exit_info = R.read_json(staging / "exit.json")
                if exit_info != {"cause": "exiting with last active thread context", "code": 0}:
                    raise RuntimeError("gem5 workload did not exit normally")
                stats = (staging / "stats.txt").read_text()
                def value(name):
                    values = re.findall(r"^" + name + r"\s+(\d+)", stats, re.M)
                    if len(values) != 1 or int(values[0]) <= 0:
                        raise RuntimeError("incomplete gem5 stat: " + name)
                    return int(values[0])
                cycles, instructions = value("simTicks"), value("simInsts")
            if cycles <= 0 or instructions <= 0:
                raise RuntimeError("empty transfer execution")
            record = {"identity": identity, "frontend": frontend, "std": "DDR5", "model": model,
                "workload": workload, "cycles_or_ticks": cycles, "instructions": instructions,
                "wall_seconds": time.time() - start, "resolved_config": config,
                "process_evidence": evidence,
                "request_trace_scope": frontend + "_dram_controller_lifecycle",
                "request_trace_timebase": "ramulator_controller_cycles", "optimization": "-O3",
                "raw_scope": "controller admissions/completions, not all logical frontend reads",
                "trace_roi": "simulation lifecycle (not ROI-only); warmup may bypass DRAM in ChampSim",
                "input_access_after_candidate_load": "specific_frontend_inputs_readable; unrelated_files_and_network_denied"}
            C.publish_trace_and_manifest(staging / "trace.csv.ch0", out / "trace.csv.ch0", manifest, record)
            AR.compress([out / "trace.csv.ch0"], out / "archive_manifest.json", 3)
            AR.verify(out / "archive_manifest.json")
            return R.read_json(manifest)
        except Exception as exc:
            atomic_write_json(staging / "failure.json", {"error": str(exc)[-4000:], "eligible_for_metrics": False})
            partial = staging / "trace.csv.ch0"
            if partial.exists():
                AR.compress([partial], staging / "failed_archive_manifest.json", 3, allow_incomplete=True)
            raise


def request_metrics(oracle, model, destination, frontend):
    """Exact identity/physical pairing; incomplete coverage is diagnostic only."""
    import numpy as np
    matcher = functools.partial(matchlib.match, champsim_filter_physical_mismatches=True) if frontend == "champsim" else matchlib.match
    result = matcher(oracle, model)
    # Reuse the established normalization/cache schema, plus signed extremes.
    payload = metrics.load_or_compute_request(destination, model, oracle, force=True,
        matcher=lambda *_: result, request_trace_scope=frontend + "_dram_controller_lifecycle",
        request_trace_timebase="ramulator_controller_cycles")
    dv = result["dv"]
    L = payload["oracle_read_mean_latency"]
    exact = (payload["match_mode"] == "stable_id" and payload["matched"] == payload["n_oracle"] ==
             payload["n_model"] == payload["stable_eligible_o"] == payload["stable_eligible_m"] and
             not payload.get("address_mismatch_pairs"))
    payload.update(most_negative_cycles=float(np.min(dv)), most_positive_cycles=float(np.max(dv)),
        most_negative_over_L=float(np.min(dv)) / L, most_positive_over_L=float(np.max(dv)) / L,
        request_metric_eligible=exact, request_metric_status="exact" if exact else "diagnostic_incomplete_pairing",
        eligibility_rule="exact bidirectional stable-ID coverage of all reads and physical-address agreement")
    atomic_write_json(destination, payload)
    return payload


def summarize(root, frontend, workloads, config):
    root = pathlib.Path(root)
    models = {}
    for model in MODELS[1:]:
        rows = {}
        for workload in workloads:
            base = root / "transfer" / frontend / "DDR5" / workload
            oracle, candidate = (R.read_json(base / label / "manifest.json") for label in ("oracle", model))
            if frontend == "gem5" and candidate["instructions"] != oracle["instructions"]:
                raise RuntimeError("gem5 instruction populations differ: " + workload)
            signed = 100 * (candidate["cycles_or_ticks"] - oracle["cycles_or_ticks"]) / oracle["cycles_or_ticks"]
            request = request_metrics(base / "oracle/trace.csv.ch0", base / model / "trace.csv.ch0",
                                      base / model / "request.json", frontend)
            rows[workload] = {"core_signed_error_pct": signed, "core_abs_error_pct": abs(signed),
                "request": request, "family_group": W.transfer_group(config, workload),
                "oracle_instructions": oracle["instructions"], "model_instructions": candidate["instructions"]}
        eligible = all(row["request"]["request_metric_eligible"] for row in rows.values())
        request = {"request_macro_mae_over_L": sum(r["request"]["mae"] for r in rows.values()) / len(rows),
            "request_macro_abs_signed_drift_over_L": sum(abs(r["request"]["sgn"]) for r in rows.values()) / len(rows),
            "request_worst_paired_p99_over_L": max(r["request"]["tail"] for r in rows.values()),
            "request_most_negative_over_L": min(r["request"]["most_negative_over_L"] for r in rows.values()),
            "request_most_positive_over_L": max(r["request"]["most_positive_over_L"] for r in rows.values())}
        models[model] = {"aggregate": {"n_workloads": len(rows),
            "cycle_macro_mae_pct": sum(r["core_abs_error_pct"] for r in rows.values()) / len(rows),
            "cycle_worst_core_abs_pct": max(r["core_abs_error_pct"] for r in rows.values()),
            "request_metric_eligible": eligible, "request_headline": request if eligible else None,
            "request_diagnostic": request, "minimum_coverage_oracle": min(r["request"]["cov_o"] for r in rows.values()),
            "minimum_coverage_model": min(r["request"]["cov_m"] for r in rows.values())}, "per_workload": rows}
    report = {"schema_version": 1, "frontend": frontend, "std": "DDR5", "models": models,
        "selection_use": "none; frozen-source transfer only", "cross_frontend_pooling": False,
        "core_metric": "ROI core cycles" if frontend == "champsim" else "whole-program simTicks at fixed CPU frequency",
        "request_metric": "matched controller read latency difference / full-oracle mean controller read latency",
        "request_coverage_rule": "exact bidirectional stable IDs required for headline; otherwise diagnostic only"}
    atomic_write_json(root / "transfer/reports" / (frontend + ".json"), report)
    from tools.chia_loop.transfer_report import write
    write(root, report)
    return report


def run(root, state, *, workers=2):
    root = pathlib.Path(root).resolve()
    config = W.load(root)
    if not config["transfer"]:
        return {}
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("transfer workers must be in [1, 12]")
    # This stage also has its own lock, in addition to each simulator case.
    with R.exclusive_lock(root / "transfer/.transfer.lock"):
        return _run(root, state, config, workers)


def _run(root, state, config, workers):
    frozen = selection(root, state)
    inputs = verify_inputs(root)
    _publish_immutable(root / "transfer/selection.json", frozen)
    status = {"status": "running", "selection": frozen, "started_at": time.time(),
        "verified_cases": 0, "total_cases": sum(len(s["workloads"]) * len(MODELS) for s in config["transfer"].values())}
    atomic_write_json(root / "transfer/status.json", status)
    reports = {}
    try:
        for frontend, settings in config["transfer"].items():
            status.update(frontend=frontend, phase="simulating")
            atomic_write_json(root / "transfer/status.json", status)
            jobs = [(w, m) for w in settings["workloads"] for m in MODELS]
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_case, root, inputs, state, frontend, w, m, frozen) for w, m in jobs]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    status["verified_cases"] += 1
                    atomic_write_json(root / "transfer/status.json", status)
            status["phase"] = "matching_and_reporting"
            atomic_write_json(root / "transfer/status.json", status)
            reports[frontend] = summarize(root, frontend, settings["workloads"], config)
        verify_inputs(root)
        if selection(root, state) != frozen:
            raise RuntimeError("selection changed during transfer")
        atomic_write_json(root / "transfer/status.json", {**status, "status": "completed", "phase": "completed", "selection": frozen,
            "finished_at": time.time(), "report_sha256": {f: P.sha((root / "transfer/reports" / (f + ".json")).read_bytes()) for f in reports}})
        return reports
    except BaseException as exc:
        atomic_write_json(root / "transfer/status.json", {**status, "status": "failed", "selection": frozen,
            "error": str(exc)[-4000:], "training_must_not_resume": True})
        raise
