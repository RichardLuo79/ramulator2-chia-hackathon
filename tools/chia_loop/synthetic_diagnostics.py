"""Agent-callable, training-only experiments with the existing generic generator.

No canned modeling cases or prior designs are inputs. Every case runs the same
deterministic generator against the configured oracle and this run's parent DSO.
This adapter executes serially inside the runner's existing shared CPU lease.
Synthetic measurements never participate in promotion or final-test selection.
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import time

from tools.chia_loop import loop_config as L, real_core as P, real_eval as E, recovery as R
from tools.chia_loop.core import atomic_write_json
from tools.eval import synthetic as synthetic_metrics

GENERATOR = "src/ramulator/frontend/impl/memory_trace/synthetic_pattern.cpp"
AXES = {"mlp": 16, "dep_frac": 0.0, "think_time": 0, "jitter": 0, "row_run": 32,
        "bank_spread": 1, "bank_random": False, "row_random": False, "wfrac_pct": 0,
        "wb_mode": 0, "wb_batch": 1, "phase_on": 0, "phase_off": 0,
        "phase_off_random": 0, "pp_lead": 40}
RANGES = {"mlp": (1, 64), "think_time": (0, 10000), "jitter": (0, 10000),
          "row_run": (1, 1024), "bank_spread": (1, 1024), "wfrac_pct": (0, 100),
          "wb_mode": (0, 4), "wb_batch": (1, 16), "phase_on": (0, 1000000),
          "phase_off": (0, 1000000), "phase_off_random": (0, 1000000), "pp_lead": (0, 10000)}


def contract(root):
    return {"request": {"tool": "synthetic_diagnostics", "cases": ["parameter object, not a filename"], "limit": 8},
        "limits": L.load(root)["synthetic"], "defaults": AXES,
        "integer_ranges": RANGES, "dep_frac_range": [0, 1],
        "additional_parameters": {"num_requests": "reads per stream, >=1000; defaults to run profile",
            "streams": "1..8", "seed": "integer 0..4294967295 (default 1)", "share_region": "boolean, default false",
            "mode_switch": "0 disables; otherwise reads per alternating A/B block",
            "mode_b": "object of axis overrides, not a string",
            "stream_params": "one object of axis overrides per stream, not a string"},
        "interpretation": "closed-loop synthetic frontend, no CPU/cache; both clocks are DRAM cycles; diagnostic only",
        "matching": "per-stream read ordinal sorted by unique admission time; require exact population and address identity",
        "candidate": "supplied parent source/plugin only; no candidate paths or overrides accepted",
        "axes": {"row_run": "consecutive cache lines in one row before advancing",
            "bank_spread": "banks per stream; bank groups cycle fastest; bounded by resolved organization",
            "bank_random": "random rather than round-robin bank choice", "row_random": "random rather than sequential rows",
            "mlp": "maximum outstanding reads per stream", "dep_frac": "probability of dependent reads",
            "think_time": "minimum inter-read issue delay (dependencies also wait for outstanding reads)", "jitter": "random extra issue delay",
            "wfrac_pct": "percentage probability of one writeback per read; ignored by mode 0",
            "wb_mode": "0 read-only, 1 same-line read-modify-write callbacks, 2 disjoint-row writes, 3 split read/write bank regions, 4 lead writes plus optional writebacks",
            "wb_batch": "queued-write release threshold", "phase_on": "active reads per phase; 0 disables",
            "phase_off": "idle cycles per phase", "phase_off_random": "extra random idle cycles",
            "pp_lead": "mode-4 lead delay in cycles"},
        "caveats": "cold start and full drain; adaptive write timing can differ between sides; only reads are paired; not a SimpleO3 core-cycle score"}


def integer(value, lo, hi, name):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return value


def axes(value):
    if not isinstance(value, dict) or set(value) - set(AXES):
        raise ValueError("synthetic axes must be an object containing only generator axis names")
    result = {}
    for key, item in value.items():
        if key in {"bank_random", "row_random"}:
            if type(item) is not bool:
                raise ValueError(key + " must be boolean")
        elif key == "dep_frac":
            if type(item) not in {int, float} or not math.isfinite(item) or not 0 <= item <= 1:
                raise ValueError("dep_frac must be finite and in [0, 1]")
            item = float(item)
        else:
            integer(item, *RANGES[key], key)
        result[key] = item
    return result


def parameters(value, limits, layout):
    extra = {"num_requests", "streams", "seed", "share_region", "mode_switch", "mode_b", "stream_params"}
    if not isinstance(value, dict) or set(value) - set(AXES) - extra:
        raise ValueError("case must contain only generic generator parameters; no paths, models, or scripts")
    result = {**AXES, **axes({k: v for k, v in value.items() if k in AXES})}
    result["streams"] = integer(value.get("streams", 1), 1, 8, "streams")
    result["num_requests"] = integer(value.get("num_requests", limits["default_requests_per_stream"]),
        1000, limits["max_total_reads_per_case"] // result["streams"], "num_requests")
    result["seed"] = integer(value.get("seed", 1), 0, 2**32 - 1, "seed")
    result["mode_switch"] = integer(value.get("mode_switch", 0), 0, result["num_requests"], "mode_switch")
    result["share_region"] = value.get("share_region", False)
    if type(result["share_region"]) is not bool:
        raise ValueError("share_region must be boolean")
    result["mode_b"] = axes(value.get("mode_b", {}))
    streams = value.get("stream_params", [{} for _ in range(result["streams"])])
    if not isinstance(streams, list) or len(streams) != result["streams"]:
        raise ValueError("stream_params must have exactly one axis object per stream")
    result["stream_params"] = [axes(v) for v in streams]
    banks = layout["total_bank_units"] // (1 if result["share_region"] else result["streams"])
    for stream in result["stream_params"]:
        for mode in ({}, result["mode_b"]):
            effective = {**result, **stream, **mode}
            if effective["row_run"] > layout["num_cls"] or effective["bank_spread"] > banks:
                raise ValueError("row_run/bank_spread exceeds the resolved per-stream DRAM layout")
            if effective["wb_mode"] == 3 and effective["bank_spread"] > banks // 2:
                raise ValueError("split read/write regions require two disjoint bank regions per stream")
    return result


def generator_parameters(params):
    # Never pass an agent-provided mini-language, config tree, or filename.
    def encoded(values):
        return ";".join(f"{k}={int(v) if type(v) is bool else v}" for k, v in sorted(values.items()))
    return {**params, "mode_b": encoded(params["mode_b"]),
            "stream_params": "|".join(encoded(v) for v in params["stream_params"])}


def own_file(root, path):
    resolved = pathlib.Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RuntimeError("synthetic diagnostic input escapes its own run")
    return resolved


def parent_identity(root, parent):
    source, plugin = (own_file(root, parent[k]) for k in ("source_path", "plugin"))
    if (plugin.parent not in {source.parent, source.parent / "build"}
            or source.name != "atomic_controller.cpp" or plugin.name != "candidate.so"
            or not (source.parent == root / "seed" or source.is_relative_to(root / "candidates"))):
        raise RuntimeError("synthetic diagnostics require the supplied own-run candidate build")
    build = R.read_json(own_file(root, plugin.parent / "build.json"))
    if (P.sha(source.read_bytes()) != parent["sha256"] or build["source_sha256"] != parent["sha256"]
            or P.sha(own_file(root, plugin.parent / "atomic_controller.cpp").read_bytes()) != parent["sha256"]
            or build["plugin_sha256"] != P.sha(plugin.read_bytes()) or build["optimization"] != "-O3"):
        raise RuntimeError("synthetic parent source/plugin build identity changed")
    runtime_path = own_file(root, root / "runtime_manifest.json")
    runtime = R.read_json(runtime_path)
    for name, field in (("libramulator.so", "library_sha256"), ("isolated_sim", "executable_sha256")):
        if P.sha(own_file(root, root / "runtime" / name).read_bytes()) != runtime[field]:
            raise RuntimeError("synthetic trusted runtime identity changed")
    if runtime["optimization"] != "-O3" or runtime.get("synthetic_generator_sha256") != P.sha((E.REPO / GENERATOR).read_bytes()):
        raise RuntimeError("synthetic generator/runtime is not the pinned optimized implementation")
    return {"source_sha256": parent["sha256"], "plugin_sha256": build["plugin_sha256"],
        "runtime_manifest_sha256": P.sha(runtime_path.read_bytes()), "generator_sha256": runtime["synthetic_generator_sha256"],
        "adapter_sha256": P.sha(pathlib.Path(__file__).read_bytes()),
        "layout_helper_sha256": P.sha((E.REPO / "tests/utils.py").read_bytes()),
        "loop_configuration_sha256": L.identity(L.load(root))}


def verify_side(directory, identity, side):
    try:
        return _verify_side(directory, identity, side)
    except (RuntimeError, OSError, ValueError) as exc:
        raise R.OperationalPause("synthetic cached evidence failed integrity verification: " + str(exc), retryable=False) from exc


def _verify_side(directory, identity, side):
    from eval import archive_results as AR
    record = R.read_json(directory / "manifest.json")
    if record.get("identity") != identity or record.get("side") != side:
        raise RuntimeError("cached synthetic simulation identity changed")
    E.archive_completed_run(directory)
    archive = directory / "archive_manifest.json"
    AR.verify(archive)
    entries = list(R.read_json(archive)["artifacts"].values())
    if len(entries) != 1 or (entries[0]["raw_sha256"], entries[0]["raw_size"]) != (
            record["raw_trace"]["sha256"], record["raw_trace"]["size"]):
        raise RuntimeError("synthetic trace archive differs from its completed simulation")
    return record


def run_side(root, directory, parent, identity, side, dram, layout, lease_fd):
    import ramulator
    import yaml
    if (directory / "manifest.json").exists():
        return verify_side(directory, identity, side)
    staging = E.C.stage_run_directory(directory)
    trace = staging / "controller_trace.csv"
    ctrl = E.S._build_controller(ramulator, side, dram, "DDR5", trace, {})
    ctrl.addr_mapper = ramulator.addr_mapper.PassThroughAddrMapper()
    frontend = ramulator.frontend.SyntheticPattern(clock_ratio=1, **layout, **generator_parameters(identity["parameters"]))
    memory = ramulator.memory_system.GenericDRAM(clock_ratio=1, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper())
    config = {"frontend": frontend.to_config(), "memory_system": memory.to_config()}
    config_path = directory / (staging.name + ".config.json")
    atomic_write_json(config_path, config)
    runtime = root / "runtime"
    plugin = str(own_file(root, parent["plugin"])) if side == "candidate" else None
    stats_file = staging / "stats.yaml"
    try:
        evidence = E.sandbox_command([runtime / "isolated_sim", config_path, plugin or "-", staging, stats_file],
            read=[runtime, config_path, *([plugin] if plugin else [])], write=[staging], cwd=staging,
            log=directory / (staging.name + ".log"), library_path=runtime,
            cpu=L.load(root)["synthetic"]["cpu_seconds_per_simulation"], file_bytes=E.SIM_FILE_BYTES,
            lease_fd=lease_fd)
        stats = yaml.safe_load(stats_file.read_text())
        frontend_stats, controller_stats = stats["frontend"], stats["memory_system"]["controller"]
        if frontend_stats["reads_sent"] != identity["parameters"]["num_requests"] * identity["parameters"]["streams"]:
            raise RuntimeError("synthetic simulation did not complete the requested read population")
        if any(frontend_stats[f"{kind}s_sent"] != frontend_stats[f"{kind}s_completed"] for kind in ("read", "write")):
            raise RuntimeError("synthetic frontend callbacks were not fully drained")
        if side == "candidate":
            for kind in ("read", "write"):
                if controller_stats[f"num_{kind}_reqs"] != controller_stats[f"num_{kind}_reqs_served"]:
                    raise RuntimeError("synthetic candidate callbacks were not fully drained")
                if controller_stats[f"peak_inflight_{kind}s"] > 64:
                    raise RuntimeError("synthetic candidate exceeded fixed admission capacity")
            E.audit_candidate_trace(staging / "controller_trace.csv.ch0", controller_stats)
        payload = {"identity": identity, "side": side, "eligible_for_promotion": False,
            "frontend": "SyntheticPattern", "std": "DDR5", "optimization": "-O3", "clock": "DRAM cycles",
            "frontend_stats": frontend_stats, "controller_stats": controller_stats,
            "resolved_config": config, "process_evidence": evidence, "completed_at": time.time()}
        shutil.move(stats_file, directory / "stats.yaml")
        E.C.publish_trace_and_manifest(staging / "controller_trace.csv.ch0", directory / "controller_trace.csv.ch0",
            directory / "manifest.json", payload)
        staging.rmdir()
        E.archive_completed_run(directory)
    except Exception as exc:
        if not (directory / "manifest.json").exists():
            atomic_write_json(directory / "failure.json", {"complete": False, "eligible_for_metrics": False,
                "error": str(exc)[-4000:], "time": time.time()})
            E.archive_failed_run(directory)
        raise
    return verify_side(directory, identity, side)


def read_population(directory, params, record):
    return synthetic_metrics.read_population(directory / "controller_trace.csv.ch0", params,
                                             record["frontend_stats"])


def summarize(oracle, candidate, records, limit):
    return synthetic_metrics.summarize(oracle, candidate, records, limit)


def inspect(root, parent, request):
    """Persist a standalone action receipt as well as the model conversation."""
    root = pathlib.Path(root).resolve(strict=True)
    R.assert_training_open(root)
    R.check_stop(root)
    if not L.enabled(root, "synthetic_diagnostics"):
        raise ValueError("synthetic diagnostics are disabled for this run")
    binding = parent_identity(root, parent)
    calls = root / "diagnostics/synthetic_calls"
    calls.mkdir(parents=True, exist_ok=True)
    with R.exclusive_lock(calls / ".calls.lock"):
        number = len(list(calls.glob("*.json"))) + 1
        path = calls / f"call_{number:06d}.json"
        receipt = {"schema_version": 1, "run_id": root.name, "started_at": time.time(),
            "parent_source_sha256": binding["source_sha256"], "parent_plugin_sha256": binding["plugin_sha256"],
            "request": request, "status": "started", "eligible_for_promotion": False}
        atomic_write_json(path, receipt)
    try:
        result = _inspect(root, parent, request)
        receipt.update(status="completed", case_ids=[r["case_id"] for r in result["results"]],
            cached_cases=sum(r["cached"] for r in result["results"]),
            response_sha256=L.identity(result), response=result)
        return result
    except Exception as exc:
        receipt.update(status="failed", error_kind=type(exc).__name__, error=str(exc)[-4000:])
        raise
    finally:
        receipt["finished_at"] = time.time()
        atomic_write_json(path, receipt)


def _inspect(root, parent, request):
    root = pathlib.Path(root).resolve(strict=True)
    R.assert_training_open(root)
    R.check_stop(root)
    if not L.enabled(root, "synthetic_diagnostics"):
        raise ValueError("synthetic diagnostics are disabled for this run")
    if set(request) - {"tool", "cases", "limit"} or request.get("tool") != "synthetic_diagnostics":
        raise ValueError("synthetic tool accepts only cases and a bounded trace-slice limit")
    limits = L.load(root)["synthetic"]
    cases = request.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= limits["max_cases_per_call"]:
        raise ValueError("cases must be a nonempty bounded list of parameter objects")
    limit = integer(request.get("limit", min(8, limits["max_trace_rows"])), 0, limits["max_trace_rows"], "limit")
    import ramulator
    from tests.utils import extract_dram_layout
    dram = ramulator.dram.DDR5(org_preset=E.C.STD["DDR5"]["org"], timing_preset=E.C.STD["DDR5"]["timing"])
    layout = extract_dram_layout(dram)
    normalized = [parameters(case, limits, layout) for case in cases]
    binding = parent_identity(root, parent)
    base = root / "diagnostics/synthetic"
    results = []
    # Serialize calls and case reservations; simulators inherit this lock's FD.
    with R.exclusive_lock(base / ".diagnostics.lock") as lease:
        for params in normalized:
            R.assert_training_open(root)
            R.check_stop(root)
            R.check_storage(root)
            identity = {"schema_version": 1, **binding, "parameters": params, "layout": layout,
                        "standard": E.C.STD["DDR5"], "reference": E.C.REFERENCE,
                        "candidate_resources": E.C.CANDIDATE_RESOURCES}
            case_id = L.identity(identity)
            directory = base / case_id
            if not directory.exists():
                if len(list(base.glob("*/identity.json"))) >= limits["max_unique_cases_per_run"]:
                    raise ValueError("synthetic unique-case limit reached for this run")
                directory.mkdir()
                atomic_write_json(directory / "identity.json", identity)
            elif R.read_json(directory / "identity.json") != identity:
                raise RuntimeError("synthetic case identity changed")
            if (directory / "failure.json").exists():
                raise ValueError("this synthetic case previously failed; evidence retained under case " + case_id)
            cached = (directory / "result.json").exists()
            try:
                records = {side: run_side(root, directory / side, parent, identity, side, dram, layout, lease.fileno())
                           for side in ("oracle", "candidate")}
                if parent_identity(root, parent) != binding:
                    raise RuntimeError("synthetic inputs changed during simulation")
                if cached:
                    receipt = R.read_json(directory / "result_receipt.json")
                    if (receipt["result_sha256"] != P.sha((directory / "result.json").read_bytes()) or
                            receipt["manifests"] != {s: P.sha((directory / s / "manifest.json").read_bytes()) for s in records}):
                        raise R.OperationalPause("cached synthetic report or evidence changed", retryable=False)
                    result = R.read_json(directory / "result.json")
                else:
                    populations = {s: read_population(directory / s, params, records[s]) for s in records}
                    result = {"case_id": case_id, "identity": identity, "eligible_for_promotion": False,
                        "matching": "exact_per_stream_read_admission_ordinal_and_address_v1",
                        **summarize(populations["oracle"], populations["candidate"], records, limits["max_trace_rows"])}
                    atomic_write_json(directory / "result.json", result)
                    atomic_write_json(directory / "result_receipt.json", {
                        "result_sha256": P.sha((directory / "result.json").read_bytes()),
                        "manifests": {s: P.sha((directory / s / "manifest.json").read_bytes()) for s in records}})
                result["slices"] = {k: rows[:limit] for k, rows in result["slices"].items()}
                results.append({**result, "cached": cached})
            except Exception as exc:
                if not cached:
                    atomic_write_json(directory / "failure.json", {"time": time.time(), "eligible_for_promotion": False,
                        "error_kind": type(exc).__name__, "error": str(exc)[-4000:]})
                raise
    return {"tool": "synthetic_diagnostics", "purpose": "training diagnostic only, excluded from promotion",
            "case_count": len(results), "results": results}
