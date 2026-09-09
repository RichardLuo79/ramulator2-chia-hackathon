"""SimpleO3 closed-loop runner: raw-first output.

For every (workload, model) this writes
eval_out/simpleo3/<std>/<wl>/<model>/:
  manifest.json          - git revision, full config echo, RAW per-core cycles,
                           controller stats, trace scopes and checksums
  trace.csv.ch0          - scored logical SimpleO3 LLC-boundary request trace,
                           including an LLC hit/merge/owner audit classifier
  controller_trace.csv.ch0 - diagnostic DRAM-controller lifecycle trace

No aggregation happens here; postprocess.py owns all metric computation.
Models are ``oracle``, ``candidate``, ``fixedlat``, ``md1``, ``wmg1``, and
``mess``. Usage: python tools/eval/run_simpleo3.py --std DDR5 [--force]
"""
import argparse
import csv
import json
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from tools.eval import config as C  # noqa: E402, I001
from tools.eval import artifacts as A  # noqa: E402, I001


from tools.eval.simpleo3 import (  # noqa: E402, F401
    ALL_MODELS, CONTROLLER_TRACE_SCOPE, FIXED_ISSUE_ROI, LOGICAL_PATH_CODES,
    LOGICAL_TRACE_COLUMNS, REQUEST_TRACE_SCOPE, REQUEST_TRACE_TIMEBASE, SAFE_LABEL,
    _build_controller, _controller_config, _validate_fixed_roi_frontend_stats,
    _validate_logical_request_trace,
)


def _publish_simpleo3_traces(staged_request_trace, request_trace_path,
                             staged_controller_trace, controller_trace_path,
                             manifest_path, payload):
    """Publish both raw traces, then the checksum-bearing commit manifest.

    A filesystem cannot atomically rename two files together.  The absent
    manifest is therefore the transaction marker: both staged artifacts are
    validated before either replacement, and the manifest is published only
    after both replacements and hashes have completed.
    """
    staged_request_trace = pathlib.Path(staged_request_trace)
    request_trace_path = pathlib.Path(request_trace_path)
    staged_controller_trace = pathlib.Path(staged_controller_trace)
    controller_trace_path = pathlib.Path(controller_trace_path)
    manifest_path = pathlib.Path(manifest_path)
    _validate_logical_request_trace(staged_request_trace)
    if (not staged_controller_trace.is_file() or
            staged_controller_trace.stat().st_size == 0):
        raise RuntimeError(
            "simulation produced no non-empty controller trace: "
            f"{staged_controller_trace}"
        )
    manifest_path.unlink(missing_ok=True)
    os.replace(staged_controller_trace, controller_trace_path)
    os.replace(staged_request_trace, request_trace_path)
    raw_trace = C.file_provenance(request_trace_path)
    controller_trace = C.file_provenance(controller_trace_path)
    C.atomic_write_json(manifest_path, {
        **payload,
        "raw_trace": raw_trace,
        "controller_trace": controller_trace,
    })
    return raw_trace, controller_trace


def _simpleo3_expectations(wl, std, model, provenance, trace_inputs,
                           insts_per_core, candidate_overrides, output_label=None):
    traces = C.MIXES.get(wl, [wl])
    controller_config = _controller_config(model, candidate_overrides)
    external_inputs = list(trace_inputs)
    if model == "mess":
        try:
            external_inputs.append(C.file_provenance(C.MESS_CURVES[std]))
        except KeyError as exc:
            raise ValueError(
                f"MESS has no calibrated curve for {std}; calibrate it before evaluation"
            ) from exc
    return {
        "manifest_schema_version": C.RUN_MANIFEST_SCHEMA_VERSION,
        "rev": provenance["rev"],
        "dirty": provenance["dirty"],
        "dirty_sha256": provenance["dirty_sha256"],
        "provenance_id": provenance.get(
            "identity", C.provenance_identity(provenance)),
        "frontend": "SimpleO3",
        "std": std,
        "workload": wl,
        "model": model,
        "label": output_label or model,
        # Retained as an explicit compatibility field for downstream readers.
        "side": output_label or model,
        "traces": traces,
        "trace_inputs": trace_inputs,
        "external_inputs": external_inputs,
        "insts_per_core": insts_per_core,
        "mshr": C.STD[std]["mshr"],
        "request_trace_scope": REQUEST_TRACE_SCOPE,
        "request_trace_timebase": REQUEST_TRACE_TIMEBASE,
        "request_trace_columns": list(LOGICAL_TRACE_COLUMNS),
        "request_trace_llc_path_codes": LOGICAL_PATH_CODES,
        "controller_trace_scope": CONTROLLER_TRACE_SCOPE,
        "fixed_issue_roi": FIXED_ISSUE_ROI,
        "reference_config": C.REFERENCE,
        "controller_config": controller_config,
        "standard_config": C.STD[std],
    }


def _validate_cached_run(manifest_path, request_trace_path,
                         controller_trace_path, expected, provenance):
    try:
        cached = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read cached manifest {manifest_path}: {exc}; rerun with --force"
        ) from exc
    mismatches = [
        key for key, value in expected.items() if cached.get(key) != value
    ]
    if cached.get("native_library", {}).get("sha256") != (
            provenance["native_library"]["sha256"]):
        mismatches.append("native_library.sha256")
    if cached.get("python_extension", {}).get("sha256") != (
            provenance["python_extension"]["sha256"]):
        mismatches.append("python_extension.sha256")
    if (cached.get("optimized_build", {}).get("target_flags", {}).get("sha256")
            != provenance["optimized_build"]["target_flags"]["sha256"]):
        mismatches.append("optimized_build.target_flags.sha256")
    logical_rows = None
    if not isinstance(cached.get("frontend_stats"), dict):
        mismatches.append("frontend_stats")
    if not isinstance(cached.get("controller_stats"), dict):
        mismatches.append("controller_stats")
    if not A.exists(request_trace_path):
        mismatches.append(f"missing raw trace {request_trace_path.name}")
    else:
        try:
            if not A.raw_provenance_matches(
                    request_trace_path, cached.get("raw_trace")):
                mismatches.append("raw_trace.sha256")
            logical_rows = _validate_logical_request_trace(request_trace_path)
        except (OSError, RuntimeError) as exc:
            mismatches.append(f"invalid raw trace: {exc}")
    if not A.exists(controller_trace_path):
        mismatches.append(
            f"missing controller trace {controller_trace_path.name}"
        )
    else:
        try:
            if not A.raw_provenance_matches(
                    controller_trace_path, cached.get("controller_trace")):
                mismatches.append("controller_trace.sha256")
        except OSError as exc:
            mismatches.append(f"unreadable controller trace: {exc}")
    if logical_rows is not None and isinstance(cached.get("frontend_stats"), dict):
        try:
            _validate_fixed_roi_frontend_stats(
                cached["frontend_stats"],
                core_count=len(expected["traces"]),
                insts_per_core=expected["insts_per_core"],
                logical_rows=logical_rows,
            )
        except RuntimeError as exc:
            mismatches.append(str(exc))
    if mismatches:
        raise RuntimeError(
            f"stale/incompatible SimpleO3 cache at {manifest_path} "
            f"({', '.join(mismatches)}); rerun with --force"
        )
    return cached


def run_one(job):
    (wl, std, model, output_label, force, provenance, trace_inputs,
     insts_per_core, candidate_overrides) = job
    outdir = C.OUT / "simpleo3" / std / wl / output_label
    manifest = outdir / "manifest.json"
    request_trace = outdir / "trace.csv.ch0"
    controller_trace = outdir / "controller_trace.csv.ch0"
    expected = _simpleo3_expectations(
        wl, std, model, provenance, trace_inputs, insts_per_core,
        candidate_overrides, output_label,
    )
    if manifest.exists() and not force:
        _validate_cached_run(
            manifest, request_trace, controller_trace, expected, provenance,
        )
        return f"cached {wl} {output_label}"
    staging = C.stage_run_directory(outdir, "manifest.json")
    import ramulator
    s = C.STD[std]
    traces = C.MIXES.get(wl, [wl])
    dram = getattr(ramulator.dram, std)(org_preset=s["org"], timing_preset=s["timing"],
                                        **s["extra"])
    request_tracefile = staging / "trace.csv.ch0"
    controller_tracefile = staging / "controller_trace.csv"
    ctrl = _build_controller(
        ramulator, model, dram, std, controller_tracefile,
        candidate_overrides,
    )
    fe = ramulator.frontend.SimpleO3(
        clock_ratio=8, traces=[C.trace_path(t) for t in traces],
        num_expected_insts=insts_per_core, llc_num_mshr_per_core=s["mshr"],
        request_trace_path=str(request_tracefile),
        translation=ramulator.translation.NoTranslation(max_addr=2**33))
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=3, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    t0 = time.time()
    sim = ramulator.Simulation(fe, mem)
    sim.run()
    wall = time.time() - t0
    st = sim.stats
    per_core = [st["frontend"][f"cycles_recorded_core_{i}"] for i in range(len(traces))]
    payload = {
        **expected,
        "native_library": provenance["native_library"],
        "python_extension": provenance["python_extension"],
        "optimized_build": provenance["optimized_build"],
        "wall_s": round(wall, 2),
        "per_core_cycles": per_core,
        "frontend_stats": {k: v for k, v in st["frontend"].items()},
        "controller_stats": {k: v for k, v in st["memory_system"]["controller"].items()},
    }
    sim.finalize()
    del sim, mem, ctrl, fe
    _publish_simpleo3_traces(
        request_tracefile, request_trace,
        staging / "controller_trace.csv.ch0", controller_trace,
        manifest, payload,
    )
    try:
        staging.rmdir()
    except OSError:
        pass
    return f"done {wl} {output_label} ({wall:.1f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", default="DDR5", choices=list(C.STD))
    ap.add_argument("--workloads", nargs="*", default=None)
    ap.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help="comma-separated subset of " + ",".join(ALL_MODELS),
    )
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument(
        "--insts-per-core", type=int, default=None,
        help="override the full campaign ROI (intended for smoke runs)",
    )
    ap.add_argument(
        "--candidate-kw", default="{}",
        help="JSON object of non-reserved Atomic candidate parameters",
    )
    ap.add_argument(
        "--candidate-label", default="candidate",
        help="safe output label; preserves multiple candidate iterations",
    )
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    wls = a.workloads or (C.CHRONUS + C.DPC4 + list(C.MIXES))
    models = [model.strip() for model in a.models.split(",") if model.strip()]
    if not wls or not models or any(model not in ALL_MODELS for model in models):
        ap.error(
            "workloads and models must be non-empty; models must be drawn from "
            + ",".join(ALL_MODELS)
        )
    if len(set(models)) != len(models):
        ap.error("--models contains duplicates")
    if not SAFE_LABEL.fullmatch(a.candidate_label):
        ap.error("--candidate-label must be a safe path component")
    if a.candidate_label in {"oracle", "fixedlat", "md1", "wmg1", "mess"}:
        ap.error("--candidate-label collides with a protected model label")
    if not 1 <= a.workers <= 12:
        ap.error("--workers must be in [1, 12]")
    if a.insts_per_core is not None and a.insts_per_core <= 0:
        ap.error("--insts-per-core must be positive")
    try:
        candidate_overrides = json.loads(a.candidate_kw)
    except json.JSONDecodeError as exc:
        ap.error(f"--candidate-kw is not valid JSON: {exc}")
    if not isinstance(candidate_overrides, dict):
        ap.error("--candidate-kw must decode to an object")
    provenance = C.runtime_provenance()
    trace_inputs = C.simpleo3_trace_inputs(wls)
    jobs = [
        (
            wl,
            a.std,
            model,
            a.candidate_label if model == "candidate" else model,
            a.force,
            provenance,
            trace_inputs[wl],
            a.insts_per_core
            if a.insts_per_core is not None
            else C.INSTS_MIX if wl in C.MIXES else C.INSTS_SINGLE,
            candidate_overrides if model == "candidate" else {},
        )
        for wl in wls for model in models
    ]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for msg in pool.map(run_one, jobs, chunksize=1):
            print(msg, flush=True)
    C.validate_simpleo3_trace_inputs(trace_inputs)
    C.validate_runtime_provenance(provenance)
    print("SIMPLEO3_EVAL_DONE")


if __name__ == "__main__":
    main()
