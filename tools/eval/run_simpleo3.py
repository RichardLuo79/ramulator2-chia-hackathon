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

from eval import config as C  # noqa: E402, I001
from eval import artifacts as A  # noqa: E402, I001


REQUEST_TRACE_SCOPE = "simpleo3_logical_llc"
CONTROLLER_TRACE_SCOPE = "dram_controller"
REQUEST_TRACE_TIMEBASE = "simpleo3_frontend_cycles"
FIXED_ISSUE_ROI = True
ALL_MODELS = ("oracle",) + C.MODEL_ORDER
SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
LOGICAL_TRACE_COLUMNS = (
    "arrive", "depart", "type", "source", "addr", "frontend_id",
    "frontend_sub_id", "admission_ordinal", "llc_path",
)
LOGICAL_PATH_CODES = {"0": "hit", "1": "mshr_merge", "2": "miss_owner"}


def _controller_config(model, candidate_overrides):
    if model == "oracle":
        return {
            **C.REFERENCE,
            "scheduler": "FRFCFSRowHit",
            "refresh_manager": "NoRefresh",
            "row_policy": "Open",
        }
    if model == "candidate":
        return {
            **C.CANDIDATE_RESOURCES,
            "refresh": "none",
            **candidate_overrides,
        }
    if model == "fixedlat":
        return {"latency": -1, "pipe": 1}
    if model == "md1":
        return {"phase_ticks": 10_000, "smoothing": 0.5}
    if model == "wmg1":
        return {"window_ns": 10_000}
    if model == "mess":
        return {"window_accesses": 1_000, "converge": 0.05}
    raise ValueError(f"unknown SimpleO3 model {model!r}")


def _build_controller(ramulator, model, dram, std, trace_path,
                      candidate_overrides):
    config = _controller_config(model, candidate_overrides)
    if model == "oracle":
        return ramulator.controller.GenericDDR(
            dram=dram,
            scheduler=ramulator.scheduler.FRFCFSRowHit(),
            refresh_manager=ramulator.refresh_manager.NoRefresh(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
            controller_plugins=[ramulator.controller_plugin.ReqTraceRecorder(
                path=str(trace_path))],
            **C.REFERENCE,
        )
    if model == "candidate":
        reserved = {"dram", "addr_mapper", "trace_path", "refresh"}
        overlap = reserved & set(candidate_overrides)
        if overlap:
            raise ValueError(
                "candidate overrides contain evaluator-owned keys: "
                + ", ".join(sorted(overlap))
            )
        return ramulator.controller.Atomic(
            dram=dram,
            addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
            trace_path=str(trace_path),
            **config,
        )

    cls = getattr(ramulator.controller, C.MODEL_IMPL[model])
    kwargs = {**config, "dram": dram, "trace_path": str(trace_path)}
    if model == "mess":
        try:
            curve = C.MESS_CURVES[std]
        except KeyError as exc:
            raise ValueError(
                f"MESS has no calibrated curve for {std}; calibrate it before evaluation"
            ) from exc
        kwargs["curve_path"] = str(curve)
    return cls(**kwargs)


def _validate_logical_request_trace(path):
    """Require the fixed, matcher-facing SimpleO3 logical trace contract."""
    path = pathlib.Path(path)
    if not A.exists(path):
        raise RuntimeError(
            f"simulation produced no non-empty logical request trace: {path}"
        )
    with A.open_text(path, newline="") as stream:
        rows = csv.reader(stream)
        try:
            header = tuple(next(rows))
        except StopIteration as exc:
            raise RuntimeError(f"logical request trace is empty: {path}") from exc
        if header != LOGICAL_TRACE_COLUMNS:
            raise RuntimeError(
                f"logical request trace {path} has columns {header!r}; "
                f"expected exactly {LOGICAL_TRACE_COLUMNS!r}"
            )
        row_count = 0
        for line_number, row in enumerate(rows, start=2):
            row_count += 1
            if len(row) != len(LOGICAL_TRACE_COLUMNS):
                raise RuntimeError(
                    f"logical request trace {path}:{line_number} has {len(row)} "
                    f"columns; expected {len(LOGICAL_TRACE_COLUMNS)}"
                )
            try:
                llc_path = int(row[-1])
            except ValueError as exc:
                raise RuntimeError(
                    f"logical request trace {path}:{line_number} has a "
                    f"non-integer llc_path {row[-1]!r}"
                ) from exc
            if llc_path not in (0, 1, 2):
                raise RuntimeError(
                    f"logical request trace {path}:{line_number} has invalid "
                    f"llc_path {llc_path}; expected 0 (hit), 1 (merge), or 2 (owner)"
                )
        if row_count == 0:
            raise RuntimeError(
                f"logical request trace contains no request rows: {path}"
            )
    return row_count


def _validate_fixed_roi_frontend_stats(
    stats,
    *,
    core_count,
    insts_per_core,
    logical_rows,
):
    """Fail closed unless the frontend reports a fully drained fixed ROI."""
    if not isinstance(stats, dict):
        raise RuntimeError("frontend_stats must be an object")

    def exact_nonnegative_int(key):
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"frontend_stats.{key} must be a non-negative integer")
        return value

    for core_id in range(core_count):
        issued = exact_nonnegative_int(f"insts_issued_core_{core_id}")
        if issued != insts_per_core:
            raise RuntimeError(
                f"frontend_stats.insts_issued_core_{core_id}={issued}; "
                f"expected fixed ROI {insts_per_core}"
            )
    live = exact_nonnegative_int("logical_requests_live")
    if live != 0:
        raise RuntimeError(
            f"frontend_stats.logical_requests_live={live}; expected a drained LLC"
        )
    completed = exact_nonnegative_int("logical_requests_completed")
    if completed != logical_rows:
        raise RuntimeError(
            f"frontend_stats.logical_requests_completed={completed}; "
            f"trace contains {logical_rows} rows"
        )
    path_total = sum(
        exact_nonnegative_int(key)
        for key in (
            "logical_requests_hit",
            "logical_requests_mshr_merge",
            "logical_requests_miss_owner",
        )
    )
    if path_total != completed:
        raise RuntimeError(
            f"frontend logical path counters sum to {path_total}; "
            f"expected {completed} completed requests"
        )
    internal_generated = exact_nonnegative_int("internal_writebacks_generated")
    internal_completed = exact_nonnegative_int("internal_writebacks_completed")
    internal_live = exact_nonnegative_int("internal_writebacks_live")
    if internal_live != 0 or internal_completed != internal_generated:
        raise RuntimeError(
            "frontend internal writebacks were not fully drained: "
            f"generated={internal_generated}, completed={internal_completed}, "
            f"live={internal_live}"
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
