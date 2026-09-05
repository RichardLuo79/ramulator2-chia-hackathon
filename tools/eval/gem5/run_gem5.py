"""gem5 SE runner: 12 benchmarks x {oracle, candidate}, DDR5, raw-first.

Each run writes eval_out/gem5/DDR5/<bench>/<side>/:
  gem5 outdir (stats.txt etc.), trace.csv.ch0 (per-request), manifest.json
  with RAW simTicks/simInsts and wall seconds.

Usage: python tools/eval/gem5/run_gem5.py [--workers N] [--force]
       [--bench a b ...]
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))
from eval import artifacts as A
from eval import config as C
from eval import metrics

BOARD = pathlib.Path(__file__).resolve().parent / "se_board.py"
DEFAULT_REQUEST_COVERAGE_THRESHOLD = 0.995
REQUEST_TRACE_SCOPE = "gem5_dram_controller_lifecycle"
REQUEST_TRACE_TIMEBASE = "ramulator_controller_cycles"
GEM5_SOURCE_FILES = (
    "src/mem/ramulator2/Ramulator2.py",
    "src/mem/ramulator2/Ramulator2VectorPorts.py",
    "src/mem/ramulator2/SConscript",
    "src/mem/ramulator2/ramulator2.cc",
    "src/mem/ramulator2/ramulator2.hh",
    "src/mem/ramulator2/ramulator2_base.cc",
    "src/mem/ramulator2/ramulator2_base.hh",
    "src/mem/ramulator2/ramulator2_identity.hh",
    "src/mem/ramulator2/ramulator2_vector_ports.cc",
    "src/mem/ramulator2/ramulator2_vector_ports.hh",
    # These define and attach the O3 sequence used by the bridge. Fingerprint
    # them explicitly because a gem5 binary hash alone cannot show which
    # checkout the untracked integration sources came from.
    "src/mem/request.hh",
    "src/cpu/o3/lsq.cc",
    "src/mem/cache/prefetch/base.hh",
    "src/mem/cache/prefetch/base.cc",
    "src/mem/cache/prefetch/queued.cc",
)


def _annotate_request_eligibility(
        request, *, minimum_coverage=DEFAULT_REQUEST_COVERAGE_THRESHOLD):
    """Label a request score as eligible or diagnostic, with exact coverage.

    A stable-ID label alone is insufficient: independently paced simulations
    can populate nominal IDs that barely overlap. Treat a latency score as a
    headline metric only when the matcher is stable-ID based and both sides
    meet the same minimum for full-population coverage, stable-population
    coverage, and stable-ID eligibility.
    """
    threshold = metrics.request_coverage(
        minimum_coverage, "minimum request coverage")
    matched = metrics.request_count(
        request.get("matched"), "matched", positive=True)
    n_oracle = metrics.request_count(
        request.get("n_oracle"), "n_oracle", positive=True)
    n_model = metrics.request_count(
        request.get("n_model"), "n_model", positive=True)
    eligible_oracle = metrics.request_count(
        request.get("stable_eligible_o"), "stable_eligible_o")
    eligible_model = metrics.request_count(
        request.get("stable_eligible_m"), "stable_eligible_m")

    coverage = {
        "full": {
            "oracle": {
                "matched": matched,
                "population": n_oracle,
                "fraction": metrics.request_coverage(
                    request.get("cov_o"), "cov_o"),
            },
            "model": {
                "matched": matched,
                "population": n_model,
                "fraction": metrics.request_coverage(
                    request.get("cov_m"), "cov_m"),
            },
        },
        "stable": {
            "oracle": {
                "matched": matched,
                "eligible_population": eligible_oracle,
                "fraction": metrics.request_coverage(
                    request.get("stable_cov_o"), "stable_cov_o"),
            },
            "model": {
                "matched": matched,
                "eligible_population": eligible_model,
                "fraction": metrics.request_coverage(
                    request.get("stable_cov_m"), "stable_cov_m"),
            },
        },
        "stable_eligibility": {
            "oracle": {
                "eligible_population": eligible_oracle,
                "full_population": n_oracle,
                "fraction": eligible_oracle / n_oracle,
            },
            "model": {
                "eligible_population": eligible_model,
                "full_population": n_model,
                "fraction": eligible_model / n_model,
            },
        },
    }

    reasons = []
    mode = request.get("match_mode")
    if mode != "stable_id":
        reasons.append({
            "code": "match_mode_not_stable_id",
            "actual": mode,
            "required": "stable_id",
        })
    for population in ("full", "stable", "stable_eligibility"):
        for direction in ("oracle", "model"):
            actual = coverage[population][direction]["fraction"]
            if actual < threshold:
                reasons.append({
                    "code": "coverage_below_minimum",
                    "population": population,
                    "direction": direction,
                    "actual": actual,
                    "required_minimum": threshold,
                    "shortfall": threshold - actual,
                })

    annotated = dict(request)
    annotated.update({
        "stable_eligibility_fraction_o": (
            coverage["stable_eligibility"]["oracle"]["fraction"]),
        "stable_eligibility_fraction_m": (
            coverage["stable_eligibility"]["model"]["fraction"]),
        "request_metric_eligible": not reasons,
        "request_metric_status": (
            "eligible" if not reasons else "diagnostic_ineligible"),
        "request_metric_headline": (
            {
                "mae_over_L": request["mae"],
                "signed_drift_over_L": request["sgn"],
                "paired_tail_over_L": request["tail"],
            }
            if not reasons else None),
        "request_metric_minimum_coverage": threshold,
        "request_metric_ineligibility_reasons": reasons,
        "coverage_audit": coverage,
    })
    return annotated


def _aggregate_request_diagnostics(requests):
    """Aggregate valid request scores without headlining ineligible data."""
    diagnostics = metrics.aggregate_request_metrics(requests)
    eligible = sorted(
        workload for workload, request in requests.items()
        if request["request_metric_eligible"])
    ineligible = {
        workload: request["request_metric_ineligibility_reasons"]
        for workload, request in requests.items()
        if not request["request_metric_eligible"]
    }
    all_eligible = not ineligible
    result = {
        "request_metric_eligible": all_eligible,
        "request_metric_status": (
            "eligible" if all_eligible else "diagnostic_ineligible"),
        "request_metric_minimum_coverage": (
            DEFAULT_REQUEST_COVERAGE_THRESHOLD),
        "eligible_workloads": eligible,
        "ineligible_workloads": ineligible,
        "per_workload_status": {
            workload: request["request_metric_status"]
            for workload, request in requests.items()
        },
        "diagnostic_metrics": diagnostics,
        "headline_metrics": diagnostics if all_eligible else None,
    }
    if all_eligible:
        result.update(diagnostics)
    return result


def _request_postprocessor_inputs():
    """Fingerprint the code that derives request eligibility and metrics."""
    return {
        "tools/eval/config.py": C.file_provenance(C.__file__),
        "tools/eval/metrics.py": C.file_provenance(metrics.__file__),
        "tools/eval/gem5/run_gem5.py": C.file_provenance(__file__),
    }


def _gem5_source_inputs(gem5_binary):
    """Fingerprint the active installed bridge and its identity field sources."""
    binary = pathlib.Path(gem5_binary).resolve()
    source_root = next(
        (parent for parent in binary.parents
         if (parent / "SConstruct").is_file() and
         (parent / "src" / "mem" / "ramulator2").is_dir()),
        None,
    )
    if source_root is None:
        raise RuntimeError(
            f"cannot locate gem5 source tree containing {binary}"
        )
    return {
        f"gem5_source:{relative}": C.file_provenance(source_root / relative)
        for relative in GEM5_SOURCE_FILES
    }


def _execution_inputs_for_benches(benches, common_inputs):
    """Bind every benchmark binary and exact argument string to run identity."""
    return {
        bench: {
            **common_inputs,
            "benchmark_binary": C.file_provenance(C.GEM5_BENCH_DIR / bench),
            "benchmark_args": C.GEM5_BENCH[bench],
        }
        for bench in benches
    }


def _validate_execution_inputs(execution_inputs):
    for bench, inputs in execution_inputs.items():
        if inputs.get("benchmark_args") != C.GEM5_BENCH.get(bench):
            raise RuntimeError(
                f"gem5 benchmark arguments changed during run: {bench}"
            )
        for label, fingerprint in inputs.items():
            if label == "benchmark_args":
                continue
            valid = False
            path = fingerprint.get("path") if isinstance(fingerprint, dict) else None
            if isinstance(path, str) and path:
                try:
                    valid = C.file_provenance_matches(path, fingerprint)
                except OSError:
                    valid = False
            if not valid:
                raise RuntimeError(
                    f"gem5 execution input changed during run: {label}"
                )


def _gem5_expectations(bench, side, model, candidate_overrides, provenance,
                       execution_inputs=None):
    applied_overrides = candidate_overrides if model == "candidate" else {}
    expected = {
        "manifest_schema_version": C.RUN_MANIFEST_SCHEMA_VERSION,
        "rev": provenance["rev"],
        "dirty": provenance["dirty"],
        "dirty_sha256": provenance["dirty_sha256"],
        "provenance_id": provenance.get(
            "identity", C.provenance_identity(provenance)),
        "frontend": "gem5-SE",
        "std": "DDR5",
        "workload": bench,
        "side": side,
        "label": side,
        "model": model,
        "reference_config": C.REFERENCE,
        "candidate_overrides": applied_overrides,
        "request_trace_scope": REQUEST_TRACE_SCOPE,
        "request_trace_timebase": REQUEST_TRACE_TIMEBASE,
    }
    if execution_inputs is not None:
        expected["execution_inputs"] = execution_inputs
    return expected


def _gem5_reference_expectations(bench, execution_inputs):
    """Identity fields that let an immutable oracle outlive model revisions."""
    return {
        "manifest_schema_version": C.RUN_MANIFEST_SCHEMA_VERSION,
        "frontend": "gem5-SE",
        "std": "DDR5",
        "workload": bench,
        "side": "oracle",
        "label": "oracle",
        "model": "oracle",
        "reference_config": C.REFERENCE,
        "candidate_overrides": {},
        "request_trace_scope": REQUEST_TRACE_SCOPE,
        "request_trace_timebase": REQUEST_TRACE_TIMEBASE,
        "execution_inputs": execution_inputs,
    }


def _validate_cached_run(manifest_path, trace_path, expected, provenance=None):
    try:
        cached = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read cached manifest {manifest_path}: {exc}; rerun with --force"
        ) from exc
    mismatches = [
        key for key, value in expected.items() if cached.get(key) != value
    ]
    if provenance is not None:
        if cached.get("native_library", {}).get("sha256") != (
                provenance["native_library"]["sha256"]):
            mismatches.append("native_library.sha256")
        if cached.get("python_extension", {}).get("sha256") != (
                provenance["python_extension"]["sha256"]):
            mismatches.append("python_extension.sha256")
    else:
        if not isinstance(cached.get("provenance_id"), str):
            mismatches.append("provenance_id")
        if not isinstance(cached.get("native_library", {}).get("sha256"), str):
            mismatches.append("native_library.sha256")
        if not isinstance(cached.get("python_extension", {}).get("sha256"), str):
            mismatches.append("python_extension.sha256")
    if not A.exists(trace_path):
        mismatches.append(f"missing raw trace {trace_path.name}")
    elif not A.raw_provenance_matches(trace_path, cached.get("raw_trace")):
        mismatches.append("raw_trace.sha256")
    if mismatches:
        raise RuntimeError(
            f"stale/incompatible gem5 cache at {manifest_path} "
            f"({', '.join(mismatches)}); rerun with --force"
        )
    return cached


def run_one(job):
    bench, side, model, candidate_overrides, force, provenance, execution_inputs = job
    outdir = C.OUT / "gem5" / "DDR5" / bench / side
    manifest = outdir / "manifest.json"
    trace = outdir / "trace.csv.ch0"
    expected = _gem5_expectations(
        bench, side, model, candidate_overrides, provenance, execution_inputs,
    )
    if manifest.exists() and not force:
        _validate_cached_run(manifest, trace, expected, provenance)
        return f"cached {bench} {side}"
    staging = C.stage_run_directory(outdir, "manifest.json", "request.json")
    import os as _os
    env = dict(**_os.environ,
               R2_MODEL=model,
               R2_BINARY=str(C.GEM5_BENCH_DIR / bench),
               R2_ARGS=C.GEM5_BENCH[bench],
               R2_REPO=str(C.REPO),
               R2_TRACE=str(staging / "trace.csv"))
    env.pop("R2_CANDIDATE_KW", None)
    if model == "candidate" and candidate_overrides:
        env["R2_CANDIDATE_KW"] = json.dumps(candidate_overrides, sort_keys=True)
    t0 = time.time()
    r = subprocess.run([C.GEM5_BIN, f"--outdir={outdir}", str(BOARD)],
                       env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if r.returncode != 0:
        (outdir / "stderr.log").write_text(r.stderr[-20000:])
        raise RuntimeError(
            f"gem5 failed for {bench}/{side} with exit code {r.returncode}; "
            f"see {outdir / 'stderr.log'}"
        )
    staged_trace = staging / "trace.csv.ch0"
    if not staged_trace.is_file():
        raise RuntimeError(
            f"gem5 produced no raw request trace for {bench}/{side}: {staged_trace}"
        )
    stats = (outdir / "stats.txt").read_text()
    ticks_match = re.search(r"simTicks\s+(\d+)", stats)
    insts_match = re.search(r"simInsts\s+(\d+)", stats)
    if ticks_match is None or insts_match is None:
        raise RuntimeError(f"gem5 stats are incomplete for {bench}/{side}")
    ticks = int(ticks_match.group(1))
    insts = int(insts_match.group(1))
    if ticks <= 0 or insts <= 0:
        raise RuntimeError(f"gem5 stats are non-positive for {bench}/{side}")
    payload = {
        **expected,
        "native_library": provenance["native_library"],
        "python_extension": provenance["python_extension"],
        "wall_s": round(wall, 1),
        "sim_ticks": ticks, "sim_insts": insts,
    }
    C.publish_trace_and_manifest(
        staged_trace, trace, manifest, payload,
    )
    try:
        staging.rmdir()
    except OSError:
        pass
    return f"done {bench} {side} ({wall:.0f}s, {ticks} ticks)"


def aggregate_runs(benches, model_side, candidate_overrides, provenance,
                   execution_inputs, *, force=False):
    """Validate and aggregate complete gem5 core and request artifacts."""
    rows = {}
    oracle_artifacts = {}
    model_artifacts = {}
    for bench in benches:
        base = C.OUT / "gem5" / "DDR5" / bench
        oracle_dir = base / "oracle"
        model_dir = base / model_side
        paths = {
            "oracle": oracle_dir / "manifest.json",
            "model": model_dir / "manifest.json",
            "oracle_trace": oracle_dir / "trace.csv.ch0",
            "model_trace": model_dir / "trace.csv.ch0",
        }
        missing = [
            str(path) for name, path in paths.items()
            if not (A.exists(path) if name.endswith("_trace") else path.is_file())
        ]
        if missing:
            raise RuntimeError(
                f"refusing partial gem5 aggregation for {bench}; missing: {missing}"
            )
        oracle = _validate_cached_run(
            paths["oracle"], paths["oracle_trace"],
            _gem5_reference_expectations(bench, execution_inputs[bench]),
        )
        model = _validate_cached_run(
            paths["model"], paths["model_trace"],
            _gem5_expectations(
                bench, model_side, "candidate", candidate_overrides, provenance,
                execution_inputs[bench],
            ),
            provenance,
        )
        for metric in ("sim_ticks", "sim_insts"):
            values = (oracle.get(metric), model.get(metric))
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in values):
                raise RuntimeError(
                    f"invalid {metric} in gem5 manifests for {bench}"
                )
        if oracle["sim_insts"] != model["sim_insts"]:
            raise RuntimeError(
                f"gem5 executed instruction populations differ for {bench}: "
                f"oracle={oracle['sim_insts']} model={model['sim_insts']}"
            )
        signed_core_dev = (
            100 * (model["sim_ticks"] - oracle["sim_ticks"]) /
            oracle["sim_ticks"]
        )
        request_path = model_dir / "request.json"
        request = metrics.load_or_compute_request(
            request_path, paths["model_trace"],
            paths["oracle_trace"], force=force,
            request_trace_scope=REQUEST_TRACE_SCOPE,
            request_trace_timebase=REQUEST_TRACE_TIMEBASE,
        )
        # _load_or_compute_request already rejects an empty match, but retaining
        # this check here keeps aggregate_runs fail-closed if that API changes.
        if request.get("matched", 0) <= 0:
            raise RuntimeError(f"no matched gem5 requests for {bench}")
        request = _annotate_request_eligibility(request)
        C.atomic_write_json(request_path, request)
        rows[bench] = {
            "core": {
                "oracle_sim_ticks": oracle["sim_ticks"],
                "model_sim_ticks": model["sim_ticks"],
                "oracle_sim_insts": oracle["sim_insts"],
                "model_sim_insts": model["sim_insts"],
                "signed_dev_pct": signed_core_dev,
                "abs_dev_pct": abs(signed_core_dev),
            },
            "request": request,
        }
        artifact_keys = (
            "provenance_id", "rev", "dirty_sha256", "native_library",
            "python_extension", "raw_trace", "request_trace_scope",
            "request_trace_timebase",
        )
        oracle_artifacts[bench] = {
            key: oracle.get(key) for key in artifact_keys
        }
        model_artifacts[bench] = {
            key: model.get(key) for key in artifact_keys
        }
    if set(rows) != set(benches):
        raise RuntimeError("refusing partial gem5 aggregation")

    core_abs = {bench: row["core"]["abs_dev_pct"]
                for bench, row in rows.items()}
    worst_core_wl = max(core_abs, key=core_abs.get)
    request_metrics = {
        bench: row["request"] for bench, row in rows.items()
    }
    aggregate = {
        "core": {
            "mean_abs_dev_pct": sum(core_abs.values()) / len(core_abs),
            "worst_abs_dev_pct": core_abs[worst_core_wl],
            "worst_abs_dev_wl": worst_core_wl,
        },
        "request": _aggregate_request_diagnostics(request_metrics),
    }
    return rows, aggregate, oracle_artifacts, model_artifacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--candidate-kw", default="",
                    help="JSON overrides for the candidate side")
    ap.add_argument("--label", default="candidate",
                    help="output side label for overridden runs")
    a = ap.parse_args()
    if not 1 <= a.workers <= 12:
        ap.error("--workers must be in [1, 12]")
    override_requested = bool(a.candidate_kw.strip())
    try:
        candidate_overrides = (
            json.loads(a.candidate_kw) if override_requested else {})
    except json.JSONDecodeError as exc:
        ap.error(f"--candidate-kw is not valid JSON: {exc}")
    if not isinstance(candidate_overrides, dict):
        ap.error("--candidate-kw must decode to a JSON object")
    if (not a.label or pathlib.Path(a.label).name != a.label or
            a.label in (".", "..")):
        ap.error("--label must be a non-empty safe path component")
    if override_requested and a.label == "oracle":
        ap.error("--label oracle would overwrite the reference run")
    benches = a.bench or list(C.GEM5_BENCH)
    if (not benches or len(set(benches)) != len(benches) or
            any(bench not in C.GEM5_BENCH for bench in benches)):
        ap.error(
            "--bench must contain unique configured gem5 benchmark names"
        )
    model_side = a.label if override_requested else "candidate"
    summary_out = C.OUT / "gem5" / "DDR5" / f"summary_{model_side}.json"
    # A failed rerun or aggregation must not leave an older success artifact.
    summary_out.unlink(missing_ok=True)
    provenance = C.runtime_provenance()
    common_inputs = {
        "gem5_binary": C.file_provenance(C.GEM5_BIN),
        "board": C.file_provenance(BOARD),
        **_gem5_source_inputs(C.GEM5_BIN),
    }
    execution_inputs = _execution_inputs_for_benches(benches, common_inputs)
    if override_requested:
        for bench in benches:
            oracle_dir = C.OUT / "gem5" / "DDR5" / bench / "oracle"
            oracle_manifest = oracle_dir / "manifest.json"
            if not oracle_manifest.is_file():
                raise RuntimeError(
                    f"missing gem5 oracle for {bench}; run the baseline/oracle "
                    "evaluation before an override scan"
                )
            _validate_cached_run(
                oracle_manifest, oracle_dir / "trace.csv.ch0",
                _gem5_reference_expectations(bench, execution_inputs[bench]),
            )
        jobs = [(bench, a.label, "candidate", candidate_overrides, a.force,
                 provenance, execution_inputs[bench])
                for bench in benches]
    else:
        jobs = [(bench, side, side, {}, a.force, provenance, execution_inputs[bench])
                for bench in benches for side in ("oracle", "candidate")]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for msg in pool.map(run_one, jobs):
            print(msg, flush=True)
    rows, aggregate, oracle_artifacts, model_artifacts = aggregate_runs(
        benches, model_side, candidate_overrides, provenance, execution_inputs,
        force=a.force,
    )
    _validate_execution_inputs(execution_inputs)
    C.validate_runtime_provenance(provenance)
    summary = {
        "manifest_schema_version": C.RUN_MANIFEST_SCHEMA_VERSION,
        "request_metric_schema_version": C.REQUEST_METRIC_SCHEMA_VERSION,
        "matcher_schema_version": metrics.matcher_schema_version(),
        "request_metric_minimum_coverage": (
            DEFAULT_REQUEST_COVERAGE_THRESHOLD),
        "request_trace_scope": REQUEST_TRACE_SCOPE,
        "request_trace_timebase": REQUEST_TRACE_TIMEBASE,
        "request_postprocessor_inputs": _request_postprocessor_inputs(),
        "rev": provenance["rev"],
        "dirty": provenance["dirty"],
        "dirty_sha256": provenance["dirty_sha256"],
        "provenance_id": provenance["identity"],
        "native_library": provenance["native_library"],
        "python_extension": provenance["python_extension"],
        "frontend": "gem5-SE",
        "std": "DDR5",
        "model_side": model_side,
        "candidate_overrides": candidate_overrides,
        "workloads": benches,
        "execution_inputs": execution_inputs,
        "oracle_artifacts": oracle_artifacts,
        "model_artifacts": model_artifacts,
        "per_benchmark": rows,
        "aggregate": aggregate,
    }
    C.atomic_write_json(summary_out, summary)
    display = {}
    for bench, row in rows.items():
        request = row["request"]
        display[bench] = {
            "core_signed_dev_pct": row["core"]["signed_dev_pct"],
            "request_metric_status": request["request_metric_status"],
            "match_mode": request["match_mode"],
            "coverage_audit": request["coverage_audit"],
        }
        if request["request_metric_eligible"]:
            display[bench].update({
                "request_mae_over_L": request["mae"],
                "request_signed_drift_over_L": request["sgn"],
            })
        else:
            display[bench]["request_diagnostic"] = {
                "mae_over_L": request["mae"],
                "signed_drift_over_L": request["sgn"],
                "paired_tail_over_L": request["tail"],
                "ineligibility_reasons": (
                    request["request_metric_ineligibility_reasons"]),
            }
    print(json.dumps(display, indent=1))
    core_text = (
        "gem5 aggregate "
        f"core mean|dev|={aggregate['core']['mean_abs_dev_pct']:.3f}% "
        f"worst={aggregate['core']['worst_abs_dev_pct']:.3f}%")
    if aggregate["request"]["request_metric_eligible"]:
        print(
            core_text +
            f" request mean MAE/L={aggregate['request']['mean_mae']:.4f} "
            f"worst MAE/L={aggregate['request']['worst_mae']:.4f}")
    else:
        print(
            core_text + " request=diagnostic/ineligible "
            f"(minimum coverage "
            f"{DEFAULT_REQUEST_COVERAGE_THRESHOLD:.3f})")
    print(f"summary -> {summary_out}")
    print("GEM5_EVAL_DONE")


if __name__ == "__main__":
    main()
