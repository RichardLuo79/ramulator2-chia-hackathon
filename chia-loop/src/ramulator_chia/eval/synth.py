"""Synthetic-pattern response characterization: oracle vs candidate.

Each scenario is a SyntheticPattern parameter dict. Both sides run the same
timing-invariant address sequence closed-loop; reads are matched 1:1 with the
occurrence-keyed matcher. Per scenario: oracle response (mean latency, p50/99,
achieved read bandwidth) and model disagreement (per-request MAE/L, signed
drift/L, paired tail, run-time deviation).

Usage: python tools/eval/synth.py scenarios.json --std DDR5 [--tag t] [--workers N]
Scenario file: {"name": {params...}, ...}; "sweep" entries expand a listed axis.
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

from ramulator_chia.eval import artifacts as A
from ramulator_chia.eval import config as C


def expand(scenarios):
    """Expand {"sweep": {axis: [v1, v2...]}, **base} entries into concrete scenarios."""
    out = {}
    for name, sc in scenarios.items():
        sc = dict(sc)
        sweep = sc.pop("sweep", None)
        if not sweep:
            out[name] = sc
            continue
        (axis, values), = sweep.items()
        for v in values:
            out[f"{name}_{axis}{v}"] = {**sc, axis: v}
    return out


def _controller_config(side, candidate_overrides):
    if side == "candidate":
        return {**C.CANDIDATE_RESOURCES, "refresh": "none", **candidate_overrides}
    return {
        **C.REFERENCE,
        "scheduler": "FRFCFSRowHit",
        "refresh_manager": "NoRefresh",
        "row_policy": "Open",
    }


def _synth_expectations(name, params, std, side, tag, candidate_overrides, provenance):
    applied_overrides = candidate_overrides if side == "candidate" else {}
    return {
        "manifest_schema_version": C.RUN_MANIFEST_SCHEMA_VERSION,
        "rev": provenance["rev"],
        "dirty": provenance["dirty"],
        "dirty_sha256": provenance["dirty_sha256"],
        "provenance_id": provenance.get(
            "identity", C.provenance_identity(provenance)),
        "tag": tag,
        "std": std,
        "scenario": name,
        "params": params,
        "side": side,
        "reference_config": C.REFERENCE,
        "candidate_overrides": applied_overrides,
        "controller_config": _controller_config(side, applied_overrides),
        "standard_config": C.STD[std],
        "address_mapper": "PassThroughAddrMapper",
        "channel_mapper": "PassThroughChannelMapper",
        "frontend_clock_ratio": 1,
        "memory_clock_ratio": 1,
    }


def _validate_cached_run(manifest_path, trace_path, expected, provenance):
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
    if not A.exists(trace_path):
        mismatches.append(f"missing raw trace {trace_path.name}")
    elif not A.raw_provenance_matches(trace_path, cached.get("raw_trace")):
        mismatches.append("raw_trace.sha256")
    if mismatches:
        raise RuntimeError(
            f"stale/incompatible synthetic cache at {manifest_path} "
            f"({', '.join(mismatches)}); rerun with --force"
        )
    return cached


def run_one(job):
    name, params, std, side, tag, candidate_overrides, force, provenance = job
    outdir = C.OUT / f"synth_{tag}" / std / name / side
    manifest = outdir / "manifest.json"
    trace = outdir / "trace.csv.ch0"
    expected = _synth_expectations(
        name, params, std, side, tag, candidate_overrides, provenance,
    )
    if manifest.exists() and not force:
        _validate_cached_run(manifest, trace, expected, provenance)
        return f"cached {name} {side}"
    staging = C.stage_run_directory(outdir, "manifest.json")
    import ramulator
    from tests.utils import extract_dram_layout
    s = C.STD[std]
    dram = getattr(ramulator.dram, std)(org_preset=s["org"], timing_preset=s["timing"],
                                        **s["extra"])
    layout = extract_dram_layout(dram)
    tracefile = staging / "trace.csv"
    if side == "oracle":
        ctrl = ramulator.controller.GenericDDR(
            dram=dram, scheduler=ramulator.scheduler.FRFCFSRowHit(),
            refresh_manager=ramulator.refresh_manager.NoRefresh(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
            controller_plugins=[ramulator.controller_plugin.ReqTraceRecorder(
                path=str(tracefile))],
            **C.REFERENCE)
    else:
        controller_config = _controller_config(side, candidate_overrides)
        ctrl = ramulator.controller.Atomic(
            dram=dram, addr_mapper=ramulator.addr_mapper.PassThroughAddrMapper(),
            trace_path=str(tracefile), **controller_config)
    fe = ramulator.frontend.SyntheticPattern(clock_ratio=1, **layout, **params)
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper())
    sim = ramulator.Simulation(fe, mem)
    sim.run()
    st = sim.stats
    payload = {
        **expected,
        "native_library": provenance["native_library"],
        "python_extension": provenance["python_extension"],
        "optimized_build": provenance["optimized_build"],
        "frontend_cycles": st["frontend"]["cycles"],
        "avg_read_latency": st["frontend"]["avg_read_latency"],
        "reads": st["frontend"]["reads_sent"], "writes": st["frontend"]["writes_sent"],
        "controller_stats": {
            key: value
            for key, value in st["memory_system"]["controller"].items()
        },
    }
    del sim, mem, ctrl, fe
    C.publish_trace_and_manifest(
        staging / "trace.csv.ch0", trace, manifest, payload,
    )
    try:
        staging.rmdir()
    except OSError:
        pass
    return f"done {name} {side}"


def main():
    import numpy as np
    from ramulator_chia.eval import matchlib
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios")
    ap.add_argument("--std", default="DDR5", choices=list(C.STD))
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--candidate-kw", default="{}",
                    help="JSON overrides applied to the candidate controller")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    raw_scenarios = json.load(open(a.scenarios))
    if not isinstance(raw_scenarios, dict) or not raw_scenarios or not all(
            isinstance(name, str) and name and pathlib.Path(name).name == name
            and isinstance(params, dict)
            for name, params in raw_scenarios.items()):
        ap.error(
            "scenarios must be a non-empty JSON object mapping safe names "
            "to parameter objects"
        )
    try:
        candidate_overrides = json.loads(a.candidate_kw)
    except json.JSONDecodeError as exc:
        ap.error(f"--candidate-kw is not valid JSON: {exc}")
    if not isinstance(candidate_overrides, dict):
        ap.error("--candidate-kw must decode to a JSON object")
    if not 1 <= a.workers <= 12:
        ap.error("--workers must be in [1, 12]")
    scenarios = expand(raw_scenarios)
    if not scenarios:
        ap.error("scenario expansion produced no scenarios")
    root = C.OUT / f"synth_{a.tag}" / a.std
    out = root / "summary.json"
    root.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    provenance = C.runtime_provenance()
    scenarios_input = C.file_provenance(a.scenarios)
    jobs = [(n, p, a.std, side, a.tag, candidate_overrides, a.force, provenance)
            for n, p in scenarios.items() for side in ("oracle", "candidate")]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for msg in pool.map(run_one, jobs, chunksize=1):
            print(msg, flush=True)

    rows = {}
    for name, params in scenarios.items():
        base = root / name
        manifests = {
            side: base / side / "manifest.json" for side in ("oracle", "candidate")
        }
        missing = [str(path) for path in manifests.values() if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"refusing partial synthetic summary for {name}; "
                f"missing manifests: {missing}"
            )
        om = _validate_cached_run(
            manifests["oracle"], base / "oracle" / "trace.csv.ch0",
            _synth_expectations(
                name, params, a.std, "oracle", a.tag, candidate_overrides, provenance,
            ),
            provenance,
        )
        mm = _validate_cached_run(
            manifests["candidate"], base / "candidate" / "trace.csv.ch0",
            _synth_expectations(
                name, params, a.std, "candidate", a.tag, candidate_overrides, provenance,
            ),
            provenance,
        )
        r = matchlib.match(base / "oracle" / "trace.csv.ch0",
                           base / "candidate" / "trace.csv.ch0")
        dv = r["dv"]
        if not len(dv):
            raise RuntimeError(
                f"no matched read requests for synthetic scenario {name}; "
                "refusing a partial/cycles-only summary"
            )
        if not len(r["all_o"]):
            raise RuntimeError(f"oracle synthetic scenario {name} contains no reads")
        all_oracle_latencies = r["all_o"]
        L = float(all_oracle_latencies.mean())
        if not np.isfinite(L) or L <= 0:
            raise RuntimeError(
                f"oracle synthetic scenario {name} has invalid mean read latency"
            )
        rows[name] = {
            "params": params,
            "oracle": {"L": round(L, 1),
                       "p50": float(np.percentile(all_oracle_latencies, 50)),
                       "p99": float(np.percentile(all_oracle_latencies, 99)),
                       "cycles": om["frontend_cycles"]},
            "run_dev": round(100 * (mm["frontend_cycles"] - om["frontend_cycles"])
                             / om["frontend_cycles"], 3),
            "mae": round(float(np.abs(dv).mean()) / L, 4),
            "sgn": round(float(dv.mean()) / L, 4),
            "tail": round(float(np.percentile(np.abs(dv), 99)) / L, 3),
            "matched": int(len(dv)),
            "n_oracle": int(r["n_o"]), "n_model": int(r["n_m"]),
            "cov": round(r["cov_o"], 4),
            "cov_o": round(r["cov_o"], 4), "cov_m": round(r["cov_m"], 4),
            "matcher_schema_version": r.get(
                "matcher_schema_version", matchlib.MATCHER_SCHEMA_VERSION),
            "match_mode": r.get("match_mode", "unspecified"),
        }
        for key in ("stable_eligible_o", "stable_eligible_m",
                    "stable_cov_o", "stable_cov_m", "legacy_fallback_reason"):
            if key in r:
                rows[name][key] = r[key]
    if set(rows) != set(scenarios):
        missing = sorted(set(scenarios) - set(rows))
        raise RuntimeError(
            f"refusing partial synthetic summary; missing scenarios: {missing}"
        )
    if not C.file_provenance_matches(a.scenarios, scenarios_input):
        raise RuntimeError("synthetic scenario configuration changed during the run; rerun")
    C.validate_runtime_provenance(provenance)
    C.atomic_write_json(out, {
        "manifest_schema_version": C.RUN_MANIFEST_SCHEMA_VERSION,
        "request_metric_schema_version": C.REQUEST_METRIC_SCHEMA_VERSION,
        "matcher_schema_version": matchlib.MATCHER_SCHEMA_VERSION,
        "rev": provenance["rev"], "dirty": provenance["dirty"],
        "dirty_sha256": provenance["dirty_sha256"],
        "provenance_id": provenance["identity"],
        "native_library": provenance["native_library"],
        "python_extension": provenance["python_extension"],
        "tag": a.tag, "std": a.std, "standard_config": C.STD[a.std],
        "reference_config": C.REFERENCE,
        "scenarios_input": scenarios_input,
        "candidate_overrides": candidate_overrides,
        "scenarios": scenarios,
        "results": rows,
    })
    print(f"{'scenario':34s} {'oracleL':>8s} {'o.p99':>7s} {'dev%':>7s} "
          f"{'mae/L':>6s} {'sgn/L':>7s} {'tail':>6s}")
    for n, r in rows.items():
        print(f"{n:34s} {r['oracle']['L']:8.1f} {r['oracle']['p99']:7.0f} {r['run_dev']:+7.2f} "
              f"{r['mae']:6.3f} {r['sgn']:+7.3f} {r['tail']:6.2f}")
    print(f"summary -> {out}")
    print("SYNTH_DONE")


if __name__ == "__main__":
    main()
