"""Full-window parity between an existing Python binding and a staged native runner.

This is an explicit infrastructure check, not an optimization campaign. It makes
no model calls and refuses to overwrite a previous check. The old binary remains
untouched so extracting the shared lifecycle can be checked against its behavior.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.framework.archive import describe_payload
from tools.chia_loop.framework.build import file_sha256


def reference_worker(native_case: Path, destination: Path):
    import ramulator._ramulator as binding
    import yaml

    from tools.eval import archive_results

    destination.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(json.loads((native_case / "config.json").read_text()))
    config["frontend"]["request_trace_path"] = str(destination / "trace.csv.ch0")
    for controller in config["memory_system"]["controllers"]:
        if "trace_path" in controller:
            controller["trace_path"] = str(destination / "controller_trace.csv")
        for plugin in controller.get("controller_plugins", []):
            if plugin["impl"] != "ReqTraceRecorder":
                raise ValueError("unexpected observation plugin in native parity fixture")
            plugin["path"] = str(destination / "controller_trace.csv")
    atomic_write_json(destination / "config.json", config)
    simulation = binding.Simulation(config)
    simulation.run()
    # Compare the same print_stats serialization used by the native runner.
    # get_stats() takes a different float-formatting path through ConfigNode.
    stats = yaml.safe_load(simulation.get_stats_yaml())
    simulation.finalize()
    atomic_write_json(destination / "stats.json", stats)
    archive_results.compress(
        [destination / "trace.csv.ch0", destination / "controller_trace.csv.ch0"],
        destination / "archive_manifest.json",
        3,
    )


def check(
    runtime_root: Path,
    destination: Path,
    *,
    workers: int,
    workload: str,
    reuse_native: Path | None = None,
) -> dict:
    from tools.chia_loop import real_eval
    from tools.eval import config as evaluation_config

    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("workers must be in [1, 12]")
    destination.mkdir(parents=True, exist_ok=False)
    reference_build = evaluation_config.optimized_build_provenance()
    runtime_manifest = json.loads((runtime_root / "runtime_manifest.json").read_text())
    if runtime_manifest["optimization"] != "-O3":
        raise ValueError("native runtime is not an optimized build")
    runtime = runtime_root / "runtime"
    for name, key in (
        ("libramulator.so", "library_sha256"),
        ("isolated_sim", "executable_sha256"),
        ("candidate.so", "seed_plugin_sha256"),
    ):
        if file_sha256(runtime / name) != runtime_manifest[key]:
            raise ValueError(f"native runtime identity changed: {name}")
    models = ["oracle", "candidate", *real_eval.COMPARISONS]
    script = Path(__file__).resolve()

    def compare(model):
        plugin = runtime / "candidate.so" if model == "candidate" else None
        native_case = (reuse_native or destination) / "training/simpleo3/DDR5" / workload / model
        if reuse_native:
            manifest = real_eval.completed_evaluation(
                reuse_native, native_case, workload, model, model, plugin, runtime_root
            )
        else:
            manifest = real_eval.run_one(
                destination,
                workload,
                model,
                model,
                plugin,
                split="training",
                runtime_root=runtime_root,
            )
        reference_case = destination / "python-reference" / model
        log = destination / f"python-{model}.log"
        environment = {
            **os.environ,
            **{
                key: "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        }
        with log.open("xb") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--reference-worker",
                    str(native_case),
                    "--output",
                    str(reference_case),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=1800,
            )
        if result.returncode:
            raise RuntimeError(f"Python reference failed for {model}; see {log}")
        stats = json.loads((reference_case / "stats.json").read_text())
        if stats["frontend"] != manifest["frontend_stats"]:
            raise AssertionError(f"frontend statistics differ for {model}")
        if stats["memory_system"]["controller"] != manifest["controller_stats"]:
            raise AssertionError(f"controller statistics differ for {model}")
        trace_hashes = {}
        for filename in ("trace.csv.ch0.gz", "controller_trace.csv.ch0.gz"):
            native = describe_payload(native_case / filename, filename, codec="gzip").member
            reference = describe_payload(reference_case / filename, filename, codec="gzip").member
            if (native.logical_sha256, native.logical_bytes) != (
                reference.logical_sha256,
                reference.logical_bytes,
            ):
                raise AssertionError(f"full trace differs for {model}: {filename}")
            trace_hashes[filename] = native.logical_sha256
        return {
            "model": model,
            "core_cycles": manifest["per_core_cycles"],
            "full_trace_sha256": trace_hashes,
            "instructions_per_core": manifest["insts_per_core"],
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(compare, models))
    report = {
        "passed": True,
        "paid_calls": 0,
        "workload": workload,
        "workers": workers,
        "native_results_reused_from": None if reuse_native is None else str(reuse_native),
        "native_runtime_manifest_sha256": file_sha256(runtime_root / "runtime_manifest.json"),
        "reference_build": reference_build,
        "reference_library": evaluation_config.file_provenance(
            evaluation_config.REPO / "libramulator.so"
        ),
        "reference_extension": evaluation_config.python_extension_provenance(),
        "results": results,
    }
    atomic_write_json(destination / "parity.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--workload", default="429.mcf")
    parser.add_argument(
        "--reuse-native", type=Path, help="verify and reuse completed native measurements"
    )
    parser.add_argument("--reference-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference_worker:
        reference_worker(args.reference_worker, args.output)
    elif args.runtime:
        print(
            json.dumps(
                check(
                    args.runtime,
                    args.output,
                    workers=args.workers,
                    workload=args.workload,
                    reuse_native=args.reuse_native,
                ),
                indent=2,
            )
        )
    else:
        parser.error("--runtime is required")


if __name__ == "__main__":
    main()
