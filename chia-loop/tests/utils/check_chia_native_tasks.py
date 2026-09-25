"""Full-window native task qualification on a private, bounded CHIA cluster.

No proposer, reviewer or provider is instantiated. This checks task dispatch,
snapshot loading, complete observations, compression and unrounded scoring. It
is not a model optimization campaign or a substitute for the complete cohort.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

import ray
from chia.base.ChiaFunction import get

from ramulator_chia.framework.archive import describe_payload
from ramulator_chia.framework.build import file_sha256
from ramulator_chia.framework.candidate import BuildLimits, compile_snapshot
from ramulator_chia.framework.evaluation import (
    CandidateBuild,
    SimpleO3Case,
    SimulationLimits,
    compare_simpleo3,
    measure_simpleo3,
)
from ramulator_chia.framework.identity import canonical_json
from ramulator_chia.framework.snapshots import ModelFiles, publish_bytes, snapshot
from ramulator_chia.eval import artifacts, config


def check(
    runtime: Path,
    output: Path,
    *,
    workload: str,
    workers: int,
    reference_parity: Path | None = None,
) -> dict:
    if type(workers) is not int or not 1 <= workers <= 12:
        raise ValueError("qualification workers must be in [1, 12]")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    contract = ModelFiles(("model.cpp",), "parameters.json")
    workspace = output / "seed-workspace"
    publish_bytes(
        workspace / "model.cpp",
        (runtime / "runtime-source/tools/chia_loop/model/seed.cpp").read_bytes(),
    )
    publish_bytes(workspace / "parameters.json", b"{}")
    submitted = snapshot(workspace, output / "snapshots", contract, maximum_bytes=1024**2)
    compile_snapshot(
        runtime,
        output / "snapshots",
        submitted["candidate_id"],
        contract,
        output / "seed-build",
        limits=BuildLimits(
            cpus=1,
            timeout_seconds=120,
            source_bytes=1024**2,
            memory_bytes=2 * 1024**3,
            file_bytes=128 * 1024**2,
        ),
    )
    seed = CandidateBuild(
        output / "snapshots",
        submitted["candidate_id"],
        contract,
        output / "seed-build",
        file_sha256(output / "seed-build/build.json"),
    )
    input_path = artifacts.resolve(Path(config.trace_path(workload)))
    trace = describe_payload(
        input_path,
        "inputs/simpleo3.trace" + (".gz" if input_path.suffix == ".gz" else ""),
        codec="gzip" if input_path.suffix == ".gz" else "none",
    )
    curve = describe_payload(Path(config.MESS_CURVES["DDR5"]), "inputs/mess.txt")
    case = SimpleO3Case(workload, "training", 20_000_000, (trace,))
    limits = SimulationLimits(
        timeout_seconds=1800,
        memory_bytes=4 * 1024**3,
        file_bytes=8 * 1024**3,
        source_bytes=1024**2,
        gzip_level=3,
    )
    models = ["oracle", "candidate", "fixedlat", "md1", "wmg1", "mess"]
    # Directly use CHIA's tasks and object references, not a parallel evaluator
    # or a custom process scheduler. This cluster cannot attach to a live run.
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    temporary = tempfile.mkdtemp(prefix="chia-native-ray.")
    repo = Path(__file__).resolve().parents[2]
    ray.init(
        address="local",
        num_cpus=workers,
        object_store_memory=128 * 1024**2,
        include_dashboard=False,
        log_to_driver=False,
        namespace="chia-native-" + uuid4().hex,
        _temp_dir=temporary,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(repo) + ":" + str(repo / "python"),
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
        },
    )
    try:
        refs = {
            model: measure_simpleo3.chia_remote(
                runtime,
                case,
                model,
                output / model,
                limits=limits,
                candidate=seed if model == "candidate" else None,
                mess_curve=curve if model == "mess" else None,
            )
            for model in models
        }
        measured = {model: get(ref) for model, ref in refs.items()}
        hashes = {model: file_sha256(output / model / "measurement.json") for model in models}
        scored = {
            model: compare_simpleo3.chia_remote(
                output / "oracle",
                output / model,
                oracle_receipt_sha256=hashes["oracle"],
                model_receipt_sha256=hashes[model],
                minimum_oracle_owner_reads=10_000,
            )
            for model in models
            if model != "oracle"
        }
        scores = {model: get(ref) for model, ref in scored.items()}
        reference = None
        if reference_parity is not None:
            previous = json.loads(reference_parity.read_text())
            if not previous.get("passed") or previous["workload"] != workload:
                raise ValueError("reference parity fixture is not a passed check of this workload")
            for old in previous["results"]:
                new = measured[old["model"]]
                if new["per_core_cycles"] != old["core_cycles"]:
                    raise AssertionError("core cycles differ from the existing parity fixture")
                for before, after in (
                    ("trace.csv.ch0.gz", "logical.csv.ch0"),
                    ("controller_trace.csv.ch0.gz", "controller.csv.ch0"),
                ):
                    if old["full_trace_sha256"][before] != new["traces"][after]["logical_sha256"]:
                        raise AssertionError("full trace differs from the existing parity fixture")
            reference = {
                "sha256": file_sha256(reference_parity),
                "full_trace_and_cycle_parity": True,
            }
        report = {
            "passed": True,
            "paid_calls": 0,
            "workers": workers,
            "ray_logs": temporary,
            "workload": workload,
            "instructions_per_core": case.instructions_per_core,
            "scope": (
                "one full-window task qualification; not a complete training cohort or campaign"
            ),
            "runtime_sha256": file_sha256(runtime / "runtime_manifest.json"),
            "measurement_receipts": hashes,
            "source_snapshot": submitted,
            "reference_parity": reference,
            "scores": scores,
        }
        publish_bytes(output / "native-tasks.json", canonical_json(report).encode())
        return report
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--workload", default="429.mcf")
    parser.add_argument("--reference-parity", type=Path)
    args = parser.parse_args()
    result = check(
        args.runtime.absolute(),
        args.output.absolute(),
        workload=args.workload,
        workers=args.workers,
        reference_parity=args.reference_parity,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "paid_calls", "workers", "reference_parity")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
