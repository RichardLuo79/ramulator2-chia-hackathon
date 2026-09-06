"""No-LLM integration validation of the DDR5 evaluation/transfer pipeline.

The unchanged seed is deliberately frozen as a fixture, never as an evolved
design. --transfer-smoke reduces case COUNT, never the instruction window or
program length. A fixture root cannot be resumed as a paid optimization run.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from tools.chia_loop import evaluation_config as W, real_eval as E, real_core as P, recovery as R, transfer, artifacts
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.traffic import traffic_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--evaluation-config", type=pathlib.Path, default=W.DEFAULT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--transfer-smoke", action="store_true", help="one full-length case per transfer frontend")
    parser.add_argument("--inputs-only", action="store_true", help="bind/hash inputs without any simulation")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be in [1, 12]")
    root = args.root.resolve()
    with R.cpu_lease(E.REPO / "eval_out/chia/.cpu_leases", max(3, args.workers)):
        root.mkdir(parents=True, exist_ok=False)
        config = W.validate(json.loads(args.evaluation_config.read_text()))
        if args.transfer_smoke:
            config["name"] += "_full_window_transfer_fixture"
            for settings in config["transfer"].values():
                settings["workloads"] = settings["workloads"][:1]
        atomic_write_json(root / "evaluation_config.json", config)
        atomic_write_json(root / "window_policy.json", {"instructions_per_core": config["simpleo3"]["instructions_per_core"]})
        manifest = {"record_type": "evaluation_integration_fixture", "status": "preparing",
                    "paid_generation_calls": 0, "evolved_design": False, "started_at": time.time()}
        atomic_write_json(root / "validation.json", manifest)
        try:
            W.inventory(root, E.C.trace_path, E.C.file_provenance)
            transfer.prepare(root)
            if args.inputs_only:
                manifest.update(status="inputs_verified", simulation_cases=0)
                return
            source = (E.REPO / P.MUTABLE).read_text()
            _, regions = P.regions(source)
            expected = "voidinit_model(){}Clk_tpredict_departure(constRequest&req){returnm_clk+m_latency;}"
            if regions["INCLUDES"].strip() or re.sub(r"\s+", "", regions["CODE"]) != expected:
                raise RuntimeError("validation requires the clean fixed-delay CHIA seed")
            E.prepare_runtime(root)
            plugin = E.compile_candidate(root, source, root / "seed")
            for split in ("training", "test"):
                E.evaluate(root, ["oracle", *E.COMPARISONS], split=split, workers=args.workers)
                E.evaluate(root, ["candidate"], plugin=plugin, label="seed", split=split, workers=args.workers)
            populations = [traffic_population(root / "training/simpleo3/DDR5" / w / "oracle",
                config["simpleo3"]["minimum_oracle_owner_reads"]) for w in E.workloads(root, "training")]
            atomic_write_json(root / "training_traffic_coverage.json", populations)
            if not all(p["owner_count_pass"] for p in populations):
                raise RuntimeError("insufficient training traffic: " + ", ".join(p["workload"] for p in populations if not p["owner_count_pass"]))
            state = {"status": "frozen", "termination": "iteration_limit", "incumbent": "seed",
                "frozen_at": time.time(), "selected": {"source_path": str(root / "seed/atomic_controller.cpp"),
                "plugin": plugin, "sha256": P.sha(source)}}
            atomic_write_json(root / "selection_frozen.json", {"run_id": root.name, "source_sha256": P.sha(source),
                "incumbent": "seed", "frozen_at": state["frozen_at"]})
            atomic_write_json(root / "fixture_state.json", state)
            result = transfer.run(root, state, workers=args.workers)
            E.verify_run_archives(root)
            artifacts.compress(root, root / "aux_archive_manifest.json", min_bytes=16384)
            artifacts.verify(root / "aux_archive_manifest.json")
            manifest.update(status="completed", transfer_reports=list(result), archives_verified=True)
        except BaseException as exc:
            manifest.update(status="failed", error=str(exc)[-4000:])
            raise
        finally:
            manifest["finished_at"] = time.time()
            atomic_write_json(root / "validation.json", manifest)
            print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
