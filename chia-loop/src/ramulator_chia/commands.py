#!/usr/bin/env python3
"""Small operator entry points; no cloud provisioning or implicit model calls."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import venv

from .layout import ROOT
from .configuration import load_config


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(argv, **kwargs):
    subprocess.run([str(v) for v in argv], check=True, **kwargs)


def setup(args):
    parser = argparse.ArgumentParser(
        description="Prepare a clean environment or optimized native runtime. No inference."
    )
    parser.add_argument(
        "--component",
        choices=[
            "analysis",
            "campaign",
            "context",
            "runtime",
            "champsim",
            "speed-inputs",
        ],
        default="analysis",
    )
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".work/runtime")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cores", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument("--work-root", type=Path, default=ROOT / ".work")
    parser.add_argument("--runtime", type=Path, default=ROOT / ".work/runtime")
    a = parser.parse_args(args)
    if a.workers < 1 or a.workers > 16:
        parser.error("build workers must be in [1,16]")
    if a.component == "champsim":
        from .frontend_build import build

        print(
            build(ROOT, a.work_root.resolve(), a.runtime.resolve(), a.cores, a.workers)
        )
        return
    if a.component == "speed-inputs":
        run(
            [
                "cmake",
                "--build",
                a.runtime / "runtime-build",
                "--target",
                "ramulator-speed-inputs",
                "--parallel",
                a.workers,
            ]
        )
        return
    if a.component == "runtime":
        from ramulator_chia.framework.build import prepare_runtime

        result = prepare_runtime(
            ROOT / "ramulator", a.output, cpus=a.workers, timeout_seconds=3600
        )
        print(
            json.dumps(
                {"runtime": str(a.output), "optimization": result["optimization"]}
            )
        )
        return
    target = a.venv or ROOT / (".venv-" + a.component)
    if not (target / "bin/python").exists():
        venv.EnvBuilder(with_pip=True).create(target)
    python = target / "bin/python"
    requirements = (
        ROOT
        / {
            "analysis": "analyses/requirements.lock.txt",
            "campaign": "chia-loop/requirements.lock.txt",
            "context": "chia-loop/requirements.context.lock.txt",
        }[a.component]
    )
    run([python, "-m", "pip", "install", "-r", requirements])
    if a.component == "campaign":
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                ROOT / "chia-loop/vendor/chia",
                "-e",
                ROOT / "chia-loop",
                "-e",
                ROOT / "ramulator",
            ]
        )
    print("Prepared", target)


def fetch_data(args):
    parser = argparse.ArgumentParser(
        description="Prepare or verify the DPC4 prefixes and placement maps for a configuration."
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "results/manifests/inputs.json"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data")
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "chia-loop/configs/astra.json"
    )
    parser.add_argument("--work-root", type=Path, default=ROOT / ".work")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--disk-reserve-gib", type=float, default=32)
    parser.add_argument(
        "--extract",
        type=Path,
        help="verify and extract an archive into a NEW directory",
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--sha256", help="required external archive SHA256")
    a = parser.parse_args(args)
    if a.extract:
        if not a.destination or not a.sha256:
            parser.error("--extract requires --destination and --sha256")
        from ramulator_chia.framework.archive import extract

        print(extract(a.extract, a.destination, expected_sha256=a.sha256))
        return
    from .input_data import prepare, safe_path

    if a.download:
        cfg = load_config(a.config, a.data_root, a.work_root, prepared=False)
        receipt = prepare(
            cfg,
            a.data_root,
            a.work_root,
            workers=a.workers,
            reserve_gib=a.disk_reserve_gib,
        )
        print(
            json.dumps(
                {
                    "verified": len(receipt["files"]),
                    "receipt": str(a.data_root / "prepared-inputs.json"),
                }
            )
        )
        return
    manifest = json.loads(a.manifest.read_text())
    if a.manifest.resolve() == (ROOT / "results/manifests/inputs.json").resolve():
        cfg = load_config(a.config, a.data_root, a.work_root, prepared=False)
        cohort = cfg.experiment.evaluation.champsim
        chosen = list(cohort.traces.values()) + [
            case.placement for case in cohort.cases.values()
        ]
        names = {
            Path(item.path).relative_to(a.data_root.absolute()).as_posix()
            for item in chosen
        }
        manifest["files"] = [
            item for item in manifest["files"] if item["path"] in names
        ]
    missing = []
    checked = 0
    prepared_path = a.data_root / "prepared-inputs.json"
    prepared = (
        json.loads(prepared_path.read_text())["files"] if prepared_path.exists() else {}
    )
    for item in manifest["files"]:
        base = (
            a.archive_dir
            if item.get("kind") == "archive" and a.archive_dir
            else a.data_root
        )
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe manifest path")
        path = safe_path(base, relative)
        if not path.exists():
            missing.append(item["path"])
            continue
        stored = item["sha256"]
        if item["path"] in prepared:
            row = prepared[item["path"]]
            if row["reference_sha256"] != stored or row["decoded_sha256"] != item.get(
                "decoded_sha256"
            ):
                raise ValueError("preparation identity mismatch: " + item["path"])
            stored = row["sha256"]
        if sha(path) != stored:
            raise ValueError("checksum mismatch: " + item["path"])
        checked += 1
    print(json.dumps({"verified": checked, "missing": missing}, indent=2))
    if missing:
        raise SystemExit(2)


def reproduce_figures(args):
    parser = argparse.ArgumentParser(
        description="Reproduce the eight paper figures from checked local evidence, offline."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "analyses/chia_campaign_results.ipynb"
    )
    parser.add_argument("--figure-output", type=Path, default=ROOT / "analyses/figures")
    a = parser.parse_args(args)
    sys.path.insert(0, str(ROOT / "analyses"))
    from paper_notebook import build

    print(build(a.output, figure_output=a.figure_output))


def evaluate(args):
    if "--suite" in args:
        at = args.index("--suite")
        if at + 1 >= len(args):
            raise SystemExit("--suite needs lat-tp or speed")
        suite = args[at + 1]
        rest = args[:at] + args[at + 2 :]
        from ramulator_chia.eval.frozen import diagnostic_cli

        diagnostic_cli(suite, rest, ROOT)
        return
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen ChampSim case; never calls an LLM."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "chia-loop/configs/astra.json"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / ".data")
    parser.add_argument("--work-root", type=Path, default=ROOT / ".work")
    parser.add_argument("--runtime", type=Path, default=ROOT / ".work/runtime")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--with-oracle",
        action="store_true",
        help="also measure the reference and write verified accuracy metrics",
    )
    parser.add_argument(
        "--organization", choices=["DDR5_16Gb_x8", "DDR5_8Gb_x8", "DDR5_16Gb_x16"]
    )
    parser.add_argument("--queue-entries", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "oracle",
            "fixedlat",
            "md1",
            "wmg1",
            "mess",
            "astra_single_core",
            "astra_multicore",
            "deepseek_single_core",
            "deepseek_multicore",
            "opus_single_core",
            "gemini_single_core",
        ],
    )
    a = parser.parse_args(args)
    from ramulator_chia.framework.inputs import _staged_champsim
    from ramulator_chia.framework.evaluation import (
        measure_native,
        SimulationLimits,
        CandidateBuild,
    )
    from ramulator_chia.framework.candidate import BuildLimits, compile_snapshot
    from ramulator_chia.framework.snapshots import ModelFiles, snapshot
    from ramulator_chia.framework.archive import describe_payload

    cfg = load_config(a.config, a.data_root, a.work_root)
    cohort = cfg.experiment.evaluation.champsim
    if a.case not in cohort.cases:
        parser.error("case is not in this frozen cohort")
    spec = cohort.cases[a.case]
    group = next(
        g for g in ("training", "validation", "test") if a.case in getattr(cohort, g)
    )
    update = {
        g: ((a.case,) if g == group else ()) for g in ("training", "validation", "test")
    }
    update.update(
        cases={a.case: spec},
        traces={p: cohort.traces[p] for p in spec.programs},
        builds={
            k: v
            for k, v in cohort.builds.items()
            if k in ("1", str(len(spec.programs)))
        },
    )
    a.output.mkdir(parents=True, exist_ok=False)
    cases, hosts = _staged_champsim(cohort.model_copy(update=update), a.output)
    case = cases[group][0]
    resources = cfg.run.resources
    if a.organization:
        if a.model not in (
            "oracle",
            "astra_single_core",
            "astra_multicore",
            "deepseek_single_core",
            "deepseek_multicore",
        ):
            parser.error(
                "the frozen hardware-transfer protocol includes oracle and Astra/DeepSeek endpoints only"
            )
        from dataclasses import fields
        from ramulator_chia.eval import final_study
        from ramulator_chia.eval.hardware_sweep import HardwarePoint

        final_study.RUNTIME = a.runtime
        capacity = (8 if a.organization == "DDR5_16Gb_x8" else 4) * 1024**3
        placement = final_study.remap(
            case, hosts["champsim:" + str(len(spec.programs))], a.output, capacity
        )
        values = {f.name: getattr(case, f.name) for f in fields(case)}
        values.update(
            placement=placement, hardware=HardwarePoint(a.organization, a.queue_entries)
        )
        case = final_study.HardwareMix(**values)
    limits = SimulationLimits(
        resources.simulation_timeout_seconds,
        resources.memory_bytes,
        resources.file_bytes,
        resources.source_bytes,
        resources.gzip_level,
    )
    candidate = None
    model = a.model
    if a.model not in ("oracle", "fixedlat", "md1", "wmg1", "mess"):
        contract = ModelFiles(("model.cpp",), "parameters.json")
        selected = snapshot(
            ROOT / "results/models" / a.model,
            a.output / "candidates",
            contract,
            maximum_bytes=resources.source_bytes,
        )
        out = a.output / "model-build"
        receipt = compile_snapshot(
            a.runtime,
            a.output / "candidates",
            selected["candidate_id"],
            contract,
            out,
            limits=BuildLimits(
                1,
                resources.build_timeout_seconds,
                resources.source_bytes,
                resources.memory_bytes,
                resources.file_bytes,
            ),
        )
        if not receipt["passed"]:
            raise RuntimeError("frozen model build failed")
        candidate = CandidateBuild(
            a.output / "candidates",
            selected["candidate_id"],
            contract,
            out,
            sha(out / "build.json"),
        )
        model = "candidate"
    curve = (
        describe_payload(
            ROOT / "chia-loop/src/ramulator_chia/eval/calibration/mess_DDR5.txt",
            "inputs/mess.txt",
        )
        if model == "mess"
        else None
    )
    result = measure_native(
        a.runtime,
        case,
        model,
        a.output / "measurement",
        limits=limits,
        candidate=candidate,
        mess_curve=curve,
        host=hosts["champsim:" + str(len(spec.programs))],
    )
    if a.with_oracle and model != "oracle":
        from .framework.measurement_reports import compare_transfer
        from .framework.snapshots import publish_bytes
        from .framework.identity import canonical_json

        reference = a.output / "oracle"
        measure_native(
            a.runtime,
            case,
            "oracle",
            reference,
            limits=limits,
            host=hosts["champsim:" + str(len(spec.programs))],
        )
        comparison = compare_transfer(
            reference,
            a.output / "measurement",
            oracle_receipt_sha256=sha(reference / "measurement.json"),
            model_receipt_sha256=sha(a.output / "measurement/measurement.json"),
            frontend="champsim",
            minimum_oracle_owner_reads=cohort.minimum_oracle_owner_reads,
            include_tails=True,
        )
        publish_bytes(a.output / "comparison.json", canonical_json(comparison).encode())
    print(json.dumps(result, indent=2))


def run_campaign(args):
    if not any(flag in args for flag in ("--help", "-h", "--allow-paid", "--check")):
        raise SystemExit(
            "No inference started: run-campaign requires explicit --allow-paid."
        )
    parser = argparse.ArgumentParser(
        description="Check, start, or resume a local CHIA campaign."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=ROOT / ".work/runtime")
    parser.add_argument("--tariff", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("RAMULATOR_CHIA_DATA", ROOT / ".data")),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.environ.get("RAMULATOR_CHIA_WORK", ROOT / ".work")),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="local checks only; no inference, authentication, or campaign creation",
    )
    parser.add_argument(
        "--allow-paid", action="store_true", help="explicitly authorize model calls"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate selected endpoints on test cases after search",
    )
    parser.add_argument(
        "--evaluation-workers", type=int, help="host-only override, 1–120"
    )
    parser.add_argument(
        "--context-compaction",
        type=Path,
        help="explicit API compaction policy; defaults to the matching public template",
    )
    parser.add_argument("--vertex-execution-project")
    parser.add_argument("--capacity-cooldown-seconds", type=int, default=3600)
    parser.add_argument("--round-finish-deadline")
    a = parser.parse_args(args)
    from .configuration import compaction_policy
    from .framework.config import ExecutionOverrides
    from .preflight import check

    configuration = load_config(a.config, a.data_root, a.work_root)
    overrides = ExecutionOverrides(
        evaluation_workers=a.evaluation_workers,
        vertex_project=a.vertex_execution_project,
    )
    overrides.validate_backend(configuration)
    policy = compaction_policy(configuration, a.context_compaction)
    report = check(
        configuration,
        a.runtime,
        a.output,
        a.tariff,
        workers=a.evaluation_workers,
        compaction=policy,
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)
    if a.check:
        return
    if not a.allow_paid:
        parser.error("inference requires explicit --allow-paid")
    output = a.output.absolute()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    from ramulator_chia.framework.snapshots import publish_bytes
    from ramulator_chia.framework.identity import canonical_json

    effective = output / "resolved-launch-config.json"
    publish_bytes(
        effective, canonical_json(configuration.model_dump(mode="json")).encode()
    )
    resolved = [
        "--config",
        str(effective),
        "--output",
        str(output),
        "--runtime",
        str(a.runtime.absolute()),
        "--tariff",
        str(a.tariff.absolute()),
        "--allow-paid",
        "--capacity-cooldown-seconds",
        str(a.capacity_cooldown_seconds),
    ]
    if policy is not None:
        policy_path = output / "resolved-compaction.json"
        publish_bytes(
            policy_path, canonical_json(policy.model_dump(mode="json")).encode()
        )
        resolved += ["--context-compaction", str(policy_path)]
    for flag, value in (
        ("--evaluation-workers", a.evaluation_workers),
        ("--vertex-execution-project", a.vertex_execution_project),
        ("--round-finish-deadline", a.round_finish_deadline),
    ):
        if value is not None:
            resolved += [flag, str(value)]
    if a.evaluate:
        resolved.append("--evaluate")
    # The existing launcher retains all session, isolation and visibility checks.
    sys.argv = ["run-campaign", *resolved]
    runpy.run_module("ramulator_chia.framework.launch", run_name="__main__")


def verify_artifact(args):
    parser = argparse.ArgumentParser(
        description="Verify release file identities and missing public inputs."
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "results/manifests/release.json"
    )
    a = parser.parse_args(args)
    data = json.loads(a.manifest.read_text())
    bad = []
    for rel, digest in data["files"].items():
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe release manifest path")
        p = ROOT / rel
        if not p.is_file() or p.is_symlink() or sha(p) != digest:
            bad.append(rel)
    print(
        json.dumps(
            {"verified": len(data["files"]) - len(bad), "mismatches": bad}, indent=2
        )
    )
    if bad:
        raise SystemExit(1)


COMMANDS = {
    "setup": setup,
    "fetch-data": fetch_data,
    "reproduce-figures": reproduce_figures,
    "evaluate": evaluate,
    "run-campaign": run_campaign,
    "verify-artifact": verify_artifact,
}


def main(command=None):
    command = command or (sys.argv.pop(1) if len(sys.argv) > 1 else "")
    if command not in COMMANDS:
        raise SystemExit("Commands: " + ", ".join(COMMANDS))
    COMMANDS[command](sys.argv[1:])


if __name__ == "__main__":
    main()
