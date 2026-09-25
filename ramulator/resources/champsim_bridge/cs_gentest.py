#!/usr/bin/env python3
"""Run a hash-recorded candidate-versus-oracle ChampSim transfer matrix.

The runner rejects dirty tracked source trees and records every binary hash,
but it cannot prove that a pre-existing untracked library was built from the
adjacent revision. Rebuild each library from its clean worktree immediately
before invoking this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TRACES = (
    "603.bwaves_s-1080B",
    "605.mcf_s-1554B",
    "619.lbm_s-2676B",
    "623.xalancbmk_s-165B",
    "649.fotonik3d_s-1176B",
    "654.roms_s-1021B",
)
CYCLE_PATTERN = re.compile(r"Simulation finished CPU 0 instructions: \d+ cycles: (\d+)")
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
BRIDGE_SOURCE_DIR = Path(__file__).resolve().parent
BRIDGE_SOURCE_LOCATIONS = {
    "ramulator_bridge.h": Path("inc/ramulator_bridge.h"),
    "ramulator_bridge_identity.h": Path("inc/ramulator_bridge_identity.h"),
    "ramulator_bridge_iface.h": Path("inc/ramulator_bridge_iface.h"),
    "ramulator_bridge.cc": Path("src/ramulator_bridge.cc"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_output(command: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, text=True, env=env).strip()


def git_provenance(path: Path) -> dict[str, object]:
    root = Path(checked_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"]))
    revision = checked_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = bool(
        checked_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ]
        )
    )
    if dirty:
        raise RuntimeError(f"refusing a dirty tracked source tree: {root}")
    return {"git_root": str(root), "revision": revision, "dirty": False}


def nearest_git_provenance(path: Path) -> dict[str, object]:
    """Resolve provenance from a file/directory or one of its parents."""
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        try:
            return git_provenance(candidate)
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError(f"no git worktree contains {path}")


def bridge_source_provenance(champsim_root: Path) -> dict[str, dict[str, object]]:
    """Hash and verify the canonical and installed ChampSim bridge sources."""
    result = {}
    for name, installed_relative in BRIDGE_SOURCE_LOCATIONS.items():
        canonical = (BRIDGE_SOURCE_DIR / name).resolve()
        installed = (champsim_root / installed_relative).resolve()
        for path in (canonical, installed):
            if not path.is_file():
                raise FileNotFoundError(path)
        canonical_sha256 = sha256(canonical)
        installed_sha256 = sha256(installed)
        if installed_sha256 != canonical_sha256:
            raise RuntimeError(
                f"installed ChampSim bridge source differs from canonical copy: "
                f"{installed} != {canonical}"
            )
        result[name] = {
            "canonical_path": str(canonical),
            "installed_path": str(installed),
            "size": canonical.stat().st_size,
            "sha256": canonical_sha256,
        }
    return result


def runtime_environment(library_dir: Path, config: Path, ticks_per_8: int) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LD_LIBRARY_PATH": str(library_dir),
        "RAMULATOR_CONFIG": str(config),
        "RAMULATOR_TICKS_PER_8": str(ticks_per_8),
    }


def verify_library_resolution(
    champsim: Path, library_dir: Path, config: Path, ticks_per_8: int
) -> tuple[Path, str]:
    expected = (library_dir / "libramulator.so").resolve()
    if not expected.is_file():
        raise RuntimeError(f"missing library: {expected}")
    output = checked_output(
        ["ldd", str(champsim)],
        env=runtime_environment(library_dir, config, ticks_per_8),
    )
    match = re.search(r"libramulator\.so\s*=>\s*(\S+)", output)
    if not match:
        raise RuntimeError("ChampSim does not resolve a linked libramulator.so")
    resolved = Path(match.group(1)).resolve()
    if resolved != expected:
        raise RuntimeError(f"resolved {resolved}, expected {expected}")
    return resolved, output


def parse_candidate(value: str) -> tuple[str, Path]:
    try:
        label, directory = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate must be LABEL=LIBRARY_DIR") from error
    if label == "oracle" or not LABEL_PATTERN.fullmatch(label):
        raise argparse.ArgumentTypeError(f"invalid candidate label: {label!r}")
    return label, Path(directory).resolve()


def signed_deviation(candidate: int, oracle: int) -> float:
    if oracle <= 0:
        raise ValueError("oracle cycles must be positive")
    return 100.0 * (candidate - oracle) / oracle


def summarize(runs: dict[str, dict[str, object]], label: str) -> dict[str, object]:
    deviations = [
        signed_deviation(int(row[label]["cycles"]), int(row["oracle"]["cycles"]))
        for row in runs.values()
    ]
    worst_index = max(range(len(deviations)), key=lambda index: abs(deviations[index]))
    return {
        "mean_absolute_core_cycle_deviation_pct": statistics.mean(map(abs, deviations)),
        "mean_signed_core_cycle_deviation_pct": statistics.mean(deviations),
        "worst_absolute_core_cycle_deviation_pct": abs(deviations[worst_index]),
        "worst_trace": tuple(runs)[worst_index],
        "summed_process_wall_seconds": sum(
            float(row[label]["wall_seconds"]) for row in runs.values()
        ),
    }


def run_one(
    *,
    trace_name: str,
    label: str,
    library_dir: Path,
    config: Path,
    champsim: Path,
    trace_dir: Path,
    output_dir: Path,
    ticks_per_8: int,
    warmup: int,
    roi: int,
) -> tuple[str, str, dict[str, object]]:
    trace = (trace_dir / f"{trace_name}.champsimtrace.xz").resolve()
    output = output_dir / f"{trace_name}_{label}.txt"
    command = [
        str(champsim),
        "-w",
        str(warmup),
        "-i",
        str(roi),
        str(trace),
    ]
    start = time.perf_counter()
    with output.open("w") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=runtime_environment(library_dir, config, ticks_per_8),
            check=False,
        )
    wall_seconds = time.perf_counter() - start
    text = output.read_text(errors="replace")
    match = CYCLE_PATTERN.search(text)
    if completed.returncode or not match:
        raise RuntimeError(
            f"{label}/{trace_name} failed with rc={completed.returncode}; see {output}"
        )
    return (
        trace_name,
        label,
        {
            "cycles": int(match.group(1)),
            "wall_seconds": wall_seconds,
            "output": str(output),
            "output_sha256": sha256(output),
            "config_used": str(config),
            "config_used_sha256": sha256(config),
            "command": command,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champsim", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--oracle-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--oracle-library-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        metavar="LABEL=LIBRARY_DIR",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ticks-per-8", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2_000_000)
    parser.add_argument("--roi", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--record-requests", action="store_true",
                        help="derive per-run configs that record per-request "
                             "traces (oracle: ReqTraceRecorder plugin; candidate: "
                             "trace_path) into the output directory")
    args = parser.parse_args()

    champsim = args.champsim.resolve()
    trace_dir = args.trace_dir.resolve()
    oracle_config = args.oracle_config.resolve()
    candidate_config = args.candidate_config.resolve()
    oracle_library = args.oracle_library_dir.resolve()
    candidates = dict(args.candidate)
    output_dir = args.output_dir.resolve()

    if len(candidates) != len(args.candidate):
        raise ValueError("candidate labels must be unique")
    if (args.ticks_per_8 <= 0 or args.warmup < 0 or args.roi <= 0 or
            not 1 <= args.workers <= 12):
        raise ValueError(
            "ticks and ROI must be positive, warmup non-negative, and workers in [1, 12]")
    for path in (champsim, oracle_config, candidate_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    champsim_provenance = nearest_git_provenance(champsim)
    bridge_sources = bridge_source_provenance(
        Path(str(champsim_provenance["git_root"]))
    )
    traces = {name: (trace_dir / f"{name}.champsimtrace.xz").resolve() for name in TRACES}
    for path in traces.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {"oracle": oracle_library, **candidates}
    provenance = {}
    ldd_outputs = {}
    for label, library_dir in variants.items():
        config = oracle_config if label == "oracle" else candidate_config
        resolved, ldd_output = verify_library_resolution(
            champsim, library_dir, config, args.ticks_per_8
        )
        provenance[label] = {
            **git_provenance(library_dir),
            "library": str(resolved),
            "library_sha256": sha256(resolved),
        }
        ldd_outputs[label] = ldd_output
        (output_dir / f"ldd_{label}.txt").write_text(ldd_output + "\n")

    def derive_request_config(base: Path, trace_name: str, label: str) -> Path:
        d = json.loads(base.read_text())
        ctrl = d["memory_system"]["controllers"][0]
        req_prefix = output_dir / f"{trace_name}_{label}_req.csv"
        if ctrl.get("impl") == "GenericDDR":
            ctrl.setdefault("controller_plugins", []).append(
                {"impl": "ReqTraceRecorder", "path": str(req_prefix)})
        else:
            ctrl["trace_path"] = str(req_prefix)
        p = output_dir / "request_configs" / f"{trace_name}_{label}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=1))
        return p

    jobs = []
    for trace_name in TRACES:
        for label, library_dir in variants.items():
            base_cfg = oracle_config if label == "oracle" else candidate_config
            job_cfg = (derive_request_config(base_cfg, trace_name, label)
                       if args.record_requests else base_cfg)
            jobs.append(
                {
                    "trace_name": trace_name,
                    "label": label,
                    "library_dir": library_dir,
                    "config": job_cfg,
                    "champsim": champsim,
                    "trace_dir": trace_dir,
                    "output_dir": output_dir,
                    "ticks_per_8": args.ticks_per_8,
                    "warmup": args.warmup,
                    "roi": args.roi,
                }
            )

    runs: dict[str, dict[str, object]] = {name: {} for name in TRACES}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for trace_name, label, result in pool.map(lambda job: run_one(**job), jobs):
            runs[trace_name][label] = result
            print(f"{trace_name} {label}: {result['cycles']:,} cycles", flush=True)

    for row in runs.values():
        oracle_cycles = int(row["oracle"]["cycles"])
        for label in candidates:
            row[label]["signed_core_cycle_deviation_pct"] = signed_deviation(
                int(row[label]["cycles"]), oracle_cycles
            )

    payload = {
        "experiment": "candidate_ddr5_champsim_transfer",
        "parameters": {
            "ticks_per_8_champsim_mc_steps": args.ticks_per_8,
            "warmup_instructions": args.warmup,
            "roi_instructions": args.roi,
            "workers": args.workers,
        },
        "champsim": {
            **champsim_provenance,
            "binary": str(champsim),
            "binary_sha256": sha256(champsim),
            "bridge_sources": bridge_sources,
        },
        "configs": {
            "oracle": {"path": str(oracle_config), "sha256": sha256(oracle_config)},
            "candidate": {
                "path": str(candidate_config),
                "sha256": sha256(candidate_config),
            },
        },
        "traces": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in traces.items()
        },
        "sources": provenance,
        "runs": runs,
        "summaries": {label: summarize(runs, label) for label in candidates},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
