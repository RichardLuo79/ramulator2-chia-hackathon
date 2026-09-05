"""Trusted, isolated adapter around the existing SimpleO3 measurements.

Metric calculation and logical trace validation are imported unchanged. The
simulation interleave is copied verbatim from the pinned Python binding. The
candidate DSO is loaded only after input traces have been closed and Landlock
has removed access to them. This is not a C++ memory-safety proof: immutable
lifecycle code, static checks, and semantic review additionally restrict code.
"""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.real_core import MUTABLE, sha

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "python"), str(REPO / "tools")]
from eval import config as C
from eval import run_simpleo3 as S
from eval import artifacts as A

COMPARISONS = ["fixedlat", "md1", "wmg1", "mess"]
TRAIN = ["429.mcf", "519.lbm"]
TEST = ["433.milc", "450.soplex", "459.GemsFDTD", "549.fotonik3d"]
# Use the established full single-core ROI, not the infrastructure-smoke ROI.
INSTS = C.INSTS_SINGLE
SIM_CPU_SECONDS = 600
SIM_FILE_BYTES = 8 * 1024**3
SYSTEM_READ = ["/usr", "/lib", "/lib64", "/bin", "/dev/null"]


def evaluation_insts(root):
    """Operator-owned configuration, shared by evaluation and agent prompts."""
    path = pathlib.Path(root) / "window_policy.json"
    value = json.loads(path.read_text())["instructions_per_core"] if path.exists() else INSTS
    if isinstance(value, bool) or not isinstance(value, int) or value < C.INSTS_SINGLE:
        raise ValueError("real evaluations require at least 20,000,000 instructions per core")
    return value


def command(args, log, *, cwd=REPO, timeout=240, env=None):
    start = time.time()
    proc = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=env)
    log = pathlib.Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(shlex.join([str(a) for a in args]) + "\n" + proc.stdout + "\n" + proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"command exited {proc.returncode}: {proc.stderr[-6000:]} {proc.stdout[-2000:]}")
    return {"command": [str(a) for a in args], "wall_seconds": time.time() - start,
            "log_sha256": sha(log.read_bytes())}


def sandbox_command(args, *, read, write, cwd, log, library_path="", cpu=120,
                    file_bytes=256 * 1024**2):
    policy = {"read": SYSTEM_READ + [str(p) for p in read],
              "write": [str(p) for p in write], "cwd": str(cwd),
              "cpu_seconds": cpu, "memory_bytes": 4 * 1024**3,
              "library_path": str(library_path), "file_bytes": file_bytes}
    policy_path = pathlib.Path(log).with_suffix(".policy.json")
    atomic_write_json(policy_path, policy)
    return command([sys.executable, REPO / "tools/chia_loop/sandbox.py", policy_path,
                    "--", *args], log, cwd=cwd, timeout=cpu + 60)


def prepare_runtime(root):
    root = pathlib.Path(root)
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    include = root / "export"
    # Export headers only, not old results, arbitrary working-tree files, or git.
    for directory in ("src", "ext/fmt/include", "ext/yaml-cpp/include"):
        for source in (REPO / directory).rglob("*"):
            if source.is_file() and source.suffix in {".h", ".hpp"}:
                dest = include / source.relative_to(REPO)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
    build = REPO / "build-bench"
    C.optimized_build_provenance(build)
    link = shlex.split((build / "CMakeFiles/ramulator.dir/link.txt").read_text())
    new_link = []
    objects = {}
    i = 0
    while i < len(link):
        token = link[i]
        if token == "-o":
            new_link.extend(["-o", str(runtime / "libramulator.so")]); i += 2; continue
        if token.endswith("atomic_controller.cpp.o"):
            i += 1; continue
        if token.endswith((".o", ".a")):
            source = (build / token).resolve()
            objects[str(source.relative_to(build))] = sha(source.read_bytes())
            token = str(source)
        new_link.append(token); i += 1
    command(new_link, root / "logs/trusted_link.log", cwd=build)
    binding = (REPO / "src/ramulator/python/bindings.cpp").read_text()
    body = binding.split("  void run() {", 1)[1].split("\n  void finalize()", 1)[0].rsplit("\n  }", 1)[0]
    driver_template = (REPO / "tools/chia_loop/isolated_sim.cpp").read_text()
    driver = runtime / "isolated_sim.cpp"
    driver.write_text(driver_template.replace("  // CHIA_INTERLEAVE_BODY", body))
    flags = ["-O3", "-DNDEBUG", "-std=c++20", "-I" + str(include / "src"),
             "-I" + str(include / "ext/fmt/include"), "-I" + str(include / "ext/yaml-cpp/include")]
    command(["/usr/bin/g++", *flags, driver, "-L" + str(runtime), "-lramulator",
             "-ldl", "-l:libseccomp.so.2", "-Wl,-rpath," + str(runtime),
             "-o", runtime / "isolated_sim"], root / "logs/driver_build.log")
    payload = {"object_sha256": objects, "binding_sha256": sha(binding),
               "interleave_body_sha256": sha(body), "driver_sha256": sha(driver.read_bytes()),
               "library_sha256": sha((runtime / "libramulator.so").read_bytes()),
               "executable_sha256": sha((runtime / "isolated_sim").read_bytes()),
               "optimization": "-O3", "compile_flags": flags,
               "export_hashes": {str(p.relative_to(include)): sha(p.read_bytes())
                                 for p in include.rglob("*") if p.is_file()}}
    atomic_write_json(root / "runtime_manifest.json", payload)
    return payload


def compile_candidate(root, source, output_dir):
    root, output_dir = pathlib.Path(root), pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file = output_dir / "atomic_controller.cpp"
    file.write_text(source)
    manifest = json.loads((root / "runtime_manifest.json").read_text())
    runtime = root / "runtime"
    args = ["/usr/bin/g++", *manifest["compile_flags"], "-fPIC", "-shared",
            "-Wl,-z,defs", file, "-L" + str(runtime), "-lramulator",
            "-o", output_dir / "candidate.so"]
    evidence = sandbox_command(args, read=[root / "export", runtime, output_dir],
        write=[output_dir], cwd=output_dir, log=output_dir / "compile.log", cpu=180)
    evidence.update({"source_sha256": sha(source), "plugin_sha256": sha((output_dir / "candidate.so").read_bytes()),
                     "optimization": "-O3"})
    atomic_write_json(output_dir / "build.json", evidence)
    return str(output_dir / "candidate.so")


def audit_candidate_trace(path, stats):
    """Check committed departures without retaining millions of CSV objects."""
    counts = {"0": 0, "1": 0}
    latency_sum = 0
    with pathlib.Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            counts[row["type"]] += 1
            latency = int(row["depart"]) - int(row["arrive"])
            if latency <= 0:
                raise RuntimeError("candidate committed a non-positive latency")
            if row["type"] == "0":
                latency_sum += latency
    if counts["0"] != stats["num_read_reqs"] or counts["1"] != stats["num_write_reqs"]:
        raise RuntimeError("candidate admission trace/count mismatch")
    if latency_sum != stats["read_latency"]:
        raise RuntimeError("candidate committed departures disagree with callback latency sum")


def run_one(root, wl, model, label, plugin, *, split, runtime_root=None, candidate_overrides=None):
    try:
        result = _run_one(root, wl, model, label, plugin, split=split,
            runtime_root=runtime_root, candidate_overrides=candidate_overrides)
        archive_completed_run(pathlib.Path(root) / split / "simpleo3/DDR5" / wl / label)
        return result
    except Exception as exc:
        directory = pathlib.Path(root) / split / "simpleo3/DDR5" / wl / label
        # Preserve failed output too, but never publish it as a scored run.
        # subprocess.run has reaped the process before this handler executes.
        if directory.exists() and not (directory / "manifest.json").exists():
            atomic_write_json(directory / "failure.json", {"error": str(exc)[-8000:], "time": time.time(),
                "complete": False, "eligible_for_metrics": False})
            partials = list(directory.glob(".incomplete-*/*.ch0"))
            if partials:
                try:
                    archive_failed_run(directory)
                except Exception as archive_error:
                    # Storage failure must not hide the original simulator error.
                    atomic_write_json(directory / "archive_failure.json", {
                        "error": str(archive_error), "time": time.time(),
                        "eligible_for_metrics": False})
        raise


def _run_one(root, wl, model, label, plugin, *, split, runtime_root=None, candidate_overrides=None):
    import ramulator
    import yaml
    root = pathlib.Path(root)
    insts = evaluation_insts(root)
    runtime_root = pathlib.Path(runtime_root) if runtime_root else root
    out = root / split / "simpleo3/DDR5" / wl / label
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise RuntimeError("refusing to overwrite a completed evaluation")
    staging = C.stage_run_directory(out, "manifest.json")
    trace = C.trace_path(wl)
    trace_input = C.file_provenance(trace)
    dram = ramulator.dram.DDR5(org_preset=C.STD["DDR5"]["org"], timing_preset=C.STD["DDR5"]["timing"])
    ctrl = S._build_controller(ramulator, model, dram, "DDR5", staging / "controller_trace.csv", candidate_overrides or {})
    frontend = ramulator.frontend.SimpleO3(clock_ratio=8, traces=[trace], num_expected_insts=insts,
        llc_num_mshr_per_core=16, request_trace_path=str(staging / "trace.csv.ch0"),
        translation=ramulator.translation.NoTranslation(max_addr=2**33))
    memory = ramulator.memory_system.GenericDRAM(clock_ratio=3, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    config = {"frontend": frontend.to_config(), "memory_system": memory.to_config()}
    atomic_write_json(out / "config.json", config)
    runtime = runtime_root / "runtime"
    read = [runtime, trace, out / "config.json"]
    if plugin:
        read.append(plugin)
    if model == "mess":
        read.append(C.MESS_CURVES["DDR5"])
    stats_file = staging / "stats.yaml"
    evidence = sandbox_command([runtime / "isolated_sim", out / "config.json", plugin or "-", staging, stats_file],
        read=read, write=[staging], cwd=staging, library_path=runtime, log=out / "simulation.log",
        cpu=SIM_CPU_SECONDS, file_bytes=SIM_FILE_BYTES)
    stats = yaml.safe_load(stats_file.read_text())
    logical_rows = S._validate_logical_request_trace(staging / "trace.csv.ch0")
    S._validate_fixed_roi_frontend_stats(stats["frontend"], core_count=1, insts_per_core=insts, logical_rows=logical_rows)
    ctrlstats = stats["memory_system"]["controller"]
    if model == "candidate":
        for kind in ("read", "write"):
            if ctrlstats[f"num_{kind}_reqs"] != ctrlstats[f"num_{kind}_reqs_served"]:
                raise RuntimeError("candidate callbacks not fully drained")
            if ctrlstats[f"peak_inflight_{kind}s"] > 64:
                raise RuntimeError("candidate exceeded fixed admission capacity")
        audit_candidate_trace(staging / "controller_trace.csv.ch0", ctrlstats)
    payload = {"manifest_schema_version": C.RUN_MANIFEST_SCHEMA_VERSION,
        "frontend": "SimpleO3", "std": "DDR5", "workload": wl, "model": model, "label": label,
        "insts_per_core": insts, "fixed_issue_roi": True,
        "request_trace_scope": S.REQUEST_TRACE_SCOPE, "request_trace_timebase": S.REQUEST_TRACE_TIMEBASE,
        "reference_config": C.REFERENCE, "standard_config": C.STD["DDR5"],
        "trace_inputs": [trace_input], "resolved_config": config,
        "per_core_cycles": [stats["frontend"]["cycles_recorded_core_0"]],
        "frontend_stats": stats["frontend"], "controller_stats": ctrlstats,
        "wall_s": stats["simulation_wall_s"],
        "construction_and_simulation_wall_s": stats["construction_and_simulation_wall_s"],
        "process_evidence": evidence,
        "native_library": C.file_provenance(runtime / "libramulator.so"),
        "candidate_plugin": C.file_provenance(plugin) if plugin else None,
        "runtime_manifest_sha256": sha((runtime_root / "runtime_manifest.json").read_bytes()),
        "optimization": "-O3", "input_access_after_candidate_load": "denied_by_landlock" if plugin else "trusted_model"}
    if C.file_provenance(trace)["sha256"] != trace_input["sha256"]:
        raise RuntimeError("trace input changed during evaluation")
    S._publish_simpleo3_traces(staging / "trace.csv.ch0", out / "trace.csv.ch0",
        staging / "controller_trace.csv.ch0", out / "controller_trace.csv.ch0", out / "manifest.json", payload)
    shutil.move(stats_file, out / "stats.yaml")
    staging.rmdir()
    return payload


def evaluate(root, models, *, plugin=None, label=None, split="training", workers=2):
    root = pathlib.Path(root)
    workloads = TRAIN if split == "training" else TEST
    jobs = [(wl, model) for wl in workloads for model in models]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, root, wl, model, label if model == "candidate" else model,
                               plugin if model == "candidate" else None, split=split) for wl, model in jobs]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            archive_completed_run(root / split / "simpleo3/DDR5" / result["workload"] / result["label"])
    labels = [label if m == "candidate" else m for m in models if m != "oracle"]
    if not labels:
        return {}
    report = root / split / "reports" / ((label or "comparisons") + ".json")
    env = os.environ.copy()
    env["EVAL_OUT"] = str(root / split)
    command([sys.executable, REPO / "tools/eval/postprocess.py", "--frontend", "simpleo3", "--std", "DDR5",
             "--workloads", *workloads, "--models", ",".join(labels), "--output", report],
            root / "logs" / (split + "_" + (label or "comparisons") + "_metrics.log"), env=env, timeout=3600)
    return json.loads(report.read_text())["models"]


def archive_failed_run(directory):
    """Archive only a closed failed run, never promote its output to metrics."""
    from eval import archive_results as AR
    directory = pathlib.Path(directory)
    if (directory / "manifest.json").exists():
        raise RuntimeError("failed-run archival cannot accept a completed run")
    failure = json.loads((directory / "failure.json").read_text())
    if failure.get("complete") is not False or failure.get("eligible_for_metrics") is not False:
        raise RuntimeError("failed-run archival requires explicit failure evidence")
    partials = list(directory.glob(".incomplete-*/*.ch0"))
    archive = directory / "failed_archive_manifest.json"
    if partials:
        AR.compress(partials, archive, 3, allow_incomplete=True)
    if archive.exists():
        AR.verify(archive)


def archive_completed_run(directory):
    """Only a completed run is eligible; each run owns its archive manifest."""
    from eval import archive_results as AR
    directory = pathlib.Path(directory)
    if not (directory / "manifest.json").exists():
        raise RuntimeError("cannot archive an incomplete evaluation")
    raw = [p for p in (directory / "trace.csv.ch0", directory / "controller_trace.csv.ch0") if p.exists()]
    if raw:
        AR.compress(raw, directory / "archive_manifest.json", 3)


def verify_run_archives(root):
    """Check per-run archives, then publish a root-level restoration manifest."""
    from eval import archive_results as AR
    root = pathlib.Path(root)
    manifest = AR._new_manifest()
    count = 0
    for split in ("training", "test", "parity", "preflight"):
        for run in sorted((root / split).glob("simpleo3/DDR5/*/*/manifest.json")):
            directory = run.parent
            archive_completed_run(directory)
            archive = directory / "archive_manifest.json"
            AR.verify(archive)
            for entry in json.loads(archive.read_text())["artifacts"].values():
                raw, stored = AR._entry_paths(entry, archive)
                relative = str(raw.relative_to(root))
                manifest["artifacts"][relative] = {**entry, "raw_path": relative,
                    "archive_path": str(stored.relative_to(root))}
                count += 1
    if not count:
        raise RuntimeError("no completed trace archives to verify")
    atomic_write_json(root / "archive_manifest.json", manifest)
    return count


def training_diagnostics(root, label, workload, kind="extremes", limit=40, *,
                         start_row=0, arrival_min=None, arrival_max=None, request_type=None):
    """Only training IDs, caller-owned label; paths never come from model input."""
    if workload not in TRAIN or kind not in {"extremes", "logical", "controller", "stats"}:
        raise ValueError("unknown training diagnostic")
    limit = max(1, min(int(limit), 200))
    start_row = max(0, int(start_row))
    arrival_min = None if arrival_min is None else int(arrival_min)
    arrival_max = None if arrival_max is None else int(arrival_max)
    if request_type not in (None, 0, 1):
        raise ValueError("request_type must be 0 (read), 1 (write), or omitted")
    if arrival_min is not None and arrival_max is not None and arrival_min > arrival_max:
        raise ValueError("arrival_min exceeds arrival_max")
    base = pathlib.Path(root) / "training/simpleo3/DDR5" / workload
    import pandas as pd
    if kind == "stats":
        result = {}
        for side in ("oracle", label):
            manifest = json.loads((base / side / "manifest.json").read_text())
            result[side] = {key: manifest.get(key) for key in
                ("per_core_cycles", "frontend_stats", "controller_stats", "wall_s", "insts_per_core")}
        return result
    if kind in {"logical", "controller"}:
        filename = "trace.csv.ch0" if kind == "logical" else "controller_trace.csv.ch0"
        result = {}
        for side in ("oracle", label):
            selected, skipped = [], 0
            with pd.read_csv(A.resolve(base / side / filename), chunksize=100_000) as chunks:
                for chunk in chunks:
                    if arrival_min is not None:
                        chunk = chunk[chunk.arrive >= arrival_min]
                    if arrival_max is not None:
                        chunk = chunk[chunk.arrive <= arrival_max]
                    if request_type is not None:
                        chunk = chunk[chunk.type == request_type]
                    if skipped + len(chunk) <= start_row:
                        skipped += len(chunk)
                        continue
                    offset = max(0, start_row - skipped)
                    selected.extend(chunk.iloc[offset:offset + limit-len(selected)].to_dict("records"))
                    skipped += len(chunk)
                    if len(selected) == limit:
                        break
            result[side] = selected
        return result
    oracle = pd.read_csv(A.resolve(base / "oracle/trace.csv.ch0"))
    model = pd.read_csv(A.resolve(base / label / "trace.csv.ch0"))
    keys = ["source", "frontend_id", "frontend_sub_id"]
    paired = oracle[oracle.type == 0].merge(model[model.type == 0], on=keys, suffixes=("_o", "_m"), validate="one_to_one")
    paired["latency_oracle"] = paired.depart_o - paired.arrive_o
    paired["latency_candidate"] = paired.depart_m - paired.arrive_m
    paired["error"] = paired.latency_candidate - paired.latency_oracle
    return {"timebase": S.REQUEST_TRACE_TIMEBASE,
            "negative": paired.nsmallest(limit, "error").to_dict("records"),
            "positive": paired.nlargest(limit, "error").to_dict("records")}
