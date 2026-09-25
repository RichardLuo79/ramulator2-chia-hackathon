"""Frozen-model multiprogrammed transfer using the existing native evaluator.

Each core measures 20M instructions after a 2M warmup barrier. Legacy finite
jobs park completed cores. The optional background-replay policy freezes each
core's measurements but keeps its traffic active until the last core finishes.
Neither policy models communicating threads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import ray
from chia.base.ChiaFunction import ChiaFunction, get

from ramulator_chia.framework.candidate import BuildLimits
from ramulator_chia.framework.dram import MODEL_FILES, build_model, measurement
from ramulator_chia.framework.evaluation import SimulationLimits
from ramulator_chia.framework.external_frontends import (
    ExternalHost, TransferCase, describe_champsim_trace, prepare_host,
)
from ramulator_chia.framework.identity import canonical_json, file_sha256
from ramulator_chia.framework.archive import Payload
from ramulator_chia.framework.measurement_reports import compare_transfer
from ramulator_chia.framework.scoring import aggregate
from ramulator_chia.framework.snapshots import publish_bytes, replace_file, snapshot
from ramulator_chia.recovery import exclusive_lock

SEED = "dpc4-multiprogrammed-heldout-20260912-v1"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    publish_bytes(path, canonical_json(value).encode())


def status(root, stage, **values):
    record = dict(stage=stage, time=time.time(), pid=os.getpid(), **values)
    replace_file(root, "status.json", canonical_json(record).encode())
    print(canonical_json(record), flush=True)


def select_mixes(rows):
    """Eight mixes per core count; stratification never consults model errors."""
    def ordered(group):
        return sorted((r["name"] for r in rows if group(r)),
                      key=lambda n: hashlib.sha256((SEED + "\0" + n).encode()).hexdigest())

    non_google = ordered(lambda r: r["category"] != "google")
    sierra = ordered(lambda r: r["family"] == "sierra")
    tahoe = ordered(lambda r: r["family"] == "tahoe")
    if (len(non_google), len(sierra), len(tahoe)) != (9, 8, 8):
        raise ValueError("this declared mix design requires the frozen 9 + 8 + 8 cohort")
    mixes = []
    for cores in (2, 4):
        position = 0

        def take_non_google(count):
            nonlocal position
            result = [non_google[(position + i) % len(non_google)] for i in range(count)]
            position += count
            return result

        def add(group, members):
            name = f"c{cores}_{group}_{1 + sum(m['cores'] == cores for m in mixes):02d}"
            # Rotate workload placement deterministically, not by measured results.
            members = sorted(members, key=lambda n: hashlib.sha256(
                (SEED + "\0" + name + "\0" + n).encode()).hexdigest())
            if len(members) != cores or len(set(members)) != cores:
                raise ValueError("each mix must have one distinct program per core")
            mixes.append(dict(name=name, cores=cores, group=group, programs=members))

        for _ in range(2):
            add("non_google", take_non_google(cores))
        add("google", sierra[:cores])
        add("google", tahoe[:cores])
        for i in range(4):
            google = ([sierra[4 + i], tahoe[4 + i]] if cores == 4 else
                      [sierra[2 + i // 2] if i % 2 == 0 else tahoe[2 + i // 2]])
            add("mixed", take_non_google(cores // 2) + google)
    if {n for m in mixes for n in m["programs"]} != {r["name"] for r in rows}:
        raise ValueError("mixes must cover every qualified held-out trace")
    return mixes


@dataclass(frozen=True, kw_only=True)
class MultiProgramCase(TransferCase):
    companions: tuple[TransferCase, ...]
    placement: Payload | None = None
    completion_policy: str = "finite"

    def __post_init__(self):
        super().__post_init__()
        if self.frontend != "champsim" or self.stage not in ("training", "validation", "test") or self.cores not in (1, 2, 4, 8):
            raise ValueError("evaluation requires 1, 2, 4 or 8 training/validation/test ChampSim jobs")
        if self.completion_policy not in ("finite", "background-replay"):
            raise ValueError("unknown ChampSim completion policy")
        for item in self.companions:
            if (item.frontend, item.stage, item.warmup_instructions, item.roi_instructions) != (
                self.frontend, self.stage, self.warmup_instructions, self.roi_instructions
            ) or item.instruction_inventory["host_sha256"] != self.instruction_inventory["host_sha256"]:
                raise ValueError("programs must share the same frontend, stage and instruction windows")

    @property
    def cores(self):
        return 1 + len(self.companions)

    def identity(self):
        first = super().identity()
        others = [item.identity() for item in self.companions]
        result = {"frontend": "champsim", "std": "DDR5", "workload": self.workload,
                "stage": self.stage, "num_cores": self.cores,
                "warmup_instructions": self.warmup_instructions,
                "roi_instructions": self.roi_instructions, "allow_trace_wrap": False,
                "programs": [{"core": i, **{k: r[k] for k in (
                    "input_sha256", "input_bytes", "instruction_record_bytes")}}
                             for i, r in enumerate([first, *others])],
                "instruction_identity": "per-stream-ordinal-times-core-count-plus-core-v1",
                "stopping": "finite-jobs-warmup-barrier-v1"}
        if self.placement is not None:
            result["placement"] = {"policy": "canonical-data-and-page-table-frames-v1",
                                   "sha256": self.placement.member.logical_sha256,
                                   "bytes": self.placement.member.logical_bytes}
        if self.completion_policy == "background-replay":
            result.update(stopping="background-replay-warmup-barrier-v1",
                          completion_policy=self.completion_policy,
                          allow_trace_wrap="background_only",
                          request_window="per-core-admission-ordinal-v1",
                          no_retirement_windows=2, retirement_sample_cpu_cycles=10_000_000)
        return result

    def input_files(self):
        files = self.trace_files()
        if self.placement is not None:
            files["placement.txt"] = self.placement
        return files

    def trace_files(self):
        files = list(super().input_files().items())
        for item in self.companions:
            files.extend(item.input_files().items())
        return {f"core-{i}-{name}": payload for i, (name, payload) in enumerate(files)}

    def check_statistics(self, stats, observations, *, candidate):
        if self.completion_policy == "background-replay":
            windows = stats["frontend"].get("admission_windows")
            background = stats["frontend"].get("background")
            if (not windows or len(windows) != self.cores or not background or len(background) != self.cores):
                raise ValueError("continuous contention requires complete per-core phase records")
            from ramulator_chia.eval import matchlib
            reads, _ = matchlib._load_frame(observations / "controller.csv.ch0", windows=windows)
            for window in windows:
                if len(reads[reads["src"] == window["core"]]) > window["admitted_reads"]:
                    raise ValueError("recorded foreground reads exceed admitted population")
        else:
            super().check_statistics(stats, observations, candidate=candidate)
        if self.placement is not None:
            if "[VMEM] Fixed placement loaded:" not in (observations.parent / "simulation.log").read_text():
                raise ValueError("frontend did not load the requested fixed placement")
        counts = stats["frontend"].get("per_core_instructions", [])
        if len(counts) != self.cores or any(
            not self.roi_instructions <= n < self.roi_instructions + 5 for n in counts
        ):
            raise ValueError("every finite job must finish its own 20M ROI without running ahead")


def freeze(args):
    previous = read(args.prior / "protocol.json")
    extra = read(args.gemini / "protocol.json")
    eligibility = read(args.prior / "eligibility.json")
    rows = [r for r in previous["cases"] if eligibility[r["name"]]["eligible"]]
    if extra["source_protocol_sha256"] != file_sha256(args.prior / "protocol.json"):
        raise ValueError("Gemini test and original test cohorts differ")
    if file_sha256(args.runtime / "runtime_manifest.json") != previous["runtime_sha256"]:
        raise ValueError("reuse the original frozen simulator runtime")
    selections = {**previous["selections"], "gemini_flash": extra["selection"]}
    roots = {name: args.prior for name in previous["selections"]}
    roots["gemini_flash"] = args.gemini
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, selection in selections.items():
        copied = snapshot(roots[name] / "candidates" / selection["candidate"]["candidate_id"],
                          args.output / "candidates", MODEL_FILES,
                          maximum_bytes=previous["resources"]["source_bytes"])
        if copied != selection["candidate"]:
            raise ValueError("frozen selected source changed: " + name)
    sources = {2: args.source2, 4: args.source4}
    hosts = {}
    for cores, source in sources.items():
        destination = args.output / "hosts" / f"c{cores}"
        if (destination / "host.json").exists():
            host = ExternalHost(destination, file_sha256(destination / "host.json"))
            host.record()
        else:
            host = prepare_host(destination, "champsim", source / f"bin/champsim-{cores}core", source)
        if host.record()["settings"]["num_cores"] != cores:
            raise ValueError("host core count differs from the requested configuration")
        hosts[str(cores)] = dict(root=str(host.root), receipt_sha256=host.receipt_sha256)
    protocol = dict(source_study=str(args.prior), source_protocol_sha256=file_sha256(args.prior / "protocol.json"),
        gemini_source_study=str(args.gemini), driver_sha256=file_sha256(Path(__file__)),
        runtime=str(args.runtime), runtime_sha256=previous["runtime_sha256"],
        selections=selections, cases=rows, mixes=select_mixes(rows), seed=SEED, hosts=hosts,
        resources=previous["resources"], cpus=args.cpus, optimization="-O3",
        minimum_oracle_owner_reads=previous["minimum_oracle_owner_reads"],
        stopping="Each core stops at its own phase target; warmup barrier; outstanding memory traffic may drain",
        warmup_instructions=2_000_000, roi_instructions=20_000_000,
        cache_scaling="Private structures unchanged; shared LLC capacity, queues, MSHRs and throughput scale with core count; latency unchanged",
        memory_scaling="One DDR5-4800 channel; original 32 banks and 64-entry read/write queues",
        no_llm_calls=True, no_model_edits=True, no_agent_feedback=True,
        address_mapping="Original seed-1 first-touch physical placement; exact physical equality required for request pairs",
        request_metric="Pooled paired reads within each mix, normalized by all recorded oracle reads; equal mix weight")
    save(args.output / "protocol.json", protocol)
    return protocol


def cases_from_protocol(root, protocol):
    prior = Path(protocol["source_study"])
    cases = {}
    for core_count, descriptor in protocol["hosts"].items():
        host = ExternalHost(Path(descriptor["root"]), descriptor["receipt_sha256"])
        needed = {n for mix in protocol["mixes"] if mix["cores"] == int(core_count) for n in mix["programs"]}
        traces = {}
        for name in sorted(needed):
            old = read(prior / "inputs" / name / "verified-input.json")
            payload, inventory = describe_champsim_trace(Path(old["path"]), host)
            for key in ("stored_sha256", "decoded_sha256", "instructions", "record_bytes"):
                if inventory[key] != old["inventory"][key]:
                    raise ValueError("held-out trace changed: " + name)
            save(root / "inputs" / f"c{core_count}" / (name + ".json"),
                 dict(path=old["path"], inventory=inventory))
            traces[name] = TransferCase("champsim", name, payload, instruction_inventory=inventory, stage="test")
        for mix in protocol["mixes"]:
            if mix["cores"] != int(core_count):
                continue
            members = [traces[n] for n in mix["programs"]]
            cases[mix["name"]] = (MultiProgramCase(
                frontend="champsim", workload=mix["name"], stage="test", payload=members[0].payload,
                instruction_inventory=members[0].instruction_inventory, companions=tuple(members[1:])), host)
    return cases


@ChiaFunction(num_cpus=1, max_retries=0)
def score_pair(root, name, arm, oracle, candidate, floor):
    row = compare_transfer(root / oracle["directory"], root / candidate["directory"],
        oracle_receipt_sha256=oracle["receipt_sha256"], model_receipt_sha256=candidate["receipt_sha256"],
        frontend="champsim", minimum_oracle_owner_reads=floor)
    row["cycles"]["worst_abs_per_core_pct"] = max(abs(v) for v in row["cycles"]["per_core_dev_pct"])
    save(root / "scores" / arm / (name + ".json"), row)
    return row


def collect(root, jobs, stage):
    pending = {ref: key for key, ref in jobs.items()}
    results, failures = {}, {}
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1, timeout=30)
        for ref in ready:
            key = pending.pop(ref)
            try:
                results[key] = get(ref)
                print(f"{stage} DONE {key}", flush=True)
            except Exception as exc:
                failures[str(key)] = str(exc)
                print(f"{stage} FAILED {key}: {exc}", flush=True)
        status(root, stage, completed=len(results), failed=len(failures), total=len(jobs))
    if failures:
        save(root / "failures" / (stage + ".json"), failures)
        raise RuntimeError(f"{stage} has failed jobs; successful receipts preserved")
    return results


def run(args, protocol):
    root = args.output
    status(root, "verifying_inputs")
    cases = cases_from_protocol(root, protocol)
    ray.init(address="local", num_cpus=args.cpus, include_dashboard=False, log_to_driver=False,
             object_store_memory=128 * 1024**2, _temp_dir=tempfile.mkdtemp(prefix="chia-mixes-ray."))
    resources = protocol["resources"]
    limits = SimulationLimits(resources["simulation_timeout_seconds"], resources["memory_bytes"],
                              resources["file_bytes"], resources["source_bytes"], resources["gzip_level"])
    build_limits = BuildLimits(1, resources["build_timeout_seconds"], resources["source_bytes"],
                               resources["memory_bytes"], resources["file_bytes"])
    builds = collect(root, {a: build_model.options().remote(args.runtime, root, s["candidate"], build_limits)
                           for a, s in protocol["selections"].items()}, "building")
    if not all(b["passed"] for b in builds.values()):
        raise RuntimeError("a frozen model did not build")
    # A full-window four-core case verifies the new path before the full matrix.
    first = next(m["name"] for m in protocol["mixes"] if m["cores"] == 4 and m["group"] == "mixed")
    phases = [([first], "preflight"), ([n for n in cases if n != first], "remaining")]
    scores = {}
    for names, label in phases:
        oracle = collect(root, {n: measurement.options().remote(args.runtime, root, cases[n][0],
            "oracle", limits, None, None, cases[n][1]) for n in names}, label + "_oracles")
        measured = collect(root, {(a, n): measurement.options().remote(args.runtime, root, cases[n][0],
            "candidate", limits, b, None, cases[n][1]) for n in names for a, b in builds.items()}, label + "_models")
        part = collect(root, {(a, n): score_pair.options().remote(root, n, a, oracle[n], measured[a, n],
            protocol["minimum_oracle_owner_reads"]) for n in names for a in builds}, label + "_matching")
        if label == "preflight" and any(row["pairing_error"] or row["request"] is None for row in part.values()):
            raise RuntimeError("multicore preflight did not produce valid stable request pairs")
        scores.update(part)
        save(root / (label + "-scores.json"), {a: {n: part[a, n] for n in names} for a in builds})
    reports = {}
    for cores in (2, 4):
        groups = {"all": [m["name"] for m in protocol["mixes"] if m["cores"] == cores]}
        for group in ("non_google", "google", "mixed"):
            groups[group] = [m["name"] for m in protocol["mixes"] if m["cores"] == cores and m["group"] == group]
        reports[str(cores)] = {a: {group: aggregate({n: scores[a, n] for n in names},
            expected_workloads=names, stage="test", request_objective="champsim_exact_physical_filter")
            for group, names in groups.items()} for a in builds}
    save(root / "report.json", dict(protocol_sha256=file_sha256(root / "protocol.json"), reports=reports))
    status(root, "complete", mixes=len(cases), model_comparisons=len(scores), no_llm_calls=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "gemini", "output", "runtime", "source2", "source4"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cpus <= 16:
        parser.error("cpus must be between 1 and 16")
    with exclusive_lock(args.output / "evaluation.lock"):
        try:
            protocol = freeze(args)
            if not args.prepare_only:
                run(args, protocol)
        except BaseException as exc:
            status(args.output, "stopped", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()


if __name__ == "__main__":
    main()
