"""Evaluate frozen CHIA models across DDR5 organizations and queue capacities.

This is an offline experiment, not another agent loop. It reuses the native
CHIA executor, source snapshots, compressed inputs, and request matcher. The
input is a completed held-out evaluation; its models and workload membership
are not selected again. Run with the same frozen Python code/SDK as that study.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import ray
from chia.base.ChiaFunction import ChiaFunction, get

from ramulator_chia.framework.archive import describe_payload
from ramulator_chia.framework.candidate import BuildLimits
from ramulator_chia.framework.dram import MODEL_FILES, build_model, measurement
from ramulator_chia.framework.evaluation import SimulationLimits
from ramulator_chia.framework.external_frontends import ExternalHost, TransferCase
from ramulator_chia.framework.identity import canonical_json, digest_json, file_sha256
from ramulator_chia.framework.measurement_reports import compare_transfer, verify_native_measurement
from ramulator_chia.framework.scoring import aggregate
from ramulator_chia.framework.snapshots import publish_bytes, replace_file, snapshot
from ramulator_chia.recovery import exclusive_lock


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    publish_bytes(path, canonical_json(value).encode())


@dataclass(frozen=True)
class HardwarePoint:
    organization: str
    queue_entries: int

    def __post_init__(self):
        if self.organization not in {"DDR5_8Gb_x8", "DDR5_16Gb_x8", "DDR5_16Gb_x16"}:
            raise ValueError("organization is outside this experiment's declared grid")
        if type(self.queue_entries) is not int or self.queue_entries not in {32, 64, 128}:
            raise ValueError("queue capacity is outside this experiment's declared grid")

    @property
    def name(self):
        return f"{self.organization}_q{self.queue_entries}"

    def settings(self):
        import ramulator

        # Let the standard resolve width/density-dependent secondary timings.
        # Never edit the serialized bank counts without resolving the standard.
        return {
            "dram": ramulator.dram.DDR5(
                org_preset=self.organization, timing_preset="DDR5_4800AN"
            ).to_config(),
            "read_buffer_size": self.queue_entries,
            "write_buffer_size": self.queue_entries,
            "wr_low_watermark": 0.5,
            "wr_high_watermark": 0.8,
        }

    def identity(self):
        return {
            "organization_preset": self.organization,
            "timing_preset": "DDR5_4800AN",
            "resolved_controller_settings": self.settings(),
        }


BASELINE = HardwarePoint("DDR5_16Gb_x8", 64)
GRID = tuple(HardwarePoint(org, q) for org in (
    "DDR5_16Gb_x8", "DDR5_8Gb_x8", "DDR5_16Gb_x16"
) for q in (64, 32, 128))


@dataclass(frozen=True, kw_only=True)
class HardwareCase(TransferCase):
    hardware: HardwarePoint

    def identity(self):
        return {**super().identity(), "hardware": self.hardware.identity()}

    def configuration(self, model, inputs, observations, parameters, curve):
        if model not in {"oracle", "candidate"}:
            raise ValueError("this sweep compares frozen candidates with the cycle-level oracle")
        config = super().configuration(model, inputs, observations, parameters, curve)
        config["memory_system"]["controllers"][0].update(self.hardware.settings())
        return config

    def check_statistics(self, stats, observations, *, candidate):
        super().check_statistics(stats, observations, candidate=candidate)
        config = read(observations.parent / "config.json")
        controller = config["memory_system"]["controllers"][0]
        if any(controller.get(k) != v for k, v in self.hardware.settings().items()):
            raise ValueError("executed controller configuration differs from hardware identity")


def status(root, stage, **values):
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **values)
    replace_file(root, "status.json", canonical_json(value).encode())
    print(canonical_json(value), flush=True)


def freeze(prior, root, runtime, cpus):
    previous = read(prior / "protocol.json")
    eligibility = read(prior / "eligibility.json")
    protocol = dict(
        source_study=str(prior), source_protocol_sha256=file_sha256(prior / "protocol.json"),
        source_report_sha256=file_sha256(prior / "report.json"),
        driver_sha256=file_sha256(Path(__file__)),
        runtime=str(runtime), runtime_sha256=file_sha256(runtime / "runtime_manifest.json"),
        host=previous["host"], host_sha256=previous["host_sha256"],
        selections=previous["selections"], resources=previous["resources"], cpus=cpus,
        cases=[r for r in previous["cases"] if eligibility[r["name"]]["eligible"]],
        grid={p.name: p.identity() for p in GRID}, baseline=BASELINE.name,
        warmup_instructions=previous["warmup_instructions"],
        roi_instructions=previous["roi_instructions"],
        minimum_oracle_owner_reads=previous["minimum_oracle_owner_reads"],
        request_objective=previous["request_objective"],
        no_llm_calls=True, no_model_edits=True, no_parameter_retuning=True,
        no_agent_feedback=True, optimization="-O3",
        membership="Fixed previous test cohort; report a common oracle-qualified subset across hardware points",
        baseline_evidence="Reuse verified original baseline measurements; do not relabel their identities",
        organization_caveat="Preset changes also change capacity/device width and derived timings",
        queue_caveat="Oracle waiting buffers exclude active/issued requests; atomic caps cover outstanding requests",
    )
    if protocol["runtime_sha256"] != previous["runtime_sha256"]:
        raise ValueError("runtime differs from the completed held-out study")
    # Immutable publication refuses a resume with changed inputs or settings.
    save(root / "protocol.json", protocol)
    for selection in protocol["selections"].values():
        copied = snapshot(prior / "candidates" / selection["candidate"]["candidate_id"],
                          root / "candidates", MODEL_FILES,
                          maximum_bytes=protocol["resources"]["source_bytes"])
        if copied != selection["candidate"]:
            raise ValueError("source snapshot differs from the frozen selected model")
    return protocol


def baseline_receipts(prior, protocol):
    """Locate the old native receipts by their actual input/model identities."""
    candidates = {v["candidate"]["candidate_id"]: arm
                  for arm, v in protocol["selections"].items()}
    names = {r["name"] for r in protocol["cases"]}
    result = {}
    for path in (prior / "measurements").glob("*/result.json"):
        receipt = read(path)
        identity = receipt["identity"]
        name = identity["case"]["workload"]
        if name not in names:
            continue
        arm = "oracle" if identity["model"] == "oracle" else candidates.get(
            identity["candidate"]["candidate_id"]
        )
        if arm is None:
            continue
        key = (BASELINE.name, name, arm)
        if key in result:
            raise ValueError("ambiguous baseline measurement")
        result[key] = {**receipt, "directory": str(prior / receipt["directory"])}
    expected = {(BASELINE.name, n, a) for n in names for a in ("oracle", *candidates.values())}
    if set(result) != expected:
        raise ValueError("baseline measurements do not cover the frozen cohort")
    return result


@ChiaFunction(num_cpus=1, max_retries=0)
def verify_baseline(receipt, expected_case, hardware):
    directory = Path(receipt["directory"])
    verified = verify_native_measurement(directory, receipt["receipt_sha256"],
        frontend="champsim", observations=TransferCase.observation_names)
    if verified["case"] != expected_case:
        raise ValueError("baseline trace/window differs from sweep input")
    controller = read(directory / "config.json")["memory_system"]["controllers"][0]
    if any(controller.get(k) != v for k, v in hardware.settings().items()):
        raise ValueError("old baseline hardware is not the declared baseline")
    return receipt


@ChiaFunction(num_cpus=1, max_retries=0)
def score_pair(root, point, name, arm, oracle, candidate, floor):
    row = compare_transfer(Path(oracle["directory"]), Path(candidate["directory"]),
        oracle_receipt_sha256=oracle["receipt_sha256"],
        model_receipt_sha256=candidate["receipt_sha256"],
        frontend="champsim", minimum_oracle_owner_reads=floor)
    save(root / "scores" / point / arm / (name + ".json"), row)
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
        raise RuntimeError(f"{stage}: {len(failures)} failures; successful receipts retained")
    return results


def write_report(root, protocol, scores, measurements, *, complete):
    names = [r["name"] for r in protocol["cases"]]
    arms = tuple(protocol["selections"])
    points = [p for p in GRID if all((p.name, n, a) in scores for n in names for a in arms)]
    floor = protocol["minimum_oracle_owner_reads"]
    common = [n for n in names if all(
        measurements[p.name, n, "oracle"]["observation"]["controller_stats"]["observed_read_records"] >= floor
        for p in points)]
    groups = {"all": common}
    for label in ("non_google", "google", "sierra", "tahoe"):
        groups[label] = [r["name"] for r in protocol["cases"] if r["name"] in common and (
            r["category"] != "google" if label == "non_google" else
            r["category"] == "google" if label == "google" else r["family"] == label)]
    summaries = {}
    table = []
    for point in points:
        summaries[point.name] = {}
        for arm in arms:
            summaries[point.name][arm] = {group: aggregate(
                {n: scores[point.name, n, arm] for n in members}, expected_workloads=members,
                stage="test", request_objective=protocol["request_objective"])
                for group, members in groups.items() if members}
            for n in names:
                value = scores[point.name, n, arm]
                req = value["request"]
                table.append(dict(configuration=point.name, model=arm, workload=n,
                    common_eligible=n in common, oracle_reads=value["oracle_owner_reads"],
                    core_error_pct=value["cycles"]["mean_abs_per_core_pct"],
                    request_mae_over_L=req["mae"], signed_drift_over_L=req["sgn"],
                    paired_p99_over_L=req["tail"],
                    minimum_error_cycles=req["extreme_min_cycles"], maximum_error_cycles=req["extreme_max_cycles"],
                    L=req["oracle_read_mean_latency"], oracle_coverage=req["cov_o"], model_coverage=req["cov_m"]))
    report = dict(complete=complete, protocol_sha256=file_sha256(root / "protocol.json"),
        common_eligible_workloads=common, configured_workloads=names,
        excluded_from_common_mean=[n for n in names if n not in common], summaries=summaries)
    replace_file(root, "report.json", canonical_json(report).encode())
    text = io.StringIO()
    if table:
        writer = csv.DictWriter(text, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    replace_file(root, "per-workload.csv", text.getvalue().encode())
    lines = ["# Frozen-model DDR5 configuration sweep", "",
        f"Status: {'complete' if complete else 'partial'}. {len(points)}/9 configurations; {len(common)} common oracle-qualified workloads.", "",
        "No LLM calls, parameter retuning, or feedback to agents. Existing core and request metrics are unchanged.", "",
        "| Configuration | Model | Core error (%) | Request MAE/L |", "| --- | --- | ---: | ---: |"]
    for point in points:
        for arm in arms:
            if common:
                row = summaries[point.name][arm]["all"]["aggregate"]
                lines.append(f"| {point.name} | {arm} | {row['cycle_macro_mae_pct']:.4f} | {row['request_macro_mae_over_L']:.6f} |")
    lines += ["", "Means give each workload equal weight. Cohort-specific results, paired coverage, tails, and signed extrema are in report.json and per-workload.csv.", "",
        "Organizations also differ in capacity/device width and derived timings. Queue capacities have different admission semantics in the oracle and atomic wrapper; this is part of the evaluated approximation.", ""]
    replace_file(root, "summary.md", "\n".join(lines).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=8, choices=range(1, 13))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prior, root, runtime = args.prior.absolute(), args.output.absolute(), args.runtime.absolute()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    protocol = freeze(prior, root, runtime, args.cpus)
    if not args.run:
        print(canonical_json(dict(protocol=str(root / "protocol.json"), configurations=len(GRID),
                                 workloads=len(protocol["cases"]), new_simulations=8 * 3 * len(protocol["cases"]))))
        return
    with exclusive_lock(root / "evaluation.lock"):
        try:
            status(root, "preparing_inputs")
            host = ExternalHost(Path(protocol["host"]), protocol["host_sha256"])
            host.record()
            cases = {}
            for row in protocol["cases"]:
                saved = read(prior / "inputs" / row["name"] / "verified-input.json")
                path = Path(saved["path"])
                payload = describe_payload(path, path.name)
                # The compressed SHA binds the previous full-stream CRC/decoded
                # checksum verification. No trace conversion or new download.
                for point in GRID:
                    cases[point.name, row["name"]] = HardwareCase(
                        frontend="champsim", workload=row["name"], payload=payload,
                        instruction_inventory=saved["inventory"], stage="test", hardware=point,
                        warmup_instructions=protocol["warmup_instructions"],
                        roi_instructions=protocol["roi_instructions"])
            ray.init(address="local", num_cpus=args.cpus, include_dashboard=False,
                     log_to_driver=False, object_store_memory=128 * 1024**2,
                     namespace=root.name, _temp_dir=tempfile.mkdtemp(prefix="chia-hardware-ray."))
            old = baseline_receipts(prior, protocol)
            measured = collect(root, {k: verify_baseline.options().remote(v,
                TransferCase.identity(cases[k[0], k[1]]), BASELINE) for k, v in old.items()}, "verify_baseline")
            r = protocol["resources"]
            builds = collect(root, {arm: build_model.options().remote(runtime, root,
                v["candidate"], BuildLimits(1, r["build_timeout_seconds"], r["source_bytes"],
                                            r["memory_bytes"], r["file_bytes"]))
                for arm, v in protocol["selections"].items()}, "build_models")
            if not all(b["passed"] for b in builds.values()):
                raise RuntimeError("a frozen model did not build; evidence retained")
            limits = SimulationLimits(r["simulation_timeout_seconds"], r["memory_bytes"],
                                     r["file_bytes"], r["source_bytes"], r["gzip_level"])
            scores = {}
            # Publish a complete result after each hardware point. Native jobs
            # and scores are immutable/cacheable, so an interrupted sweep resumes.
            for point in GRID:
                if point != BASELINE:
                    jobs = {(point.name, row["name"], arm): measurement.options().remote(
                        runtime, root, cases[point.name, row["name"]],
                        "oracle" if arm == "oracle" else "candidate", limits,
                        None if arm == "oracle" else builds[arm], None, host)
                        for row in protocol["cases"] for arm in ("oracle", *builds)}
                    receipts = collect(root, jobs, point.name)
                    for key, receipt in receipts.items():
                        receipt["directory"] = str(root / receipt["directory"])
                        measured[key] = receipt
                paired = collect(root, {(point.name, row["name"], arm): score_pair.options().remote(
                    root, point.name, row["name"], arm,
                    measured[point.name, row["name"], "oracle"], measured[point.name, row["name"], arm],
                    protocol["minimum_oracle_owner_reads"])
                    for row in protocol["cases"] for arm in builds}, "match_" + point.name)
                scores.update(paired)
                write_report(root, protocol, scores, measured, complete=point == GRID[-1])
                status(root, "configuration_complete", configuration=point.name, completed=GRID.index(point)+1, total=9)
            save(root / "measurement_index.json", {"|".join(k): v for k, v in measured.items()})
            status(root, "complete", configurations=9, comparisons=len(scores), no_llm_calls=True)
        except BaseException as exc:
            status(root, "stopped", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()


if __name__ == "__main__":
    main()
