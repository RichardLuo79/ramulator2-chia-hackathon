"""Frozen campaign Lat–Tp comparison, using the native suite and checked executor.

No agents, model tuning or workload traces are involved. Every load point is
an independent simulation; completed receipts can be resumed without reruns.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

from ramulator_chia.eval.lat_tp_models import CONFIG, LatTpCase, READ_RATIOS
from ramulator_chia.eval.lat_tp_suite import resolve_spec
from ramulator_chia.framework.archive import describe_payload
from ramulator_chia.framework.candidate import BuildLimits
from ramulator_chia.framework.dram import MODEL_FILES, build_model, measurement
from ramulator_chia.framework.evaluation import IsolatedRamulator, SimulationLimits
from ramulator_chia.framework.identity import canonical_json, file_sha256
from ramulator_chia.framework.measurement_reports import verify_native_measurement
from ramulator_chia.framework.snapshots import publish_bytes, replace_file, snapshot
from ramulator_chia.recovery import exclusive_lock

ENCODING = "ro-ba-ra-co-ch-byte"
BASELINES = ("oracle", "fixedlat", "md1", "wmg1", "mess")
ENDPOINTS = ("astra_single_core", "deepseek_single_core", "astra_multicore", "deepseek_multicore")
ARMS = BASELINES + ENDPOINTS
LIMITS = SimulationLimits(1800, 4 * 1024**3, 4 * 1024**3, 1024**2, 3)
FRONTEND = "src/ramulator/frontend/impl/memory_trace/latency_throughput_trace.cpp"
HIGH_MEMORY_ARMS = ("fixedlat", "mess")


class UnobservedLatTpCase(LatTpCase):
    """Omit optional CSV only; statistics, traffic and callbacks are unchanged."""
    observation_names = ()

    def identity(self):
        return {**super().identity(), "request_observations": False,
                "logging_adapter_sha256": file_sha256(Path(__file__))}

    def configuration(self, model, inputs, observations, parameters, curve):
        config = super().configuration(model, inputs, observations, parameters, curve)
        controller = config["memory_system"]["controllers"][0]
        controller.pop("trace_path", None)
        if model == "oracle":
            controller["controller_plugins"] = []
        return config


class CloudQualificationCase(UnobservedLatTpCase):
    def identity(self):
        return {**super().identity(), "purpose": "fresh-cloud-counter-parity"}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    publish_bytes(Path(path), canonical_json(value).encode())


def status(root, phase, **fields):
    value = dict(phase=phase, time=time.time(), pid=os.getpid(), **fields)
    replace_file(root, "status.json", canonical_json(value).encode())
    print(canonical_json(value), flush=True)


def points(pilot=False):
    ratios = (100, 50) if pilot else READ_RATIOS
    nops = (10000, 1) if pilot else CONFIG["nop_counters"]
    return [LatTpCase(n, r, address_encoding=ENCODING) for r in ratios for n in nops] + [
        LatTpCase(1, 100, True, ENCODING)]


def point_name(case):
    return "streaming" if case.streaming_only else f"rr{case.read_ratio}_nop{case.nop_counter}"


def effective_limits(root, arm):
    policy = root / "execution-policy.json"
    if not policy.exists():
        return LIMITS
    settings = read(policy)
    size = settings["high_memory_bytes"] if arm in HIGH_MEMORY_ARMS else LIMITS.memory_bytes
    return replace(LIMITS, memory_bytes=size)


def numerical_signature(value):
    controller = dict(value["controller_stats"])
    controller.pop("controller_plugin", None)  # Observer metadata, not a numerical statistic.
    return dict(frontend_stats=value["frontend_stats"], controller_stats=controller)


def verify_saved_point(root, case, arm, *, approved_sha256=None):
    path = root / "study/points" / arm / (point_name(case)+".json")
    if approved_sha256 and file_sha256(path) != approved_sha256:
        raise ValueError("preserved point changed: " + str(path))
    value = read(path)
    if value["arm"] != arm or any(value[k] != v for k, v in asdict(case).items()):
        raise ValueError("point identity changed")
    receipt = value["receipt"]
    runtime_hash = file_sha256(root / "runtime/runtime_manifest.json")
    if receipt["identity"]["runtime"] != runtime_hash or receipt["observation"]["runtime_sha256"] != runtime_hash:
        raise ValueError("runtime identity changed")
    expected_candidate = read(root / "builds.json").get(arm, {}).get("candidate")
    if receipt["identity"]["candidate"] != expected_candidate:
        raise ValueError("frozen candidate changed")
    if canonical_json(receipt["identity"]["case"]["suite"]) != canonical_json(CONFIG):
        raise ValueError("suite settings changed")
    for key in ("frontend_stats", "controller_stats"):
        if value[key] != receipt["observation"][key]:
            raise ValueError("point counters disagree with receipt")
    verify_native_measurement(root / "study" / receipt["directory"], receipt["receipt_sha256"],
        frontend="LatencyThroughputTrace", observations=tuple(receipt["observation"]["traces"]))
    return value


def prepare(root, repository, previous_runtime, endpoints_path):
    """Stage an already-qualified runtime and frozen endpoint sources."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    previous = read(previous_runtime / "runtime_manifest.json")
    for name, digest in previous["source_inventory"].items():
        if file_sha256(previous_runtime / "runtime-source" / name) != digest:
            raise ValueError("frozen source changed: " + name)
    runtime = root / "runtime"
    if not runtime.exists():
        shutil.copytree(previous_runtime, runtime)
    if file_sha256(runtime / "runtime_manifest.json") != file_sha256(previous_runtime / "runtime_manifest.json"):
        raise ValueError("staged runtime identity changed")
    endpoints = read(endpoints_path)
    builds = {}
    for arm in ENDPOINTS:
        candidate = endpoints[arm]["candidate"]
        copied = snapshot(repository / endpoints[arm]["model_path"], root / "study/candidates",
                          MODEL_FILES, maximum_bytes=1024**2)
        if copied != candidate:
            raise ValueError("endpoint changed: " + arm)
        build = build_model._chia_original(runtime, root / "study", candidate,
            BuildLimits(1, 300, 1024**2, 4 * 1024**3, 4 * 1024**3))
        if not build["passed"]:
            raise RuntimeError("endpoint build failed: " + arm)
        builds[arm] = build
    curve = Path(__file__).resolve().parent / "calibration/mess_DDR5.txt"
    save(root / "builds.json", builds)
    publish_bytes(root / "mess_DDR5.txt", curve.read_bytes())
    save(root / "endpoints.json", endpoints)
    save(root / "protocol.json", dict(schema_version=1,arms=list(ARMS),points_per_arm=157,
        expected_points=len(ARMS)*157,case=points()[0].identity(),address_policy=ENCODING,
        source_sha256=file_sha256(Path(__file__)),limits=asdict(LIMITS),
        runtime_sha256=file_sha256(runtime / "runtime_manifest.json"),
        model_sources={n:endpoints[n]["candidate"] for n in ENDPOINTS},
        curve_sha256=file_sha256(root / "mess_DDR5.txt")))
    status(root, "prepared", expected=len(ARMS)*157)


def run_point(root, case, arm, *, publish=True):
    runtime, store = root / "runtime", root / "study"
    policy_path = root / "execution-policy.json"
    if policy_path.exists():
        policy = read(policy_path)
        existing = store / "points" / arm / (point_name(case)+".json")
        if publish and existing.exists():
            expected = policy["reused_points"].get(str(existing.relative_to(root)))
            if expected is None:
                saved = read(existing)
                if saved.get("execution_policy_sha256") != file_sha256(policy_path):
                    raise ValueError("unapproved existing point")
            return verify_saved_point(root, case, arm, approved_sha256=expected)
        if type(case) is LatTpCase and not policy["request_observations"]:
            case = UnobservedLatTpCase(**asdict(case))
    build = read(root / "builds.json").get(arm)
    model = "candidate" if build else arm
    curve = describe_payload(root / "mess_DDR5.txt", "inputs/mess.txt") if arm == "mess" else None
    receipt = measurement._chia_original(runtime, store, case, model, effective_limits(root, arm),
                                         build, curve, IsolatedRamulator())
    observed = receipt["observation"]
    frontend, controller = observed["frontend_stats"], observed["controller_stats"]
    if bool(frontend.get("physical_byte_addresses")) != (case.address_encoding == ENCODING):
        raise ValueError("runtime did not apply the requested address encoding")
    spec = resolve_spec(CONFIG)
    bandwidth = ((controller["num_read_reqs_served"] + controller["num_write_reqs_served"])
                 * spec.bytes_per_req / (controller["cycles"] * spec.time_unit_ns))
    latency = (None if case.streaming_only else frontend["total_probe_latency"]
               / frontend["probe_requests_completed"] * spec.time_unit_ns)
    if not math.isfinite(bandwidth) or bandwidth <= 0 or (latency is not None and not math.isfinite(latency)):
        raise ValueError("nonfinite curve point")
    value = dict(arm=arm, **asdict(case), throughput_GBps=bandwidth, probe_latency_ns=latency,
        frontend_stats=frontend, controller_stats=controller,
        receipt=receipt, candidate_id=build["candidate"]["candidate_id"] if build else None)
    if policy_path.exists():
        value.update(execution_policy_sha256=file_sha256(policy_path),
                     request_observations=bool(case.observation_names))
    suffix = "" if case.address_encoding == ENCODING else "-legacy"
    if publish:
        save(store / "points" / arm / (point_name(case)+suffix+".json"), value)
    return value


def observation_rows(root, value):
    receipt = value["receipt"]
    member = receipt["observation"]["traces"]["controller.csv.ch0"]
    path = root / "study" / receipt["directory"] / member["name"]
    opener = gzip.open if member["codec"] == "gzip" else open
    with opener(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream)


def verify_encoding(row):
    address = int(row["addr"])
    if address < 0 or address >= 8 * 1024**3 or address % 64:
        raise ValueError("physical address is out of range or unaligned")
    slot = address >> 6
    # Independent decoder follows pinned RoBaRaCoCh for this frozen geometry.
    expected = {"Column": (slot & 63) * 16, "Rank": 0,
                "BankGroup": (slot >> 6) & 7, "Bank": (slot >> 9) & 3,
                "Row": slot >> 11, "Channel": 0}
    for key, value in expected.items():
        if int(row[key]) != value:
            raise ValueError(f"physical address/coordinate mismatch: {key}")


def scientific_signature(value):
    observation = value["receipt"]["observation"]
    return dict(frontend_stats=value["frontend_stats"], controller_stats=value["controller_stats"],
        observations={name:member["logical_sha256"] for name,member in observation["traces"].items()})


def prepare_memory_retry(root):
    """Record the approved resource-only retry, without reclassifying old failures."""
    with exclusive_lock(root / "worker.lock"):
        logging = read(root / "logging-parity.json")
        if not logging["passed"] or {c["arm"] for c in logging["cases"]} != set(ARMS):
            raise ValueError("logging parity is not qualified for all nine arms")
        if not all(c["complete"] and c["counters_equal"] for c in logging["cases"]):
            raise ValueError("logging changed simulated counters")
        protocol = read(root / "protocol.json")
        if protocol["runtime_sha256"] != file_sha256(root / "runtime/runtime_manifest.json"):
            raise ValueError("frozen runtime changed")
        before, after = dict(protocol["case"]), points()[0].identity()
        before.pop("driver_sha256"); after.pop("driver_sha256")
        if canonical_json(before) != canonical_json(after):
            raise ValueError("scientific suite changed")
        reused = {}
        for arm in ARMS:
            for case in points():
                path = root / "study/points" / arm / (point_name(case)+".json")
                if path.exists():
                    verify_saved_point(root, case, arm)
                    reused[str(path.relative_to(root))] = file_sha256(path)
        address_checks = []
        for case in points(pilot=True):
            new = read(root / "study/points/oracle" / (point_name(case)+".json"))
            old = read(root / "study/points/oracle" / (point_name(case)+"-legacy.json"))
            a, b = numerical_signature(new), numerical_signature(old)
            a["frontend_stats"] = dict(a["frontend_stats"])
            b["frontend_stats"] = dict(b["frontend_stats"])
            a["frontend_stats"].pop("physical_byte_addresses", None)
            b["frontend_stats"].pop("physical_byte_addresses", None)
            if a != b:
                raise ValueError("address encoding changed oracle counters")
            banks = set(); count = 0
            for encoded, legacy in zip(observation_rows(root, new), observation_rows(root, old), strict=True):
                verify_encoding(encoded)
                if {k:v for k,v in encoded.items() if k != "addr"} != {k:v for k,v in legacy.items() if k != "addr"}:
                    raise ValueError("address encoding changed oracle request order")
                banks.add((encoded["BankGroup"], encoded["Bank"])); count += 1
            if len(banks) != 32:
                raise ValueError("not all banks covered")
            address_checks.append(dict(point=asdict(case), requests=count, banks=32, parity=True))
        failures = [v for values in failed_attempts(root).values() for v in values]
        if any("std::bad_alloc" not in f["reason"] for f in failures):
            raise ValueError("unexplained failure must not be treated as resource exhaustion")
        save(root / "prior-failures.json", failures)
        policy = dict(schema_version=1, reason="User-approved larger-memory resource-only retry",
            original_protocol_sha256=file_sha256(root / "protocol.json"),
            runtime_sha256=protocol["runtime_sha256"], request_observations=False,
            logging_parity_sha256=file_sha256(root / "logging-parity.json"),
            high_memory_bytes=64*1024**3, high_memory_arms=list(HIGH_MEMORY_ARMS),
            high_memory_workers=4, workers=8, minimum_available_memory_bytes=16*1024**3,
            timeout_seconds=1800, reused_points=reused,
            no_model_changes=True, no_admission_changes=True, no_traffic_changes=True)
        save(root / "execution-policy.json", policy)
        save(root / "qualification.json", dict(passed=True, integrity_passed=True,
            all_arms_completed=False, successful_points=len(reused), resource_failed_points=len(failures),
            oracle_address_parity=address_checks, logging_parity=True,
            runtime_sha256=protocol["runtime_sha256"],
            acceptance="Resource-failed corners remain unscored; retry with more RAM while independent points proceed."))
        save(root / "qualification-reference.json", {
            arm+"/rr100_nop10000":numerical_signature(read(root / "study/points" / arm / "rr100_nop10000.json"))
            for arm in ARMS})
        status(root, "prepared_memory_retry", successful=len(reused), failed=len(failures), cloud_state="not_provisioned")
        export_compact(root)


def qualify_cloud(root):
    """Fresh full-window low-load runs; never reuse the local measurement cache."""
    expected = read(root / "qualification-reference.json")
    checked = []
    with exclusive_lock(root / "worker.lock"):
        status(root, "qualifying")
        case = CloudQualificationCase(10000, 100, address_encoding=ENCODING)
        for arm in ARMS:
            value = run_point(root, case, arm, publish=False)
            if numerical_signature(value) != expected[arm+"/rr100_nop10000"]:
                raise ValueError("cloud simulated counters differ: " + arm)
            checked.append(dict(arm=arm, equal=True, receipt_sha256=value["receipt"]["receipt_sha256"],
                                directory=value["receipt"]["directory"]))
        save(root / "cloud-qualification.json", dict(passed=True, cases=checked))
        status(root, "qualified", points=len(checked))


def qualify(root):
    """Full-size corner points count toward production; check legacy parity too."""
    with exclusive_lock(root / "worker.lock"):
        status(root, "qualifying")
        checked = []
        for case in points(pilot=True):
            current = run_point(root, case, "oracle")
            legacy = run_point(root, replace(case, address_encoding="bank-major-line"), "oracle")
            for key in ("frontend_stats", "controller_stats"):
                a, b = dict(current[key]), dict(legacy[key])
                a.pop("physical_byte_addresses", None); b.pop("physical_byte_addresses", None)
                if a != b:
                    raise ValueError("address representation changed oracle timing/counters")
            count = 0; banks = set()
            for new, old in zip(observation_rows(root, current), observation_rows(root, legacy), strict=True):
                verify_encoding(new)
                if {k:v for k,v in new.items() if k != "addr"} != {k:v for k,v in old.items() if k != "addr"}:
                    raise ValueError("address representation changed oracle request order/timing")
                banks.add((new["BankGroup"],new["Bank"])); count += 1
            if len(banks) != 32:
                raise ValueError("qualification did not exercise every bank")
            checked.append(dict(point=asdict(case), oracle_requests=count, banks=len(banks), parity=True))
        # Keep compilation and qualification evidence on failures as on success.
        with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_point, root, p, arm) for p in points(pilot=True) for arm in ARMS if arm != "oracle"]
            for future in as_completed(futures):
                future.result()
        signatures = {arm+"/"+point_name(p):scientific_signature(read(root / "study/points" / arm / (point_name(p)+".json")))
                      for arm in ARMS for p in points(pilot=True)}
        reference = root / "qualification-reference.json"
        if reference.exists() and read(reference) != signatures:
            raise ValueError("cross-host qualification counters or decoded observations differ")
        save(root / "qualification-signatures.json", signatures)
        save(root / "qualification.json", dict(passed=True, oracle_address_parity=checked,
             all_arms_completed=True, points_per_arm=5, runtime_sha256=read(root / "protocol.json")["runtime_sha256"]))
        status(root, "qualified", points=45)


def failed_attempts(root):
    """Read durable native failures, including qualification failures.

    A missing successful pointer is not evidence of a failed simulation. Only
    preserved terminal receipts are classified here; unstarted points remain
    pending, and partial statistics are never converted into accuracy data.
    """
    failures = {}
    if (root / "prior-failures.json").exists():
        for attempt in read(root / "prior-failures.json"):
            case = LatTpCase(**{k:attempt[k] for k in ("nop_counter", "read_ratio", "streaming_only", "address_encoding")})
            failures.setdefault((attempt["arm"], point_name(case)), []).append(attempt)
    candidates = {item["candidate"]["candidate_id"]:arm
                  for arm,item in read(root / "builds.json").items()}
    for path in sorted((root / "study/measurements").glob("*/*/measurement.json")):
        receipt = read(path)
        point = receipt.get("case", {}).get("point", {})
        if receipt.get("complete") is True or point.get("address_encoding") != ENCODING:
            continue
        arm = receipt.get("model")
        if arm == "candidate":
            candidate = receipt.get("candidate") or {}
            arm = candidates.get(candidate.get("candidate_id"))
        if arm not in ARMS:
            continue
        case = LatTpCase(**point)
        reason = receipt.get("error", "native simulation did not complete")
        log = path.parent / "simulation.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 1024))
                ending = stream.read().decode(errors="replace")
            if "std::bad_alloc" in ending:
                reason = "std::bad_alloc under the recorded address-space limit; measurement incomplete"
        attempt = dict(arm=arm, **point, status="failed", reason=reason,
                       limits=receipt.get("limits", asdict(LIMITS)),
                       throughput_GBps=None, probe_latency_ns=None,
                       receipt=str(path.relative_to(root)), receipt_sha256=file_sha256(path))
        failures.setdefault((arm, point_name(case)), []).append(attempt)
    # Postprocessing or launcher errors can fail outside the native receipt.
    for path in sorted((root / "failures").glob("*.json")):
        record = read(path)
        if record.get("native_failure"):
            continue
        case = LatTpCase(**record["point"])
        failures.setdefault((record["arm"], point_name(case)), []).append(dict(
            arm=record["arm"], **asdict(case), status="failed",
            reason=record.get("error", "point processing failed"), limits=record.get("limits", asdict(LIMITS)),
            throughput_GBps=None, probe_latency_ns=None,
            receipt=str(path.relative_to(root)), receipt_sha256=file_sha256(path)))
    return {key:list({v["receipt_sha256"]:v for v in values}.values()) for key,values in failures.items()}


def export_compact(root, *, held_reason=None):
    if held_reason is not None:
        status(root, "held", reason=held_reason, running=False,
               cloud_state="not_provisioned" if not (root / "vm.json").exists() else "recorded")
    execution = read(root / "status.json") if (root / "status.json").exists() else {"phase":"prepared"}
    attempts = failed_attempts(root)
    rows = []; records = []
    for arm in ARMS:
        for point in points():
            failed = attempts.get((arm, point_name(point)), [])
            record = dict(arm=arm, **asdict(point), status="pending",
                          reason="not yet evaluated", limits=asdict(effective_limits(root, arm)),
                          throughput_GBps=None, probe_latency_ns=None, failed_attempts=failed)
            path = root / "study/points" / arm / (point_name(point)+".json")
            if path.exists():
                raw = read(path)
                receipt = raw.pop("receipt")
                if receipt.get("observation", {}).get("complete") is True:
                    raw.update(status="complete", reason=None,
                               limits=receipt["observation"].get("limits", asdict(LIMITS)),
                               receipt_sha256=receipt["receipt_sha256"],
                               wall_seconds=receipt["observation"]["process"]["wall_seconds"])
                    rows.append(raw)
                    record.update(raw)
                else:
                    record.update(status="incomplete", reason="no completed native measurement receipt")
            elif failed:
                record.update(status="failed", reason=failed[-1]["reason"], limits=failed[-1]["limits"])
            records.append(record)
    failed_points = [r for r in records if r["status"] == "failed"]
    value = dict(schema_version=2, protocol=read(root / "protocol.json"), execution=execution,
        running=execution.get("phase") in {"qualifying", "sweep"},
        cloud_provisioned=(root / "vm.json").exists(),
        qualification=read(root / "qualification.json") if (root / "qualification.json").exists() else None,
        points=rows, point_statuses=records, successful=len(rows), expected=len(records),
        failed=len(failed_points), incomplete=sum(r["status"] == "incomplete" for r in records),
        pending=sum(r["status"] == "pending" for r in records), failures=failed_points,
        failed_attempt_count=sum(map(len, attempts.values())),
        metrics_policy="Only complete native measurements have metrics; failed/incomplete/pending points use null, never zero.")
    if (root / "execution-policy.json").exists():
        value["execution_policy"] = read(root / "execution-policy.json")
    if (root / "cloud-qualification.json").exists():
        value["cloud_qualification"] = read(root / "cloud-qualification.json")
    replace_file(root, "notebook-data.json", canonical_json(value).encode())
    return value


def run(root, workers=8):
    if not read(root / "qualification.json")["passed"]:
        raise ValueError("qualification has not passed")
    with exclusive_lock(root / "worker.lock"):
        status(root, "sweep", total=1413)
        completed = 0
        if workers < 2 or workers > 8:
            raise ValueError("use 2–8 workers for the memory-bounded sweep")
        policy_hash = file_sha256(root / "execution-policy.json") if (root / "execution-policy.json").exists() else None
        old_failures = failed_attempts(root)
        jobs = [(point,arm) for point in points() for arm in ARMS]
        jobs.sort(key=lambda item: ((item[1],point_name(item[0])) not in old_failures,
                                   item[1] in ENDPOINTS[2:], points().index(item[0]), ARMS.index(item[1])))
        heavy_workers = min(4, workers//2)
        with ProcessPoolExecutor(max_workers=heavy_workers, mp_context=multiprocessing.get_context("spawn")) as heavy, \
             ProcessPoolExecutor(max_workers=workers-heavy_workers, mp_context=multiprocessing.get_context("spawn")) as ordinary:
            futures = {}
            for point, arm in jobs:
                failure = root / "failures" / (arm+"-"+point_name(point)+".json")
                if failure.exists() and read(failure).get("execution_policy_sha256") == policy_hash:
                    completed += 1  # One preserved attempt per approved resource policy; no automatic reroll.
                    continue
                pool = heavy if arm in HIGH_MEMORY_ARMS else ordinary
                futures[pool.submit(run_point, root, point, arm)] = (point,arm)
            for future in as_completed(futures):
                point, arm = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    record = dict(arm=arm, point=asdict(point), error_type=type(exc).__name__, error=str(exc),
                                  limits=asdict(effective_limits(root, arm)), execution_policy_sha256=policy_hash)
                    # The executor already preserves native failures. Keep this
                    # scheduling receipt too, without duplicating their count.
                    native = getattr(exc, "receipt", None)
                    if native is not None:
                        record["native_failure"] = True
                    save(root / "failures" / (arm+"-"+point_name(point)+".json"), record)
                completed += 1
                if completed % 50 == 0 or completed == 1413:
                    compact = export_compact(root)
                    status(root, "sweep", settled=completed, complete=compact["successful"], failed=compact["failed"], total=1413)
        compact = export_compact(root)
        status(root, "complete" if compact["successful"]==1413 else "complete_with_failures",
               complete=compact["successful"], failed=compact["failed"], total=1413)
        export_compact(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "qualify", "memory-retry", "qualify-cloud", "run", "export"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=__import__("ramulator_chia.layout", fromlist=["RAMULATOR"]).RAMULATOR)
    parser.add_argument("--previous-runtime", type=Path)
    parser.add_argument("--endpoints", type=Path)
    parser.add_argument("--held-reason", help="Record an operator hold when exporting; does not start or stop work")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.repository, args.previous_runtime, args.endpoints)
    elif args.action == "qualify": qualify(args.root)
    elif args.action == "memory-retry": prepare_memory_retry(args.root)
    elif args.action == "qualify-cloud": qualify_cloud(args.root)
    elif args.action == "run": run(args.root)
    else: export_compact(args.root, held_reason=args.held_reason)


if __name__ == "__main__":
    main()
