"""Frozen single-/multicore placement study, using native measurements.

Preparation reads traces, not oracle output. Each host runs its own ordinary
CHIA task queue; only immutable inputs and completed records cross hosts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import ray
from chia.base.ChiaFunction import get

from ramulator_chia.framework.archive import Member, Payload, describe_payload
from ramulator_chia.framework.candidate import BuildLimits
from ramulator_chia.framework.dram import COMPARISONS, MODEL_FILES, build_model, measurement
from ramulator_chia.framework.evaluation import SimulationLimits
from ramulator_chia.framework.external_frontends import ExternalHost, TransferCase, prepare_host
from ramulator_chia.framework.measurement_reports import verify_native_measurement
from ramulator_chia.framework.identity import canonical_json, file_sha256
from ramulator_chia.framework.scoring import aggregate
from ramulator_chia.framework.snapshots import publish_bytes, replace_file, snapshot
from ramulator_chia.recovery import exclusive_lock
from ramulator_chia.eval.champsim_mixes import MultiProgramCase, score_pair
from ramulator_chia.eval import config as C

SEED = "dpc4-placement-validation-test-20260914-v1"
CAPACITY = 8 << 30
ARMS = ("oracle", "astra_xhigh", "deepseek_max", "gemini_flash", *COMPARISONS)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    publish_bytes(Path(path), canonical_json(value).encode())


def progress(root, **values):
    replace_file(root, "status.json", canonical_json(dict(time=time.time(), pid=os.getpid(), **values)).encode())


def record_stop(root, exc):
    """Do not turn a finished queue with failed jobs into an unfinished queue."""
    path = root / "status.json"
    state = read(path) if path.exists() else {}
    if state.get("pid") == os.getpid() and state.get("terminal"):
        return
    progress(root, phase="stopped", terminal=False,
             error_type=type(exc).__name__, error=str(exc))


def order_key(*parts):
    return hashlib.sha256((SEED + "\0" + "\0".join(map(str, parts))).encode()).hexdigest()


def balanced_mixes(names, stage, cores):
    """Balance program exposure first, repeated pairings second; never use errors."""
    names = sorted(names)
    usage, pairs, result = Counter(), Counter(), []
    for index in range(8):
        chosen = []
        while len(chosen) < cores:
            name = min((n for n in names if n not in chosen), key=lambda n: (
                usage[n], sum(pairs[tuple(sorted((n, p)))] for p in chosen),
                order_key(stage, cores, index, len(chosen), n)))
            chosen.append(name)
        for n in chosen:
            usage[n] += 1
        for i, a in enumerate(chosen):
            for b in chosen[i + 1:]:
                pairs[tuple(sorted((a, b)))] += 1
        result.append(dict(name=f"{stage}-c{cores}-{index + 1:02d}", stage=stage,
                           cores=cores, programs=chosen))
    if set(usage) != set(names) or len({tuple(sorted(m["programs"])) for m in result}) != 8:
        raise ValueError("mix selector failed coverage or uniqueness checks")
    return result


def select_matrix(previous, validation):
    test = [row["name"] for row in previous["cases"]]
    if len(test) != 25 or len(validation) != 14 or set(test) & set(validation):
        raise ValueError("expected disjoint frozen 14-validation/25-test cohorts")
    result = []
    for stage, names in (("validation", validation), ("test", test)):
        for cores in (4, 8):
            if (stage, cores) == ("test", 4):
                group = [dict(m, name="test-" + m["name"], stage="test")
                         for m in previous["mixes"] if m["cores"] == 4]
            else:
                group = balanced_mixes(names, stage, cores)
            if len(group) != 8 or {n for m in group for n in m["programs"]} != set(names):
                raise ValueError("each group must contain eight mixes covering its pool")
            local = {m["name"] for m in sorted(group, key=lambda m: order_key(m["name"]))[:2]}
            result += [dict(m, site="local" if m["name"] in local else "cloud") for m in group]
    return result


def scan_trace(root, scanner, name, source):
    destination = root / "traces" / (name + ".gz")
    receipt_path = root / "inputs" / (name + ".json")
    if receipt_path.exists():
        receipt = read(receipt_path)
        if file_sha256(destination) != receipt["inventory"]["stored_sha256"]:
            raise ValueError("prepared trace changed: " + name)
        return receipt
    original = Path(source["path"])
    if file_sha256(original) != source["sha256"]:
        raise ValueError("source trace checksum mismatch: " + name)
    started = time.monotonic()
    digest, size = hashlib.sha256(), 0
    process = subprocess.Popen([str(scanner)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        with gzip.open(original, "rb") as stream:
            while data := stream.read(1024**2):
                digest.update(data)
                size += len(data)
                process.stdin.write(data)
        process.stdin.close()
        raw = process.stdout.read()
        if process.wait() != 0:
            raise RuntimeError("trace page scanner failed")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    count, pages = raw.split(b"\n", 1)
    if int(count) != 23_000_000 or size != 23_000_000 * 64 or digest.hexdigest() != source["decoded_sha256"]:
        raise ValueError("decoded trace identity/window mismatch: " + name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, destination)
    if file_sha256(destination) != source["sha256"]:
        raise ValueError("source changed while copying: " + name)
    pages_path = root / "pages" / (name + ".txt.gz")
    publish_bytes(pages_path, gzip.compress(pages, compresslevel=3, mtime=0))
    inventory = dict(stored_sha256=source["sha256"], decoded_sha256=digest.hexdigest(),
                     decoded_bytes=size, record_bytes=64, instructions=int(count),
                     complete_compressed_integrity=True, complete_xz_integrity=False, compression="gzip")
    receipt = dict(path=str(destination.relative_to(root)), inventory=inventory,
                   stored_bytes=destination.stat().st_size, source=source,
                   pages_sha256=file_sha256(pages_path), pages=len(pages.splitlines()),
                   scan_wall_seconds=time.monotonic() - started)
    save(receipt_path, receipt)
    print("SCANNED", name, receipt["pages"], flush=True)
    return receipt


def prepare(args):
    if args.completion_policy == "background-replay":
        from ramulator_chia.eval.champsim_contention import prepare as prepare_contention
        return prepare_contention(args)
    root = args.output
    previous = read(args.prior / "protocol.json")
    validation = read(args.validation_config)["experiment"]["evaluation"]["champsim"]
    mixes = select_matrix(previous, validation["validation"])
    sources = {n: validation["traces"][n] for n in validation["validation"]}
    for row in previous["cases"]:
        old = read(Path(previous["source_study"]) / "inputs" / row["name"] / "verified-input.json")
        sources[row["name"]] = dict(path=old["path"], sha256=old["inventory"]["stored_sha256"],
                                   decoded_sha256=old["inventory"]["decoded_sha256"],
                                   family=row["family"], category=row["category"])
    root.mkdir(parents=True, exist_ok=True)
    save(root / "mixes.json", mixes)  # Freeze membership before any new oracle measurements.
    with ThreadPoolExecutor(max_workers=args.cpus) as pool:
        futures = {n: pool.submit(scan_trace, root, args.scanner, n, s) for n, s in sources.items()}
        inputs = {n: f.result() for n, f in futures.items()}
    for arm, selection in previous["selections"].items():
        copied = snapshot(args.prior / "candidates" / selection["candidate"]["candidate_id"],
                          root / "candidates", MODEL_FILES, maximum_bytes=1 << 20)
        if copied != selection["candidate"]:
            raise ValueError("frozen model changed: " + arm)
    curve = describe_payload(C.MESS_CURVES["DDR5"], "mess.txt")
    publish_bytes(root / "mess.txt", curve.source.read_bytes())
    for mix in mixes:
        name = mix["name"]
        compressed = root / "maps" / (name + ".txt.gz")
        if not compressed.exists():
            inventory = f"CHAMPSIM_PAGES_V1 {mix['cores']} 4096 {CAPACITY}\n".encode()
            for core, n in enumerate(mix["programs"]):
                inventory += b"".join(str(core).encode() + b" " + line + b"\n" for line in
                                      gzip.decompress((root / "pages" / (n + ".txt.gz")).read_bytes()).splitlines())
            page_file = root / "maps" / (name + ".pages")
            map_file = root / "maps" / (name + ".txt")
            publish_bytes(page_file, inventory)
            binary = getattr(args, f"source{mix['cores']}") / f"bin/champsim-{mix['cores']}core"
            env = {k: v for k, v in os.environ.items() if k not in ("CHAMPSIM_PLACEMENT_FILE", "RAMULATOR_CONFIG")}
            with (root / "maps" / (name + ".prepare.log")).open("w") as log:
                subprocess.run([str(binary), "--prepare-placement", str(page_file), str(map_file)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            raw = map_file.read_bytes()
            publish_bytes(compressed, gzip.compress(raw, compresslevel=3, mtime=0))
            if gzip.decompress(compressed.read_bytes()) != raw:
                raise ValueError("placement compression failed")
            page_file.unlink()
            map_file.unlink()
        mix["placement"] = asdict(describe_payload(compressed, "placement.txt", codec="gzip").member)
    pilots = [next(m["name"] for m in mixes if m["stage"] == "test" and m["cores"] == c and m["site"] == "local")
              for c in (4, 8)]
    save(root / "protocol.json", dict(schema_version=1, seed=SEED, mixes=mixes, inputs=inputs,
         selections=previous["selections"], arms=ARMS, pilots=pilots, capacity_bytes=CAPACITY,
         mess_sha256=curve.member.stored_sha256, source_protocol_sha256=file_sha256(args.prior / "protocol.json"),
         runtime_source_inventory=read(args.runtime / "runtime_manifest.json")["source_inventory"],
         warmup_instructions=2_000_000, roi_instructions=20_000_000,
         no_llm_calls=True, no_model_edits=True, no_agent_feedback=True))
    print("PREPARED", len(mixes), "mixes; pilots", pilots, flush=True)


def cases_for_host(args, protocol):
    root = args.output
    cases = {}
    for cores in sorted({m["cores"] for m in protocol["mixes"]}):
        destination = root / args.site / "hosts" / f"c{cores}"
        if destination.exists():
            host = ExternalHost(destination, file_sha256(destination / "host.json"))
            host.record()
        else:
            source = getattr(args, f"source{cores}")
            host = prepare_host(destination, "champsim", source / f"bin/champsim-{cores}core", source)
        for mix in (m for m in protocol["mixes"] if m["cores"] == cores):
            traces = []
            for name in mix["programs"]:
                row = protocol["inputs"][name]
                inv = dict(row["inventory"], host_sha256=host.receipt_sha256)
                payload = Payload(root / row["path"], Member("trace.gz", inv["stored_sha256"], row["stored_bytes"],
                                  inv["stored_sha256"], row["stored_bytes"], "none"))
                traces.append(TransferCase("champsim", name, payload, instruction_inventory=inv, stage=mix["stage"]))
            placement = Payload(root / "maps" / (mix["name"] + ".txt.gz"), Member(**mix["placement"]))
            cases[mix["name"]] = (MultiProgramCase(frontend="champsim", workload=mix["name"], stage=mix["stage"],
                payload=traces[0].payload, instruction_inventory=traces[0].instruction_inventory,
                companions=tuple(traces[1:]), placement=placement,
                completion_policy=protocol.get("completion_policy", "finite")), host)
    return cases


def available_bytes():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))) * 1024


def matching_batch(pending, results, available):
    """Reserve room for pandas frames, copies and joins, plus the host margin.

    Twelve times decoded CSV size is a conservative working-set estimate, not
    a measurement of pandas RSS. At most two matches run concurrently.
    """
    remaining = max(0, available - (10 << 30))
    batch = []
    for name, arm in pending[:2]:
        decoded = sum(results[name, a]["observation"]["traces"]["controller.csv.ch0"]["logical_bytes"]
                      for a in ("oracle", arm))
        estimate = 12 * decoded + (1 << 30)
        if estimate > remaining:
            break
        batch.append((name, arm))
        remaining -= estimate
    if not batch:
        raise RuntimeError("insufficient RAM for one conservative matching reservation; completed runs are retained")
    return batch


def infrastructure_failure(exc):
    """Retry only recognized execution infrastructure, not simulator failures."""
    text = type(exc).__name__ + ": " + str(exc)
    return any(marker in text for marker in (
        "WorkerCrashedError", "NodeDiedError", "LocalRayletDiedError",
        "Input/output error", "Resource temporarily unavailable"))


def run(args):
    # Fail before dispatching jobs if the launch environment lacks the config package.
    import ramulator  # noqa: F401
    protocol = read(args.output / "protocol.json")
    if read(args.runtime / "runtime_manifest.json")["source_inventory"] != protocol["runtime_source_inventory"]:
        raise ValueError("Ramulator runtime sources differ from the frozen study")
    root = args.output / args.site
    root.mkdir(parents=True, exist_ok=True)
    progress(root, phase=args.phase + "_preparation", terminal=False)
    for selection in protocol["selections"].values():
        copied = snapshot(args.output / "candidates" / selection["candidate"]["candidate_id"],
                          root / "candidates", MODEL_FILES, maximum_bytes=1 << 20)
        if copied != selection["candidate"]:
            raise ValueError("candidate source identity changed")
    cases = cases_for_host(args, protocol)
    ray.init(address="local", num_cpus=args.cpus, include_dashboard=False, log_to_driver=False,
             object_store_memory=128 * 1024**2)
    builds = {a: get(build_model.options().remote(args.runtime, root, s["candidate"], BuildLimits(1, 1800, 1 << 20, 4 << 30, 16 << 30)))
              for a, s in protocol["selections"].items()}
    if not all(b["passed"] for b in builds.values()):
        raise RuntimeError("a frozen model failed to build")
    curve = describe_payload(args.output / "mess.txt", "mess.txt")
    if curve.member.stored_sha256 != protocol["mess_sha256"]:
        raise ValueError("MESS calibration changed")
    if args.phase == "qualification":
        names = protocol["pilots"]
        arms = ARMS if args.site == "local" else ("oracle", "fixedlat")
    else:
        release = args.output / "qualification-passed.json"
        if not release.exists():
            raise ValueError("bulk release requires the cross-host qualification receipt")
        approval = read(release)
        if approval.get("passed") is not True or approval.get("protocol_sha256") != file_sha256(args.output / "protocol.json"):
            raise ValueError("qualification receipt belongs to another experiment")
        names = [m["name"] for m in protocol["mixes"] if m["site"] == args.site]
        arms = ARMS
    todo = [(name, arm) for name in names for arm in arms]
    active, results, failures = {}, {}, {}
    attempts = {}
    pending = []
    for name, arm in todo:
        done = root / "completed" / name / (arm + ".json")
        job = root / "jobs" / name / (arm + ".json")
        prior = read(job) if job.exists() else {}
        attempts[name, arm] = prior.get("attempts", 0)
        if done.exists():
            row = read(done)
            receipt = verify_native_measurement(root / row["directory"], row["receipt_sha256"],
                frontend="champsim", observations=("controller.csv.ch0",))
            if receipt["case"] != cases[name][0].identity():
                raise ValueError("completed job belongs to another protocol")
            results[name, arm] = row
        elif prior.get("state") == "failed":
            failures[name + "/" + arm] = prior["error"]
        else:
            pending.append((name, arm))
    todo = pending
    limits = SimulationLimits(None, 4 << 30, 16 << 30, 1 << 20, 3)
    while todo or active:
        while todo and len(active) < args.cpus and available_bytes() > 10 << 30:
            name, arm = todo.pop(0)
            attempts[name, arm] += 1
            replace_file(root, f"jobs/{name}/{arm}.json", canonical_json(dict(
                state="running", attempts=attempts[name, arm], started=time.time())).encode())
            case, host = cases[name]
            ref = measurement.options().remote(args.runtime, root, case,
                "candidate" if arm in builds else arm, limits, builds.get(arm), curve if arm == "mess" else None, host)
            active[ref] = (name, arm)
        if not active and todo:
            raise RuntimeError("less than 10 GiB RAM available before launch")
        ready, _ = ray.wait(list(active), num_returns=1, timeout=30)
        for ref in ready:
            name, arm = active.pop(ref)
            try:
                row = get(ref)
                save(root / "completed" / name / (arm + ".json"), row)
                results[name, arm] = row
                replace_file(root, f"jobs/{name}/{arm}.json", canonical_json(dict(
                    state="complete", attempts=attempts[name, arm], receipt_sha256=row["receipt_sha256"])).encode())
                print("DONE", name, arm, row["observation"]["process"]["wall_seconds"], flush=True)
            except Exception as exc:
                retry = infrastructure_failure(exc) and attempts[name, arm] < 2
                replace_file(root, f"jobs/{name}/{arm}.json", canonical_json(dict(
                    state="pending" if retry else "failed", attempts=attempts[name, arm], error=str(exc))).encode())
                if retry:
                    todo.append((name, arm))
                else:
                    failures[name + "/" + arm] = str(exc)
                print("FAILED", name, arm, str(exc), flush=True)
        progress(root, phase=args.phase, completed=len(results), failed=len(failures), active=len(active), queued=len(todo))
    save(root / "attempts" / (args.phase + f"-{time.time_ns()}-failures.json"), failures)
    # Keep pandas joins out of the simulation wave, with at most two per host.
    pending = []
    for name in names:
        for arm in arms:
            if arm != "oracle" and (name, arm) in results and (name, "oracle") in results:
                pending.append((name, arm))
    scored, scoring_failures = {}, {}
    while pending:
        batch = matching_batch(pending, results, available_bytes())
        del pending[:len(batch)]
        refs = [score_pair.options().remote(root, n, a, results[n, "oracle"], results[n, a], 10_000) for n, a in batch]
        for (name, arm), ref in zip(batch, refs):
            try:
                score = get(ref)
                if score["pairing_error"] or score["request_pairing"]["address_mismatch_pairs"] != 0:
                    raise RuntimeError("request qualification failed for " + name + "/" + arm)
                scored.setdefault(arm, {})[name] = score
            except Exception as exc:
                scoring_failures[name + "/" + arm] = str(exc)
        progress(root, phase=args.phase + "_matching", scored=sum(map(len, scored.values())), queued=len(pending))
    # Per-measurement and per-pair receipts are immutable. This index is a
    # replaceable view, so a resumed phase can add newly completed jobs.
    replace_file(root, args.phase + "-scores.json", canonical_json(scored).encode())
    progress(root, phase=args.phase + "_complete", completed=len(results), failures=failures,
             scoring_failures=scoring_failures, terminal=True, active=0, queued=0, no_llm_calls=True)
    ray.shutdown()
    if failures or scoring_failures:
        raise RuntimeError("failed measurements retained; inspect before retrying")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    for name in ("output", "runtime", "source4", "source8"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--source1", type=Path)
    parser.add_argument("--completion-policy", choices=("finite", "background-replay"), default="finite")
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--validation-config", type=Path)
    parser.add_argument("--scanner", type=Path)
    parser.add_argument("--site", choices=("local", "cloud"), default="local")
    parser.add_argument("--phase", choices=("qualification", "production"), default="qualification")
    args = parser.parse_args()
    if not 1 <= args.cpus <= len(os.sched_getaffinity(0)):
        parser.error("cpus must fit the CPUs available on this host")
    with exclusive_lock(args.output / (args.mode + "-" + args.site + ".lock")):
        try:
            (prepare if args.mode == "prepare" else run)(args)
        except BaseException as exc:
            if args.mode == "run":
                record_stop(args.output / args.site, exc)
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()


if __name__ == "__main__":
    main()
