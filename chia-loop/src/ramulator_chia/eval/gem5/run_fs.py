"""Run the frozen full-ROI FS matrix on one host, with resumable records.

This is an evaluation queue, not an agent loop. There are no model calls,
instruction limits, or spending cutoffs. Cost thresholds only produce reports.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from fs_board import save_json, sha256


def make_jobs(workloads):
    jobs = []
    for suite, spec in workloads["suites"].items():
        for cores in workloads["cores"]:
            group = f"{suite}-c{cores}"
            prepare = f"prepare-{group}"
            jobs.append(dict(id=prepare, suite=suite, cores=cores, kind="prepare", deps=[], priority=0,
                             slots=cores+1 if cores > 1 else 1))
            qualifications = [f"qualification-{group}-{m}" for m in ("oracle", "fixedlat")]
            for model, key in zip(("oracle", "fixedlat"), qualifications):
                jobs.append(dict(id=key, suite=suite, cores=cores, kind="qualification", model=model,
                                 deps=[prepare], priority=1, slots=1, **spec["qualification"]))
            for benchmark in spec["benchmarks"]:
                pilot = benchmark == spec["pilot"]
                for model in workloads["models"]:
                    jobs.append(dict(id=f"production-{group}-{benchmark}-{model}", suite=suite,
                        cores=cores, kind="production", benchmark=benchmark, model=model,
                        graph_scale=spec.get("graph_scale", 20), input_size=spec.get("input_size", "simlarge"),
                        deps=qualifications if pilot else [*qualifications, f"production-{group}-{spec['pilot']}-{model}"],
                        priority=2 if pilot else 4, slots=1, pilot=pilot))
            jobs.append(dict(id=f"repeat-{group}", suite=suite, cores=cores, kind="repeat", model="oracle",
                benchmark=spec["pilot"], graph_scale=spec.get("graph_scale", 20),
                input_size=spec.get("input_size", "simlarge"),
                deps=[f"production-{group}-{spec['pilot']}-oracle"], priority=3, slots=1))
    return sorted(jobs, key=lambda job: (job["priority"], job["id"]))


def checkpoint_path(root, suite, cores, records):
    record = records[f"prepare-{suite}-c{cores}"]
    return root / record["directory"] / "checkpoint"


def command(root, host, job, directory, records):
    result = [host["gem5"], "--outdir=" + str(directory), str(root / "fs_board.py"),
              "prepare" if job["kind"] == "prepare" else "run", "--suite=" + job["suite"],
              "--cores=" + str(job["cores"]), "--resources=" + host["resources"][job["suite"]]]
    if job["kind"] != "prepare":
        result += ["--checkpoint=" + str(checkpoint_path(root, job["suite"], job["cores"], records)),
                   "--benchmark=" + job["benchmark"], "--ramulator-python=" + host["ramulator_python"],
                   "--memory-config=" + str(root / "effective-configs" / (job["model"] + ".json")),
                   "--graph-scale=" + str(job.get("graph_scale", 20)),
                   "--input-size=" + job.get("input_size", "simlarge")]
    return result


def archive(directory, compressor="pigz -p 1 -3"):
    target = directory / "records.tar.gz"
    # Writing inside the archived directory changes its mtime while tar is
    # reading it, even if the output file itself is excluded.
    temporary = directory.parent / (directory.name + ".records.tar.gz.partial")
    subprocess.run(["tar", "--exclude=./records.tar.gz", "--exclude=./records.tar.gz.partial",
                    "--exclude=./records.tar.gz.sha256",
                    "--exclude=*.socket", "-I", compressor, "-cf", str(temporary),
                    "-C", str(directory), "."], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["gzip", "-t", str(temporary)], check=True)
    temporary.replace(target)
    (directory / "records.tar.gz.sha256").write_text(sha256(target) + "  records.tar.gz\n")
    return sha256(target)


def memory_counts(directory):
    import yaml
    data = yaml.safe_load((directory / "roi.ramulator.yaml").read_text())
    found = []

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("num_read_reqs", "num_write_reqs", "s_read_reqs", "s_write_reqs"):
                    found.append({"name": prefix+key, "value": child})
                else:
                    walk(child, prefix+str(key)+".")
        elif isinstance(value, list):
            for i, child in enumerate(value):
                walk(child, prefix+str(i)+".")
    walk(data)
    return found


def finish(root, job, record):
    directory = root / record["directory"]
    if job["kind"] == "prepare":
        checkpoint = directory / "checkpoint"
        stage = json.loads((checkpoint / "stage.json").read_text())
        if stage["stage"] != "prelaunch" or stage["benchmark_supplied"]:
            raise RuntimeError("invalid pre-launch checkpoint")
        files = {str(p.relative_to(checkpoint)): sha256(p) for p in checkpoint.rglob("*") if p.is_file()}
        save_json(directory / "checkpoint-files.json", files)
        result = dict(stage=stage, files=files)
    else:
        result = json.loads((directory / "result.json").read_text())
        if result["roi_ticks"] <= 0 or result["cores"] != job["cores"]:
            raise RuntimeError("invalid completed ROI")
        result["dram_counters"] = memory_counts(directory)
        runtime_file = directory / "loaded-libraries.json"
        if runtime_file.exists():
            result["loaded_libraries"] = json.loads(runtime_file.read_text())
        # Record the actual SimObjects, not just the requested command line.
        config = (directory / "config.ini").read_text()
        if config.count("type=BaseO3CPU") != job["cores"] or "type=Ramulator2\n" not in config:
            raise RuntimeError("run did not instantiate the requested O3/Ramulator board")
        if "type=CowDiskImage" not in config or "read_only=true" not in config:
            raise RuntimeError("missing read-only base image and COW overlay")
    save_json(directory / "measurement.json", result)
    record.update(status="completed", result=result, completed_at=time.time(),
                  archive_sha256=archive(directory), measurement_sha256=sha256(directory / "measurement.json"))


def estimated_cost(host, now=None):
    now = time.time() if now is None else now
    hours = max(0, now-host["created_unix"])/3600
    components = {name: rate*hours for name, rate in host["hourly_usd"].items()}
    components["known_transfers"] = host.get("known_transfer_usd", 0)
    return dict(estimated_usd=sum(components.values()), hours=hours, components=components,
                basis="gross list-rate estimate before credits/tax; billing reconciliation pending")


def compare_rows(jobs, records):
    production = {j["id"]: j for j in jobs if j["kind"] == "production"}
    rows = []
    for key, job in production.items():
        record = records.get(key, {})
        if record.get("status") != "completed":
            continue
        result = record["result"]
        reference = records.get(f"production-{job['suite']}-c{job['cores']}-{job['benchmark']}-oracle", {})
        ticks = reference.get("result", {}).get("roi_ticks") if reference.get("status") == "completed" else None
        signed = 100*(result["roi_ticks"]-ticks)/ticks if ticks else None
        reads = [v["value"] for v in result.get("dram_counters", []) if "read" in v["name"]]
        rows.append(dict(suite=job["suite"], benchmark=job["benchmark"], cores=job["cores"], model=job["model"],
            roi_ticks=result["roi_ticks"], signed_error_pct=signed,
            absolute_error_pct=abs(signed) if signed is not None else None,
            oracle_roi_ticks=ticks, wall_seconds=result["wall_seconds"],
            setup_wall_seconds=result["phases"]["wall_seconds"]["setup"],
            roi_wall_seconds=result["phases"]["wall_seconds"]["roi"],
            verification_wall_seconds=result["phases"]["wall_seconds"]["verification"],
            max_rss_kib=result["max_rss_kib"], recorded_read_count=max(reads) if reads else None,
            low_dram_traffic=(max(reads) < 10000) if reads else None))
    return rows


def report(root, jobs, records, host, active=()):
    ledger = root / "transfer-ledger.json"
    if ledger.exists():
        host = {**host, "known_transfer_usd": json.loads(ledger.read_text())["estimated_usd"]}
    cost = estimated_cost(host)
    counts = Counter(r["status"] for r in records.values())
    rows = compare_rows(jobs, records)
    status = dict(time=datetime.now(timezone.utc).isoformat(), cost=cost, jobs=dict(counts),
                  production_completed=len(rows), production_total=480, active=list(active))
    save_json(root / "status.json", status)
    save_json(root / "results.json", rows)
    if rows:
        with (root / "results.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Full-ROI gem5 FS evaluation", "", f"Updated {status['time']}", "",
        f"Completed primary runs: **{len(rows)}/480**. States: {dict(counts)}.",
        f"Estimated experiment cost: **USD {cost['estimated_usd']:.2f}** after {cost['hours']:.2f} hours.",
        "This is gross resource usage, not an invoice or remaining-credit balance. No cost/time cutoff.", "",
        "The measured interval is the original full ROI. Setup and built-in warmup run under O3; verification follows ROI-end.",
        "No gem5 request matching. Numerical verification is checked for NPB/GAPBS; PARSEC checks ROI and successful process completion.", "",
        "## Accuracy on shared completed workloads", "",
        "Averages compare all seven immediate-response models on the same completed workloads within each suite/core group.", "",
        "| Suite | Cores | Shared workloads | Model | Mean absolute ROI-time error (%) | Worst (%) |",
        "|---|---:|---:|---|---:|---:|"]
    models = [m for m in json.loads((root / "fs_workloads.json").read_text())["models"] if m != "oracle"]
    indexed = {(r["suite"], r["cores"], r["benchmark"], r["model"]): r for r in rows}
    for suite in ("npb", "gapbs", "parsec"):
        for cores in (1, 4):
            names = {r["benchmark"] for r in rows if r["suite"] == suite and r["cores"] == cores}
            shared = sorted(n for n in names if all((suite, cores, n, m) in indexed and
                indexed[suite, cores, n, m]["absolute_error_pct"] is not None for m in models))
            for model in models if shared else []:
                errors = [indexed[suite, cores, n, model]["absolute_error_pct"] for n in shared]
                lines.append(f"| {suite} | {cores} | {len(shared)} | {model} | {statistics.mean(errors):.4f} | {max(errors):.4f} |")
    lines += ["", "Per-workload values, including results not yet in a complete comparison group, are in results.csv.",
              "", "## Runtime and repeatability", ""]
    for suite in ("npb", "gapbs", "parsec"):
        values = [r for r in rows if r["suite"] == suite]
        if values:
            medians = {phase: round(statistics.median(r[phase+"_wall_seconds"] for r in values), 2)
                       for phase in ("setup", "roi", "verification")}
            lines.append(f"- {suite}: {len(values)} completed; median phase wall seconds {medians}.")
    samples = defaultdict(list)
    for job in jobs:
        record = records.get(job["id"], {})
        if job["kind"] == "production" and record.get("status") == "completed":
            samples[job["suite"], job["cores"], job["model"]].append(record["result"]["wall_seconds"])
    estimated_work_seconds, unestimated = 0.0, 0
    for job in jobs:
        record = records.get(job["id"], {})
        if job["kind"] != "production" or record.get("status") not in ("pending", "running"):
            continue
        values = samples[job["suite"], job["cores"], job["model"]]
        if not values:
            unestimated += 1
        else:
            elapsed = time.time()-record["started_at"] if record["status"] == "running" else 0
            estimated_work_seconds += max(0, statistics.median(values)-elapsed)
    extra_hours = estimated_work_seconds / host.get("slots", 46) / 3600
    extra_cost = extra_hours * sum(host["hourly_usd"].values())
    lines += [f"- Sample-based remaining work: {extra_hours:.2f} fully occupied host-hours / USD {extra_cost:.2f}; {unestimated} jobs have no comparable completed sample.",
              "  This is a provisional throughput extrapolation, not an upper bound: unfinished long ROIs, dependencies and low end-of-batch parallelism can make completion substantially later."]
    for job in jobs:
        record = records.get(job["id"], {})
        if job["kind"] == "repeat" and record.get("status") == "completed":
            key = f"production-{job['suite']}-c{job['cores']}-{job['benchmark']}-oracle"
            original = records[key]["result"]["roi_ticks"]
            delta = 100*(record["result"]["roi_ticks"]-original)/original
            lines.append(f"- Oracle repeat {job['suite']}/{job['benchmark']}/c{job['cores']}: {delta:+.6f}% ROI-time change.")
    speedups = []
    for row in rows:
        one = indexed.get((row["suite"], 1, row["benchmark"], row["model"]))
        if row["cores"] == 4 and one:
            speedups.append(dict(suite=row["suite"], benchmark=row["benchmark"], model=row["model"],
                                 speedup=one["roi_ticks"]/row["roi_ticks"]))
    save_json(root / "speedups.json", speedups)
    lines += ["", "## Running jobs", ""]
    for item in active:
        lines.append(f"- {item['id']}: {item.get('phase', 'starting')}; {item.get('wall_seconds', 0):.0f}s; RSS {item.get('rss_kib', 0)/1024:.0f} MiB.")
    lines += ["", "## Failed or blocked jobs", ""]
    lines += [f"- {key}: {r['status']}: {r.get('error', '')}" for key, r in records.items()
              if r["status"] in ("failed", "blocked")]
    content = "\n".join(lines) + "\n"
    (root / "report.md").write_text(content)
    threshold = 150
    while threshold <= cost["estimated_usd"]:
        milestone = root / f"review-usd-{threshold}.md"
        if not milestone.exists():
            milestone.write_text(content)
            print(f"COST_REVIEW USD{threshold}: {milestone}; continuing as authorized", flush=True)
        threshold += 50
    return status


def run(root):
    host = json.loads((root / "host.json").read_text())
    workload = json.loads((root / "fs_workloads.json").read_text())
    jobs = make_jobs(workload)
    records_file = root / "jobs.json"
    records = json.loads(records_file.read_text()) if records_file.exists() else {}
    finished = root / "finished.json"
    if finished.exists():
        finished.rename(root / f"finished-previous-{int(time.time())}.json")
    for job in jobs:
        record = records.setdefault(job["id"], dict(status="pending", attempts=0))
        if record["status"] == "running":
            record.update(status="pending" if record["attempts"] < 2 else "failed", error="previous worker interrupted")
        elif record["status"] == "completed":
            directory = root / record["directory"]
            if sha256(directory / "measurement.json") != record["measurement_sha256"] or sha256(directory / "records.tar.gz") != record["archive_sha256"]:
                raise RuntimeError("completed measurement checksum changed: " + job["id"])
        elif record["status"] == "failed" and record.get("returncode") == 0 and (
                root / record.get("directory", "") / "measurement.json").is_file():
            # A successful simulator run must not be rerun merely because
            # collection/compression failed. Retain failure.json and recollect.
            finish(root, job, record)
            record["collection_recovered"] = True
        elif record["status"] == "blocked":
            record["status"] = "pending"  # Re-evaluate dependencies after recovery.
    save_json(root / "job-manifest.json", jobs)
    save_json(records_file, records)
    active = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_report = 0
    while True:
        for job in jobs:
            key = job["id"]
            if key not in active:
                continue
            process, log, submitted = active[key]
            code = process.poll()
            if code is None:
                continue
            log.close()
            record = records[key]
            record["returncode"] = code
            try:
                if code:
                    raise RuntimeError(f"gem5 exited with status {code}")
                finish(root, job, record)
                print("COMPLETED " + key, flush=True)
            except Exception as exc:
                record.update(status="failed", error=str(exc), completed_at=time.time())
                # One retry for process interruption, not deterministic model,
                # verification or segmentation-fault failures.
                if code in (-signal.SIGTERM, -signal.SIGHUP) and record["attempts"] < 2:
                    record["status"] = "pending"
                save_json(root / record["directory"] / "failure.json", record)
                print("FAILED " + key + ": " + str(exc), flush=True)
            active.pop(key)
            save_json(records_file, records)
        if not stopping:
            used = sum(next(j["slots"] for j in jobs if j["id"] == key) for key in active)
            available_memory = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
            for job in jobs:
                record = records[job["id"]]
                if record["status"] != "pending":
                    continue
                bad = [dep for dep in job["deps"] if records[dep]["status"] in ("failed", "blocked")]
                if bad:
                    record.update(status="blocked", error="dependency failed: " + ", ".join(bad))
                    continue
                if used + job["slots"] > host.get("slots", 46) or available_memory < 16*1024**2:
                    continue
                if any(records[dep]["status"] != "completed" for dep in job["deps"]):
                    continue
                record["attempts"] += 1
                directory = root / "jobs" / job["id"] / f"attempt-{record['attempts']}"
                directory.mkdir(parents=True, exist_ok=False)
                argv = command(root, host, job, directory, records)
                save_json(directory / "launch.json", dict(job=job, argv=argv, driver_sha256=sha256(root / "fs_board.py"),
                    config_sha256=sha256(root / "effective-configs" / (job["model"]+".json")) if "model" in job else None))
                log = (directory / "simulation.log").open("w")
                environment = dict(os.environ, LD_LIBRARY_PATH=host["ramulator_library_dir"],
                    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=root)
                record.update(status="running", directory=str(directory.relative_to(root)), pid=process.pid, started_at=time.time())
                active[job["id"]] = (process, log, time.time())
                used += job["slots"]
                available_memory -= 4*1024**2  # Conservative admission until RSS is observed.
                save_json(records_file, records)
                print("STARTED " + job["id"], flush=True)
        if time.time() - last_report >= 60:
            processes = []
            for key, (process, _, submitted) in active.items():
                progress = root / records[key]["directory"] / "progress.json"
                entry = json.loads(progress.read_text()) if progress.exists() else {}
                try:
                    rss = int(next(line.split()[1] for line in Path(f"/proc/{process.pid}/status").read_text().splitlines() if line.startswith("VmRSS:")))
                except (FileNotFoundError, StopIteration):
                    rss = 0
                processes.append(dict(id=key, pid=process.pid, phase=entry.get("phase", entry.get("event")),
                                      tick=entry.get("tick"), wall_seconds=time.time()-submitted, rss_kib=rss))
            save_json(records_file, records)
            report(root, jobs, records, host, processes)
            last_report = time.time()
        if not active and (stopping or not any(r["status"] == "pending" for r in records.values())):
            break
        time.sleep(2)
    save_json(records_file, records)
    final = report(root, jobs, records, host)
    if not stopping:
        save_json(root / "finished.json", final)
    return 1 if stopping or any(r["status"] in ("failed", "blocked") for r in records.values()) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report(args.root, json.loads((args.root / "job-manifest.json").read_text()),
               json.loads((args.root / "jobs.json").read_text()), json.loads((args.root / "host.json").read_text()))
    else:
        with (args.root / "worker.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(run(args.root.resolve()))
