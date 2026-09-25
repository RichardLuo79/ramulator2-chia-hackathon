"""Scheduling and summaries for standalone throughput measurements.

The public executor is scripts/evaluate --suite speed. These helpers retain
the published matrix order and aggregate completed attempt receipts.
"""

from datetime import datetime
import math
import os
import random
import statistics
import time

from .records import read, save

ARMS = (
    "oracle",
    "astra_single_core",
    "deepseek_single_core",
    "astra_multicore",
    "deepseek_multicore",
    "fixedlat",
    "md1",
    "wmg1",
    "mess",
)
CRITICAL = ARMS[:6]
RATIOS = (100, 75, 50)
INTERVALS = (1, 4, 16, 64, 256)
WARMUP, MEASURED, REPEATS = 100000, 5000000, 5
STOP = float(os.environ.get("RAMULATOR_CHIA_STOP_AT", "inf"))


def settings(root):
    """Optional new-study manifest; absent manifests retain the original study."""
    path = root / "study.json"
    value = read(path) if path.exists() else {}
    arms = tuple(value.get("models", ARMS))
    allowed = set(ARMS) | {"opus_single_core", "gemini_single_core"}
    if (
        not arms
        or len(set(arms)) != len(arms)
        or not set(arms) <= allowed
        or "oracle" not in arms
    ):
        raise ValueError("invalid standalone speed model population")
    critical = tuple(value.get("critical_models", [a for a in CRITICAL if a in arms]))
    if len(set(critical)) != len(critical) or not set(critical) <= set(arms):
        raise ValueError("invalid standalone speed priority population")
    return {**value, "models": arms, "critical_models": critical}


def cutoff(root):
    value = settings(root)
    if "stop_utc" not in value:
        return STOP
    return (
        datetime.fromisoformat(value["stop_utc"].replace("Z", "+00:00")).timestamp()
        if value["stop_utc"]
        else None
    )


def points():
    rows = [
        dict(pattern=p, read_percent=r, interval=i)
        for p in ("streaming", "random")
        for r in RATIOS
        for i in INTERVALS
    ]
    random.Random(12345).shuffle(rows)
    return rows


def point_name(p):
    return f"{p['pattern']}-r{p['read_percent']}-i{p['interval']}"


def schedule(arms=ARMS, critical=CRITICAL):
    traffic = points()
    corners = [
        p
        for p in traffic
        if p["read_percent"] in (100, 50) and p["interval"] in (1, 256)
    ]
    other = [p for p in traffic if p not in corners]
    jobs = []
    for repeat in range(1, REPEATS + 1):
        # User priority: complete the key six-model population before the other
        # baselines. Complete the whole first pass before collecting repeats.
        for group in (critical, tuple(a for a in arms if a not in critical)):
            if not group:
                continue
            for j, p in enumerate((corners + other) if repeat == 1 else traffic):
                order = list(group)
                offset = (j + repeat - 1) % len(order)
                order = order[offset:] + order[:offset]
                for arm in order:
                    jobs.append(dict(arm=arm, point=p, repeat=repeat))
    return jobs


def simulation_identity(stats, *, legacy_precision=False, ignore_retries=False):
    """Exact native integer counters; floats compared at legacy print precision."""

    def clean(value):
        if isinstance(value, dict):
            return {
                k: clean(v)
                for k, v in value.items()
                if not k.endswith(("_wall_s", "_cpu_s"))
                and not (ignore_retries and k == "send_rejects")
            }
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite simulation statistic")
            return float(format(value, ".6g")) if legacy_precision else value
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return clean(stats)


def export(root):
    study = settings(root)
    expected = len(study["models"]) * len(points()) * REPEATS
    rows = [read(p) for p in (root / "production/runs").glob("*/receipt.json")]
    good = [r for r in rows if r["complete"]]
    summary = []
    for p in points():
        oracle = [
            r["requests_per_second"]
            for r in good
            if r["arm"] == "oracle" and r["point"] == p
        ]
        for arm in study["models"]:
            chosen = [r for r in good if r["arm"] == arm and r["point"] == p]
            values = [r["requests_per_second"] for r in chosen]
            summary.append(
                dict(
                    **p,
                    arm=arm,
                    n=len(values),
                    median=statistics.median(values) if values else None,
                    mean=statistics.mean(values) if values else None,
                    standard_deviation=statistics.stdev(values)
                    if len(values) > 1
                    else None,
                    oracle_relative_speedup=statistics.median(values)
                    / statistics.median(oracle)
                    if values and oracle
                    else None,
                    repetitions=[
                        dict(
                            repeat=r["repeat"],
                            requests_per_second=r["requests_per_second"],
                            wall_seconds=r["batch"]["measured_wall_s"],
                            cpu_seconds=r["batch"]["measured_cpu_s"],
                            drain_wall_seconds=r["batch"]["drain_wall_s"],
                            peak_rss_kib=r["resources"]["peak_rss_kib"],
                            warmup_wall_seconds=r["batch"]["warmup_wall_s"],
                            warmup_cpu_seconds=r["batch"]["warmup_cpu_s"],
                            drain_cpu_seconds=r["batch"]["drain_cpu_s"],
                            input_loading_wall_seconds=r["input_loading_wall_s"],
                            total_process_wall_seconds=r["resources"]["wall_seconds"],
                            total_process_cpu_seconds=r["resources"]["user_seconds"]
                            + r["resources"]["system_seconds"],
                            simulated_cycles=r["batch"]["cycles"],
                            measured_simulated_cycles=r["batch"]["cycles"]
                            - r["batch"]["measurement_start_cycle"],
                            drain_simulated_cycles=r["batch"]["cycles"]
                            - r["batch"]["final_admission_cycle"],
                            admission_wait_sum=r["batch"][
                                "measured_admission_wait_sum"
                            ],
                            admission_wait_max=r["batch"][
                                "measured_admission_wait_max"
                            ],
                            warmup_reads_outstanding=r["batch"][
                                "warmup_reads_outstanding"
                            ],
                            warmup_writes_outstanding=r["batch"][
                                "warmup_writes_outstanding"
                            ],
                            measured_reads=r["batch"]["measured_reads"],
                            measured_writes=r["batch"]["measured_writes"],
                            measured_reads_completed=r["batch"][
                                "measured_reads_completed"
                            ],
                            measured_writes_completed=r["batch"][
                                "measured_writes_completed"
                            ],
                        )
                        for r in sorted(chosen, key=lambda r: r["repeat"])
                    ],
                )
            )
    save(
        root / "production/results.json",
        dict(
            time=time.time(),
            expected=expected,
            successful=len(good),
            failed=len(rows) - len(good),
            pending=expected - len(rows),
            summary=summary,
            failures=[r for r in rows if not r["complete"]],
            protocol=read(root / "production/protocol.json"),
        ),
    )
