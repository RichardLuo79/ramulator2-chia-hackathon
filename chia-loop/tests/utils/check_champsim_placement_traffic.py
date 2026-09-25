"""Short placement correctness fixtures, never accuracy-study measurements.

Run against the same source-built 4-/8-core executables as the study. This
exercises normal trace reading, translation, caches and the Ramulator bridge;
it does not change production instruction windows or model parameters.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import struct
import subprocess

from ramulator_chia.eval import config as C, matchlib, simpleo3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", type=int, choices=(4, 8), required=True)
    parser.add_argument("--reference-binary", type=Path, help="optional pre-patch executable for the second no-placement run")
    parser.add_argument("--patterns", nargs="+", choices=("streaming", "random", "mixed_rw", "page_crossing"),
                        default=("streaming", "random", "mixed_rw", "page_crossing"))
    args = parser.parse_args()
    import ramulator

    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.source / f"bin/champsim-{args.cores}core"
    env = dict(os.environ, LD_LIBRARY_PATH=str(args.runtime / "runtime"), RAMULATOR_TICKS_PER_8="12")
    env.pop("CHAMPSIM_PLACEMENT_FILE", None)
    env.pop("RAMULATOR_CONFIG", None)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = {}
    for pattern in args.patterns:
        root = args.output / pattern
        root.mkdir(exist_ok=True)
        rng = random.Random(20260914)
        pages, records = set(), bytearray()
        for i in range(200_000):
            ip = 0x400000 + (i % 256) * 4
            address = 0x10000000 + ((rng.randrange(262144) if pattern == "random" else i % 262144) * 64)
            if pattern == "page_crossing":
                address = 0x10000000 + (i // 2 % 4096) * 4096 + (4095 if i % 2 == 0 else 0)
            store = pattern == "mixed_rw" and i % 4 == 0
            records += struct.pack("<QBB2B4B2Q4Q", ip, 0, 0, 1, 0, 0, 0, 0, 0,
                                   address if store else 0, 0, 0 if store else address, 0, 0, 0)
            pages.update((ip >> 12, address >> 12))
        trace = root / "fixture.gz"
        trace.write_bytes(gzip.compress(records, compresslevel=3, mtime=0))
        inventory, placement = root / "pages.txt", root / "placement.txt"
        inventory.write_text(f"CHAMPSIM_PAGES_V1 {args.cores} 4096 {8 << 30}\n" +
                             "".join(f"{core} {page}\n" for core in range(args.cores) for page in sorted(pages)))
        subprocess.run([str(binary), "--prepare-placement", str(inventory), str(placement)], env=env,
                       stdout=subprocess.DEVNULL, check=True, timeout=60)
        runs = {}
        for name, model, fixed in (("oracle", "oracle", True), ("fixedlat", "fixedlat", True),
                                   ("legacy1", "fixedlat", False), ("legacy2", "fixedlat", False)):
            output = root / name
            output.mkdir(exist_ok=True)
            dram = ramulator.dram.DDR5(org_preset=C.STD["DDR5"]["org"], timing_preset=C.STD["DDR5"]["timing"])
            controller = simpleo3._build_controller(ramulator, model, dram, "DDR5", output / "controller.csv", {}, mess_curve=None)
            config = {"frontend": {"impl": "External", "clock_ratio": 1}, "memory_system":
                      ramulator.memory_system.GenericDRAM(clock_ratio=1, controllers=[controller],
                          channel_mapper=ramulator.channel_mapper.CacheLineInterleave()).to_config()}
            config_path = output / "config.json"
            config_path.write_text(json.dumps(config))
            live_env = dict(env, RAMULATOR_CONFIG=str(config_path))
            if fixed:
                live_env["CHAMPSIM_PLACEMENT_FILE"] = str(placement)
            run_binary = args.reference_binary if name == "legacy2" and args.reference_binary else binary
            with (output / "simulation.log").open("w") as log:
                subprocess.run([str(run_binary), "-w", "1000", "-i", "10000", *([str(trace)] * args.cores)],
                               env=live_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            text = (output / "simulation.log").read_text()
            assert "Reached end of trace" not in text
            timing = re.findall(r"Simulation finished CPU (\d+) instructions: (\d+) cycles: (\d+)", text)
            assert len(timing) == args.cores and all(10_000 <= int(row[1]) < 10_005 for row in timing)
            observation = output / "controller.csv.ch0"
            runs[name] = dict(timing=sorted(timing), csv_sha256=hashlib.sha256(observation.read_bytes()).hexdigest())
        assert runs["legacy1"] == runs["legacy2"], "no-placement mode is not repeatable"
        paired = matchlib.match(root / "oracle/controller.csv.ch0", root / "fixedlat/controller.csv.ch0",
                               champsim_filter_physical_mismatches=True)
        assert paired["address_mismatch_pairs"] == 0 and len(paired["dv"]) > 0
        result[pattern] = dict(runs=runs, matched=len(paired["dv"]), physical_mismatches=0,
                               trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        print("PASS", args.cores, pattern, flush=True)
    # Trusted preparation still fails closed on corrupt maps and insufficient
    # capacity. These negative cases must not become a timing-based fallback.
    original = placement.read_text().splitlines()
    data = [i for i, line in enumerate(original) if line.startswith("D ")]
    negative = {}
    for kind in ("alias", "out_of_range", "missing_root"):
        lines = original.copy()
        if kind == "missing_root":
            index = next(i for i, line in enumerate(lines) if line.startswith("P 0 5 "))
            del lines[index]
            header = lines[0].split()
            header[-1] = str(int(header[-1]) - 1)
            lines[0] = " ".join(header)
        else:
            item = lines[data[0]].split()
            item[-1] = lines[data[1]].split()[-1] if kind == "alias" else str((8 << 30) // 4096)
            lines[data[0]] = " ".join(item)
        bad = args.output / (kind + ".txt")
        bad.write_text("\n".join(lines) + "\n")
        trial = subprocess.run([str(binary), "--prepare-placement", str(inventory), str(args.output / "unused.map")],
                               env=dict(env, CHAMPSIM_PLACEMENT_FILE=str(bad)), capture_output=True, timeout=60)
        assert trial.returncode != 0
        negative[kind] = dict(returncode=trial.returncode, stderr=trial.stderr.decode())
    small = args.output / "capacity.pages"
    lines = inventory.read_text().splitlines()
    lines[0] = f"CHAMPSIM_PAGES_V1 {args.cores} 4096 {(1 << 20) + 4096}"
    small.write_text("\n".join(lines) + "\n")
    trial = subprocess.run([str(binary), "--prepare-placement", str(small), str(args.output / "unused.map")],
                           env=env, capture_output=True, timeout=60)
    assert trial.returncode != 0 and b"exceed physical memory capacity" in trial.stderr
    negative["capacity"] = dict(returncode=trial.returncode, stderr=trial.stderr.decode())
    (args.output / "negative-checks.json").write_text(json.dumps(negative, indent=2) + "\n")
    (args.output / "passed.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
