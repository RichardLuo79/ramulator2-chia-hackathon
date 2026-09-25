"""gem5 v25.1 FS driver: KVM pre-launch checkpoint, then full-ROI O3 execution.

The guest receives its benchmark command only after restoration. Checkpoint
draining therefore cannot run any benchmark work under KVM. All restores use
gem5's private in-memory COW disk over the same read-only base image.
"""

import argparse
import configparser
import copy
import hashlib
import json
from pathlib import Path
import re
import resource
import sys
import time

FS_START_TICK = 13_121_004_000_177
PAYLOAD_HEADER = "# GEM5_FULL_ROI_PAYLOAD_V1"
IMAGES = {"npb": "x86-ubuntu-24.04-npb-img-5.0.0",
          "gapbs": "x86-ubuntu-24.04-gapbs-img-1.0.0", "parsec": "x86-parsec-1.0.0"}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guest_script(suite, cores, benchmark=None, graph_scale=20, input_size="simlarge"):
    """A preparation trampoline or a benchmark payload, never both."""
    bridge = "m5" if suite == "parsec" else "gem5-bridge"
    if benchmark is None:
        # An early marker is deliberately not used: the wrapper's existing FS
        # boot workaround requires restoring after 13.121 simulated seconds.
        marker = "m5 exit" if suite == "parsec" else "gem5-bridge hypercall 6"
        return (f"#!/bin/bash\nsleep 10\n{marker}\n"
                "while true; do\n"
                f"  {bridge} readfile > /tmp/full-roi-payload.sh\n"
                f"  if head -n 1 /tmp/full-roi-payload.sh | grep -qx '{PAYLOAD_HEADER}'; then\n"
                "    exec /bin/bash /tmp/full-roi-payload.sh\n"
                "  fi\n  sleep 0.01\ndone\n")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", benchmark):
        raise ValueError("invalid benchmark name")
    if suite == "npb":
        if not re.fullmatch(r"[a-z]+\.[A-Z]\.x", benchmark):
            raise ValueError("expected NPB executable")
        command = f"/home/gem5/NPB3.4-OMP/bin/{benchmark}"
    elif suite == "gapbs":
        if benchmark not in ("bfs", "bc", "cc", "cc_sv", "pr", "pr_spmv", "sssp", "tc"):
            raise ValueError("unknown GAPBS kernel")
        command = f"/home/gem5/gapbs/bin/{benchmark} -g {graph_scale} -k 16 -n 1 -s -v"
    else:
        if input_size not in ("simsmall", "simmedium", "simlarge"):
            raise ValueError("unknown PARSEC input")
        command = ("cd /home/gem5/parsec-benchmark && source env.sh && "
                   f"parsecmgmt -a run -p {benchmark} -c gcc-hooks -i {input_size} -n {cores}")
    return (f"{PAYLOAD_HEADER}\nexport OMP_NUM_THREADS={cores}\n"
            "export OMP_DYNAMIC=FALSE\nexport OMP_DISPLAY_ENV=TRUE\n"
            "{\n"
            "echo FULL_ROI_PAYLOAD_STARTED\necho GUEST_CPUS=$(nproc)\n"
            f"test \"$(nproc)\" = {cores} || exit 2\n{command}\n"
            "workload_status=$?\necho FULL_ROI_EXIT_CODE=$workload_status\n"
            "} > /tmp/full-roi-output.txt 2>&1\n"
            f"{bridge} writefile /tmp/full-roi-output.txt workload-output.txt\n"
            + ("m5 exit\n" if suite == "parsec" else "gem5-bridge hypercall 3\n")
            + "exit $workload_status\n")


def check_workload_output(suite, text, exit_marker="FULL_ROI_EXIT_CODE"):
    if not re.search(r"^" + re.escape(exit_marker) + r"=0$", text, re.MULTILINE):
        raise RuntimeError("guest benchmark did not exit successfully")
    if suite == "npb" and not re.search(r"Verification\s*=\s*SUCCESSFUL", text, re.I):
        raise RuntimeError("NPB numerical verification failed")
    if suite == "gapbs" and not re.search(r"Verification\s*:?\s+PASS\b", text):
        raise RuntimeError("GAPBS numerical verification failed")
    return "numerical_verification" if suite != "parsec" else "roi_and_process_completion"


def loaded_libraries(model_library=None):
    """Check what this process actually loaded, not only its requested config."""
    mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if len(line.split()) >= 6 and line.split()[-1].startswith("/")}
    libraries = [Path(path) for path in mapped if Path(path).name == "libramulator.so"]
    if not libraries:
        raise RuntimeError("Ramulator shared library is not mapped")
    if model_library:
        model = Path(model_library).resolve()
        if str(model) not in mapped:
            raise RuntimeError("requested frozen model was not loaded; check RAMULATOR_CHIA_MODEL_API")
        libraries.append(model)
    return {str(path): sha256(path) for path in libraries}


def main():
    import m5
    from gem5.components.boards.x86_board import X86Board
    from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import PrivateL1PrivateL2CacheHierarchy
    from gem5.components.memory.simple import SingleChannelSimpleMemory
    from gem5.components.processors.cpu_types import CPUTypes
    from gem5.components.processors.simple_switchable_processor import SimpleSwitchableProcessor
    from gem5.isas import ISA
    from gem5.resources.resource import DiskImageResource, KernelResource
    from gem5.simulate.exit_handler import register_exit_handler
    from gem5.simulate.exit_event import ExitEvent
    from gem5.simulate.simulator import Simulator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--suite", choices=tuple(IMAGES), required=True)
    parser.add_argument("--cores", type=int, choices=(1, 4), required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ramulator-python", type=Path)
    parser.add_argument("--memory-config", type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--graph-scale", type=int, default=20)
    parser.add_argument("--input-size", default="simlarge")
    args = parser.parse_args()
    preparing = args.mode == "prepare"
    legacy = args.suite == "parsec"
    out = Path(m5.options.outdir)
    started = time.monotonic()
    phases = {}
    done = False
    script = guest_script(args.suite, args.cores, None if preparing else args.benchmark,
                          args.graph_scale, args.input_size)
    (out / "guest-script.sh").write_text(script)

    class Processor(SimpleSwitchableProcessor):
        def incorporate_processor(self, board):
            super().incorporate_processor(board)
            if args.cores == 1:
                for core in self._all_cores():
                    for obj in core.get_simobject().descendants():
                        obj.eventq_index = 0
                    core.get_simobject().eventq_index = 0

        def _pre_instantiate(self, root):
            super()._pre_instantiate(root)
            root.sim_quantum = 1_000_000_000 if preparing and args.cores == 4 else 0

    def record(event, **fields):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        entry = dict(event=event, tick=int(m5.curTick()), wall_seconds=time.monotonic()-started,
                     cpu_seconds=usage.ru_utime+usage.ru_stime, max_rss_kib=usage.ru_maxrss, **fields)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        save_json(out / "progress.json", entry)
        return entry

    def marker(simulator, payload, *, name):
        nonlocal done
        entry = record(name)
        if preparing:
            if name == "prelaunch":
                checkpoint = out / "checkpoint"
                simulator.save_checkpoint(checkpoint)
                if int(m5.curTick()) < FS_START_TICK:
                    raise RuntimeError("prelaunch checkpoint precedes FS startup threshold")
                save_json(checkpoint / "stage.json", dict(stage="prelaunch", suite=args.suite,
                    cores=args.cores, tick=int(m5.curTick()), marker_tick=entry["tick"],
                    drain_ticks=int(m5.curTick())-entry["tick"], benchmark_supplied=False,
                    disk_overlay="private in-memory CowDiskImage on restoration",
                    preparation_quantum_ticks=1_000_000_000 if args.cores == 4 else 0))
                done = True
            elif name not in ("kernel_booted", "after_boot"):
                raise RuntimeError("benchmark/event reached during preparation: " + name)
        elif name == "roi_begin":
            if "roi_begin" in phases or "roi_end" in phases:
                raise RuntimeError("unexpected repeated ROI begin")
            phases[name] = entry
            m5.stats.reset()  # Statistics only: no cache flush or model reinitialization.
        elif name == "roi_end":
            if "roi_begin" not in phases or "roi_end" in phases:
                raise RuntimeError("ROI end without exactly one begin")
            phases[name] = entry
            m5.stats.dump()
            # This dump is immutable even though verification continues.
            (out / "roi.stats.txt").write_bytes((out / "stats.txt").read_bytes())
            snapshot = out / f"ramulator_stats.{entry['tick']}.yaml"
            if not snapshot.is_file():
                raise RuntimeError("missing ROI-end Ramulator statistics")
            (out / "roi.ramulator.yaml").write_bytes(snapshot.read_bytes())
        elif name == "script_ended":
            done = True
        elif name not in ("kernel_booted", "after_boot"):
            raise RuntimeError("unexpected event during detailed execution: " + name)
        return done

    for number, name in ((1, "kernel_booted"), (2, "after_boot"), (3, "script_ended"),
                         (4, "roi_begin"), (5, "roi_end"), (6, "prelaunch")):
        register_exit_handler(number, lambda sim, payload, name=name: marker(sim, payload, name=name), name)
    if preparing:
        if args.checkpoint or args.benchmark:
            parser.error("preparation never receives a benchmark or an old checkpoint")
        memory = SingleChannelSimpleMemory(latency="50ns", latency_var="0ns", bandwidth="38.4GiB/s", size="3GiB")
    else:
        if not all((args.checkpoint, args.benchmark, args.ramulator_python, args.memory_config)):
            parser.error("run requires checkpoint, benchmark, ramulator-python, and memory-config")
        stage = json.loads((args.checkpoint / "stage.json").read_text())
        if (stage["stage"], stage["suite"], stage["cores"], stage["benchmark_supplied"]) != (
                "prelaunch", args.suite, args.cores, False):
            raise RuntimeError("incompatible pre-launch checkpoint")
        state = configparser.ConfigParser(interpolation=None)
        state.read(args.checkpoint / "m5.cpt")
        if state.getint("root.globals", "curTick") < FS_START_TICK:
            raise RuntimeError("checkpoint precedes FS startup threshold")
        sys.path.insert(0, str(args.ramulator_python))
        import ramulator
        config = json.loads(args.memory_config.read_text())
        config = copy.deepcopy(config["memory_system"])
        if any(c.get("controller_plugins") or c.get("trace_path") for c in config["controllers"]):
            raise RuntimeError("raw request tracing must be disabled for this study")
        memory = ramulator.gem5.Memory(config, size="3GiB")
        save_json(out / "memory_config.json", config)
    processor = Processor(starting_core_type=CPUTypes.KVM if preparing else CPUTypes.O3,
                          switch_core_type=CPUTypes.ATOMIC, isa=ISA.X86, num_cores=args.cores)
    if preparing:
        for core in processor.get_cores():
            core.get_simobject().usePerf = False
    board = X86Board(clk_freq="3.2GHz", processor=processor, memory=memory,
        cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(l1d_size="32KiB", l1i_size="32KiB", l2_size="1MiB"))
    kernel = "x86-linux-kernel-4.19.83-1.0.0" if legacy else "x86-linux-kernel-6.8.0-52-generic-1.0.0"
    root = f"{board.get_disk_device()}1" if legacy else "/dev/sda2"
    board.set_kernel_disk_workload(kernel=KernelResource(local_path=str(args.resources / kernel)),
        disk_image=DiskImageResource(local_path=str(args.resources / IMAGES[args.suite]), root_partition="1" if legacy else "2"),
        kernel_args=["earlyprintk=ttyS0", "console=ttyS0", "lpj=7999923", f"root={root}"],
        readfile_contents=script, checkpoint=args.checkpoint)

    def legacy_event(name):
        while True:
            yield marker(simulator, {}, name=name)

    events = {}
    if legacy:
        events = {ExitEvent.WORKBEGIN: legacy_event("roi_begin"), ExitEvent.WORKEND: legacy_event("roi_end"),
                  ExitEvent.EXIT: legacy_event("prelaunch" if preparing else "script_ended")}
    simulator = Simulator(board=board, max_ticks=10**12 if preparing else 10**9, on_exit_event=events)
    record("start", mode=args.mode, suite=args.suite, cores=args.cores, benchmark=args.benchmark)
    last_report = started
    runtime_verified = False
    while not done:
        simulator.run()  # Bounded stepping for heartbeats, not a stopping limit.
        if not preparing and not runtime_verified:
            save_json(out / "loaded-libraries.json", loaded_libraries(
                config["controllers"][0].get("model_library")))
            runtime_verified = True
        if time.monotonic() - last_report >= 30:
            record("progress", phase="verification" if "roi_end" in phases else "roi" if phases else "setup",
                   instructions=[int(c.get_simobject().getCurrentInstCount(0)) for c in processor.get_cores()])
            last_report = time.monotonic()
    if preparing:
        record("checkpoint_saved")
        return
    if set(phases) != {"roi_begin", "roi_end"}:
        raise RuntimeError("benchmark did not complete exactly one ROI")
    verification = check_workload_output(args.suite, (out / "workload-output.txt").read_text(errors="replace"))
    finished = record("passed", verification=verification)
    begin, end = phases["roi_begin"], phases["roi_end"]
    result = dict(suite=args.suite, benchmark=args.benchmark, cores=args.cores,
                  graph_scale=args.graph_scale if args.suite == "gapbs" else None,
                  input_size=args.input_size if legacy else None,
                  roi_ticks=end["tick"]-begin["tick"], roi_seconds=(end["tick"]-begin["tick"])/10**12,
                  roi_begin_tick=begin["tick"], roi_end_tick=end["tick"], verification=verification,
                  wall_seconds=finished["wall_seconds"], max_rss_kib=finished["max_rss_kib"], phases={})
    if result["roi_ticks"] <= 0:
        raise RuntimeError("non-positive ROI")
    for field in ("wall_seconds", "cpu_seconds"):
        result["phases"][field] = dict(setup=begin[field], roi=end[field]-begin[field], verification=finished[field]-end[field])
    save_json(out / "result.json", result)
    m5.stats.dump()


if __name__ == "__m5_main__" or __name__ == "__main__":
    main()
