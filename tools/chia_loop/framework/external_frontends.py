"""DDR5 transfer cases for the shared native evaluator.

This module describes installed frontend inputs, commands and observations. CHIA
still owns dispatch; evaluation.measure_native owns execution, compression and
receipts. It neither selects a model nor authorizes access to held-out workloads.
The caller must bind these inputs before training and require a selection freeze.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from resources.champsim_bridge import cs_gentest as champsim
from tools.eval import config as C
from tools.eval import matchlib, simpleo3

from .archive import Member, Payload, describe_payload, safe_name, stage_payload
from .build import file_sha256, run_build_command
from .candidate import runtime_inputs
from .evaluation import Invocation
from .identity import canonical_json
from .measurement_reports import CONTROLLER_OBSERVATIONS
from .measurement_reports import compare_transfer as compare_transfer
from .snapshots import publish_bytes, read_file, regular_path

OBSERVATIONS = CONTROLLER_OBSERVATIONS
REPO = Path(__file__).resolve().parents[3]


def prepare_host(destination: Path, frontend: str, binary: Path, source: Path) -> "ExternalHost":
    """Freeze installed optimized builds for local evaluation, without inference.

    This is an installed-build attestation, not a claim to have rebuilt gem5 or
    ChampSim. The captured inputs are content-bound copies; installation paths
    remain provenance only. These compiled files are local build products, not
    final artifact payloads. No frontend checkout or result is a model grant.
    """
    if frontend not in {"champsim", "gem5"}:
        raise ValueError("unknown external frontend")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    binary, source = binary.resolve(strict=True), source.resolve(strict=True)
    files = {"binary": C.file_provenance(binary)}
    details = {}
    if frontend == "champsim":
        bridge = champsim.bridge_source_provenance(source)
        files.update(
            {name: C.file_provenance(row["installed_path"]) for name, row in bridge.items()}
        )
        for name in (
            "global.options",
            "champsim_config.json",
            ".csconfig/core_inst.cc.inc",
            "inc/trace_instruction.h",
            "inc/repeatable.h",
        ):
            files[name] = C.file_provenance(source / name)
        if 'fmt::print("*** Reached end of trace:' not in (source / "inc/repeatable.h").read_text():
            raise ValueError("the installed frontend lacks the checked trace-wrap event")
        flags = re.findall(r"-O(?:[0-3sgz]|fast)\b", (source / "global.options").read_text())
        if not flags or flags[-1] != "-O3":
            raise ValueError("ChampSim build recipe must use -O3")
        settings = json.loads((source / "champsim_config.json").read_text())
        clock = re.search(
            r"DRAM\{\s*champsim::chrono::picoseconds\{\d+\},\s*champsim::chrono::picoseconds\{(\d+)\}",
            (source / ".csconfig/core_inst.cc.inc").read_text(),
        )
        if (
            settings.get("num_cores") != 1
            or settings.get("block_size") != 64
            or clock is None
            or int(clock[1]) != 625
        ):
            raise ValueError("DDR5 ChampSim needs one core, 64B lines and a 625ps memory tick")
        probe_source = REPO / "tools/chia_loop/champsim_trace_format.cpp"
        publish_bytes(destination / "trace_format.cpp", probe_source.read_bytes())
        process = run_build_command(
            [
                "/usr/bin/g++",
                "-O3",
                "-DNDEBUG",
                "-std=c++17",
                "-I" + str(source / "inc"),
                str(destination / "trace_format.cpp"),
                "-o",
                str(destination / "trace_format"),
            ],
            destination / "trace-format-build.log",
            cwd=destination,
            cpus=1,
            timeout_seconds=60,
        )
        record_bytes = int(subprocess.check_output([str(destination / "trace_format")], text=True))
        if record_bytes <= 0:
            raise ValueError("invalid native instruction record size")
        details = {
            "instruction_record_bytes": record_bytes,
            "format_probe": process,
            "clocking": {"memory_period_ps": 625, "ramulator_ticks_per_8": 12},
        }
    else:
        from tools.eval.gem5.run_gem5 import _gem5_source_inputs

        if binary.name not in {"gem5.opt", "gem5.fast"}:
            raise ValueError("gem5 transfer requires an optimized binary")
        files.update(_gem5_source_inputs(binary))
        for name, record in files.items():
            if not name.startswith("gem5_source:src/mem/ramulator2/"):
                continue
            canonical = REPO / "resources/gem5_wrappers" / Path(record["path"]).name
            if canonical.name != "SConscript" and file_sha256(canonical) != record["sha256"]:
                raise ValueError("installed gem5 bridge differs from the declared source")
        recipe = source / "src/SConscript"
        if "['-O3']" not in recipe.read_text() and "['-O3'," not in recipe.read_text():
            raise ValueError("cannot verify the gem5 optimized build recipe")
        files["build_recipe"] = C.file_provenance(recipe)
        publish_bytes(
            destination / "board.py", (REPO / "tools/chia_loop/transfer_gem5_board.py").read_bytes()
        )
        shutil.copytree(
            REPO / "python/ramulator",
            destination / "python/ramulator",
            ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__"),
        )
        details = {"cpu": "O3", "cpu_clock": "3.2GHz", "cores": 1, "run_to_exit": True}
    # These files are protected runtime input, not generated model output.
    assets = {
        p.relative_to(destination).as_posix(): asdict(
            describe_payload(
                p, p.relative_to(destination).as_posix(), executable=bool(p.stat().st_mode & 0o111)
            ).member
        )
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    }
    record = {
        "schema_version": 2,
        "frontend": frontend,
        "files": capture_host_inputs(destination, files),
        "assets": assets,
        "settings": details,
        "optimization": "-O3",
        "driver_sha256": file_sha256(Path(__file__)),
        "build_attestation": (
            "existing executable hashes and optimized build recipe; not a fresh frontend rebuild"
        ),
    }
    publish_bytes(destination / "host.json", canonical_json(record).encode())
    return ExternalHost(destination, file_sha256(destination / "host.json"))


def capture_host_inputs(destination: Path, origins: dict) -> dict:
    """Copy only the attested files through the existing checked staging helper.

    The complete Member is retained alongside original provenance. Timestamp
    and installation path describe the source; neither is a replay location or
    a condition for later inspecting the captured bytes.
    """
    captured = {}
    for index, (name, origin) in enumerate(sorted(origins.items())):
        source = Path(origin["path"])
        relative = safe_name(f"inputs/{index:03d}/{source.name}")
        payload = describe_payload(source, relative, executable=bool(source.stat().st_mode & 0o111))
        if (payload.member.stored_sha256, payload.member.stored_bytes) != (
            origin["sha256"],
            origin["size"],
        ):
            raise ValueError("frontend source changed before capture: " + name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        stage_payload(payload, target)
        target.chmod(0o500 if payload.member.executable else 0o400)
        captured[name] = {"origin": origin, "member": asdict(payload.member)}
    return captured


def describe_champsim_trace(path: Path, host: "ExternalHost") -> tuple[Payload, dict]:
    """CRC-check the entire xz input and bind decoded content without expanding it.

    ChampSim reads xz directly. The shared staging layer copies those compressed
    bytes; it does not keep a many-gigabyte expansion per simulation. The case's
    scientific identity uses decoded content, independently of xz encoding.
    """
    record = host.record()
    if record["frontend"] != "champsim":
        raise ValueError("the trace format must belong to ChampSim")
    payload = describe_payload(path, "inputs/trace.champsimtrace.xz")
    digest, count = hashlib.sha256(), 0
    with lzma.open(path, "rb") as stream:
        while data := stream.read(1024**2):
            count += len(data)
            digest.update(data)
    if describe_payload(path, payload.member.name).member != payload.member:
        raise ValueError("input changed while validating its xz stream")
    size = record["settings"]["instruction_record_bytes"]
    if not count or count % size:
        raise ValueError("input is not an integral nonempty instruction trace")
    return payload, {
        "stored_sha256": payload.member.stored_sha256,
        "decoded_sha256": digest.hexdigest(),
        "decoded_bytes": count,
        "record_bytes": size,
        "instructions": count // size,
        "host_sha256": host.receipt_sha256,
        "complete_xz_integrity": True,
    }


@dataclass(frozen=True)
class TransferCase:
    frontend: str
    workload: str
    payload: Payload
    arguments: tuple[str, ...] = ()
    warmup_instructions: int = 2_000_000
    roi_instructions: int = 20_000_000
    instruction_inventory: dict | None = None
    observation_names = OBSERVATIONS
    stage = "transfer"

    def __post_init__(self):
        if self.frontend not in {"champsim", "gem5"} or not simpleo3.SAFE_LABEL.fullmatch(
            self.workload
        ):
            raise ValueError("unknown frontend or unsafe workload label")
        if type(self.arguments) is not tuple or any(
            not isinstance(a, str) or "\0" in a for a in self.arguments
        ):
            raise ValueError("guest arguments must be an explicit string tuple")
        if self.frontend == "champsim":
            inv = self.instruction_inventory
            if (
                self.payload.member.executable
                or self.payload.member.codec != "none"
                or self.arguments
            ):
                raise ValueError("ChampSim requires a compressed trace, not a guest executable")
            if (
                any(type(v) is not int for v in (self.warmup_instructions, self.roi_instructions))
                or self.warmup_instructions < 2_000_000
                or self.roi_instructions < 20_000_000
            ):
                raise ValueError(
                    "transfer requires at least 2M warmup and 20M measured instructions"
                )
            if (
                not inv
                or inv.get("complete_xz_integrity") is not True
                or inv["stored_sha256"] != self.payload.member.stored_sha256
            ):
                raise ValueError("trace lacks its complete source-bound xz validation")
            if (
                inv["record_bytes"] <= 0
                or inv["decoded_bytes"] != inv["instructions"] * inv["record_bytes"]
            ):
                raise ValueError("invalid instruction population")
            if inv["instructions"] < self.warmup_instructions + self.roi_instructions:
                raise ValueError("trace cannot cover the full window without wrapping")
        elif not self.payload.member.executable or self.instruction_inventory is not None:
            raise ValueError("gem5 needs an explicit whole-program executable")

    def identity(self):
        common = {
            "frontend": self.frontend,
            "std": "DDR5",
            "workload": self.workload,
            "stage": self.stage,
            "arguments": list(self.arguments),
        }
        if self.frontend == "champsim":
            common.update(
                warmup_instructions=self.warmup_instructions,
                roi_instructions=self.roi_instructions,
                input_sha256=self.instruction_inventory["decoded_sha256"],
                input_bytes=self.instruction_inventory["decoded_bytes"],
                instruction_record_bytes=self.instruction_inventory["record_bytes"],
                allow_trace_wrap=False,
            )
        else:
            common.update(
                input_sha256=self.payload.member.logical_sha256,
                input_bytes=self.payload.member.logical_bytes,
                run_to_exit=True,
            )
        return common

    def input_files(self):
        return {
            "trace.champsimtrace.xz" if self.frontend == "champsim" else "benchmark": self.payload
        }

    def configuration(self, model, inputs, observations, parameters, curve):
        import ramulator

        dram = ramulator.dram.DDR5(
            org_preset=C.STD["DDR5"]["org"], timing_preset=C.STD["DDR5"]["timing"]
        )
        controller = simpleo3._build_controller(
            ramulator,
            model,
            dram,
            "DDR5",
            observations / "controller.csv",
            parameters,
            mess_curve=curve,
        )
        memory = ramulator.memory_system.GenericDRAM(
            clock_ratio=1 if self.frontend == "champsim" else 3,
            controllers=[controller],
            channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
        )
        return {
            "frontend": {"impl": "External", "clock_ratio": 1},
            "memory_system": memory.to_config(),
        }

    def check_statistics(self, stats, observations, *, candidate):
        if any(
            type(stats["frontend"].get(key)) is not int or stats["frontend"][key] <= 0
            for key in ("cycles_or_ticks", "instructions")
        ):
            raise ValueError("empty or incomplete frontend execution")
        if (
            self.frontend == "champsim"
            and stats["frontend"]["instructions"] < self.roi_instructions
        ):
            raise ValueError("ChampSim did not complete the full measured window")
        # Reuse the common raw parser; do not infer full callback drain from a
        # valid file. External frontends terminate at different lifecycle points.
        matchlib._load_frame(observations / OBSERVATIONS[0])

    def result_fields(self, stats):
        return {
            "per_core_cycles": [stats["frontend"]["cycles_or_ticks"]],
            "core_metric": "ROI core cycles"
            if self.frontend == "champsim"
            else "whole-program simTicks at 3.2GHz",
            "request_population_verified": False,
            "request_population_note": (
                "controller trace only; complete admission/drain population not established"
            ),
            "input_access_after_candidate_load": (
                "specific frontend inputs remain readable; no C++ address-space isolation"
            ),
        }


@dataclass(frozen=True)
class ExternalHost:
    root: Path
    receipt_sha256: str

    def __post_init__(self):
        if not self.root.is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", self.receipt_sha256):
            raise ValueError("frontend host needs an absolute root and SHA-256 identity")

    @property
    def evidence_files(self):
        return (
            "observations/stats.txt",
            "observations/exit.json",
            "observations/ramulator_stats.yaml",
            "observations/config.ini",
            "observations/config.json",
        )

    def record(self):
        raw = read_file(self.root, "host.json", maximum_bytes=64 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != self.receipt_sha256:
            raise ValueError("frontend host receipt changed")
        record = json.loads(raw)
        if record.get("schema_version") != 2:
            raise ValueError("frontend host must be prepared as a captured-input snapshot")
        if (
            record.get("frontend") not in {"champsim", "gem5"}
            or record.get("optimization") != "-O3"
        ):
            raise ValueError("frontend host requires a declared optimized frontend")
        if record["driver_sha256"] != file_sha256(Path(__file__)):
            raise ValueError("frontend driver changed after preparation")
        for payload in self._payloads(record):
            member = payload.member
            actual = describe_payload(
                payload.source,
                member.name,
                codec=member.codec,
                executable=bool(payload.source.stat().st_mode & 0o111),
            )
            if actual.member != member:
                raise ValueError("frontend runtime input changed: " + member.name)
        for value in record["files"].values():
            member, origin = Member(**value["member"]), value["origin"]
            if (member.stored_sha256, member.stored_bytes) != (origin["sha256"], origin["size"]):
                raise ValueError("captured frontend input differs from its attested origin")
        if not record["files"]["binary"]["member"]["executable"]:
            raise ValueError("captured frontend binary is not executable")
        return record

    def _payloads(self, record):
        members = [Member(**value["member"]) for value in record["files"].values()]
        for name, value in record["assets"].items():
            member = Member(**value)
            if member.name != name:
                raise ValueError("frontend asset name differs from its inventory")
            members.append(member)
        if len({member.name for member in members}) != len(members):
            raise ValueError("frontend input and asset names overlap")
        return tuple(Payload(regular_path(self.root, member.name), member) for member in members)

    def identity(self):
        return {"kind": self.record()["frontend"], "receipt_sha256": self.receipt_sha256}

    def prepare(self, runtime, case, configuration, plugin, output):
        record = self.record()
        if not isinstance(case, TransferCase) or case.frontend != record["frontend"]:
            raise ValueError("frontend and transfer case differ")
        if (
            case.instruction_inventory
            and case.instruction_inventory["host_sha256"] != self.receipt_sha256
        ):
            raise ValueError("instruction format was probed against another frontend")
        manifest, _, _ = runtime_inputs(runtime)
        if manifest.get("external_model_loading") != "controller_model_library_v1":
            raise ValueError("runtime lacks the explicit external model loader")
        if plugin:
            configuration["memory_system"]["controllers"][0]["model_library"] = str(plugin)
        binary = str(self.root / record["files"]["binary"]["member"]["name"])
        # Reuse the established loader-resolution check, before any candidate is
        # loaded. The launched environment names this exact shared library.
        champsim.verify_library_resolution(
            Path(binary), runtime / "runtime", output / "config.json", 12
        )
        observations = output / "observations"
        env = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "TMPDIR": str(observations),
            "LD_LIBRARY_PATH": str(runtime / "runtime"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "MPLCONFIGDIR": str(observations),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        read = [Path(binary), self.root]
        if case.frontend == "champsim":
            command = (
                binary,
                "-w",
                str(case.warmup_instructions),
                "-i",
                str(case.roi_instructions),
                str(output / "inputs/trace.champsimtrace.xz"),
            )
            env.update(RAMULATOR_CONFIG=str(output / "config.json"), RAMULATOR_TICKS_PER_8="12")
        else:
            command = (binary, "--outdir=" + str(observations), str(self.root / "board.py"))
            env.update(
                CHIA_MEMORY_CONFIG=str(output / "config.json"),
                CHIA_BENCHMARK=str(output / "inputs/benchmark"),
                CHIA_BENCHMARK_ARGS=json.dumps(case.arguments),
                CHIA_PYTHON=str(self.root / "python"),
                CHIA_EXIT_RECEIPT=str(observations / "exit.json"),
            )
            read += [
                Path(p)
                for p in ("/proc/self/maps", "/proc/meminfo", "/proc/cpuinfo", "/dev/urandom")
                if Path(p).exists()
            ]
        return configuration, Invocation(command, tuple(read), env)

    def statistics(self, output):
        frontend = self.record()["frontend"]
        observations = output / "observations"
        if frontend == "champsim":
            log = (output / "simulation.log").read_text()
            if "*** Reached end of trace:" in log:
                raise ValueError("ChampSim wrapped the input, including speculative read-ahead")
            matches = re.findall(
                r"Simulation finished CPU 0 instructions: (\d+) cycles: (\d+)", log
            )
            if len(matches) != 1:
                raise ValueError("ChampSim did not complete one measured ROI")
            instructions, cycles = map(int, matches[0])
        else:
            exit_info = json.loads((observations / "exit.json").read_text())
            if exit_info != {"cause": "exiting with last active thread context", "code": 0}:
                raise ValueError("gem5 workload did not exit normally")
            stats = (observations / "stats.txt").read_text()

            def value(name):
                matches = re.findall(r"^" + re.escape(name) + r"\s+(\d+)", stats, re.M)
                if len(matches) != 1 or int(matches[0]) <= 0:
                    raise ValueError("missing or ambiguous gem5 statistic: " + name)
                return int(matches[0])

            instructions, cycles = value("simInsts"), value("simTicks")
        reads, _ = matchlib._load_frame(observations / OBSERVATIONS[0])
        return {
            "frontend": {"instructions": instructions, "cycles_or_ticks": cycles},
            "memory_system": {"controller": {"observed_read_records": len(reads)}},
        }
