"""Source-bound SimpleO3 measurements on CHIA's ordinary task primitives.

This is the DRAM task, not a campaign runner. It accepts explicit, trusted input
identities and one immutable model build. It does not discover workloads, call a
provider, select a parent, retry an operation or inspect historical campaigns.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from chia.base.ChiaFunction import ChiaFunction

from ramulator_chia.eval import archive_results, simpleo3
from ramulator_chia.eval import config as C

from . import measurement_reports
from .archive import Payload, describe_payload, safe_name, stage_payload
from .build import file_sha256, run_build_command
from .candidate import runtime_inputs, verify_build
from .identity import canonical_json, digest_json
from .measurement_reports import verify_measurement as verify_measurement
from .measurement_reports import verify_native_measurement as verify_native_measurement
from .snapshots import ModelFiles, publish_bytes


class NativeCase(Protocol):
    """Task-specific inputs/checks around one shared native execution path."""

    observation_names: tuple[str, ...]

    def identity(self) -> dict: ...
    def input_files(self) -> dict[str, Payload]: ...
    def configuration(self, model, inputs, observations, parameters, curve) -> dict: ...
    def check_statistics(self, stats, observations, *, candidate: bool): ...
    def result_fields(self, stats) -> dict: ...


@dataclass(frozen=True)
class Invocation:
    """A trusted host's command and additional read-only runtime dependencies."""

    argv: tuple[str, ...]
    read_paths: tuple[Path, ...] = ()
    environment: dict[str, str] | None = None


class NativeHost(Protocol):
    """Frontend-specific launch/statistics, not another execution or retry loop."""

    evidence_files: tuple[str, ...]

    def identity(self) -> dict: ...
    def prepare(self, runtime, case, configuration, plugin, output) -> tuple[dict, Invocation]: ...
    def statistics(self, output) -> dict: ...


@dataclass(frozen=True)
class IsolatedRamulator:
    evidence_files = ("observations/stats.yaml",)

    def identity(self):
        return {"kind": "isolated_ramulator"}

    def prepare(self, runtime, case, configuration, plugin, output):
        return configuration, Invocation(
            (
                str(runtime / "runtime/isolated_sim"),
                str(output / "config.json"),
                str(plugin) if plugin else "-",
                str(output / "observations"),
                str(output / "observations/stats.yaml"),
            )
        )

    def statistics(self, output):
        import yaml

        return yaml.safe_load((output / "observations/stats.yaml").read_text())


@dataclass(frozen=True)
class SimpleO3Case:
    workload: str
    stage: str
    instructions_per_core: int
    traces: tuple[Payload, ...]
    observation_names = measurement_reports.SIMPLEO3_OBSERVATIONS

    def __post_init__(self):
        if not isinstance(self.workload, str) or not simpleo3.SAFE_LABEL.fullmatch(self.workload):
            raise ValueError("workload must be a catalog label, not a path")
        if self.stage not in {"training", "test", "qualification"}:
            raise ValueError("unknown SimpleO3 measurement stage")
        if type(self.instructions_per_core) is not int or self.instructions_per_core <= 0:
            raise ValueError("the instruction window must be a positive integer")
        if self.stage != "qualification" and self.instructions_per_core < 20_000_000:
            raise ValueError("accuracy measurements require at least 20M instructions per core")
        if (
            type(self.traces) is not tuple
            or not self.traces
            or any(not isinstance(t, Payload) for t in self.traces)
        ):
            raise ValueError("a case needs an ordered tuple of catalogued core traces")
        if any(t.member.executable for t in self.traces):
            raise ValueError("a trace cannot be an executable input")

    def identity(self) -> dict:
        # Compression is a storage choice, not a different experiment.
        return {
            "frontend": "SimpleO3",
            "std": "DDR5",
            "workload": self.workload,
            "stage": self.stage,
            "insts_per_core": self.instructions_per_core,
            "fixed_issue_roi": True,
            "allow_trace_wrap": False,
            "trace_inputs": [
                {"logical_sha256": t.member.logical_sha256, "logical_bytes": t.member.logical_bytes}
                for t in self.traces
            ],
        }

    def input_files(self) -> dict[str, Payload]:
        return {f"core-{i}.trace": trace for i, trace in enumerate(self.traces)}

    def configuration(self, model, inputs, observations, parameters, curve):
        return simpleo3.build_config(
            model,
            standard="DDR5",
            traces=list(inputs.values()),
            instructions_per_core=self.instructions_per_core,
            request_trace=observations / "logical.csv.ch0",
            controller_trace=observations / "controller.csv",
            candidate_overrides=parameters,
            mess_curve=curve,
        )

    def check_statistics(self, stats, observations, *, candidate):
        _check_stats(
            stats,
            self,
            observations / "logical.csv.ch0",
            observations / "controller.csv.ch0",
            candidate=candidate,
        )

    def result_fields(self, stats):
        return {
            "per_core_cycles": [
                stats["frontend"][f"cycles_recorded_core_{i}"] for i in range(len(self.traces))
            ]
        }


@dataclass(frozen=True)
class SimulationLimits:
    timeout_seconds: int | None
    memory_bytes: int
    file_bytes: int
    source_bytes: int
    gzip_level: int

    def __post_init__(self):
        values = asdict(self)
        level = values.pop("gzip_level")
        timeout = values.pop("timeout_seconds")
        if timeout is not None and (type(timeout) is not int or timeout <= 0):
            raise ValueError("simulation timeout must be positive or None")
        if any(type(v) is not int or v <= 0 for v in values.values()):
            raise ValueError("simulation guards must be explicit positive integers")
        if type(level) is not int or not 0 <= level <= 9:
            raise ValueError("gzip level must be an integer in [0, 9]")


@dataclass(frozen=True)
class CandidateBuild:
    store: Path
    candidate_id: str
    contract: ModelFiles
    directory: Path
    receipt_sha256: str

    def verify(self, runtime: Path, maximum_source_bytes: int) -> tuple[dict, dict]:
        return verify_build(
            runtime,
            self.store,
            self.candidate_id,
            self.contract,
            self.directory,
            expected_receipt_sha256=self.receipt_sha256,
            maximum_source_bytes=maximum_source_bytes,
        )


class MeasurementFailed(RuntimeError):
    def __init__(self, receipt: dict):
        self.receipt = receipt
        super().__init__(receipt["error"])

    def __reduce__(self):
        # Ray uses Python's exception serialization. RuntimeError's default
        # string arguments do not match our receipt-based constructor.
        return type(self), (self.receipt,)


def _check_stats(
    stats: dict, case: SimpleO3Case, logical_trace: Path, controller_trace: Path, *, candidate: bool
):
    frontend = stats["frontend"]
    rows = simpleo3._validate_logical_request_trace(logical_trace)
    simpleo3._validate_fixed_roi_frontend_stats(
        frontend,
        core_count=len(case.traces),
        insts_per_core=case.instructions_per_core,
        logical_rows=rows,
    )
    # Checking the emitted setting also rejects older binaries that silently
    # ignore this new configuration key. C++ stream serialization emits 0/1.
    if type(frontend.get("allow_trace_wrap")) is not int or frontend["allow_trace_wrap"] != 0:
        raise RuntimeError("runtime did not establish a no-wrap SimpleO3 window")
    for core in range(len(case.traces)):
        available = frontend.get(f"trace_instructions_core_{core}")
        if type(available) is not int or available < case.instructions_per_core:
            raise RuntimeError("input trace does not cover the fixed instruction window")
    if candidate:
        check_candidate_callbacks(stats, controller_trace)


def check_candidate_callbacks(stats, controller_trace):
    controller = stats["memory_system"]["controller"]
    for kind in ("read", "write"):
        if controller[f"num_{kind}_reqs"] != controller[f"num_{kind}_reqs_served"]:
            raise RuntimeError("candidate callbacks were not fully drained")
        if controller[f"peak_inflight_{kind}s"] > C.CANDIDATE_RESOURCES[f"{kind}_buffer_size"]:
            raise RuntimeError("candidate exceeded the configured admission capacity")
    simpleo3.audit_candidate_trace(controller_trace, controller)


def _compress_traces(
    output: Path, level: int, names=SimpleO3Case.observation_names
) -> dict[str, dict]:
    """Reuse the verified gzip workflow; leave failed partials explicitly labeled."""
    traces = [path for name in names if (path := output / "observations" / name).is_file()]
    if not traces:
        return {}
    archive_results.compress_files(traces, output / "archive_manifest.json", level, keep_raw=True)
    result = {}
    for trace in traces:
        compressed = trace.with_name(trace.name + ".gz")
        result[trace.name] = asdict(
            describe_payload(
                compressed, compressed.relative_to(output).as_posix(), codec="gzip"
            ).member
        )
    # This durable inventory is published before removing any redundant raw
    # copy. It supplements, rather than changes, the historical archive format.
    publish_bytes(output / "trace_inventory.json", canonical_json(result).encode())
    for trace in traces:
        trace.unlink()
    return result


@ChiaFunction(num_cpus=1, max_retries=0)
def measure_simpleo3(
    runtime: Path,
    case: SimpleO3Case,
    model: str,
    output: Path,
    *,
    limits: SimulationLimits,
    candidate: CandidateBuild | None = None,
    mess_curve: Payload | None = None,
    lease_fd: int | None = None,
) -> dict:
    return measure_native(
        runtime,
        case,
        model,
        output,
        limits=limits,
        candidate=candidate,
        mess_curve=mess_curve,
        lease_fd=lease_fd,
    )


def measure_native(
    runtime: Path,
    case: NativeCase,
    model: str,
    output: Path,
    *,
    limits: SimulationLimits,
    candidate: CandidateBuild | None = None,
    mess_curve: Payload | None = None,
    lease_fd: int | None = None,
    host: NativeHost = IsolatedRamulator(),
) -> dict:
    """Measure one exact case with one CPU, including staging and compression.

    Paths are trusted task inputs, never arguments exposed directly to an agent.
    The caller reserves resources through CHIA and supplies a new attempt path;
    recovery/reuse requires a previously verified receipt, not this function's
    automatic replay. No inference happens here.
    """
    if model not in simpleo3.ALL_MODELS:
        raise ValueError("unknown immediate-response comparison model")
    if (model == "candidate") != (candidate is not None):
        raise ValueError("only candidate measurements must supply an immutable candidate build")
    if (model == "mess") != (mess_curve is not None):
        raise ValueError("MESS alone requires an explicitly identified calibration curve")
    runtime, output = runtime.absolute(), output.absolute()
    manifest, _, _ = runtime_inputs(runtime)
    build, snapshot = candidate.verify(runtime, limits.source_bytes) if candidate else (None, None)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    inputs, observations = output / "inputs", output / "observations"
    inputs.mkdir(mode=0o700)
    observations.mkdir(mode=0o700)
    receipt = {
        "schema_version": 1,
        "complete": False,
        "execution": "native",
        "case": case.identity(),
        "case_sha256": digest_json(case.identity()),
        "model": model,
        "runtime_sha256": file_sha256(runtime / "runtime_manifest.json"),
        "host": host.identity(),
        "limits": asdict(limits),
        "optimization": "-O3",
        "trace_inputs": [asdict(t.member) for t in case.input_files().values()],
        "calibration": asdict(mess_curve.member) if mess_curve else None,
        "candidate": snapshot,
        "build_receipt_sha256": candidate.receipt_sha256 if candidate else None,
        "binary_sha256": build["binary_sha256"] if build else manifest["library_sha256"],
    }
    publish_bytes(output / "intent.json", canonical_json(receipt).encode())
    started = time.monotonic()
    simulation_started = None
    diagnostic = case.identity()["frontend"] in {"ControllerReplay", "SyntheticPattern"}
    try:
        staged = {}
        for name, trace in case.input_files().items():
            safe_name(name)
            path = inputs / name
            stage_payload(trace, path)
            staged[name] = path
        curve_path = inputs / "mess.txt" if mess_curve else None
        if mess_curve:
            stage_payload(mess_curve, curve_path)
        parameters = (
            {
                "model_parameters": [
                    f"{name}={value}" for name, value in sorted(snapshot["parameters"].items())
                ]
            }
            if snapshot
            else {}
        )
        configuration = case.configuration(model, staged, observations, parameters, curve_path)
        plugin = candidate.directory.absolute() / "candidate.so" if candidate else None
        configuration, invocation = host.prepare(runtime, case, configuration, plugin, output)
        publish_bytes(output / "config.json", canonical_json(configuration).encode())
        binary_root = runtime / "runtime"
        sandbox = Path(__file__).resolve().parents[1] / "sandbox.py"
        policy = {
            "read": [
                "/dev/null",
                str(binary_root),
                str(inputs),
                str(output / "config.json"),
            ]
            + ["/usr", "/lib", "/lib64", "/bin"]
            + ([str(plugin)] if plugin else [])
            + [str(path) for path in invocation.read_paths],
            "write": [str(observations)],
            "cwd": str(observations),
            # Diagnostics have a parent-enforced wall deadline. Avoid racing
            # it with RLIMIT_CPU's unclassified SIGKILL at the same duration.
            "cpu_seconds": None if diagnostic else limits.timeout_seconds,
            "memory_bytes": limits.memory_bytes,
            "file_bytes": limits.file_bytes,
        }
        if invocation.environment is not None:
            policy["environment"] = invocation.environment
        publish_bytes(output / "simulation.policy.json", canonical_json(policy).encode())
        receipt["launcher_sha256"] = file_sha256(sandbox)
        simulation_started = time.monotonic()
        receipt["process"] = run_build_command(
            [
                "/usr/bin/python3",
                str(sandbox),
                *(["--lease-fd", str(lease_fd)] if lease_fd is not None else []),
                str(output / "simulation.policy.json"),
                "--",
                *invocation.argv,
            ],
            output / "simulation.log",
            cwd=output,
            cpus=1,
            timeout_seconds=limits.timeout_seconds,
            lease_fd=lease_fd,
        )
        stats = host.statistics(output)
        case.check_statistics(stats, observations, candidate=candidate is not None)
        if candidate:
            candidate.verify(runtime, limits.source_bytes)
        else:
            runtime_inputs(runtime)
        if receipt["runtime_sha256"] != file_sha256(runtime / "runtime_manifest.json"):
            raise RuntimeError("runtime manifest changed during measurement")
        if receipt["host"] != host.identity():
            raise RuntimeError("frontend runtime identity changed during measurement")
        receipt.update(
            complete=True,
            **case.result_fields(stats),
            frontend_stats=stats["frontend"],
            controller_stats=stats["memory_system"]["controller"],
            simulation_wall_s=stats.get("simulation_wall_s"),
            construction_and_simulation_wall_s=stats.get("construction_and_simulation_wall_s"),
        )
    except Exception as exc:
        receipt.update(complete=False, error=f"{type(exc).__name__}: {exc}")
        if diagnostic and simulation_started is not None and isinstance(exc, subprocess.TimeoutExpired):
            receipt["diagnostic_failure"] = {
                "reason": "runtime_limit",
                "message": "diagnostic exceeded its runtime limit",
                "diagnostic": "open_loop" if receipt["case"]["frontend"] == "ControllerReplay"
                              else "synthetic",
                "training_case": receipt["case"].get("workload"),
                "case_sha256": receipt["case_sha256"],
                "candidate_id": snapshot["candidate_id"] if snapshot else None,
                "model": model,
                "elapsed_seconds": time.monotonic() - simulation_started,
                "deadline_seconds": limits.timeout_seconds,
            }
            receipt["error"] = canonical_json(receipt["diagnostic_failure"])
    try:
        receipt["traces"] = _compress_traces(output, limits.gzip_level, case.observation_names)
    except Exception as exc:
        receipt.update(complete=False, archive_error=f"{type(exc).__name__}: {exc}")
        receipt.setdefault("error", receipt["archive_error"])
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["files"] = {
        name: file_sha256(output / name)
        for name in (
            "config.json",
            "simulation.policy.json",
            "simulation.log",
            "trace_inventory.json",
            *host.evidence_files,
        )
        if (output / name).is_file()
    }
    publish_bytes(output / "measurement.json", canonical_json(receipt).encode())
    # Storage cleanup cannot invalidate a completed observation. If the original
    # archive changed, keep our verified copy and record the closure problem for
    # the artifact assembler instead of silently deleting the only good input.
    cleanup = {"removed": [], "retained": []}
    for payload, path in [(t, inputs / name) for name, t in case.input_files().items()] + (
        [(mess_curve, inputs / "mess.txt")] if mess_curve else []
    ):
        if path.is_file():
            try:
                current = describe_payload(
                    payload.source,
                    payload.member.name,
                    codec=payload.member.codec,
                    executable=payload.member.executable,
                )
                if current.member != payload.member:
                    raise RuntimeError("original input identity changed")
                path.unlink()
                cleanup["removed"].append(path.relative_to(output).as_posix())
            except Exception as exc:
                cleanup["retained"].append(
                    {
                        "path": path.relative_to(output).as_posix(),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    publish_bytes(output / "staging_cleanup.json", canonical_json(cleanup).encode())
    if not receipt["complete"]:
        raise MeasurementFailed(receipt)
    return receipt


@ChiaFunction(num_cpus=1, max_retries=0)
def compare_simpleo3(
    oracle: Path,
    model: Path,
    *,
    oracle_receipt_sha256: str,
    model_receipt_sha256: str,
    minimum_oracle_owner_reads: int,
) -> dict:
    """CHIA resource admission around the shared read-only comparison."""
    return measurement_reports.compare_simpleo3(
        oracle,
        model,
        oracle_receipt_sha256=oracle_receipt_sha256,
        model_receipt_sha256=model_receipt_sha256,
        minimum_oracle_owner_reads=minimum_oracle_owner_reads,
    )
