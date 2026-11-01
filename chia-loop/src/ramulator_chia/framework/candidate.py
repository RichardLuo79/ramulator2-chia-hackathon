"""CMake builds of immutable models through the restricted value API.

The compiler sees system build tools, one copied API header and the submitted
files. It cannot read the checkout, observations or any other campaign. This is
a native compilation boundary, not a C++ semantic proof. CHIA owns placement,
resource admission and remote cancellation; callers reserve the stated CPUs.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .build import (
    MODEL_API_HEADER,
    MODEL_INTERFACE,
    file_sha256,
    run_build_command,
    verify_optimized_commands,
)
from .identity import canonical_json
from .snapshots import (
    InvalidSnapshot,
    ModelFiles,
    materialize,
    publish_bytes,
    read_file,
    verify_snapshot,
)

RECIPE = "tools/chia_loop/model/CMakeLists.txt"


@dataclass(frozen=True)
class BuildLimits:
    """Explicit recorded limits; their classification belongs to launch config."""

    cpus: int
    timeout_seconds: int
    source_bytes: int
    memory_bytes: int
    file_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("build limits must be positive integers")
        if self.cpus > 12:
            raise ValueError("the local build profile permits at most 12 reserved CPUs")


class BuildFailed(RuntimeError):
    """A failed build with a settled receipt, not an eligible binary."""

    def __init__(self, receipt: dict):
        self.receipt = receipt
        super().__init__(receipt["error"])

    def __reduce__(self):
        return type(self), (self.receipt,)


def _verified_file(root: Path, relative: str, expected: str) -> bytes:
    # The size comes from trusted storage; read_file still rejects intermediate
    # links and changes during reading. The expected digest binds the payload.
    data = read_file(root, relative, maximum_bytes=(root / relative).stat().st_size)
    import hashlib

    if hashlib.sha256(data).hexdigest() != expected:
        raise InvalidSnapshot(f"runtime input changed: {relative}")
    return data


def runtime_inputs(root: Path) -> tuple[dict, bytes, bytes]:
    """Verify runtime/API bytes without requiring the original build machine.

    Compiler identity remains part of the recorded build certificate. Only a
    new compilation needs to reopen that executable; inspecting or loading a
    previously verified model does not.
    """
    manifest = json.loads((root / "runtime_manifest.json").read_text())
    if (
        manifest.get("schema_version") != 3
        or manifest.get("candidate_interface") != MODEL_INTERFACE
        or manifest.get("optimization") != "-O3"
    ):
        raise InvalidSnapshot(
            "a restricted-model API runtime is required; legacy runtimes are not interchangeable"
        )
    exports = manifest["export_hashes"]
    if exports != {MODEL_API_HEADER: manifest["model_api_sha256"]}:
        raise InvalidSnapshot("the model export must contain exactly its approved API header")
    header = _verified_file(root / "export", MODEL_API_HEADER, manifest["model_api_sha256"])
    recipe = _verified_file(root / "runtime-source", RECIPE, manifest["source_inventory"][RECIPE])
    compiler = manifest["compiler"]
    if (
        set(compiler) != {"path", "sha256"}
        or not Path(compiler["path"]).is_absolute()
        or not isinstance(compiler["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", compiler["sha256"])
    ):
        raise InvalidSnapshot("invalid recorded compiler identity")
    for name, key in (("libramulator.so", "library_sha256"), ("isolated_sim", "executable_sha256")):
        _verified_file(root / "runtime", name, manifest[key])
    return manifest, header, recipe


def _verify_compiler(manifest: dict) -> None:
    compiler = manifest["compiler"]
    if file_sha256(Path(compiler["path"])) != compiler["sha256"]:
        raise InvalidSnapshot("runtime compiler identity changed")


def verify_build(
    runtime: Path,
    store: Path,
    candidate_id: str,
    contract: ModelFiles,
    output: Path,
    *,
    expected_receipt_sha256: str,
    maximum_source_bytes: int,
) -> tuple[dict, dict]:
    """Bind a published module to the expected draft, parameters and runtime.

    Evaluators receive the receipt identity from trusted campaign storage. A
    binary path or a model's claimed source hash is not a build certificate.
    """
    receipt = json.loads(_verified_file(output, "build.json", expected_receipt_sha256))
    snapshot = verify_snapshot(store, candidate_id, contract, maximum_bytes=maximum_source_bytes)
    manifest, _, _ = runtime_inputs(runtime)
    expected = {
        "schema_version": 1,
        "passed": True,
        "optimization": "-O3",
        "candidate_id": candidate_id,
        "source_sha256": snapshot["source_sha256"],
        "configuration_sha256": snapshot["configuration_sha256"],
        "runtime_sha256": file_sha256(runtime / "runtime_manifest.json"),
        "model_api_sha256": manifest["model_api_sha256"],
        "compiler": manifest["compiler"],
        "recipe_sha256": manifest["source_inventory"][RECIPE],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise InvalidSnapshot("build receipt does not identify the requested candidate and runtime")
    _verified_file(output, "candidate.so", receipt["binary_sha256"])
    return receipt, snapshot


def _compile_names(contract: ModelFiles) -> list[str]:
    # These are file names in a trusted CMake list, not agent-written CMake.
    # The grant is configured before the campaign and cannot add command syntax.
    result = []
    for name in contract.sources:
        if not re.fullmatch(r"[A-Za-z0-9_./+-]+", name) or Path(name).suffix not in {
            ".cpp",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
        }:
            raise InvalidSnapshot("model source names must be ordinary C++ source/header paths")
        result.append("model/" + name)
    if not any(Path(name).suffix in {".cpp", ".cc", ".cxx"} for name in contract.sources):
        raise InvalidSnapshot("the model grant needs at least one C++ translation unit")
    return result


def compile_snapshot(
    runtime: Path,
    store: Path,
    candidate_id: str,
    contract: ModelFiles,
    output: Path,
    *,
    limits: BuildLimits,
    lease_fd: int | None = None,
) -> dict:
    """Compile this draft, never an implicit parent. Each attempt uses a new directory.

    A completed receipt is published only after source, compiler and binary checks.
    On failure the source, policy and complete command logs remain with a failure
    receipt. Retry/reuse decisions belong to the durable operation owner.
    """
    runtime, store, output = runtime.absolute(), store.absolute(), output.absolute()
    names = _compile_names(contract)
    snapshot = verify_snapshot(store, candidate_id, contract, maximum_bytes=limits.source_bytes)
    manifest, header, recipe = runtime_inputs(runtime)
    _verify_compiler(manifest)
    output.mkdir(parents=True, exist_ok=False)
    project, build = output / "project", output / "build"
    project.mkdir()
    materialize(store, candidate_id, project / "model", contract, maximum_bytes=limits.source_bytes)
    publish_bytes(project / "CMakeLists.txt", recipe)
    publish_bytes(project / "api" / MODEL_API_HEADER.removeprefix("src/"), header)
    build.mkdir()
    # Reuse the existing Linux launcher. The policy grants neither a whole
    # repository nor the parent's runtime/source tree. Its env is minimal.
    sandbox = Path(__file__).resolve().parents[1] / "sandbox.py"
    policy = {
        "read": ["/usr", "/lib", "/lib64", "/bin", "/dev/null", str(project), str(build)],
        "write": [str(build)],
        "cwd": str(build),
        "cpu_seconds": limits.timeout_seconds,
        "memory_bytes": limits.memory_bytes,
        "file_bytes": limits.file_bytes,
    }
    publish_bytes(output / "compile.policy.json", canonical_json(policy).encode())
    launcher = ["/usr/bin/python3", str(sandbox)]
    if lease_fd is not None:
        launcher += ["--lease-fd", str(lease_fd)]
    launcher += [str(output / "compile.policy.json"), "--"]
    commands = [
        [
            "/usr/bin/cmake",
            "-S",
            str(project),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG",
            f"-DCMAKE_CXX_COMPILER={manifest['compiler']['path']}",
            "-DMODEL_SOURCES=" + ";".join(names),
        ],
        [
            "/usr/bin/cmake",
            "--build",
            str(build),
            "--parallel",
            str(limits.cpus),
            "--target",
            "candidate",
        ],
    ]
    receipt = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "source_sha256": snapshot["source_sha256"],
        "configuration_sha256": snapshot["configuration_sha256"],
        "runtime_sha256": file_sha256(runtime / "runtime_manifest.json"),
        "model_api_sha256": manifest["model_api_sha256"],
        "compiler": manifest["compiler"],
        "recipe_sha256": manifest["source_inventory"][RECIPE],
        "launcher_sha256": file_sha256(sandbox),
        "limits": asdict(limits),
        "optimization": "-O3",
        "commands": [],
        "passed": False,
    }
    started = time.monotonic()
    try:
        for phase, command in zip(("configure", "build"), commands):
            receipt["commands"].append(
                run_build_command(
                    launcher + command,
                    output / f"{phase}.log",
                    cwd=output,
                    cpus=limits.cpus,
                    timeout_seconds=limits.timeout_seconds,
                    lease_fd=lease_fd,
                )
            )
        compiled = verify_optimized_commands(build / "compile_commands.json")
        verify_snapshot(store, candidate_id, contract, maximum_bytes=limits.source_bytes)
        for name in contract.paths:
            _verified_file(project / "model", name, snapshot["files"][name]["sha256"])
        current, _, _ = runtime_inputs(runtime)
        _verify_compiler(current)
        if file_sha256(runtime / "runtime_manifest.json") != receipt["runtime_sha256"]:
            raise InvalidSnapshot("runtime identity changed during compilation")
        binary = build / "output/candidate.so"
        symbols = run_build_command(
            launcher + ["/usr/bin/nm", "--dynamic", "--defined-only", str(binary)],
            output / "symbols.log",
            cwd=output,
            cpus=limits.cpus,
            timeout_seconds=limits.timeout_seconds,
            lease_fd=lease_fd,
        )
        receipt["commands"].append(symbols)
        exported_functions = {
            parts[-1]
            for line in (output / "symbols.log").read_text().splitlines()
            if len(parts := line.split()) == 3 and parts[1] in {"T", "W"}
        }
        required = {"ramulator_atomic_model_api_version", "ramulator_create_atomic_model"}
        if not required <= exported_functions:
            raise InvalidSnapshot("candidate is missing model-API function exports")
        # A settled copy is the only artifact evaluators may load. Build scratch
        # remains evidence, not the published model location.
        with binary.open("rb") as stream:
            data = stream.read(limits.file_bytes + 1)
        if len(data) > limits.file_bytes:
            raise InvalidSnapshot("candidate binary exceeds its declared file guard")
        publish_bytes(output / "candidate.so", data)
        receipt.update(
            passed=True,
            binary_sha256=file_sha256(output / "candidate.so"),
            compile_commands_sha256=file_sha256(build / "compile_commands.json"),
            translation_units=len(compiled),
        )
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["logs"] = {
        name: file_sha256(output / name)
        for name in ("configure.log", "build.log", "symbols.log")
        if (output / name).is_file()
    }
    publish_bytes(output / "build.json", canonical_json(receipt).encode())
    if not receipt["passed"]:
        raise BuildFailed(receipt)
    return receipt
