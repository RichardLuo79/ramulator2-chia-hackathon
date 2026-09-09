"""Stage and build Ramulator's native evaluation targets with CMake.

The staging tree is trusted build input, not an agent workspace. Agents receive
only the separately approved model/API view. CMake owns object selection and
linking; this module records input/output identity and resource use.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

from tools.chia_loop.core import atomic_write_json

from .identity import file_sha256 as file_sha256

SOURCE_TREES = ("src", "ext/fmt", "ext/yaml-cpp")
SOURCE_FILES = (
    "CMakeLists.txt",
    "tools/chia_loop/CMakeLists.txt",
    "tools/chia_loop/isolated_sim.cpp",
    "tools/chia_loop/model/seed.cpp",
    "tools/chia_loop/model/CMakeLists.txt",
    "tests/utils/simulation_lifecycle_test.cpp",
    "tests/utils/batch_simulation_test.cpp",
    "tests/utils/atomic_model_api_test.cpp",
    "tests/utils/atomic_admission_test.cpp",
    "tests/utils/atomic_model_loader_test.cpp",
    "tests/utils/atomic_model_loader_fixture.cpp",
)
EXCLUDED_DIRECTORIES = frozenset({".git", "__pycache__", ".pytest_cache", "build", "_build"})
MODEL_API_HEADER = "src/ramulator/controller/atomic_model/api.h"
MODEL_INTERFACE = "atomic-model-api-v1"


def _source_files(repo: Path):
    """Enumerate the explicitly approved build roots, never repository history."""
    for relative in SOURCE_FILES:
        yield repo / relative
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        if (repo / name).is_file():
            yield repo / name
    for relative in SOURCE_TREES:
        tree = repo / relative
        if not tree.is_dir() or tree.is_symlink():
            raise ValueError(f"missing or symlinked build input: {tree}")
        for directory, names, files in os.walk(tree, followlinks=False):
            names[:] = sorted(name for name in names if name not in EXCLUDED_DIRECTORIES)
            for name in names:
                if (Path(directory) / name).is_symlink():
                    raise ValueError(f"symlinked build directory: {Path(directory) / name}")
            for name in sorted(files):
                # Git submodule administration is not source or replay input.
                if name != ".git":
                    yield Path(directory) / name


def stage_source(repo: Path, destination: Path) -> dict[str, str]:
    """Copy build inputs once, checking that source bytes did not change mid-copy."""
    repo, destination = repo.resolve(strict=True), destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"staging destination already exists: {destination}")
    paths = list(_source_files(repo))
    hashes: dict[str, str] = {}
    for source in paths:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"build input must be a regular file: {source}")
        relative = source.relative_to(repo).as_posix()
        hashes[relative] = file_sha256(source)
    destination.mkdir(parents=True)
    for source in paths:
        relative = source.relative_to(repo).as_posix()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_sha256(target) != hashes[relative] or file_sha256(source) != hashes[relative]:
            raise RuntimeError(f"build input changed during staging: {relative}")
    atomic_write_json(destination / "source_inventory.json", hashes)
    return hashes


def run_build_command(
    command: list[str],
    log: Path,
    *,
    cwd: Path,
    cpus: int,
    timeout_seconds: int,
    lease_fd: int | None = None,
) -> dict:
    """Retain complete output, including failures; never return a success receipt on error."""
    env = os.environ.copy()
    env.update(
        {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpus)
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("xb") as stream:
        stream.write((shlex.join(command) + "\n").encode())
        stream.flush()
        try:
            with subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=() if lease_fd is None else (lease_fd,),
            ) as process:
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except BaseException:
                    # CHIA tracks this process group for remote cancellation.
                    # A local deadline must also reap compiler grandchildren;
                    # killing CMake alone would leave them writing our evidence.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    # The group may outlive its leader: a compiler grandchild
                    # can ignore TERM even after CMake has exited. Always finish
                    # group cleanup before publishing a failed attempt.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    raise
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    receipt = {
        "argv": command,
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "cpus": cpus,
        "log_sha256": file_sha256(log),
    }
    if returncode:
        raise RuntimeError(f"build command failed ({returncode}); see {log}")
    return receipt


def verify_optimized_commands(path: Path) -> list[dict]:
    """Check actual translation-unit flags, not just the requested build type."""
    commands = json.loads(path.read_text())
    if not commands:
        raise ValueError("CMake emitted no compilation commands")
    for entry in commands:
        arguments = entry.get("arguments") or shlex.split(entry["command"])
        optimizations = [
            arg
            for arg in arguments
            if arg
            in {
                "-O",
                "-O0",
                "-O1",
                "-O2",
                "-O3",
                "-Os",
                "-Oz",
                "-Og",
                "-Ofast",
            }
        ]
        if not optimizations or optimizations[-1] != "-O3":
            raise ValueError(f"translation unit does not use -O3: {entry['file']}")
    return commands


def prepare_runtime(
    repo: Path, campaign: Path, *, cpus: int, timeout_seconds: int = 1800, model_api: bool = True
) -> dict:
    """Build a new, source-bound runtime; this function never launches inference.

    The caller must reserve ``cpus`` through CHIA before dispatch. A partial build
    remains evidence; it is not silently reused or overwritten on another call.
    """
    if type(cpus) is not int or not 1 <= cpus <= 12:
        raise ValueError("the local build profile permits 1 to 12 reserved CPUs")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("build timeout must be a positive integer")
    if type(model_api) is not bool:
        raise ValueError("model_api must be a boolean")
    repo, campaign = repo.resolve(strict=True), campaign.absolute()
    if (campaign / "runtime").exists() or (campaign / "runtime-source").exists():
        raise FileExistsError("runtime preparation requires a new destination")
    source, build, runtime = (
        campaign / name for name in ("runtime-source", "runtime-build", "runtime")
    )
    inventory = stage_source(repo, source)
    runtime.mkdir()
    configure = run_build_command(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG",
            "-DRAMULATOR_PYTHON_BINDINGS=OFF",
            "-DRAMULATOR_CHIA_TOOLS=ON",
            f"-DRAMULATOR_CHIA_MODEL_API={'ON' if model_api else 'OFF'}",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            "-DYAML_CPP_BUILD_TESTS=OFF",
            "-DFMT_TEST=OFF",
            f"-DRAMULATOR_OUTPUT_DIRECTORY={runtime}",
        ],
        campaign / "logs/runtime-configure.log",
        cwd=source,
        cpus=cpus,
        timeout_seconds=timeout_seconds,
    )
    compiled = run_build_command(
        [
            "cmake",
            "--build",
            str(build),
            "--parallel",
            str(cpus),
            "--target",
            "ramulator-isolated",
            "ramulator-atomic",
        ],
        campaign / "logs/runtime-build.log",
        cwd=source,
        cpus=cpus,
        timeout_seconds=timeout_seconds,
    )
    commands = verify_optimized_commands(build / "compile_commands.json")

    # The new model implementation sees only its value API. Full controller
    # headers are exported solely for the explicit historical compatibility path.
    export = campaign / "export"
    for relative in inventory:
        path = source / relative
        if relative == MODEL_API_HEADER or (not model_api and path.suffix in {".h", ".hpp"}):
            target = export / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    flags = ["-O3", "-DNDEBUG", "-std=c++20", f"-I{export / 'src'}"]
    if not model_api:
        flags += [f"-I{export / 'ext/fmt/include'}", f"-I{export / 'ext/yaml-cpp/include'}"]
    manifest = {
        "schema_version": 3 if model_api else 2,
        "build_method": "cmake_targets",
        "candidate_interface": MODEL_INTERFACE if model_api else "legacy-controller",
        "source_inventory": inventory,
        "configure": configure,
        "build": compiled,
        "optimization": "-O3",
        "compile_flags": flags,
        "compile_commands_sha256": file_sha256(build / "compile_commands.json"),
        "translation_units": len(commands),
        "library_sha256": file_sha256(runtime / "libramulator.so"),
        "executable_sha256": file_sha256(runtime / "isolated_sim"),
        "seed_plugin_sha256": file_sha256(runtime / "candidate.so"),
        "driver_sha256": inventory["tools/chia_loop/isolated_sim.cpp"],
        "lifecycle_sha256": inventory["src/ramulator/base/simulation.cpp"],
        "synthetic_generator_sha256": inventory[
            "src/ramulator/frontend/impl/memory_trace/synthetic_pattern.cpp"
        ],
        "batch_lifecycle_sha256": inventory["src/ramulator/base/batch_simulation.cpp"],
        "batch_replay_format": "ordered-controller-arrivals-v1",
        "export_hashes": {
            p.relative_to(export).as_posix(): file_sha256(p)
            for p in sorted(export.rglob("*"))
            if p.is_file()
        },
    }
    if model_api:
        manifest["external_model_loading"] = "controller_model_library_v1"
        manifest["model_api_sha256"] = inventory[MODEL_API_HEADER]
        compiler = Path((commands[0].get("arguments") or shlex.split(commands[0]["command"]))[0])
        if not compiler.is_absolute():
            raise ValueError("the runtime compiler must have a recorded absolute path")
        manifest["compiler"] = {"path": str(compiler), "sha256": file_sha256(compiler)}
    atomic_write_json(campaign / "runtime_manifest.json", manifest)
    return manifest


def main(argv=None):
    """Build the local evaluator from source; never start a model or campaign."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True, help="new runtime directory")
    parser.add_argument("--workers", type=int, choices=range(1, 13), default=6)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    manifest = prepare_runtime(
        args.repo.absolute(),
        args.output.absolute(),
        cpus=args.workers,
        timeout_seconds=args.timeout_seconds,
        model_api=True,
    )
    print(
        json.dumps(
            {
                "runtime": str(args.output.absolute()),
                "translation_units": manifest["translation_units"],
            }
        )
    )


if __name__ == "__main__":
    main()
