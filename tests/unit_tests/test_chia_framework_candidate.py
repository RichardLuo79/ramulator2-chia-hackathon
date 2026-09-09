"""Real isolated compiler checks, with no inference or application simulation.

The tiny runtime fixture binds the actual header, recipe and compiler; its
placeholder host binaries are never executed. Full native loading/measurement
parity is checked separately by check_chia_native_parity.py.
"""

import json
import pickle
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tools.chia_loop.framework import candidate as implementation
from tools.chia_loop.framework.build import MODEL_API_HEADER, MODEL_INTERFACE, file_sha256
from tools.chia_loop.framework.candidate import (
    RECIPE,
    BuildFailed,
    BuildLimits,
    compile_snapshot,
    verify_build,
)
from tools.chia_loop.framework.snapshots import InvalidSnapshot, ModelFiles, publish_bytes, snapshot

REPO = Path(__file__).resolve().parents[2]
CONTRACT = ModelFiles(("model.cpp",), "parameters.json")
LIMITS = BuildLimits(
    cpus=1,
    timeout_seconds=60,
    source_bytes=1024**2,
    memory_bytes=2 * 1024**3,
    file_bytes=128 * 1024**2,
)


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime-fixture"
    api, recipe = REPO / MODEL_API_HEADER, REPO / RECIPE
    publish_bytes(root / "export" / MODEL_API_HEADER, api.read_bytes())
    publish_bytes(root / "runtime-source" / RECIPE, recipe.read_bytes())
    for name in ("libramulator.so", "isolated_sim"):
        publish_bytes(root / "runtime" / name, b"not executable; compiler fixture only")
    compiler = Path(shutil.which("c++"))
    manifest = {
        "schema_version": 3,
        "candidate_interface": MODEL_INTERFACE,
        "optimization": "-O3",
        "model_api_sha256": file_sha256(api),
        "export_hashes": {MODEL_API_HEADER: file_sha256(api)},
        "source_inventory": {RECIPE: file_sha256(recipe)},
        "compiler": {"path": str(compiler), "sha256": file_sha256(compiler)},
        "library_sha256": file_sha256(root / "runtime/libramulator.so"),
        "executable_sha256": file_sha256(root / "runtime/isolated_sim"),
    }
    (root / "runtime_manifest.json").write_text(json.dumps(manifest))
    return root


def submit(tmp_path, source, *, parameters=b"{}", name="draft"):
    workspace = tmp_path / name
    publish_bytes(workspace / "model.cpp", source.encode())
    publish_bytes(workspace / "parameters.json", parameters)
    return snapshot(workspace, tmp_path / "snapshots", CONTRACT, maximum_bytes=LIMITS.source_bytes)


def build(runtime, tmp_path, candidate, *, name="build"):
    return compile_snapshot(
        runtime,
        tmp_path / "snapshots",
        candidate["candidate_id"],
        CONTRACT,
        tmp_path / name,
        limits=LIMITS,
    )


def seed():
    return (REPO / "tools/chia_loop/model/seed.cpp").read_text()


def test_actual_seed_compiles_from_the_snapshot(runtime, tmp_path):
    submitted = submit(tmp_path, seed())
    receipt = build(runtime, tmp_path, submitted)
    assert receipt["passed"] and receipt["translation_units"] == 1
    assert receipt["candidate_id"] == submitted["candidate_id"]
    assert receipt["source_sha256"] == submitted["source_sha256"]
    assert receipt["binary_sha256"] == file_sha256(tmp_path / "build/candidate.so")
    assert receipt == json.loads((tmp_path / "build/build.json").read_text())
    assert set(receipt["logs"]) == {"configure.log", "build.log", "symbols.log"}
    api_files = [
        p.relative_to(tmp_path / "build/project/api").as_posix()
        for p in (tmp_path / "build/project/api").rglob("*")
        if p.is_file()
    ]
    assert api_files == [MODEL_API_HEADER.removeprefix("src/")]
    policy = json.loads((tmp_path / "build/compile.policy.json").read_text())
    assert str(REPO) not in policy["read"] and str(runtime) not in policy["read"]
    assert policy["write"] == [str(tmp_path / "build/build")]
    with pytest.raises(FileExistsError):
        build(runtime, tmp_path, submitted)
    verified, snapshot = verify_build(
        runtime,
        tmp_path / "snapshots",
        submitted["candidate_id"],
        CONTRACT,
        tmp_path / "build",
        expected_receipt_sha256=file_sha256(tmp_path / "build/build.json"),
        maximum_source_bytes=LIMITS.source_bytes,
    )
    assert verified == receipt and snapshot == submitted


def test_verify_build_rejects_wrong_draft_and_changed_binary(runtime, tmp_path):
    submitted = submit(tmp_path, seed())
    build(runtime, tmp_path, submitted)
    receipt_hash = file_sha256(tmp_path / "build/build.json")
    another = submit(tmp_path, seed(), name="second", parameters=b'{"delay": 1}')
    with pytest.raises(InvalidSnapshot, match="requested candidate"):
        verify_build(
            runtime,
            tmp_path / "snapshots",
            another["candidate_id"],
            CONTRACT,
            tmp_path / "build",
            expected_receipt_sha256=receipt_hash,
            maximum_source_bytes=LIMITS.source_bytes,
        )
    with pytest.raises(InvalidSnapshot, match="changed"):
        verify_build(
            runtime,
            tmp_path / "snapshots",
            submitted["candidate_id"],
            CONTRACT,
            tmp_path / "build",
            expected_receipt_sha256="0" * 64,
            maximum_source_bytes=LIMITS.source_bytes,
        )
    binary = tmp_path / "build/candidate.so"
    binary.chmod(0o600)
    binary.write_bytes(b"not the compiled candidate")
    with pytest.raises(InvalidSnapshot, match="changed"):
        verify_build(
            runtime,
            tmp_path / "snapshots",
            submitted["candidate_id"],
            CONTRACT,
            tmp_path / "build",
            expected_receipt_sha256=receipt_hash,
            maximum_source_bytes=LIMITS.source_bytes,
        )


def test_different_draft_and_parameter_identities_are_retained(runtime, tmp_path):
    first = submit(tmp_path, seed(), name="first")
    second = submit(
        tmp_path,
        seed().replace("return now + latency;", "return now + latency + 1;"),
        name="second",
        parameters=b'{"delay": 1}',
    )
    a = build(runtime, tmp_path, first, name="first-build")
    b = build(runtime, tmp_path, second, name="second-build")
    assert a["source_sha256"] != b["source_sha256"]
    assert a["configuration_sha256"] != b["configuration_sha256"]
    assert a["binary_sha256"] != b["binary_sha256"]


@pytest.mark.parametrize(
    "source",
    [
        '#include "ramulator/base/request.h"\n',
        seed()
        .replace("return now + latency;", "return access.frontend_id;")
        .replace("Ramulator::AtomicModel::Access,", "Ramulator::AtomicModel::Access access,"),
        "int not_a_model() { return 1; }\n",
    ],
)
def test_invalid_api_use_is_not_a_successful_build(runtime, tmp_path, source):
    candidate = submit(tmp_path, source)
    with pytest.raises(BuildFailed) as exc:
        build(runtime, tmp_path, candidate)
    assert exc.value.receipt["passed"] is False
    assert (tmp_path / "build/build.json").is_file()
    assert not (tmp_path / "build/candidate.so").exists()
    assert (tmp_path / "build/project/model/model.cpp").read_text() == source


def test_compiler_cannot_read_a_private_include(runtime, tmp_path):
    private = tmp_path / "private-outside-grant.h"
    private.write_text("#error PRIVATE_CANARY_WAS_READ\n")
    submitted = submit(tmp_path, '#include "' + str(private) + '"\n' + seed())
    with pytest.raises(BuildFailed):
        build(runtime, tmp_path, submitted)
    log = (tmp_path / "build/build.log").read_text()
    assert "Permission denied" in log
    assert "PRIVATE_CANARY_WAS_READ" not in log


@pytest.mark.parametrize(
    "field", ["candidate_interface", "compiler", "model_api_sha256", "export_hashes"]
)
def test_runtime_mismatch_fails_before_compilation(runtime, tmp_path, field):
    candidate = submit(tmp_path, seed())
    file = runtime / "runtime_manifest.json"
    data = json.loads(file.read_text())
    if field == "compiler":
        data[field]["sha256"] = "0" * 64
    elif field == "export_hashes":
        data[field]["src/private.h"] = "0" * 64
    else:
        data[field] = "legacy" if field == "candidate_interface" else "0" * 64
    file.write_text(json.dumps(data))
    with pytest.raises(InvalidSnapshot):
        build(runtime, tmp_path, candidate)
    assert not (tmp_path / "build").exists()


def test_existing_build_verification_does_not_reopen_original_compiler(
    runtime, tmp_path, monkeypatch
):
    submitted = submit(tmp_path, seed())
    receipt = build(runtime, tmp_path, submitted)
    compiler = Path(receipt["compiler"]["path"])
    original_hash = implementation.file_sha256

    def compiler_unavailable(path):
        if Path(path) == compiler:
            raise FileNotFoundError("the original compiler is not on the replay host")
        return original_hash(path)

    monkeypatch.setattr(implementation, "file_sha256", compiler_unavailable)
    verified, _ = verify_build(
        runtime,
        tmp_path / "snapshots",
        submitted["candidate_id"],
        CONTRACT,
        tmp_path / "build",
        expected_receipt_sha256=file_sha256(tmp_path / "build/build.json"),
        maximum_source_bytes=LIMITS.source_bytes,
    )
    assert verified == receipt
    with pytest.raises(FileNotFoundError, match="original compiler"):
        build(runtime, tmp_path, submitted, name="new-build")
    assert not (tmp_path / "new-build").exists()


def test_runtime_manifest_cannot_change_during_a_successful_compile(runtime, tmp_path, monkeypatch):
    submitted = submit(tmp_path, seed())
    original_run = implementation.run_build_command

    def mutate_after_command(*args, **kwargs):
        result = original_run(*args, **kwargs)
        path = runtime / "runtime_manifest.json"
        record = json.loads(path.read_text())
        record["unexpected_change_during_build"] = True
        path.write_text(json.dumps(record))
        return result

    monkeypatch.setattr(implementation, "run_build_command", mutate_after_command)
    with pytest.raises(BuildFailed, match="runtime identity changed during compilation") as failed:
        build(runtime, tmp_path, submitted)
    assert not failed.value.receipt["passed"]
    assert not (tmp_path / "build/candidate.so").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpus", 13),
        ("cpus", True),
        ("source_bytes", 0),
        ("memory_bytes", -1),
        ("timeout_seconds", 1.5),
    ],
)
def test_limits_are_explicit_validated_inputs(field, value):
    with pytest.raises(ValueError):
        replace(LIMITS, **{field: value})


def test_model_api_is_independently_compilable_and_parameters_are_checked(tmp_path):
    binary = tmp_path / "api-test"
    subprocess.run(
        [
            "/usr/bin/c++",
            "-O3",
            "-DNDEBUG",
            "-std=c++20",
            "-I" + str(REPO / "src"),
            str(REPO / "tests/utils/atomic_model_api_test.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
    assert "API and parameter checks passed" in result.stdout


def test_build_failure_keeps_its_receipt_when_pickled():
    receipt = {"error": "fixture compiler error", "passed": False, "compiler_log": "example"}
    original = implementation.BuildFailed(receipt)
    recovered = pickle.loads(pickle.dumps(original))
    assert type(recovered) is type(original)
    assert recovered.receipt == receipt and str(recovered) == str(original)
