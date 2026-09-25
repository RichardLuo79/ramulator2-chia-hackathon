"""Common task contracts and opt-in native integration, with no model calls.

CHIA_NATIVE_TEST_RUNTIME points to an isolated -O3 runtime prepared by build.py.
Tiny native cases are explicitly marked qualification and cannot be scored.
Representative-window parity is a separate, explicitly launched check.
"""

import csv
import gzip
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from ramulator_chia.framework import evaluation as E
from ramulator_chia.framework.archive import describe_payload
from ramulator_chia.framework.build import file_sha256
from ramulator_chia.framework.candidate import BuildLimits, compile_snapshot
from ramulator_chia.framework.snapshots import ModelFiles, snapshot
from ramulator_chia.eval import config as C
from ramulator_chia.eval import simpleo3

from ramulator_chia.layout import RAMULATOR as REPO, NATIVE
LIMITS = E.SimulationLimits(
    timeout_seconds=60,
    memory_bytes=2 * 1024**3,
    file_bytes=128 * 1024**2,
    source_bytes=1024**2,
    gzip_level=3,
)
CONTRACT = ModelFiles(("model.cpp",), "parameters.json")
MEASURE_NODE, COMPARE_NODE = E.measure_simpleo3, E.compare_simpleo3


@pytest.fixture(autouse=True)
def local_task_bodies(monkeypatch):
    # These tests exercise native binaries, not Ray startup/profiling. The
    # explicit distributed check supplies its own bounded local CHIA cluster.
    monkeypatch.syspath_prepend(str(REPO / "python"))
    monkeypatch.setattr(E, "measure_simpleo3", MEASURE_NODE._chia_original)
    monkeypatch.setattr(E, "compare_simpleo3", COMPARE_NODE._chia_original)


def trace_case(tmp_path, *, stage="qualification", instructions=64):
    data = "".join(f"1 {64 + index * 4096}\n" for index in range(32)).encode()
    path = tmp_path / "input.trace.gz"
    path.write_bytes(gzip.compress(data, mtime=0))
    return E.SimpleO3Case(
        "generic-fixture",
        stage,
        instructions,
        (describe_payload(path, "inputs/fixture.trace.gz", codec="gzip"),),
    )


def test_short_windows_are_qualification_only(tmp_path):
    case = trace_case(tmp_path)
    assert case.identity()["allow_trace_wrap"] is False
    for stage in ("training", "test"):
        with pytest.raises(ValueError, match="20M"):
            replace(case, stage=stage)
        assert replace(case, stage=stage, instructions_per_core=20_000_000)
    with pytest.raises(ValueError, match="catalog label"):
        replace(case, workload="../../another-campaign")
    with pytest.raises(ValueError, match="ordered tuple"):
        replace(case, traces=[])
    with pytest.raises(ValueError, match="positive integer"):
        replace(case, instructions_per_core=True)


def test_storage_compression_does_not_change_case_identity(tmp_path):
    case = trace_case(tmp_path)
    path = tmp_path / "input.trace"
    path.write_bytes(gzip.decompress(case.traces[0].source.read_bytes()))
    raw_case = replace(case, traces=(describe_payload(path, "inputs/fixture.trace"),))
    assert raw_case.identity() == case.identity()
    assert raw_case.traces[0].member != case.traces[0].member


def test_shared_config_keeps_all_baselines_and_no_wrap(tmp_path):
    for model in simpleo3.ALL_MODELS:
        cfg = simpleo3.build_config(
            model,
            standard="DDR5",
            traces=[tmp_path / "trace"],
            instructions_per_core=20_000_000,
            request_trace=tmp_path / "logical",
            controller_trace=tmp_path / "controller",
            mess_curve=tmp_path / "curve",
        )
        assert cfg["frontend"]["allow_trace_wrap"] is False
        ctrl = cfg["memory_system"]["controllers"][0]
        assert ctrl["dram"]["impl"] == "DDR5"
        if model in {"oracle", "candidate"}:
            assert ctrl["read_buffer_size"] == ctrl["write_buffer_size"] == 64
        if model == "mess":
            assert ctrl["curve_path"] == str(tmp_path / "curve")


def test_measurement_nodes_reuse_chia_resource_reservations():
    assert (
        MEASURE_NODE._chia_options
        == COMPARE_NODE._chia_options
        == {"num_cpus": 1, "max_retries": 0}
    )


@pytest.fixture
def native_runtime():
    path = os.environ.get("CHIA_NATIVE_TEST_RUNTIME")
    if not path:
        pytest.skip("native runtime integration is an explicit opt-in")
    return Path(path)


def build_candidate(runtime, tmp_path, *, parameters=None, name="draft"):
    source = (NATIVE / "model/seed.cpp").read_text()
    if parameters is not None:
        changed = source.replace(
            "Ramulator::AtomicModel::Parameters&) override",
            "Ramulator::AtomicModel::Parameters& parameters) override",
        )
        changed = changed.replace(
            "latency = hardware.read_latency;",
            'latency = parameters.number("delay", hardware.read_latency, 1, 1000);',
        )
        assert changed != source
        source = changed
    workspace = tmp_path / name
    workspace.mkdir()
    (workspace / "model.cpp").write_text(source)
    (workspace / "parameters.json").write_text(json.dumps(parameters or {}))
    submitted = snapshot(
        workspace, tmp_path / "snapshots", CONTRACT, maximum_bytes=LIMITS.source_bytes
    )
    build = tmp_path / (name + "-build")
    compile_snapshot(
        runtime,
        tmp_path / "snapshots",
        submitted["candidate_id"],
        CONTRACT,
        build,
        limits=BuildLimits(
            cpus=1,
            timeout_seconds=60,
            memory_bytes=LIMITS.memory_bytes,
            file_bytes=LIMITS.file_bytes,
            source_bytes=LIMITS.source_bytes,
        ),
    )
    return E.CandidateBuild(
        tmp_path / "snapshots",
        submitted["candidate_id"],
        CONTRACT,
        build,
        file_sha256(build / "build.json"),
    )


@pytest.mark.parametrize("model", simpleo3.ALL_MODELS)
def test_native_baselines_and_snapshot_seed(native_runtime, tmp_path, model):
    case = trace_case(tmp_path)
    candidate = build_candidate(native_runtime, tmp_path) if model == "candidate" else None
    curve = (
        describe_payload(Path(C.MESS_CURVES["DDR5"]), "inputs/mess.txt")
        if model == "mess"
        else None
    )
    output = tmp_path / "measurement"
    result = E.measure_simpleo3(
        native_runtime, case, model, output, limits=LIMITS, candidate=candidate, mess_curve=curve
    )
    assert result["complete"] and result["frontend_stats"]["allow_trace_wrap"] == 0
    assert result["frontend_stats"]["trace_instructions_core_0"] == 64
    assert result["per_core_cycles"][0] > 0
    assert len(result["traces"]) == 2
    assert not list((output / "inputs").glob("*.trace"))
    assert not list((output / "observations").glob("*.ch0"))
    assert case.traces[0].source.is_file()
    checked = E.verify_measurement(output, file_sha256(output / "measurement.json"))
    assert checked == result
    with pytest.raises(FileExistsError):
        E.measure_simpleo3(
            native_runtime,
            case,
            model,
            output,
            limits=LIMITS,
            candidate=candidate,
            mess_curve=curve,
        )
    if model == "oracle":
        with pytest.raises(ValueError, match="fixtures are not accuracy"):
            E.compare_simpleo3(
                output,
                output,
                oracle_receipt_sha256=file_sha256(output / "measurement.json"),
                model_receipt_sha256=file_sha256(output / "measurement.json"),
                minimum_oracle_owner_reads=1,
            )


def test_native_draft_parameters_reach_the_actual_predictor(native_runtime, tmp_path):
    case = trace_case(tmp_path)
    results = []
    for delay in (37, 99):
        candidate = build_candidate(
            native_runtime, tmp_path, parameters={"delay": delay}, name=f"delay-{delay}"
        )
        output = tmp_path / f"measurement-{delay}"
        result = E.measure_simpleo3(
            native_runtime, case, "candidate", output, limits=LIMITS, candidate=candidate
        )
        assert result["candidate"]["parameters"] == {"delay": delay}
        assert result["candidate"]["candidate_id"] == candidate.candidate_id
        with gzip.open(output / result["traces"]["controller.csv.ch0"]["name"], "rt") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and {int(r["depart"]) - int(r["arrive"]) for r in rows} == {delay}
        results.append(result)
    assert (
        results[0]["candidate"]["configuration_sha256"]
        != results[1]["candidate"]["configuration_sha256"]
    )
    assert results[0]["per_core_cycles"][0] < results[1]["per_core_cycles"][0]


def test_native_unknown_parameter_is_failure_with_evidence(native_runtime, tmp_path):
    case = trace_case(tmp_path)
    candidate = build_candidate(native_runtime, tmp_path, parameters={"unknown": 37})
    output = tmp_path / "failed"
    with pytest.raises(E.MeasurementFailed) as error:
        E.measure_simpleo3(
            native_runtime, case, "candidate", output, limits=LIMITS, candidate=candidate
        )
    assert error.value.receipt["complete"] is False
    assert "Unknown atomic model parameter" in (output / "simulation.log").read_text()
    with pytest.raises(ValueError, match="complete native"):
        E.verify_measurement(output, file_sha256(output / "measurement.json"))


def test_native_short_trace_fails_in_parser_before_candidate_load(native_runtime, tmp_path):
    case = trace_case(tmp_path, instructions=65)
    candidate = build_candidate(native_runtime, tmp_path)
    output = tmp_path / "failed"
    with pytest.raises(E.MeasurementFailed):
        E.measure_simpleo3(
            native_runtime, case, "candidate", output, limits=LIMITS, candidate=candidate
        )
    assert "supplies 64 instructions; fixed ROI needs 65" in (output / "simulation.log").read_text()
    result = json.loads((output / "measurement.json").read_text())
    assert result["complete"] is False and "per_core_cycles" not in result


def test_native_corrupt_input_is_not_executed(native_runtime, tmp_path):
    case = trace_case(tmp_path)
    data = bytearray(case.traces[0].source.read_bytes())
    data[-8] ^= 1
    case.traces[0].source.write_bytes(data)
    output = tmp_path / "failed"
    with pytest.raises(E.MeasurementFailed) as error:
        E.measure_simpleo3(native_runtime, case, "oracle", output, limits=LIMITS)
    assert "CRC check failed" in error.value.receipt["error"]
    assert not (output / "simulation.log").exists()
    assert not (output / "inputs/core-0.trace").exists()


def test_native_legacy_wrap_default_cannot_be_scored(native_runtime, tmp_path, monkeypatch):
    case = trace_case(tmp_path, instructions=65)
    builder = simpleo3.build_config

    def legacy_config(*args, **kwargs):
        config = builder(*args, **kwargs)
        del config["frontend"]["allow_trace_wrap"]
        return config

    monkeypatch.setattr(simpleo3, "build_config", legacy_config)
    output = tmp_path / "legacy-wrap"
    with pytest.raises(E.MeasurementFailed, match="no-wrap SimpleO3 window"):
        E.measure_simpleo3(native_runtime, case, "oracle", output, limits=LIMITS)
    import yaml

    stats = yaml.safe_load((output / "observations/stats.yaml").read_text())["frontend"]
    # Historical cyclic replay still works, but it is not an eligible new run.
    assert stats["allow_trace_wrap"] == 1
    assert stats["trace_instructions_core_0"] == 64
    assert stats["insts_issued_core_0"] == 65


def test_measurement_failure_keeps_its_receipt_when_pickled():
    receipt = {"error": "fixture native error", "complete": False, "observation": "example"}
    original = E.MeasurementFailed(receipt)
    recovered = pickle.loads(pickle.dumps(original))
    assert type(recovered) is type(original)
    assert recovered.receipt == receipt and str(recovered) == str(original)
