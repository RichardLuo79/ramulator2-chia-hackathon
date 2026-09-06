import copy
import json
import pathlib
import subprocess
import sys

import pytest

from tools.chia_loop import evaluation_config as W, transfer as T
from tools.chia_loop.core import atomic_write_json


def frozen_fixture(root):
    for name in ("seed", "selected"):
        directory = root / name
        directory.mkdir()
        source, plugin = directory / "atomic_controller.cpp", directory / "candidate.so"
        source.write_text(name)
        plugin.write_bytes(name.encode())
        atomic_write_json(directory / "build.json", {"source_sha256": T.P.sha(source.read_bytes()),
            "plugin_sha256": T.P.sha(plugin.read_bytes()), "optimization": "-O3"})
    library = root / "runtime/libramulator.so"
    library.parent.mkdir()
    library.write_bytes(b"trusted runtime")
    atomic_write_json(root / "runtime_manifest.json", {"optimization": "-O3", "library_sha256": T.P.sha(library.read_bytes())})
    state = {"status": "frozen", "termination": "iteration_limit", "incumbent": "c1", "frozen_at": 7,
        "selected": {"source_path": str(root / "selected/atomic_controller.cpp"),
                     "plugin": str(root / "selected/candidate.so"), "sha256": T.P.sha("selected")}}
    atomic_write_json(root / "selection_frozen.json", {"run_id": root.name, "source_sha256": T.P.sha("selected"), "incumbent": "c1", "frozen_at": 7})
    return state


def test_transfer_freeze_binds_exact_source_plugin_and_runtime(tmp_path):
    state = frozen_fixture(tmp_path)
    assert T.selection(tmp_path, state)["source_sha256"] == T.P.sha("selected")
    running = copy.deepcopy(state)
    running["status"] = "running"
    with pytest.raises(RuntimeError, match="frozen"):
        T.selection(tmp_path, running)
    (tmp_path / "selected/candidate.so").write_bytes(b"different")
    with pytest.raises(RuntimeError, match="binding"):
        T.selection(tmp_path, state)


def test_legacy_transfer_is_disabled_without_opening_any_tests(tmp_path):
    assert T.run(tmp_path, {"status": "running"}) == {}


def write_trace(path, arrivals, identities=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    identities = identities or list(range(len(arrivals)))
    path.write_text("addr,type,source,arrive,depart,frontend_id,frontend_sub_id,admission_ordinal\n" +
        "".join(f"{i*64},0,0,{a},{a+10},{i},0,{i}\n" for i,a in zip(identities,arrivals)))


def test_exact_coverage_and_incomplete_coverage_are_not_conflated(tmp_path):
    oracle, model = tmp_path / "oracle.csv", tmp_path / "model.csv"
    write_trace(oracle, [1, 2, 3])
    write_trace(model, [1, 2, 3])
    r = T.request_metrics(oracle, model, tmp_path / "request.json", "gem5")
    assert r["request_metric_eligible"] and r["mae"] == 0
    assert r["most_negative_over_L"] == 0
    write_trace(model, [1, 2])
    r = T.request_metrics(oracle, model, tmp_path / "request.json", "gem5")
    assert not r["request_metric_eligible"]
    assert r["request_metric_status"] == "diagnostic_incomplete_pairing"
    assert r["cov_o"] == pytest.approx(2/3)


def test_transfer_keeps_all_baselines_and_fixed_ddr5(tmp_path):
    assert T.MODELS == ["oracle", "seed", "selected", "fixedlat", "md1", "wmg1", "mess"]
    for frontend in ("champsim", "gem5"):
        for model in T.MODELS:
            config = T.memory_config(tmp_path, model, tmp_path / "trace.csv", frontend)
            ctrl = config["memory_system"]["controllers"][0]
            assert ctrl["dram"]["impl"] == "DDR5"
            assert ctrl["impl"] == ("Atomic" if model in {"seed", "selected"} else "GenericDDR" if model == "oracle" else T.C.MODEL_IMPL[model])
            if model in {"seed", "selected", "oracle"}:
                assert ctrl["read_buffer_size"] == 64 and ctrl["write_buffer_size"] == 64
            if model == "mess":
                assert ctrl["curve_path"] == str(tmp_path / "transfer/runtime/mess_DDR5.txt")


@pytest.mark.parametrize("model", T.E.COMPARISONS)
def test_baseline_controller_trace_retains_frontend_identity(tmp_path, model):
    """Small unit fixture only, not an evaluation or representative ROI."""
    import csv
    import ramulator
    pytest.importorskip("ramulator._ramulator")
    trace = tmp_path / "input.trace"
    trace.write_text("1 64\n1 128\n1 192\n1 256\n")
    dram = ramulator.dram.DDR5(org_preset="DDR5_16Gb_x8", timing_preset="DDR5_4800AN")
    ctrl = T.E.S._build_controller(ramulator, model, dram, "DDR5", tmp_path / "controller.csv", {})
    fe = ramulator.frontend.SimpleO3(clock_ratio=8, traces=[str(trace)], num_expected_insts=64,
        llc_num_mshr_per_core=16, translation=ramulator.translation.NoTranslation(max_addr=2**33))
    ms = ramulator.memory_system.GenericDRAM(clock_ratio=3, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    sim = ramulator.Simulation(fe, ms)
    sim.run()
    sim.finalize()
    with (tmp_path / "controller.csv.ch0").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(int(r["frontend_id"]) >= 0 and int(r["admission_ordinal"]) >= 0 for r in rows)
    assert len({r["admission_ordinal"] for r in rows}) == len(rows)


def test_changed_transfer_asset_or_profile_is_rejected(tmp_path):
    W.install(tmp_path)
    asset = tmp_path / "runtime.txt"
    asset.write_text("original")
    atomic_write_json(tmp_path / "transfer_inputs.json", {"profile": W.load(tmp_path),
        "profile_sha256": T.P.sha((tmp_path / "evaluation_config.json").read_bytes()),
        "assets": {"runtime.txt": T.C.file_provenance(asset)}, "external": {}})
    T.verify_inputs(tmp_path)
    asset.write_text("changed")
    with pytest.raises(RuntimeError, match="input changed"):
        T.verify_inputs(tmp_path)


def test_frozen_transfer_is_outside_training_and_runs_complete_matrix(tmp_path, monkeypatch):
    state = frozen_fixture(tmp_path)
    W.install(tmp_path)
    config = W.load(tmp_path)
    monkeypatch.setattr(T, "verify_inputs", lambda root: {"profile": config})
    jobs = []
    monkeypatch.setattr(T, "run_case", lambda root, inputs, state, frontend, w, m, frozen: jobs.append((frontend, w, m)))
    def summary(root, frontend, workloads, config):
        result = {"models": {}, "workloads": workloads}
        atomic_write_json(root / "transfer/reports" / (frontend + ".json"), result)
        return result
    monkeypatch.setattr(T, "summarize", summary)
    result = T.run(tmp_path, state, workers=2)
    assert set(result) == {"champsim", "gem5"}
    assert len(jobs) == (6 + 12) * 7
    assert not (tmp_path / "training").exists()
    assert json.loads((tmp_path / "transfer/status.json").read_text())["status"] == "completed"


def test_transfer_archives_join_the_root_restore_manifest(tmp_path):
    directory = tmp_path / "transfer/champsim/DDR5/workload/seed"
    atomic_write_json(directory / "manifest.json", {"complete": True})
    write_trace(directory / "trace.csv.ch0", [1, 2, 3])
    original = T.C.file_provenance(directory / "trace.csv.ch0")
    assert T.E.verify_run_archives(tmp_path) == 1
    assert not (directory / "trace.csv.ch0").exists()
    T.AR.verify(tmp_path / "archive_manifest.json")
    assert T.A.raw_provenance(directory / "trace.csv.ch0")["sha256"] == original["sha256"]


def test_transfer_launcher_denies_other_runs_and_network(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "other_run.txt"
    secret.write_text("PRIVATE_MODEL_HISTORY")
    probe = ("import pathlib,socket;\n"
             f"p=pathlib.Path({str(secret)!r})\n"
             "try: p.read_text()\n"
             "except PermissionError: pass\n"
             "else: raise AssertionError('other run visible')\n"
             "try: socket.socket()\n"
             "except PermissionError: pass\n"
             "else: raise AssertionError('network permitted')\n"
             "print('TRANSFER_ISOLATION_PASS')\n")
    policy = tmp_path / "launch.json"
    atomic_write_json(policy, {"cwd": str(allowed), "cpu_seconds": 10,
        "read": [*T.E.SYSTEM_READ, str(allowed)], "write": [str(allowed)],
        "command": ["/usr/bin/python3", "-c", probe], "environment": {"PATH": "/usr/bin:/bin"}})
    result = subprocess.run([sys.executable, "-m", "tools.chia_loop.transfer_sandbox", str(policy)],
        capture_output=True, text=True, cwd=T.REPO)
    assert result.returncode == 0, result.stderr
    assert "TRANSFER_ISOLATION_PASS" in result.stdout
    assert secret.read_text() == "PRIVATE_MODEL_HISTORY"
