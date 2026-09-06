"""No-LLM synthetic diagnostic protocol, pairing, provenance and gzip tests."""
import pathlib

import pandas as pd
import pytest

from tools.chia_loop import synthetic_diagnostics as D, loop_config as L, real_core as P, real_eval as E
from tools.chia_loop.core import atomic_write_json

LAYOUT = {"total_bank_units": 32, "num_cls": 64}


def params(values=None):
    return D.parameters(values or {}, L.DEFAULTS["synthetic"], LAYOUT)


@pytest.mark.parametrize("value", [
    {"path": "../private"}, {"plugin": "other.so"}, {"std": "HBM4"}, {"num_requests": 1},
    {"num_requests": True}, {"streams": 9}, {"bank_spread": 33}, {"row_run": 65},
    {"share_region": "false"}, {"mlp": 65}, {"dep_frac": float("nan")}, {"dep_frac": True},
    {"mode_b": "mlp=10;path=secret"}, {"stream_params": [{"no_such_axis": 1}]},
    {"streams": 2, "stream_params": [{}]}, {"streams": 8},
    {"wb_mode": 3, "bank_spread": 17}, {"mode_b": {"row_run": 100}},
])
def test_bounded_generic_axes_only(value):
    with pytest.raises(ValueError):
        params(value)


def test_normalization_and_typed_composition():
    p = params({"streams": 2, "stream_params": [{"bank_random": True}, {"dep_frac": 1}],
                "mode_switch": 100, "mode_b": {"row_run": 4}})
    assert p["num_requests"] == 50000 and p["seed"] == 1
    generated = D.generator_parameters(p)
    assert generated["stream_params"] == "bank_random=1|dep_frac=1.0"
    assert generated["mode_b"] == "row_run=4"
    assert params({}) == params(D.AXES)


def own_parent(root, nested=False):
    directory = root / "candidates/fixture" if nested else root / "seed"
    build_dir = directory / "build" if nested else directory
    build_dir.mkdir(parents=True)
    source = "offline source fixture"
    (directory / "atomic_controller.cpp").write_text(source)
    (build_dir / "atomic_controller.cpp").write_text(source)
    plugin = build_dir / "candidate.so"
    plugin.write_bytes(b"not executed, fixture DSO")
    atomic_write_json(build_dir / "build.json", {"source_sha256": P.sha(source),
        "plugin_sha256": P.sha(plugin.read_bytes()), "optimization": "-O3"})
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "isolated_sim").write_bytes(b"fixture executable")
    (runtime / "libramulator.so").write_bytes(b"fixture library")
    atomic_write_json(root / "runtime_manifest.json", {"optimization": "-O3",
        "executable_sha256": P.sha((runtime / "isolated_sim").read_bytes()),
        "library_sha256": P.sha((runtime / "libramulator.so").read_bytes()),
        "synthetic_generator_sha256": P.sha((E.REPO / D.GENERATOR).read_bytes())})
    return {"source_path": str(directory / "atomic_controller.cpp"), "plugin": str(plugin), "sha256": P.sha(source)}


@pytest.mark.parametrize("nested", [False, True])
def test_exact_parent_source_plugin_and_runtime_binding(tmp_path, nested):
    L.install(tmp_path)
    parent = own_parent(tmp_path, nested)
    bound = D.parent_identity(tmp_path, parent)
    assert bound["source_sha256"] == parent["sha256"]
    pathlib.Path(parent["plugin"]).write_bytes(b"tampered DSO")
    with pytest.raises(RuntimeError, match="identity changed"):
        D.parent_identity(tmp_path, parent)


def test_other_run_and_escaping_symlinks_are_rejected(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    parent = own_parent(second)
    L.install(first)
    with pytest.raises(RuntimeError, match="escapes"):
        D.parent_identity(first, parent)
    (first / "seed").symlink_to(second / "seed", target_is_directory=True)
    parent["source_path"] = str(first / "seed/atomic_controller.cpp")
    parent["plugin"] = str(first / "seed/candidate.so")
    with pytest.raises(RuntimeError, match="escapes"):
        D.parent_identity(first, parent)


def test_runtime_and_generator_identity_are_checked(tmp_path):
    L.install(tmp_path)
    parent = own_parent(tmp_path)
    (tmp_path / "runtime/libramulator.so").write_bytes(b"changed trusted runtime")
    with pytest.raises(RuntimeError, match="runtime identity changed"):
        D.parent_identity(tmp_path, parent)


def frame(latencies, addresses=None):
    return pd.DataFrame({"source": [0] * len(latencies), "ordinal": list(range(len(latencies))),
        "addr": addresses or [100 + i for i in range(len(latencies))], "latency": latencies,
        "arrive": list(range(len(latencies))), "depart": [i + v for i, v in enumerate(latencies)]})


def test_metrics_use_all_oracle_reads_and_signed_paired_errors():
    records = {s: {"frontend_stats": {"cycles": v}} for s, v in (("oracle", 100), ("candidate", 110))}
    result = D.summarize(frame([10, 20, 30]), frame([5, 20, 40]), records, 1)
    assert result["L"] == 20
    assert result["request_mae_over_L"] == .25
    assert result["signed_drift_over_L"] == pytest.approx(5 / 3 / 20)
    assert result["signed_min_cycles"] == -5 and result["signed_max_cycles"] == 10
    assert result["paired_p99_over_L"] == pytest.approx(9.9 / 20)
    assert result["synthetic_elapsed_error_pct"] == 10
    assert result["slices"]["most_negative"][0]["ordinal"] == 0
    assert result["slices"]["most_positive"][0]["ordinal"] == 2
    assert "cycle_macro_mae_pct" not in result


@pytest.mark.parametrize("candidate", [frame([10]), frame([10, 20], [100, 999]), frame([])])
def test_incomplete_or_wrong_address_pairing_returns_no_score(candidate):
    with pytest.raises(RuntimeError, match="pairing failed"):
        D.summarize(frame([10, 20]), candidate, {}, 1)


def fake_simulations(monkeypatch):
    calls = []
    def fake(root, directory, parent, identity, side, dram, layout, lease_fd):
        if (directory / "manifest.json").exists():
            return D.verify_side(directory, identity, side)
        calls.append(side)
        staging = E.C.stage_run_directory(directory)
        n = identity["parameters"]["num_requests"]
        latency = 10 if side == "oracle" else 12
        # Reverse completion/file order to exercise admission-order matching.
        text = "arrive,depart,type,source,addr\n" + "".join(
            f"{i},{i+latency},0,0,{i % 10}\n" for i in reversed(range(n)))
        trace = staging / "controller_trace.csv.ch0"
        trace.write_text(text)
        E.C.publish_trace_and_manifest(trace, directory / trace.name, directory / "manifest.json", {
            "identity": identity, "side": side, "frontend_stats": {"cycles": n + latency,
                "reads_sent": n, "writes_sent": 0, "total_read_latency": n * latency}})
        staging.rmdir()
        return D.verify_side(directory, identity, side)
    monkeypatch.setattr(D, "run_side", fake)
    return calls


def test_agent_call_caches_exact_inputs_and_reads_compressed_evidence(tmp_path, monkeypatch):
    L.install(tmp_path)
    parent = own_parent(tmp_path)
    calls = fake_simulations(monkeypatch)
    request = {"tool": "synthetic_diagnostics", "cases": [{"num_requests": 1000}], "limit": 2}
    first = D.inspect(tmp_path, parent, request)["results"][0]
    assert not first["cached"] and first["matched_reads"] == 1000
    assert first["request_mae_over_L"] == .2
    second = D.inspect(tmp_path, parent, {**request, "limit": 1})["results"][0]
    assert second["cached"] and first["case_id"] == second["case_id"]
    assert calls == ["oracle", "candidate"]
    receipts = sorted((tmp_path / "diagnostics/synthetic_calls").glob("*.json"))
    assert len(receipts) == 2 and D.R.read_json(receipts[-1])["cached_cases"] == 1
    assert len(second["slices"]["first_reads"]) == 1
    case = tmp_path / "diagnostics/synthetic" / first["case_id"]
    assert (case / "oracle/controller_trace.csv.ch0.gz").exists()
    assert not (case / "oracle/controller_trace.csv.ch0").exists()
    assert E.verify_run_archives(tmp_path) == 2
    assert not (tmp_path / "training").exists() and not (tmp_path / "test").exists()
    atomic_write_json(case / "result.json", {"fabricated_score": 0})
    with pytest.raises(RuntimeError, match="report or evidence changed"):
        D.inspect(tmp_path, parent, request)


def test_case_quota_is_per_run_and_cached_calls_do_not_consume_it(tmp_path, monkeypatch):
    atomic_write_json(tmp_path / "profile.json", {"synthetic": {"max_unique_cases_per_run": 1}})
    L.install(tmp_path, tmp_path / "profile.json")
    parent = own_parent(tmp_path)
    calls = fake_simulations(monkeypatch)
    request = {"tool": "synthetic_diagnostics", "cases": [{"num_requests": 1000}]}
    D.inspect(tmp_path, parent, request)
    D.inspect(tmp_path, parent, request)
    with pytest.raises(ValueError, match="unique-case limit"):
        D.inspect(tmp_path, parent, {**request, "cases": [{"num_requests": 1000, "row_run": 1}]})
    assert len(calls) == 2


def test_native_failure_is_retained_and_never_published_as_score(tmp_path, monkeypatch):
    L.install(tmp_path)
    parent = own_parent(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("fixture simulator failure")
    monkeypatch.setattr(D, "run_side", fail)
    request = {"tool": "synthetic_diagnostics", "cases": [{}]}
    with pytest.raises(RuntimeError, match="fixture"):
        D.inspect(tmp_path, parent, request)
    cases = list((tmp_path / "diagnostics/synthetic").glob("*/identity.json"))
    assert len(cases) == 1
    assert not (cases[0].parent / "result.json").exists()
    with pytest.raises(ValueError, match="previously failed"):
        D.inspect(tmp_path, parent, request)


def test_disabled_tool_cannot_run_directly(tmp_path, monkeypatch):
    L.install(tmp_path, L.DEFAULT.with_name("loop_no_synthetic_v1.json"))
    monkeypatch.setattr(D, "parent_identity", lambda *a: pytest.fail("disabled tool accessed candidate"))
    with pytest.raises(ValueError, match="disabled"):
        D.inspect(tmp_path, {}, {"tool": "synthetic_diagnostics", "cases": [{}]})


def test_bad_archive_is_an_operational_failure_not_model_feedback(tmp_path, monkeypatch):
    L.install(tmp_path)
    parent = own_parent(tmp_path)
    calls = fake_simulations(monkeypatch)
    request = {"tool": "synthetic_diagnostics", "cases": [{"num_requests": 1000}]}
    result = D.inspect(tmp_path, parent, request)["results"][0]
    archive = tmp_path / "diagnostics/synthetic" / result["case_id"] / "oracle/controller_trace.csv.ch0.gz"
    archive.write_bytes(b"deliberately corrupted unit-test gzip fixture")
    with pytest.raises(D.R.OperationalPause, match="integrity verification"):
        D.inspect(tmp_path, parent, request)
    assert len(calls) == 2


def test_zero_latency_coalesced_write_is_valid_but_read_is_not(tmp_path):
    path = tmp_path / "controller_trace.csv.ch0"
    path.write_text("arrive,depart,type,source,addr\n0,0,1,0,8\n1,11,0,0,9\n")
    record = {"frontend_stats": {"reads_sent": 1, "writes_sent": 1, "total_read_latency": 10}}
    assert len(D.read_population(tmp_path, {"streams": 1, "num_requests": 1}, record)) == 1
    path.write_text("arrive,depart,type,source,addr\n0,0,1,0,8\n1,1,0,0,9\n")
    with pytest.raises(RuntimeError, match="latency"):
        D.read_population(tmp_path, {"streams": 1, "num_requests": 1}, record)
