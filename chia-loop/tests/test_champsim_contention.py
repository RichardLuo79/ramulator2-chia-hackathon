"""Continuous contention: measurement boundaries must not follow background work."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ramulator_chia.eval import matchlib, metrics
from ramulator_chia.eval.champsim_contention import select_cases, REPLACEMENTS
from ramulator_chia.eval.champsim_mixes import MultiProgramCase
from ramulator_chia.eval.champsim_placement import (
    progress,
    record_stop,
    infrastructure_failure,
)
from test_champsim_mixes import trace

HEADER = (
    "arrive,depart,type,source,addr,frontend_id,frontend_sub_id,admission_ordinal\n"
)


def test_foreground_window_keeps_late_departure_and_excludes_background_normalizer(
    tmp_path,
):
    oracle = tmp_path / "oracle.csv"
    model = tmp_path / "model.csv"
    oracle.write_text(
        HEADER + "0,10,0,0,64,10,0,0\n0,30,0,0,128,-1,0,1\n1,100001,0,0,192,1000,0,2\n"
    )
    model.write_text(
        HEADER + "0,20,0,0,64,10,0,0\n0,40,0,0,128,-1,0,1\n1,200001,0,0,192,1000,0,2\n"
    )
    window = [dict(core=0, begin=0, end=2)]
    paired = matchlib.match(oracle, model, oracle_windows=window, model_windows=window)
    assert paired["all_o"].tolist() == [10, 30]
    assert paired["dv"].tolist() == [10]
    assert paired["cov_o"] == 0.5
    assert (
        metrics.paired_error_statistics(paired["dv"], paired["all_o"].mean())["mae"]
        == 0.5
    )


def test_window_is_per_source_and_uses_admission_not_record_order(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(
        HEADER + "0,9,0,1,192,2,0,2\n0,10,0,0,64,0,0,0\n0,90,0,0,128,4,0,3\n"
    )
    frame, _ = matchlib._load_frame(
        path, windows=[dict(core=0, begin=0, end=1), dict(core=1, begin=0, end=4)]
    )
    assert sorted(frame["admission_ordinal"]) == [0, 2]


def test_malformed_background_is_not_hidden(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(HEADER + "0,10,0,0,64,0,0,0\n10,9,0,0,128,2,0,2\n")
    with pytest.raises(ValueError, match="including background"):
        matchlib._load_frame(path, windows=[dict(core=0, begin=0, end=1)])


def test_single_core_and_policy_have_distinct_case_identities():
    a = trace("a", "a")
    finite = MultiProgramCase(
        frontend="champsim",
        workload="one",
        stage="test",
        payload=a.payload,
        instruction_inventory=a.instruction_inventory,
        companions=(),
    )
    continuous = replace(finite, completion_policy="background-replay")
    assert continuous.cores == 1
    assert continuous.identity() != finite.identity()
    assert continuous.identity()["allow_trace_wrap"] == "background_only"
    assert continuous.identity()["request_window"] == "per-core-admission-ordinal-v1"


def test_terminal_failures_survive_top_level_exception(tmp_path):
    progress(
        tmp_path,
        phase="production_complete",
        terminal=True,
        active=0,
        queued=0,
        failures={"case/model": "preserved simulator failure"},
    )
    record_stop(tmp_path, RuntimeError("some measurements failed"))
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["terminal"] is True
    assert state["failures"] == {"case/model": "preserved simulator failure"}


def test_supervisor_interruption_is_not_completion(tmp_path):
    progress(tmp_path, phase="production", terminal=False, active=2, queued=10)
    record_stop(tmp_path, KeyboardInterrupt())
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["terminal"] is False
    assert state["phase"] == "stopped"
    assert state["error_type"] == "KeyboardInterrupt"


def test_retry_is_limited_to_identified_infrastructure():
    assert infrastructure_failure(RuntimeError("WorkerCrashedError"))
    assert not infrastructure_failure(RuntimeError("NO_PROGRESS core 2"))
    assert not infrastructure_failure(RuntimeError("simulator returned signal 11"))
