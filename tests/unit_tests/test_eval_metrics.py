import json

import numpy as np
import pytest

from tools.eval import matchlib, metrics, postprocess, run_simpleo3

TRACE_HEADER = (
    "arrive,depart,type,source,addr,frontend_id,frontend_sub_id,"
    "admission_ordinal\n")


def _trace(path, rows):
    path.write_text(TRACE_HEADER + "\n".join(rows) + "\n")


def test_request_metrics_define_L_from_full_oracle_read_population(tmp_path):
    oracle = tmp_path / "oracle.csv.ch0"
    model = tmp_path / "model.csv.ch0"
    _trace(oracle, [
        "0,10,0,0,64,1,0,0",
        "0,90,0,0,128,2,0,1",
    ])
    _trace(model, [
        "0,20,0,0,64,1,0,0",
        "0,110,0,0,128,2,0,1",
    ])

    result = postprocess.request_metrics(oracle, model)

    assert result["L_oracle_mean_lat"] == 50.0
    assert result["bulk_mae_over_L"] == 0.3
    assert result["signed_mean_over_L"] == 0.3
    assert result["extreme_min_cyc"] == 10
    assert result["extreme_max_cyc"] == 20


def test_closed_loop_scoring_rejects_incomplete_stable_coverage(tmp_path):
    oracle = tmp_path / "oracle.csv.ch0"
    model = tmp_path / "model.csv.ch0"
    _trace(oracle, [
        "0,10,0,0,64,1,0,0",
        "0,20,0,0,128,2,0,1",
    ])
    _trace(model, ["0,12,0,0,64,1,0,0"])

    with pytest.raises(ValueError, match="exact bidirectional"):
        postprocess.request_metrics(oracle, model)


def test_checksum_bound_metric_cache_recomputes_after_raw_change(tmp_path):
    oracle = tmp_path / "oracle.csv.ch0"
    model = tmp_path / "model.csv.ch0"
    output = tmp_path / "request.json"
    _trace(oracle, ["0,10,0,0,64,1,0,0"])
    _trace(model, ["0,10,0,0,64,1,0,0"])
    calls = []

    def matcher(*_):
        calls.append(None)
        delta = 10 * len(calls)
        return {
            "dv": np.array([delta]),
            "all_o": np.array([10]),
            "cov_o": 1.0,
            "cov_m": 1.0,
            "n_o": 1,
            "n_m": 1,
            "matcher_schema_version": matchlib.MATCHER_SCHEMA_VERSION,
            "match_mode": "stable_id",
            "stable_eligible_o": 1,
            "stable_eligible_m": 1,
            "stable_cov_o": 1.0,
            "stable_cov_m": 1.0,
        }

    first = metrics.load_or_compute_request(
        output, model, oracle, matcher=matcher,
        request_trace_scope=run_simpleo3.REQUEST_TRACE_SCOPE,
        request_trace_timebase=run_simpleo3.REQUEST_TRACE_TIMEBASE)
    _trace(model, ["0,11,0,0,64,1,0,0"])
    second = metrics.load_or_compute_request(
        output, model, oracle, matcher=matcher,
        request_trace_scope=run_simpleo3.REQUEST_TRACE_SCOPE,
        request_trace_timebase=run_simpleo3.REQUEST_TRACE_TIMEBASE)

    assert len(calls) == 2
    assert first["mae"] == 1.0
    assert second["mae"] == 2.0
    assert json.loads(output.read_text())["mae"] == 2.0
