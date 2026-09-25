"""Metric parity, full-population eligibility and strict Pareto regression tests."""

import numpy as np
import pytest

from ramulator_chia import pareto as established_pareto
from ramulator_chia.framework import scoring
from ramulator_chia.eval import matchlib, metrics, postprocess


def paired(tmp_path, oracle_latencies, model_latencies, *, model_ids=None):
    header = "arrive,depart,type,source,addr,frontend_id,frontend_sub_id,admission_ordinal\n"
    oracle, model = tmp_path / "oracle.csv", tmp_path / "model.csv"
    oracle.write_text(
        header
        + "".join(
            f"0,{latency},0,0,{(index + 1) * 64},{index},0,{index}\n"
            for index, latency in enumerate(oracle_latencies)
        )
    )
    ids = model_ids if model_ids is not None else range(len(model_latencies))
    model.write_text(
        header
        + "".join(
            f"0,{latency},0,0,{(identity + 1) * 64},{identity},0,{index}\n"
            for index, (identity, latency) in enumerate(zip(ids, model_latencies))
        )
    )
    return matchlib.match(oracle, model)


def score(result, oracle_cycles=(100,), model_cycles=(110,), *, minimum=1, owner_reads=None):
    oracle = {
        "per_core_cycles": list(oracle_cycles),
        "controller_stats": {
            "num_read_reqs": result["n_o"] if owner_reads is None else owner_reads
        },
    }
    return scoring.score_workload(
        oracle,
        {"per_core_cycles": list(model_cycles)},
        result,
        request_trace_scope="simpleo3_logical_llc",
        request_trace_timebase="simpleo3_frontend_cycles",
        minimum_oracle_owner_reads=minimum,
    )


def report(cycle, request, *, eligible=True):
    return {
        "eligible_for_promotion": eligible,
        "precision": "unrounded_binary64",
        "stage": "training",
        "aggregate": {"cycle_macro_mae_pct": cycle, "request_macro_mae_over_L": request},
    }


def test_full_L_signed_cancellation_paired_percentile_and_extremes(tmp_path):
    result = paired(tmp_path, [10, 90], [20, 80])
    scored = score(result)
    request = scored["request"]
    assert scored["request_headline_eligible"] is True
    assert request["oracle_read_mean_latency"] == 50
    assert request["mae"] == 0.2
    assert request["sgn"] == 0
    assert request["tail"] == 0.2
    assert request["percentile_method"] == "linear"
    assert request["extreme_min_cycles"] == -10
    assert request["extreme_max_cycles"] == 10
    assert request["extreme_min_over_L"] == -0.2
    assert request["extreme_max_over_L"] == 0.2
    assert request["paired_p99_cycles"] != abs(
        np.percentile(result["all_m"], 99) - np.percentile(result["all_o"], 99)
    )


def test_partial_match_is_conditional_and_L_still_uses_all_oracle_reads(tmp_path):
    scored = score(paired(tmp_path, [10, 100], [20]))
    assert scored["request"]["oracle_read_mean_latency"] == 55
    assert scored["request"]["mae"] == pytest.approx(10 / 55)
    assert scored["request_headline_eligible"] is False
    summary = scoring.aggregate({"one": scored}, expected_workloads=["one"], stage="training")
    assert summary["aggregate"]["request_macro_mae_over_L"] is None
    assert summary["aggregate"]["cycle_macro_mae_pct"] == 10
    assert summary["eligible_for_promotion"] is False
    assert not scoring.should_promote(summary, report(50, 50), mechanically_valid=True)


def test_zero_trusted_pairs_does_not_become_zero_error(tmp_path):
    scored = score(paired(tmp_path, [10, 100], [20], model_ids=[999]))
    assert scored["request"] is None
    assert scored["request_populations"]["matched"] == 0
    assert not scored["request_headline_eligible"]


def test_minimum_read_population_is_a_real_eligibility_guard(tmp_path):
    scored = score(paired(tmp_path, [10], [10]), minimum=10_000)
    assert scored["request"]["mae"] == 0
    assert not scored["request_headline_eligible"]
    assert any("declared minimum" in reason for reason in scored["ineligibility_reasons"])


def test_logical_llc_hits_cannot_satisfy_the_dram_traffic_guard(tmp_path):
    pair = paired(tmp_path, [10] * 100, [10] * 100)
    scored = score(pair, owner_reads=2, minimum=10)
    assert scored["request"]["n_oracle"] == 100
    assert scored["oracle_owner_reads"] == 2
    assert scored["request"]["oracle_read_mean_latency"] == 10
    assert not scored["request_headline_eligible"]
    assert any("declared minimum" in reason for reason in scored["ineligibility_reasons"])


def test_macro_metrics_have_equal_workload_weight_and_cannot_drop_a_case(tmp_path):
    pair = paired(tmp_path, [10, 90], [20, 80])
    one = score(pair, oracle_cycles=(100, 200), model_cycles=(120, 220))
    two = score(pair, oracle_cycles=(100,), model_cycles=(100,))
    summary = scoring.aggregate(
        {"one": one, "two": two}, expected_workloads=["one", "two"], stage="training"
    )
    assert summary["aggregate"]["cycle_macro_mae_pct"] == 7.5
    assert summary["aggregate"]["request_macro_mae_over_L"] == 0.2
    assert summary["aggregate"]["request_macro_absolute_signed_drift_over_L"] == 0
    with pytest.raises(ValueError, match="exactly"):
        scoring.aggregate({"one": one}, expected_workloads=["one", "two"], stage="training")
    two["observation"]["scope"] = "controller_transaction"
    with pytest.raises(ValueError, match="separate reports"):
        scoring.aggregate(
            {"one": one, "two": two}, expected_workloads=["one", "two"], stage="training"
        )


def test_heldout_metrics_cannot_be_promoted_even_if_numerically_good(tmp_path):
    scored = score(paired(tmp_path, [10, 90], [10, 90]))
    summary = scoring.aggregate({"one": scored}, expected_workloads=["one"], stage="test")
    assert summary["aggregate"]["request_macro_mae_over_L"] == 0
    assert not summary["eligible_for_promotion"]
    assert not scoring.should_promote(summary, report(50, 50), mechanically_valid=True)
    forged = {**summary, "eligible_for_promotion": True}
    with pytest.raises(ValueError, match="training measurements"):
        scoring.should_promote(forged, report(50, 50), mechanically_valid=True)


def test_aggregate_rechecks_the_recorded_eligibility(tmp_path):
    scored = score(paired(tmp_path, [10, 100], [20]))
    scored["request_headline_eligible"] = True
    with pytest.raises(ValueError, match="contradicts"):
        scoring.aggregate({"one": scored}, expected_workloads=["one"], stage="training")


def test_presentation_rounding_is_not_used_for_selection():
    incumbent, better = report(1.000049, 0.1000049), report(1.000048, 0.1000048)
    assert round(incumbent["aggregate"]["cycle_macro_mae_pct"], 4) == round(
        better["aggregate"]["cycle_macro_mae_pct"], 4
    )
    assert scoring.should_promote(better, incumbent, mechanically_valid=True)
    slightly_worse_cycles = report(1.0000491, 0.09)
    assert not scoring.should_promote(slightly_worse_cycles, incumbent, mechanically_valid=True)
    rounded = {**better, "precision": "display_rounded"}
    with pytest.raises(ValueError, match="unrounded"):
        scoring.should_promote(rounded, incumbent, mechanically_valid=True)


def test_cycle_extraction_preserves_legacy_report_rounding():
    oracle, model = {"per_core_cycles": [999999]}, {"per_core_cycles": [1000000]}
    raw, display = metrics.cycles_metrics(oracle, model), postprocess.cycles_metrics(oracle, model)
    assert raw["mean_abs_per_core_pct"] != display["mean_abs_per_core_pct"]
    for key, value in raw.items():
        expected = (
            [round(item, 4) for item in value] if isinstance(value, list) else round(value, 4)
        )
        assert display[key] == expected


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ((9, 9), True),
        ((10, 9), True),
        ((9, 10), True),
        ((10, 10), False),
        ((11, 1), False),
        ((1, 11), False),
    ],
)
def test_promotion_preserves_established_strict_pareto_rule(candidate, expected):
    assert established_pareto.dominates(candidate, (10, 10)) == expected
    assert (
        scoring.should_promote(report(*candidate), report(10, 10), mechanically_valid=True)
        == expected
    )
    assert not scoring.should_promote(report(*candidate), report(10, 10), mechanically_valid=False)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, None, True])
def test_invalid_objectives_cannot_enter_selection(bad):
    with pytest.raises(ValueError, match="finite nonnegative"):
        scoring.should_promote(report(bad, 0), report(10, 10), mechanically_valid=True)


def test_parent_selection_is_separate_from_incumbent_promotion():
    candidates = {
        "incumbent": {"metrics": report(10, 10)},
        "tradeoff": {"metrics": report(1, 11)},
        "dominated": {"metrics": report(20, 20)},
    }
    assert (
        scoring.select_parent(candidates, "incumbent", iteration=2, policy="incumbent")
        == "incumbent"
    )
    assert (
        scoring.select_parent(candidates, "incumbent", iteration=2, policy="pareto_round_robin")
        == "tradeoff"
    )
    assert established_pareto.pareto(candidates) == ["incumbent", "tradeoff"]
