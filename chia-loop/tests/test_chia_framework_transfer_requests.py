"""ChampSim pairing/reporting from saved observations; no simulator or LLM calls."""

import gzip
import json
from dataclasses import asdict

import pytest

from ramulator_chia.framework import measurement_reports, scoring
from ramulator_chia.framework.archive import describe_payload
from ramulator_chia.framework.identity import digest_json, file_sha256
from ramulator_chia.eval import matchlib

HEADER = "arrive,depart,type,source,addr,frontend_id,frontend_sub_id,admission_ordinal\n"


def measurement(root, model, rows, *, frontend="champsim", compressed=True):
    directory = root / model
    directory.mkdir(parents=True)
    name = "controller.csv.ch0" + (".gz" if compressed else "")
    data = (HEADER + "\n".join(rows) + "\n").encode()
    (directory / name).write_bytes(gzip.compress(data) if compressed else data)
    member = describe_payload(
        directory / name, name, codec="gzip" if compressed else "none"
    ).member
    case = {"frontend": frontend, "stage": "transfer", "workload": "fixture"}
    receipt = {
        "complete": True, "execution": "native", "model": model,
        "case": case, "case_sha256": digest_json(case), "runtime_sha256": "runtime",
        "host": {"kind": frontend, "receipt_sha256": "host"},
        "core_metric": "ROI core cycles" if frontend == "champsim" else "simTicks",
        "per_core_cycles": [100 if model == "oracle" else 110],
        "controller_stats": {"observed_read_records": sum(row.split(",")[2] == "0" for row in rows)},
        "request_population_verified": False,
        "traces": {"controller.csv.ch0": asdict(member)}, "files": {},
    }
    (directory / "measurement.json").write_text(json.dumps(receipt))
    return directory, file_sha256(directory / "measurement.json")


def compare(oracle, model, *, frontend="champsim"):
    return measurement_reports.compare_transfer(
        oracle[0], model[0], oracle_receipt_sha256=oracle[1], model_receipt_sha256=model[1],
        frontend=frontend, minimum_oracle_owner_reads=10_000,
    )


def partial_pair(root):
    oracle = measurement(root, "oracle", [
        "0,10,0,0,64,101,0,0", "0,30,0,0,128,102,0,1",
        "0,50,0,0,192,103,0,2", "0,70,0,0,256,104,0,3",
        "0,90,0,0,320,-1,0,4",
    ], compressed=False)
    model = measurement(root, "candidate", [
        "0,20,0,0,128,102,0,0",  # reordered ID: delta -10
        "0,30,0,0,64,101,0,1",   # delta +20
        "0,1,0,0,960,103,0,2",   # same logical ID, different physical address
        "0,60,0,0,384,105,0,3",  # no oracle counterpart
        "0,40,0,0,320,-1,0,4",   # ineligible despite matching address
        "0,-1,1,0,448,106,0,5",  # write: not in the read population
    ])
    return oracle, model


def test_reordered_partial_pairs_normalization_tails_and_population_accounting(tmp_path):
    result = compare(*partial_pair(tmp_path))
    request = result["request"]
    assert request["match_mode"] == "stable_id"
    assert request["trusted_pair_policy"] == "champsim_exact_physical_filter"
    assert request["oracle_read_mean_latency"] == 50  # all five oracle reads, not just pairs
    assert request["matched"] == 2
    assert request["mae"] == .3
    assert request["sgn"] == .1
    assert request["tail"] == pytest.approx(19.9 / 50)
    assert request["extreme_min_cycles"] == -10
    assert request["extreme_max_cycles"] == 20
    counts = result["request_pairing"]
    assert counts["stable_logical_pairs"] == 3
    assert counts["address_mismatch_pairs"] == 1
    for side in ("oracle", "model"):
        assert counts[side] == {
            "recorded_reads": 5, "matched_reads": 2, "unmatched_reads": 3,
            "coverage": .4, "without_stable_id": 1, "stable_id_without_counterpart": 1,
        }
        assert counts[side]["unmatched_reads"] == (
            counts[side]["without_stable_id"]
            + counts[side]["stable_id_without_counterpart"] + counts["address_mismatch_pairs"]
        )
    assert not result["request_headline_eligible"]
    assert result["cycles"]["mean_abs_per_core_pct"] == 10


def test_diagnostics_keep_low_traffic_cases_and_never_relax_promotion(tmp_path):
    one = compare(*partial_pair(tmp_path / "one"))
    two = compare(
        measurement(tmp_path / "two", "oracle", ["0,10,0,0,64,1,0,0"]),
        measurement(tmp_path / "two", "candidate", ["0,20,0,0,64,1,0,0"]),
    )
    report = scoring.aggregate({"one": one, "two": two}, expected_workloads=["one", "two"], stage="transfer")
    diagnostic = report["matched_request_diagnostics"]
    assert diagnostic["summary"]["mean_mae"] == .65  # equal workloads, not equal requests
    assert diagnostic["summary"]["mean_abs_sgn"] == .55
    assert diagnostic["summary"]["mean_paired_p99_over_L"] == pytest.approx((.398 + 1) / 2)
    assert diagnostic["summary"]["min_cov_o"] == .4
    assert diagnostic["below_minimum_read_workloads"] == ["one", "two"]
    assert diagnostic["unavailable_workloads"] == []
    assert report["aggregate"]["request_macro_mae_over_L"] is None
    assert not report["eligible_for_promotion"]
    assert not diagnostic["eligible_for_promotion"]
    with pytest.raises(ValueError, match="training measurements"):
        scoring.valid_objectives(report)


@pytest.mark.parametrize("model_row,logical,mismatches", [
    ("0,20,0,0,64,2,0,0", 0, 0),  # no shared ID
    ("0,20,0,0,128,1,0,0", 1, 1),  # shared ID but wrong physical address
])
def test_zero_pairs_are_unavailable_not_zero_error_or_a_dropped_workload(tmp_path, model_row, logical, mismatches):
    bad = compare(
        measurement(tmp_path / "bad", "oracle", ["0,10,0,0,64,1,0,0"]),
        measurement(tmp_path / "bad", "candidate", [model_row]),
    )
    assert bad["request"] is None
    assert bad["request_pairing"]["stable_logical_pairs"] == logical
    assert bad["request_pairing"]["address_mismatch_pairs"] == mismatches
    good = compare(*partial_pair(tmp_path / "good"))
    report = scoring.aggregate({"good": good, "bad": bad}, expected_workloads=["good", "bad"], stage="transfer")
    assert report["matched_request_diagnostics"]["summary"] is None
    assert report["matched_request_diagnostics"]["unavailable_workloads"] == ["bad"]
    assert report["aggregate"]["cycle_macro_mae_pct"] == 10


@pytest.mark.parametrize("rows,message", [
    (["0,10,0,0,64,1,0,0", "0,20,0,0,64,1,0,1"], "duplicate stable request identity"),
    (["0,10,0,0,64,-1,0,0"], "requires eligible stable IDs"),
])
def test_invalid_identity_is_reported_without_legacy_fallback(tmp_path, rows, message):
    result = compare(measurement(tmp_path, "oracle", rows), measurement(tmp_path, "candidate", rows))
    assert result["request"] is None
    assert message in result["pairing_error"]
    assert result["request_pairing"]["oracle"]["without_stable_id"] is None


@pytest.mark.parametrize("rows", [
    ["10,9,0,0,64,1,0,0"],  # negative read latency
    ["0,10,0,0,64,1,0,0", "0,10,0,0,128,2,0,0"],  # duplicate admission ordinal
])
def test_malformed_observations_fail_instead_of_becoming_core_only(tmp_path, rows):
    with pytest.raises(ValueError):
        compare(measurement(tmp_path, "oracle", rows), measurement(tmp_path, "candidate", rows))


@pytest.mark.parametrize("field,value", [
    ("stable_logical_pairs", 2), ("address_mismatch_pairs", 0),
    ("physical_consistent_pairs", 3), ("physical_consistency_rate", 1.0),
])
def test_physical_pair_accounting_is_validated_before_aggregation(tmp_path, field, value):
    result = compare(*partial_pair(tmp_path))
    result["request"][field] = value
    with pytest.raises(ValueError):
        scoring.aggregate({"one": result}, expected_workloads=["one"], stage="transfer")


def test_default_matcher_still_rejects_physical_address_divergence(tmp_path):
    oracle, model = partial_pair(tmp_path)
    with pytest.raises(matchlib.RequestIdentityError, match="different addresses"):
        matchlib.match(oracle[0] / "controller.csv.ch0", model[0] / "controller.csv.ch0.gz")


def test_pairing_distinguishes_cpu_and_transaction_sub_id(tmp_path):
    result = compare(
        measurement(tmp_path, "oracle", [
            "0,10,0,0,64,1,0,0", "0,20,0,1,64,1,0,1", "0,30,0,0,128,1,1,2",
        ]),
        measurement(tmp_path, "candidate", [
            "0,40,0,0,128,1,1,0", "0,30,0,1,64,1,0,1", "0,20,0,0,64,1,0,2",
        ]),
    )
    assert result["request"]["matched"] == 3
    assert result["request"]["mae"] == .5
    assert result["request"]["sgn"] == .5


def test_broken_gzip_is_not_downgraded_to_an_identity_error(tmp_path):
    oracle, model = partial_pair(tmp_path)
    path = model[0] / "controller.csv.ch0.gz"
    data = bytearray(path.read_bytes())
    data[-8] ^= 1  # damage the gzip CRC while leaving its compressed payload intact
    path.write_bytes(data)
    with pytest.raises(gzip.BadGzipFile, match="CRC check failed"):
        compare(oracle, model)


@pytest.mark.parametrize("change", ["trace", "case", "runtime"])
def test_corrupt_or_incompatible_evidence_is_not_reused(tmp_path, change):
    oracle, model = partial_pair(tmp_path)
    if change == "trace":
        path = model[0] / "controller.csv.ch0.gz"
        path.write_bytes(gzip.compress(b"changed trace\n"))
    else:
        path = model[0] / "measurement.json"
        receipt = json.loads(path.read_text())
        if change == "case":
            receipt["case"]["workload"] = "different"
            receipt["case_sha256"] = digest_json(receipt["case"])
        else:
            receipt["runtime_sha256"] = "different"
        path.write_text(json.dumps(receipt))
        model = model[0], file_sha256(path)
    with pytest.raises(ValueError):
        compare(oracle, model)


def test_gem5_comparison_never_calls_the_request_matcher(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("gem5 request matching must stay disabled")

    monkeypatch.setattr(matchlib, "match", forbidden)
    rows = ["0,10,0,0,64,1,0,0"]
    result = compare(
        measurement(tmp_path, "oracle", rows, frontend="gem5"),
        measurement(tmp_path, "candidate", rows, frontend="gem5"), frontend="gem5",
    )
    assert result["request"] is None
    assert result["request_matching"].startswith("disabled:")
    report = scoring.aggregate({"one": result}, expected_workloads=["one"], stage="transfer")
    assert report["aggregate"]["cycle_macro_mae_pct"] == 10
    assert "matched_request_diagnostics" not in report
