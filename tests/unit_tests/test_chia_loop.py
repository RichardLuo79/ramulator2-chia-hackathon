import json
import pathlib

import pytest

from tools.chia_loop import artifacts, core


def _training_result(label, cycle_mae, request_mae, signed=0.1, L=50.0):
    return {
        "candidate_label": label,
        "proposal": {"candidate": {"latency": 42}},
        "summary": {
            "models": {
                label: {
                    "aggregate": {
                        "cycle_macro_mae_pct": cycle_mae,
                        "request_macro_mae_over_L": request_mae,
                    },
                    "per_workload": {
                        "a": {
                            "requests": {
                                "signed_mean_over_L": signed,
                                "L_oracle_mean_lat": L,
                            }
                        },
                        "b": {
                            "requests": {
                                "signed_mean_over_L": signed * 2,
                                "L_oracle_mean_lat": L,
                            }
                        },
                    },
                }
            }
        },
    }


def test_dummy_agent_uses_bounded_explainable_intercept_correction():
    prior = _training_result("candidate_iter_000", 5.0, 0.4)

    proposal = core.dummy_proposal(1, {"latency": 42}, prior, 0.10)

    # Residuals are +5 and +10 cycles; rounded median asks for -8 cycles, but
    # the 10% trust bound limits the latency reduction to four cycles.
    assert proposal["candidate"] == {"latency": 38}
    assert proposal["change_kind"] == "bounded_intercept_correction"
    assert proposal["evidence"]["requested_latency_step_cycles"] == 8
    assert proposal["evidence"]["applied_latency_step_cycles"] == 4
    assert "constant latency intercept" in proposal["principle"]


def test_selection_never_hides_a_tradeoff_in_a_scalar_score():
    incumbent = _training_result("candidate_iter_000", 5.0, 0.4)
    tradeoff = _training_result("candidate_iter_001", 4.0, 0.5)
    dominant = _training_result("candidate_iter_002", 4.0, 0.3)

    tradeoff_selection = core.select_candidate([incumbent, tradeoff])
    dominant_selection = core.select_candidate([incumbent, dominant])

    assert tradeoff_selection["selected_label"] == "candidate_iter_000"
    assert dominant_selection["selected_label"] == "candidate_iter_002"
    assert tradeoff_selection["policy"] == (
        "incumbent_retention_unless_pareto_dominated")


def test_smoke_config_keeps_all_immediate_response_comparisons(tmp_path):
    source = core.load_config(
        pathlib.Path(__file__).resolve().parents[2]
        / "tools/chia_loop/smoke.json")
    source["comparison_models"] = ["fixedlat", "md1", "mess"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="FixedLat, MD1, WMG1, and MESS"):
        core.load_config(path)


def test_auxiliary_archive_round_trip_and_verification(tmp_path):
    raw = tmp_path / "logs" / "large.log"
    raw.parent.mkdir()
    payload = ("principled agent record\n" * 100).encode()
    raw.write_bytes(payload)
    manifest = tmp_path / "artifacts" / "aux.json"

    result = artifacts.compress(tmp_path, manifest, min_bytes=1)
    artifacts.verify(manifest)

    assert not raw.exists()
    assert (tmp_path / result["artifacts"]["logs/large.log"]["archive_path"]).is_file()
    artifacts.restore(manifest, ["logs/large.log"])
    assert raw.read_bytes() == payload
