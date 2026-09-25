"""ChampSim objectives and validation boundaries; no inference or simulator calls."""

import copy
import gzip
import json

import pytest
from test_chia_framework_campaign import FixtureResearch, configuration
from test_chia_framework_transfer_requests import compare, measurement, partial_pair

from ramulator_chia.framework.campaign import Campaign
from ramulator_chia.framework.config import CampaignConfig
from ramulator_chia.framework.identity import digest_json
from ramulator_chia.framework.scoring import aggregate, should_promote, validation_view

POLICY = "champsim_exact_physical_filter"


def search_config():
    record = configuration().model_dump(mode="json")
    record["experiment"]["validation_non_worsening"] = True
    record["experiment"]["evaluation"] = {
        "schema_version": 2, "name": "champsim-fixture",
        "champsim": {
            "training": ["train"], "validation": ["held-out"], "test": [],
            "warmup_instructions": 2_000_000, "roi_instructions": 20_000_000,
            "minimum_oracle_owner_reads": 1, "request_objective": POLICY,
            "traces": {
                name: {"path": "/private/" + name + ".gz", "sha256": digit * 64,
                       "decoded_sha256": digit * 64, "family": name}
                for name, digit in (("train", "1"), ("held-out", "2"))
            },
        },
    }
    return CampaignConfig.model_validate(record)


def test_partial_population_is_an_explicit_opt_in_not_a_global_relaxation(tmp_path):
    row = compare(*partial_pair(tmp_path))
    row["minimum_oracle_owner_reads"] = 1
    rows = {"train": row}
    full = aggregate(rows, expected_workloads=["train"], stage="training")
    partial = aggregate(rows, expected_workloads=["train"], stage="training", request_objective=POLICY)
    assert not full["eligible_for_promotion"]
    assert partial["eligible_for_promotion"]
    assert partial["aggregate"]["request_macro_mae_over_L"] == .3
    assert not row["request_headline_eligible"] and not row["request_population_verified"]
    assert row["request_pairing"]["oracle"]["coverage"] == .4
    row["minimum_oracle_owner_reads"] = 10_000
    screened = aggregate(rows, expected_workloads=["train"], stage="training", request_objective=POLICY)
    assert not screened["eligible_for_promotion"]
    assert screened["aggregate"]["request_macro_mae_over_L"] is None


def test_validation_projection_is_only_two_scores_and_aliases(tmp_path):
    row = compare(*partial_pair(tmp_path))
    row["minimum_oracle_owner_reads"] = 1
    report = aggregate({"secret": row}, expected_workloads=["secret"], stage="validation",
                       request_objective=POLICY)
    view = validation_view({"measurement": report, "failure": "/private/secret.csv"}, ("secret",))
    assert view == {
        "aggregate": {"core_error_pct": 10., "request_mae_over_L": .3},
        "workloads": {"val1": {"core_error_pct": 10., "request_mae_over_L": .3}},
    }
    missing = validation_view({"measurement": None, "failure": "secret"}, ("secret",))
    assert missing["workloads"]["val1"] == {"core_error_pct": None, "request_mae_over_L": None}
    assert "secret" not in json.dumps(missing)


def test_config_hides_input_identities_and_rejects_family_or_input_overlap():
    config = search_config()
    view = config.experiment.agent_view()
    assert "held-out" not in json.dumps(view) and "/private" not in json.dumps(view)
    assert view["validation_non_worsening"]
    for field in ("family", "decoded_sha256"):
        record = config.model_dump(mode="json")
        traces = record["experiment"]["evaluation"]["champsim"]["traces"]
        traces["held-out"][field] = traces["train"][field]
        with pytest.raises(ValueError, match="share a family or decoded input"):
            CampaignConfig.model_validate(record)


class ChampSimFixture(FixtureResearch):
    def evaluate(self, candidate, stage):
        self.calls.append(stage)
        error = candidate["parameters"]["error"]
        # First proposal improves training but regresses validation. The second
        # improves both. Complete raw validation evidence stays in state only.
        measured_error = 60 if stage == "validation" and error == 20 else error
        root = self.root / "measurements" / stage / candidate["candidate_id"]
        oracle = measurement(root, "oracle", ["0,10,0,0,64,1,0,0", "0,10,0,0,128,2,0,1"])
        model = measurement(root, "candidate", [
            f"0,{10 + measured_error},0,0,64,1,0,0", "0,10,0,0,256,2,0,1",
        ])
        row = compare(oracle, model)
        row["minimum_oracle_owner_reads"] = 1
        row["cycles"]["mean_abs_per_core_pct"] = float(measured_error)
        name = "train" if stage == "training" else "held-out"
        return {
            "candidate": candidate, "execution": "offline_fixture",
            "evaluation_sha256": digest_json(self.configuration.experiment.evaluation.model_dump(mode="json")),
            "measurement": aggregate({name: row}, expected_workloads=[name], stage=stage,
                                     request_objective=POLICY),
        }

    def train(self, candidate):
        return self.evaluate(candidate, "training")

    def validate(self, candidate):
        return self.evaluate(candidate, "validation")


def test_shared_campaign_validation_gate_feedback_and_resume(tmp_path):
    config = search_config()
    research = ChampSimFixture(tmp_path, config)
    campaign = Campaign(tmp_path, config, research)
    selected = campaign.run_search()
    first = campaign.state.get("iteration:1")
    assert not first["promoted"]
    assert first["validation"]["aggregate"]["core_error_pct"] == 60
    assert campaign.state.get("iteration:2")["promoted"]
    assert selected["candidate"]["parameters"]["error"] == 10
    assert research.calls.count("validation") == 3  # seed + one final draft each
    for session in research.sessions:
        assert "held-out" not in json.dumps(session.history)
    private = campaign.state.completed_step("validation:" + selected["candidate"]["candidate_id"])
    assert "held-out" in private["measurement"]["workloads"]
    calls = research.calls[:]
    assert campaign.run_search() == selected
    assert research.calls == calls


@pytest.mark.parametrize("regressed", ["cycle_macro_mae_pct", "request_macro_mae_over_L"])
def test_either_validation_regression_blocks_training_improvement(tmp_path, regressed):
    row = compare(*partial_pair(tmp_path))
    row["minimum_oracle_owner_reads"] = 1
    incumbent = aggregate({"train": row}, expected_workloads=["train"], stage="training",
                          request_objective=POLICY)
    candidate = copy.deepcopy(incumbent)
    candidate["aggregate"]["cycle_macro_mae_pct"] /= 2
    validation = copy.deepcopy(incumbent)
    validation["stage"] = "validation"
    assert should_promote(candidate, incumbent, mechanically_valid=True,
                          candidate_validation=validation, incumbent_validation=validation)
    bad = copy.deepcopy(validation)
    bad["aggregate"][regressed] += 1e-10  # No rounding or hidden tolerance.
    assert not should_promote(candidate, incumbent, mechanically_valid=True,
                              candidate_validation=bad, incumbent_validation=validation)


def test_gzip_instruction_inventory_is_checked_without_expansion(tmp_path):
    from ramulator_chia.framework.external_frontends import describe_champsim_trace, TransferCase

    class Host:
        receipt_sha256 = "a" * 64

        def record(self):
            return {"frontend": "champsim", "settings": {"instruction_record_bytes": 64}}

    path = tmp_path / "trace.gz"
    path.write_bytes(gzip.compress(bytes(64 * 3)))
    payload, inventory = describe_champsim_trace(path, Host())
    assert inventory["complete_compressed_integrity"] and inventory["instructions"] == 3
    assert inventory["compression"] == "gzip" and not inventory["complete_xz_integrity"]
    with pytest.raises(ValueError, match="full window"):
        TransferCase("champsim", "tiny", payload, instruction_inventory=inventory, stage="training")
    damaged = bytearray(path.read_bytes())
    damaged[-8] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(gzip.BadGzipFile):
        describe_champsim_trace(path, Host())
