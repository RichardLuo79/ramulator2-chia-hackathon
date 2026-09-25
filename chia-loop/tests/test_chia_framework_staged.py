"""Two-stage selection, numeric visibility and resume; no providers or simulators."""

import copy
import hashlib
import json
from contextlib import contextmanager

import numpy as np
import pytest
from test_chia_framework_campaign import FixtureResearch, configuration
from test_chia_framework_campaign import no_services as no_services
from test_chia_framework_scoring import paired

from ramulator_chia.framework.campaign import Campaign
from ramulator_chia.framework.config import CampaignConfig
from ramulator_chia.framework.identity import digest_json
from ramulator_chia.framework.review import (
    grouped_measurement,
    parse_decision,
    prompt_inventory,
    score_view,
)
from ramulator_chia.framework.scoring import score_workload
from ramulator_chia.eval.metrics import paired_error_statistics

pytestmark = pytest.mark.usefixtures("no_services")


def staged_config(rounds=(10, 5)):
    raw = configuration(review=False, iterations=sum(rounds)).model_dump(mode="json")
    traces, cases, groups = {}, {}, {}
    for group in ("training", "validation", "test"):
        programs = [f"{group}_program_{i}" for i in range(8)]
        for name in programs:
            digest = hashlib.sha256(name.encode()).hexdigest()
            traces[name] = dict(
                path=f"/private/{name}.gz", sha256=digest, decoded_sha256=digest, family=name
            )
        groups[group] = []
        for cores in (1, 4, 8):
            name = f"{group}_c{cores}"
            groups[group].append(name)
            cases[name] = dict(
                programs=programs[:cores],
                placement=dict(
                    path=f"/private/{name}.map.gz", sha256="1" * 64, decoded_sha256="2" * 64
                ),
            )
    raw["experiment"].update(
        stages=[
            dict(name="single_core", rounds=rounds[0], core_counts=[1]),
            dict(name="multicore", rounds=rounds[1], core_counts=[1, 4, 8]),
        ],
        promotion_policy="llm_review",
        prompt_sha256=prompt_inventory(),
        evaluation=dict(
            schema_version=2,
            name="staged-fixture",
            champsim=dict(
                **groups,
                traces=traces,
                cases=cases,
                builds={
                    str(n): dict(binary=f"/build/c{n}", source=f"/source/c{n}") for n in (1, 4, 8)
                },
                warmup_instructions=2_000_000,
                roi_instructions=20_000_000,
                minimum_oracle_owner_reads=1,
                request_objective="champsim_exact_physical_filter",
            ),
        ),
    )
    return CampaignConfig.model_validate(raw)


def decision(value="promote"):
    return dict(
        decision=value,
        rationale="Fixture trade-off judgment, not scientific evidence.",
        improvements=["lower tail"] if value == "promote" else [],
        accepted_regressions=[],
        contract_findings=[],
        contract_violation=False,
        uncertainty=[],
    )


class StagedResearch(FixtureResearch):
    def __init__(self, root, config):
        super().__init__(root, config)
        self.active_stage = None
        self.turns = []
        self.verdict = decision()
        self.bad_response = None
        self.repair_response = None
        self.fail_reflection = False
        self.unchanged = False
        self.inputs = []

    def activate_stage(self, stage):
        self.active_stage = stage

    def measured(self, candidate, group):
        experiment = self.configuration.experiment
        cohort = experiment.evaluation.champsim
        names = experiment.case_names(group, self.active_stage)
        directory = self.root / "pairs"
        directory.mkdir(exist_ok=True)
        matched = paired(directory, [100, 100], [130, 80])
        matched.update(
            trusted_pair_policy="champsim_exact_physical_filter",
            stable_logical_pairs=2,
            address_mismatch_pairs=0,
            physical_consistent_pairs=2,
            physical_consistency_rate=1.0,
        )
        rows = {}
        for name in names:
            cores = len(cohort.cases[name].programs)
            rows[name] = score_workload(
                {"per_core_cycles": [100] * cores, "controller_stats": {"num_read_reqs": 2}},
                {"per_core_cycles": [100 + candidate["parameters"]["error"]] * cores},
                matched,
                request_trace_scope="champsim_dram_controller_foreground_admissions",
                request_trace_timebase="ramulator_controller_cycles",
                minimum_oracle_owner_reads=1,
                population_verified=False,
                include_tails=True,
            )
        return dict(
            candidate=candidate,
            execution=self.configuration.execution,
            evaluation_sha256=digest_json(experiment.evaluation.model_dump(mode="json")),
            campaign_stage=self.active_stage.model_dump(mode="json"),
            measurement=grouped_measurement(
                rows,
                names=names,
                core_counts=self.active_stage.core_counts,
                cases=cohort.cases,
                stage=group,
            ),
        )

    def train(self, candidate):
        return self.measured(candidate, "training")

    def validate(self, candidate):
        return self.measured(candidate, "validation")

    @contextmanager
    def session(self, iteration, role, candidate, history, *, review_context=None):
        research = self
        draft = candidate

        class Session:
            def turn(self, phase, inputs):
                nonlocal draft
                research.turns.append((iteration, role, phase))
                research.inputs.append((role, phase, inputs))
                if phase == "explore":
                    if not research.unchanged:
                        # Intentionally worse core error: review, not Pareto, must decide.
                        draft = research.source(
                            research.root / f"draft-{iteration}", 50 + iteration
                        )
                if phase == "review":
                    assert not history and review_context is not None
                    return {"text": research.bad_response or json.dumps(research.verdict)}
                if phase == "review_format":
                    return {"text": research.repair_response or json.dumps(research.verdict)}
                if phase == "reflect" and research.fail_reflection:
                    raise RuntimeError("fixture reflection interrupted")
                return {"text": "done"}

            def snapshot(self):
                return draft

            def summary(self):
                return {"path": f"summary-{iteration}.md", "sha256": "f" * 64}

        yield Session()


def test_exact_ten_plus_five_and_two_frozen_endpoints(tmp_path):
    config = staged_config()
    research = StagedResearch(tmp_path, config)
    campaign = Campaign(tmp_path, config, research)
    result = campaign.run_search()
    assert result["iterations"] == 15
    endpoints = result["stage_selections"]
    assert endpoints["single_core"]["iteration"] == 10
    assert endpoints["single_core"]["candidate"]["parameters"]["error"] == 60
    assert endpoints["multicore"]["candidate"]["parameters"]["error"] == 65
    entry = campaign.state.get("stage-entry:multicore")
    assert entry["candidate"] == endpoints["single_core"]["candidate"]
    assert set(entry["training"]["measurement"]["groups"]) == {"1", "4", "8"}
    assert len([turn for turn in research.turns if turn[2] == "review"]) == 15
    assert all(turn[2] != "revise" for turn in research.turns)
    before = research.turns[:]
    assert campaign.run_search() == result
    assert research.turns == before
    tested = campaign.evaluate()
    assert set(tested["stage_endpoints"]) == {"single_core", "multicore"}
    for role, phase, inputs in research.inputs:
        text = json.dumps(inputs)
        assert "validation_program" not in text and "test_program" not in text
        assert "validation_c" not in text and "/private/" not in text
        assert "test_c" not in text

    from ramulator_chia.framework.staged_report import collect, markdown

    report = collect(tmp_path)
    assert len(report["rounds"]) == 15 and report["postrun"] is not None
    assert report["usage"]["upper_micro_usd"] is None  # Fixtures are not paid inference.
    assert "Completed rounds: 15/15" in markdown(report)


def test_ten_single_core_rounds_only_and_one_frozen_endpoint(tmp_path):
    raw = staged_config().model_dump(mode="json")
    raw["run"]["maximum_iterations"] = 10
    raw["experiment"]["stages"] = [dict(name="single_core", rounds=10, core_counts=[1])]
    cohort = raw["experiment"]["evaluation"]["champsim"]
    cohort["cases"] = {n: c for n, c in cohort["cases"].items() if len(c["programs"]) == 1}
    for group in ("training", "validation", "test"):
        cohort[group] = [n for n in cohort[group] if n in cohort["cases"]]
    programs = {p for c in cohort["cases"].values() for p in c["programs"]}
    cohort["traces"] = {n: t for n, t in cohort["traces"].items() if n in programs}
    cohort["builds"] = {"1": cohort["builds"]["1"]}
    config = CampaignConfig.model_validate(raw)
    research = StagedResearch(tmp_path, config)
    campaign = Campaign(tmp_path, config, research)
    result = campaign.run_search()
    assert result["iterations"] == 10
    assert set(result["stage_selections"]) == {"single_core"}
    assert result["stage_selections"]["single_core"]["iteration"] == 10
    assert campaign.state.get("stage-entry:multicore") is None
    assert len([t for t in research.turns if t[2] == "review"]) == 10
    before = research.turns[:]
    assert campaign.run_search() == result
    assert research.turns == before
    tested = campaign.evaluate()
    assert set(tested["stage_endpoints"]) == {"single_core"}
    assert set(config.experiment.evaluation.champsim.builds) == {"1"}
    assert all(len(c.programs) == 1 for c in config.experiment.evaluation.champsim.cases.values())
    for role, phase, inputs in research.inputs:
        text = json.dumps(inputs)
        assert "validation_program" not in text and "test_program" not in text
        assert "test_c" not in text and "/private/" not in text


def test_stage_transition_recovers_without_repeating_completed_rounds(tmp_path):
    config = staged_config((1, 1))
    research = StagedResearch(tmp_path, config)
    original = research.train
    interrupted = False

    def train(candidate):
        nonlocal interrupted
        if research.active_stage.name == "multicore" and not interrupted:
            interrupted = True
            raise RuntimeError("fixture measurement interruption")
        return original(candidate)

    research.train = train
    campaign = Campaign(tmp_path, config, research)
    with pytest.raises(RuntimeError, match="measurement interruption"):
        campaign.run_search()
    assert campaign.state.get("stage-selection:single_core") is not None
    campaign.run_search()
    assert research.turns.count((1, "proposer", "explore")) == 1
    assert research.turns.count((2, "reviewer", "review")) == 1


def reopen_fixture_round(campaign, *, reason="compiler_identity_changed", reviewed=False):
    """Mimic the separately audited operator transaction; never used by agents."""
    prefix = "superseded:compiler-recovery:"
    original = campaign.state.get("iteration:2")
    if reviewed:
        original = {**original, "promotion_review": decision("keep")}
    campaign.state.save(prefix + "iteration:2", original)
    for key in ("iteration:2", "selection", "stage-selection:multicore"):
        if key != "iteration:2":
            campaign.state.save(prefix + key, campaign.state.get(key))
        campaign.state._write("DELETE FROM records WHERE key=?", (key,))
    campaign.state._write(
        "UPDATE attempts SET step=? || step WHERE step LIKE 'iteration:2:%'", (prefix,)
    )
    campaign.state.save("operational-recovery:2", {
        "reason": reason, "superseded_iteration_key": prefix + "iteration:2",
        "private_evidence": "/private/TEST_EVIDENCE_MUST_STAY_PRIVATE",
    })
    return original


def test_authorized_recovery_keeps_previous_round_and_hides_private_evidence(tmp_path):
    research = StagedResearch(tmp_path, staged_config((1, 1)))
    research.unchanged = True
    campaign = Campaign(tmp_path, research.configuration, research)
    campaign.run_search()
    first = campaign.state.get("iteration:1")
    original = reopen_fixture_round(campaign)
    research.unchanged = False
    result = Campaign(tmp_path, research.configuration, research).run_search()
    assert result["iterations"] == 2
    assert campaign.state.get("iteration:1") == first
    assert campaign.state.get("superseded:compiler-recovery:iteration:2") == original
    assert research.turns.count((1, "proposer", "explore")) == 1
    assert research.turns.count((2, "proposer", "explore")) == 2
    assert research.turns.count((2, "reviewer", "review")) == 1
    recovered = [inputs for _, phase, inputs in research.inputs
                 if phase == "explore" and "operational_recovery" in inputs]
    assert len(recovered) == 1
    assert "same round" in recovered[0]["operational_recovery"]
    assert "TEST_EVIDENCE_MUST_STAY_PRIVATE" not in json.dumps(research.inputs)
    before = research.turns[:]
    assert campaign.run_search() == result
    assert research.turns == before


@pytest.mark.parametrize("reason,reviewed", [
    ("arbitrary_new_attempt", False), ("compiler_identity_changed", True),
])
def test_recovery_notice_cannot_reroll_a_judgment(tmp_path, reason, reviewed):
    from ramulator_chia.framework.records import RecordConflict

    research = StagedResearch(tmp_path, staged_config((1, 1)))
    research.unchanged = True
    campaign = Campaign(tmp_path, research.configuration, research)
    campaign.run_search()
    reopen_fixture_round(campaign, reason=reason, reviewed=reviewed)
    before = research.turns[:]
    with pytest.raises(RecordConflict, match="unchanged, unreviewed"):
        campaign.run_search()
    assert research.turns == before


@pytest.mark.parametrize("unchanged, invalid", [(True, False), (False, True)])
def test_mechanical_failure_and_unchanged_source_skip_llm_judgment(tmp_path, unchanged, invalid):
    research = StagedResearch(tmp_path, staged_config((1, 1)))
    research.unchanged = unchanged
    original_check = research.check
    research.check = lambda candidate: {
        **original_check(candidate),
        "passed": not invalid or candidate["parameters"]["error"] == 50,
    }
    result = Campaign(tmp_path, research.configuration, research).run_search()
    assert result["candidate"]["parameters"]["error"] == 50
    assert not any(turn[1] == "reviewer" for turn in research.turns)


def test_rejection_is_persisted_not_retried(tmp_path):
    research = StagedResearch(tmp_path, staged_config((1, 1)))
    research.verdict = decision("keep")
    campaign = Campaign(tmp_path, research.configuration, research)
    result = campaign.run_search()
    assert result["candidate"]["parameters"]["error"] == 50
    assert len([t for t in research.turns if t[2] == "review"]) == 2
    assert campaign.state.get("iteration:1:decision")["decision"] == "keep"


def test_only_one_format_repair_even_after_resume(tmp_path):
    research = StagedResearch(tmp_path, staged_config((1, 1)))
    research.bad_response = "not JSON"
    research.repair_response = "still not JSON"
    campaign = Campaign(tmp_path, research.configuration, research)
    for _ in range(2):
        with pytest.raises(ValueError):
            campaign.run_search()
    assert [t[2] for t in research.turns].count("review") == 1
    assert [t[2] for t in research.turns].count("review_format") == 1
    assert campaign.state.get("iteration:1") is None


def test_format_repair_cannot_reverse_an_explicit_rejection(tmp_path):
    research = StagedResearch(tmp_path, staged_config((1, 1)))
    research.bad_response = '{"decision":"keep"}'
    research.repair_response = json.dumps(decision("promote"))
    campaign = Campaign(tmp_path, research.configuration, research)
    with pytest.raises(ValueError, match="changed the original decision"):
        campaign.run_search()
    assert campaign.state.get("iteration:1:decision") is None


def test_physical_mismatch_cannot_be_waived_by_reviewer(tmp_path):
    research = StagedResearch(tmp_path, staged_config((1, 1)))
    original = research.measured

    def mismatch(candidate, group):
        value = original(candidate, group)
        if candidate["parameters"]["error"] != 50:
            for section in value["measurement"]["groups"].values():
                for row in section["workloads"].values():
                    row["request"]["address_mismatch_pairs"] = 1
        return value

    research.measured = mismatch
    # Receipts are checked/recomputed before asking the reviewer.
    with pytest.raises(ValueError):
        Campaign(tmp_path, research.configuration, research).run_search()
    assert not any(turn[1] == "reviewer" for turn in research.turns)


def test_anonymous_tails_are_numeric_and_aliases_stable_across_stages(tmp_path):
    research = StagedResearch(tmp_path, staged_config((1, 1)))
    candidate = research.prepare()["seed"]
    names = research.configuration.experiment.evaluation.validation
    for stage in research.configuration.experiment.stages:
        research.activate_stage(stage)
        receipt = research.validate(candidate)
        row = receipt["measurement"]["groups"]["1"]["workloads"]["validation_c1"]
        row["raw_path"] = "/private/validation_program_0"
        row["request"]["unknown_field"] = "VALIDATION_CANARY"
        view = score_view(receipt, names, anonymous=True)
        assert "val1" in view["groups"]["1"]["cases"]
        encoded = json.dumps(view)
        assert (
            "validation_" not in encoded and "CANARY" not in encoded and "/private" not in encoded
        )
        assert "p999_over_L" in encoded


def test_prompt_mutation_cannot_silently_resume(tmp_path, monkeypatch):
    from ramulator_chia.framework import review

    research = StagedResearch(tmp_path, staged_config((1, 1)))
    campaign = Campaign(tmp_path, research.configuration, research)
    campaign.run_search()
    monkeypatch.setattr(review, "prompt_inventory", lambda: {"task.md": "changed"})
    with pytest.raises(ValueError, match="prompt"):
        campaign.run_search()


def test_population_tail_statistics_and_zero_error():
    stats = paired_error_statistics(np.array([-600, 0, 50, 150]), 100, include_tails=True)
    assert stats["mae"] == 2
    assert stats["tail_thresholds"]["1L"]["fraction"] == 0.5
    assert stats["tail_thresholds"]["5L"]["absolute_error_share"] == 0.75
    assert stats["p999_over_L"] == pytest.approx(np.percentile([600, 0, 50, 150], 99.9) / 100)
    zero = paired_error_statistics(np.zeros(4), 100, include_tails=True)
    assert zero["tail_thresholds"]["1L"]["absolute_error_share"] == 0


def test_decision_schema_does_not_guess_or_accept_contradictions():
    assert parse_decision(json.dumps(decision("keep")))["decision"] == "keep"
    with pytest.raises(ValueError):
        parse_decision("I think we should promote this.")
    wrong = decision()
    wrong["contract_violation"] = True
    with pytest.raises(ValueError):
        parse_decision(json.dumps(wrong))


def test_schedule_and_train_validation_family_separation():
    raw = staged_config().model_dump(mode="json")
    bad = copy.deepcopy(raw)
    bad["run"]["maximum_iterations"] = 20
    with pytest.raises(ValueError, match="sum of stage rounds"):
        CampaignConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["experiment"]["evaluation"]["champsim"]["traces"]["validation_program_0"]["family"] = (
        "training_program_0"
    )
    with pytest.raises(ValueError, match="family"):
        CampaignConfig.model_validate(bad)


def test_final_review_workspace_is_readonly_and_has_no_proposer_history(tmp_path):
    from ramulator_chia.framework.build import MODEL_API_HEADER
    from ramulator_chia.framework.snapshots import publish_bytes
    from ramulator_chia.framework.workspace import Workspace

    research = StagedResearch(tmp_path, staged_config((1, 1)))
    research.runtime = tmp_path / "runtime"
    research.resources = research.configuration.run.resources
    publish_bytes(research.runtime / "export" / MODEL_API_HEADER, b"// fixture API\n")
    seed = research.prepare()["seed"]
    research.activate_stage(research.configuration.experiment.stages[0])
    view = Workspace(
        research,
        1,
        "reviewer",
        seed,
        [],
        review_context={
            "incumbent": seed,
            "comparison": {"validation": {"val1": {"core_error_pct": 1.0}}},
        },
    )
    view.set_phase("review")
    assert set(view.methods()) == {"files", "read"}
    assert view.grants()["write"] == []
    assert view.read("references/incumbent/model.cpp") == view.read("draft/model.cpp")
    assert "val1" in view.read("references/comparison.json")["text"]
    assert not (view.root / "references/history").exists()
    assert (view.root / "references/prompts/promotion_review.md").is_file()
    with pytest.raises(PermissionError):
        view.write("draft/model.cpp", "// forbidden")
    with pytest.raises(PermissionError):
        view.evaluate_training()
    with pytest.raises(PermissionError):
        view.set_phase("revise")
