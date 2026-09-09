"""Protocol fixtures, not accuracy results; no model or simulator is launched."""

import json
from contextlib import contextmanager

import pytest
from test_chia_framework_scoring import paired, score

from tools.chia_loop.framework.campaign import Campaign
from tools.chia_loop.framework.config import CampaignConfig
from tools.chia_loop.framework.identity import digest_json, file_sha256
from tools.chia_loop.framework.records import (
    CampaignState,
    RecordConflict,
    TransientFailure,
    UnresolvedStep,
)
from tools.chia_loop.framework.scoring import aggregate
from tools.chia_loop.framework.snapshots import ModelFiles, snapshot
from tools.eval.metrics import cycles_metrics

FILES = ModelFiles(("model.cpp",), "parameters.json")


def configuration(*, review=True, iterations=2):
    return CampaignConfig.model_validate(
        {
            "campaign_id": "protocol-fixture",
            "backend": {"kind": "fixture", "scenario": "two-improvements"},
            "run": {
                "maximum_iterations": iterations,
                "evaluation_execution": "offline_fixture",
                "maximum_attempts": 3,
                "retry_delay_seconds": 0,
            },
            "experiment": {
                "semantic_llm_check": review,
                "evaluation": {
                    "name": "protocol-only",
                    "simpleo3": {
                        "training": ["train"],
                        "test": ["held-out"],
                        "instructions_per_core": 20_000_000,
                        "minimum_oracle_owner_reads": 1,
                    },
                },
            },
        }
    )


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    import socket
    import subprocess

    import ray
    from chia.trace import profiler

    def denied(*args, **kwargs):
        raise AssertionError("protocol tests cannot launch a process or call a service")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(ray, "init", denied)
    monkeypatch.setattr(profiler, "get_collector", lambda namespace=None: None)
    profiler.reset_profiler()
    yield
    profiler.reset_profiler()


class FixtureResearch:
    def __init__(self, root, config):
        self.root, self.configuration = root, config
        self.calls, self.sessions = [], []
        self.fail_phase = None
        self.invalid = False
        self.adjust_report = lambda report: report

    def source(self, directory, error):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.cpp").write_text("// Inert protocol fixture, never compiled.\n")
        (directory / "parameters.json").write_text(json.dumps({"error": error}))
        return snapshot(directory, self.root / "candidates", FILES, maximum_bytes=10_000)

    def prepare(self):
        self.calls.append("prepare")
        return {"seed": self.source(self.root / "seed", 50)}

    def check(self, candidate):
        self.calls.append("check")
        return {"candidate": candidate, "passed": not self.invalid}

    def train(self, candidate):
        self.calls.append("train")
        error = candidate["parameters"]["error"]
        directory = self.root / "measurements" / candidate["candidate_id"]
        directory.mkdir(parents=True, exist_ok=True)
        row = score(paired(directory, [100, 100], [100 + error] * 2), model_cycles=[100 + error])
        measured = aggregate({"train": row}, expected_workloads=["train"], stage="training")
        return self.adjust_report(
            {
                "candidate": candidate,
                "execution": "offline_fixture",
                "measurement": measured,
                "evaluation_sha256": digest_json(
                    self.configuration.experiment.evaluation.model_dump(mode="json")
                ),
            }
        )

    def postrun(self, candidate):
        self.calls.append("postrun")
        return {"candidate": candidate, "execution": "offline_fixture", "not_accuracy_data": True}

    @contextmanager
    def session(self, iteration, role, candidate, history):
        instance = FixtureSession(self, iteration, role, candidate, history)
        self.sessions.append(instance)
        yield instance


class FixtureSession:
    def __init__(self, research, number, role, candidate, history):
        self.research, self.number, self.role = research, number, role
        self.history, self.parent = history[:], candidate
        self.phases = []
        self.directory = research.root / "sessions" / str(number) / role
        self.directory.mkdir(parents=True, exist_ok=True)

    def turn(self, phase, inputs):
        self.phases.append(phase)
        self.research.calls.append(phase)
        assert "held-out" not in json.dumps(inputs)
        if self.research.fail_phase == phase:
            raise RuntimeError("simulated lost reply")
        if phase == "explore":
            self.research.source(self.directory, 20 // self.number)
        if phase == "reflect":
            self.research.invalid = False
            (self.directory / "summary.md").write_text(
                f"Iteration {self.number}: changed the fixture; evaluated; "
                f"promoted={inputs['promoted']}. Next: inspect remaining errors.\n"
            )
        return {
            "text": "Advisory fixture: reject this design.",
            "session": f"{self.number}:{self.role}",
        }

    def snapshot(self):
        return snapshot(
            self.directory, self.research.root / "candidates", FILES, maximum_bytes=10_000
        )

    def summary(self):
        path = self.directory / "summary.md"
        return {"path": str(path.relative_to(self.research.root)), "sha256": file_sha256(path)}


def setup(tmp_path, *, review=True, iterations=2):
    config = configuration(review=review, iterations=iterations)
    research = FixtureResearch(tmp_path, config)
    return Campaign(tmp_path, config, research), research


def test_common_loop_review_continuity_summary_promotion_and_no_archive_gate(tmp_path):
    campaign, research = setup(tmp_path)
    with pytest.raises(ValueError, match="freeze"):
        campaign.evaluate()
    selected = campaign.run_search()
    assert selected["iterations"] == 2 and selected["execution"] == "offline_fixture"
    assert selected["candidate"]["parameters"] == {"error": 10}
    proposers = [s for s in research.sessions if s.role == "proposer"]
    reviewers = [s for s in research.sessions if s.role == "reviewer"]
    assert [s.phases for s in proposers] == [["explore", "revise", "reflect"]] * 2
    assert [s.phases for s in reviewers] == [["review"]] * 2
    assert not proposers[0].history and len(proposers[1].history) == 1
    assert all(not s.history for s in reviewers)
    assert campaign.state.get("iteration:1")["promoted"]
    assert campaign.state.get("iteration:2")["promoted"]
    assert all((tmp_path / h["summary"]["path"]).is_file() for h in proposers[1].history)
    assert not list(tmp_path.glob("*.zip"))
    evaluated = campaign.evaluate()
    assert evaluated["candidate"] == selected["candidate"]
    calls = research.calls[:]
    assert Campaign(tmp_path, campaign.config, research).run_search() == selected
    assert campaign.evaluate() == evaluated
    assert research.calls == calls


def test_local_checkpoints_do_not_start_the_chia_profiler_or_cluster(tmp_path, monkeypatch):
    from chia.trace import profiler

    def denied():
        raise AssertionError("local database access must not start profiling/Ray")

    monkeypatch.setattr(profiler, "get_profiler", denied)
    state = CampaignState(tmp_path / "local.sqlite")
    state.save("fixture", {"value": 1})
    assert state.get("fixture") == {"value": 1}
    assert state.completed_step("missing") is None


def test_disabled_review_never_opens_reviewer_session(tmp_path):
    campaign, research = setup(tmp_path, review=False)
    campaign.run_search()
    assert len(research.sessions) == 2
    assert all(s.phases == ["explore", "reflect"] for s in research.sessions)


def test_lower_request_error_cannot_hide_a_core_regression(tmp_path):
    campaign, research = setup(tmp_path)

    def tradeoff(result):
        if result["candidate"]["parameters"]["error"] != 50:
            row = result["measurement"]["workloads"]["train"]
            row["cycles"] = cycles_metrics({"per_core_cycles": [100]}, {"per_core_cycles": [160]})
            result["measurement"] = aggregate(
                {"train": row}, expected_workloads=["train"], stage="training"
            )
        return result

    research.adjust_report = tradeoff
    selected = campaign.run_search()
    assert selected["candidate"]["parameters"] == {"error": 50}
    for number in (1, 2):
        outcome = campaign.state.get(f"iteration:{number}")
        assert not outcome["promoted"] and outcome["summary"]


def test_failed_candidate_check_skips_training_but_still_records_summary(tmp_path):
    campaign, research = setup(tmp_path, iterations=1)
    original = research.check

    def invalid_draft(candidate):
        result = original(candidate)
        result["passed"] = candidate["parameters"]["error"] == 50
        return result

    research.check = invalid_draft
    selected = campaign.run_search()
    assert selected["candidate"]["parameters"] == {"error": 50}
    assert research.calls.count("train") == 1
    outcome = campaign.state.get("iteration:1")
    assert outcome["training"] is None and not outcome["promoted"]
    assert outcome["summary"]


def test_runtime_invalid_candidate_is_feedback_not_an_endless_retry(tmp_path):
    campaign, research = setup(tmp_path)

    def runtime_failure(result):
        if result["candidate"]["parameters"]["error"] == 20:
            result["measurement"] = None
            result["failure"] = {"kind": "candidate_runtime", "reason": "invalid departure"}
        return result

    research.adjust_report = runtime_failure
    selected = campaign.run_search()
    failed = campaign.state.get("iteration:1")
    assert not failed["promoted"] and failed["summary"]
    assert failed["training"]["failure"]["reason"] == "invalid departure"
    assert selected["candidate"]["parameters"] == {"error": 10}


def test_resume_does_not_repeat_completed_proposal_or_ambiguous_review(tmp_path):
    campaign, research = setup(tmp_path)
    research.fail_phase = "review"
    with pytest.raises(RuntimeError, match="lost reply"):
        campaign.run_search()
    calls = research.calls[:]
    research.fail_phase = None
    with pytest.raises(UnresolvedStep, match="iteration:1:review"):
        campaign.run_search()
    assert research.calls == calls
    assert campaign.state.get("selection") is None


@pytest.mark.parametrize(
    "defect", ["candidate", "execution", "stage", "workload", "traffic", "headline"]
)
def test_wrong_training_evidence_cannot_promote(tmp_path, defect):
    campaign, research = setup(tmp_path)

    def alter(result):
        if defect == "candidate":
            result["candidate"] = {}
        elif defect == "execution":
            result["execution"] = "native"
        elif defect == "stage":
            result["measurement"]["stage"] = "test"
        elif defect == "workload":
            result["measurement"]["workloads"] = {}
        elif defect == "traffic":
            result["measurement"]["workloads"]["train"]["minimum_oracle_owner_reads"] = 99
        else:
            result["measurement"]["aggregate"]["cycle_macro_mae_pct"] = 0
        return result

    research.adjust_report = alter
    with pytest.raises(ValueError):
        campaign.run_search()
    assert campaign.state.get("selection") is None
    assert not research.sessions


def test_configuration_and_session_identity_cannot_change_on_resume(tmp_path):
    campaign, research = setup(tmp_path)
    campaign.run_search()
    changed = configuration(iterations=3)
    with pytest.raises(ValueError, match="different campaign"):
        Campaign(tmp_path, changed, research)
    another = FixtureResearch(tmp_path, changed)
    with pytest.raises(RecordConflict, match="configuration"):
        Campaign(tmp_path, changed, another).run_search()


def test_state_retries_are_bounded_and_safe_recovery_is_explicit(tmp_path):
    state = CampaignState(tmp_path / "state.sqlite")
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise TransientFailure("known not dispatched")
        return {"done": True}

    settings = {"maximum_attempts": 2, "retry_delay_seconds": 0}
    assert state.step("one", {}, flaky, **settings) == {"done": True}
    assert state.step("one", {}, flaky, **settings) == {"done": True}
    assert len(calls) == 2
    with pytest.raises(RecordConflict):
        state.step("one", {"changed": True}, flaky, **settings)

    def ambiguous():
        raise RuntimeError("interrupted computation")

    with pytest.raises(RuntimeError):
        state.step("two", {}, ambiguous, **settings)
    with pytest.raises(UnresolvedStep):
        state.step("two", {}, flaky, **settings)
    assert state.step("two", {}, flaky, repeat_after_interruption=True, **settings)["done"]


def test_state_retry_limit_survives_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    calls = []

    def fail():
        calls.append(1)
        raise TransientFailure("safe but still unavailable")

    settings = {"maximum_attempts": 2, "retry_delay_seconds": 0}
    with pytest.raises(TransientFailure):
        CampaignState(path).step("one", {}, fail, **settings)
    with pytest.raises(UnresolvedStep, match="limit"):
        CampaignState(path).step("one", {}, fail, **settings)
    assert len(calls) == 2


def test_retry_delay_is_applied_only_between_attempts(tmp_path, monkeypatch):
    from tools.chia_loop.framework import records

    delays = []
    now = [0]

    def sleep(seconds):
        delays.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(records.time, "sleep", sleep)
    monkeypatch.setattr(records.time, "time", lambda: now[0])

    def fail():
        raise TransientFailure("known safe retry")

    with pytest.raises(TransientFailure):
        CampaignState(tmp_path / "state.sqlite").step(
            "one", {}, fail, maximum_attempts=3, retry_delay_seconds=7
        )
    assert delays == [7, 7]


def test_capacity_cooldown_survives_restart_and_preserves_attempt_limit(tmp_path, monkeypatch):
    from tools.chia_loop.framework import records

    now, calls, waited = [1000], [], []
    monkeypatch.setattr(records.time, "time", lambda: now[0])

    def fail():
        calls.append(now[0])
        raise TransientFailure("selected model is at capacity", retry_after_seconds=3600)

    def interrupted_sleep(seconds):
        now[0] += seconds
        raise KeyboardInterrupt

    monkeypatch.setattr(records.time, "sleep", interrupted_sleep)
    state = CampaignState(tmp_path / "state.sqlite")
    options = dict(maximum_attempts=2, retry_delay_seconds=5)
    with pytest.raises(KeyboardInterrupt):
        state.step("review", {}, fail, **options)
    rows = state._query("SELECT * FROM attempts")
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert json.loads(rows[0]["error"])["retry_not_before"] == 4600

    now[0] = 2200  # Restart twenty minutes after the failed provider call.

    def sleep(seconds):
        assert 0 < seconds <= 55
        waited.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(records.time, "sleep", sleep)
    resumed = CampaignState(tmp_path / "state.sqlite")
    with pytest.raises(TransientFailure):
        resumed.step("review", {}, fail, **options)
    assert sum(waited) == 2400 and calls == [1000, 4600]
    with pytest.raises(UnresolvedStep, match="limit"):
        resumed.step("review", {}, fail, **options)
    assert len(calls) == 2
