"""Operational migration controls do not change the experiment or native history."""

import copy

import pytest

from ramulator_chia.framework.config import ExecutionOverrides, VertexBackend, preserved_configuration
from ramulator_chia.framework.identity import digest_json
from ramulator_chia.framework.model_sessions import create_session
from test_chia_framework_campaign import configuration
from test_chia_framework_model_adapters import response, vertex_fixture


@pytest.mark.parametrize("workers", [1, 16, 120])
def test_worker_override_leaves_frozen_configuration_and_limits_unchanged(workers):
    config = configuration()
    original = config.model_dump(mode="json")
    override = ExecutionOverrides(evaluation_workers=workers)
    override.validate_backend(config)
    assert override.workers(config) == workers
    assert preserved_configuration(config, original) == original
    assert config.model_dump(mode="json") == original
    value = copy.deepcopy(original)
    value["run"]["resources"]["memory_bytes"] += 1
    changed = type(config).model_validate(value)
    with pytest.raises(ValueError, match="configuration changed"):
        preserved_configuration(changed, original)


@pytest.mark.parametrize("workers", [0, -1, 121])
def test_worker_override_has_a_real_envelope(workers):
    with pytest.raises(ValueError):
        ExecutionOverrides(evaluation_workers=workers)


def test_wrong_backend_cannot_use_a_vertex_route():
    with pytest.raises(ValueError, match="Vertex backend"):
        ExecutionOverrides(vertex_project="cloud-project").validate_backend(configuration())


def test_vertex_route_restores_the_same_native_session(tmp_path, monkeypatch):
    fixture = vertex_fixture(monkeypatch, [response()])
    profile = VertexBackend(kind="vertex_gemini", model="gemini-3.8-flash",
                            reasoning_effort="high", project="original-project", location="global")
    private = tmp_path / "private"
    private.mkdir()

    def session(project=None):
        return create_session(
            profile, session_id="unchanged:12:proposer", private_directory=private,
            workspace=tmp_path, system_message="unchanged contract", timeout_seconds=10800,
            vertex_client_kwargs={}, vertex_event_callback=lambda e, s: None,
            vertex_execution_project=project,
        )

    original = session()
    result = original.prompt_once("existing conversation", [])
    checkpoint = original.checkpoint(result)
    moved = session("fixture-alternate-project")
    assert moved.settings == original.settings
    assert digest_json(moved.settings) == checkpoint.metadata["settings_sha256"]
    assert moved.model.project == "fixture-alternate-project"
    assert original.model.project == "original-project"
    moved.restore(checkpoint)
    assert moved.checkpoint(result).files == checkpoint.files
    assert len(fixture.calls) == 1  # Restoring does not generate another response.


@pytest.mark.parametrize("project", ["", "https://elsewhere", "../project", "INVALID"])
def test_malformed_billing_destination_is_rejected(project):
    config = configuration().model_copy(update={"backend": VertexBackend(
        kind="vertex_gemini", model="fixture", reasoning_effort="high",
        project="original-project", location="global")})
    with pytest.raises(ValueError, match="project ID"):
        ExecutionOverrides(vertex_project=project).validate_backend(config)


def test_expiry_defers_only_new_rounds(tmp_path):
    from ramulator_chia.framework.records import CampaignState
    from ramulator_chia.framework.execution import HostDeadlineReached, round_admission
    state = CampaignState(tmp_path / "state.sqlite")
    admit = round_admission(state, 10 * 3600, now=lambda: 0)
    admit(1, None)
    with pytest.raises(HostDeadlineReached):
        round_admission(state, 5 * 3600, now=lambda: 0)(1, None)
    state._write("INSERT INTO attempts(step,number,input_sha256,status) VALUES(?,?,?,?)",
                 ("iteration:1:explore", 1, "existing", "started"))
    round_admission(state, 5 * 3600, now=lambda: 0)(1, None)
    assert state._query("SELECT status FROM attempts")[0]["status"] == "started"


def test_expiry_estimate_accounts_for_observed_waiting(tmp_path):
    from ramulator_chia.framework.records import CampaignState
    from ramulator_chia.framework.execution import HostDeadlineReached, round_admission
    state = CampaignState(tmp_path / "state.sqlite")
    state.save("iteration:1", {"completed": True})
    state._write("INSERT INTO attempts(step,number,input_sha256,status,started_at,finished_at) "
                 "VALUES(?,?,?,?,?,?)", ("iteration:1:explore", 1, "done", "complete",
                 "2026-09-21T00:00:00Z", "2026-09-21T10:00:00Z"))
    with pytest.raises(HostDeadlineReached, match="15.0 hours"):
        round_admission(state, 14 * 3600, now=lambda: 0)(2, None)
