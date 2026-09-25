"""Settings and positive projection for the replacement workflow."""

import json
import runpy
from pathlib import Path

import pytest
from test_chia_framework_campaign import configuration

from ramulator_chia.framework.config import CampaignConfig, load


@pytest.mark.parametrize("value", [True, 0, -1, "2"])
def test_iteration_count_is_strict(value):
    record = configuration().model_dump(mode="json")
    record["run"]["maximum_iterations"] = value
    with pytest.raises(ValueError):
        CampaignConfig.model_validate(record)


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_semantic_check_is_a_boolean(value):
    record = configuration().model_dump(mode="json")
    record["experiment"]["semantic_llm_check"] = value
    with pytest.raises(ValueError):
        CampaignConfig.model_validate(record)


def test_training_projection_omits_hidden_cohorts_and_backend():
    config = configuration()
    view = config.experiment.agent_view()
    assert view["training"] == ["train"]
    assert "held-out" not in json.dumps(view)
    assert "backend" not in view and "transfer" not in view


def test_scripted_proposer_and_native_evaluation_are_separate_settings():
    record = configuration().model_dump(mode="json")
    record["run"]["evaluation_execution"] = "native"
    config = CampaignConfig.model_validate(record)
    assert config.backend.kind == "fixture" and config.execution == "native"


@pytest.mark.parametrize("field", ["cache_mode", "episode_timeout_seconds", "parent_selection"])
def test_unimplemented_modes_are_not_accepted(field):
    record = configuration().model_dump(mode="json")
    record["run"][field] = "ignored"
    with pytest.raises(ValueError):
        CampaignConfig.model_validate(record)


def test_full_windows_and_application_family_split_remain_required():
    record = configuration().model_dump(mode="json")
    simple = record["experiment"]["evaluation"]["simpleo3"]
    simple["instructions_per_core"] = 1000
    with pytest.raises(ValueError):
        CampaignConfig.model_validate(record)
    simple["instructions_per_core"] = 20_000_000
    simple["training"], simple["test"] = ["429.mcf"], ["505.mcf_r"]
    with pytest.raises(ValueError, match="families overlap"):
        CampaignConfig.model_validate(record)


def test_duplicate_json_settings_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"campaign_id":"a","campaign_id":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        load(path)


def test_scripted_command_uses_exact_config_without_silent_overrides(tmp_path):
    command = runpy.run_path(str(Path(__file__).parent / "utils/check_chia_clean_campaign.py"))
    record = configuration().model_dump(mode="json")
    record["backend"]["scenario"] = "comment-only-mcp"
    record["run"]["evaluation_execution"] = "native"
    record["run"]["resources"]["cpus"] = 3
    record["experiment"]["semantic_llm_check"] = False
    record["experiment"]["features"]["synthetic_diagnostics"] = False
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(record))
    result = command["settings"](campaign_config=path)
    assert result.model_dump(mode="json") == record
    with pytest.raises(ValueError, match="not both"):
        command["settings"](campaign_config=path, workers=6)
    record["backend"] = {"kind": "codex_cli", "model": "fixture", "reasoning_effort": "high"}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scripted agents"):
        command["settings"](campaign_config=path)


def test_archive_options_are_checked_before_preparing_a_campaign(tmp_path):
    command = runpy.run_path(str(Path(__file__).parent / "utils/check_chia_clean_campaign.py"))
    output = tmp_path / "campaign"
    with pytest.raises(ValueError, match="used with --archive-output"):
        command["run"](
            tmp_path / "absent-runtime", output, frontend_recipes=tmp_path / "absent.json"
        )
    assert not output.exists()
    evaluation = command["settings"]().experiment.evaluation.model_dump(mode="json")
    evaluation["transfer"] = {"gem5": {"workloads": ["source-built-guest"], "run_to_exit": True}}
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(evaluation))
    with pytest.raises(ValueError, match="recipes must match"):
        command["run"](
            tmp_path / "absent-runtime",
            output,
            evaluation_file=path,
            archive_output=tmp_path / "artifacts",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "relative,name",
    [
        ("chronus/cputraces/429.mcf", "429.mcf"),
        ("dpc4/converted/605.mcf_s-1554B.trace", "605.mcf_s-1554B"),
    ],
)
def test_compressed_input_keeps_its_cohort_path(tmp_path, monkeypatch, relative, name):
    import gzip

    from ramulator_chia.eval import config as evaluation

    monkeypatch.setattr(evaluation, "TRACES", tmp_path)
    raw = tmp_path / relative
    compressed = raw.with_name(raw.name + ".gz")
    compressed.parent.mkdir(parents=True)
    compressed.write_bytes(gzip.compress(b"0 64\n"))
    assert evaluation.trace_path(name) == str(compressed)
    raw.write_bytes(b"0 64\n")
    assert evaluation.trace_path(name) == str(raw)  # Interrupted compression keeps the raw copy.
