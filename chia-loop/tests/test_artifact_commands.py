"""Offline checks of the small public command adapters; no provider calls."""
import importlib.util
import json
from pathlib import Path

import pytest

from ramulator_chia.eval.frozen import speed_memory_budget

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('artifact_commands', ROOT / 'scripts/artifact.py')
commands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commands)


def test_no_inference_without_opt_in(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('provider launcher must not run')
    monkeypatch.setattr(commands.runpy, 'run_module', forbidden)
    with pytest.raises(SystemExit, match='explicit --allow-paid'):
        commands.run_campaign([])


def test_configuration_relocates_without_scientific_changes(tmp_path):
    config = commands.load_config(ROOT / 'chia-loop/configs/astra.json', tmp_path / 'data', tmp_path / 'work')
    encoded = config.model_dump_json()
    assert '${' not in encoded
    assert str(tmp_path / 'data') in encoded
    assert config.run.resources.diagnostic_timeout_seconds == 1800
    assert config.run.resources.simulation_timeout_seconds is None


def test_missing_inputs_are_errors(tmp_path, capsys):
    manifest = tmp_path / 'inputs.json'
    manifest.write_text(json.dumps({'files': [{'path': 'missing.trace', 'sha256': '0' * 64}]}))
    with pytest.raises(SystemExit) as error:
        commands.fetch_data(['--manifest', str(manifest), '--data-root', str(tmp_path)])
    assert error.value.code == 2
    assert 'missing.trace' in capsys.readouterr().out


def test_release_rejects_parent_path(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': {'../outside': '0' * 64}}))
    with pytest.raises(ValueError, match='unsafe release manifest'):
        commands.verify_artifact(['--manifest', str(manifest)])


def test_speed_memory_reserve(tmp_path):
    meminfo = tmp_path / 'meminfo'
    meminfo.write_text('MemAvailable: 33554432 kB\n')
    assert speed_memory_budget(meminfo) == 16 << 30
    meminfo.write_text('MemAvailable: 16777216 kB\n')
    with pytest.raises(RuntimeError, match='insufficient available memory'):
        speed_memory_budget(meminfo)
