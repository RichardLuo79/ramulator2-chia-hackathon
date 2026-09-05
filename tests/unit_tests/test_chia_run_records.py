import copy
import json

import pytest

from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.run_records import reporting_view, frozen_selections


def individual(root):
    manifest = {"record_type": "optimization_run", "run_id": "pro-001", "backend": "pro",
        "model": "gemini-pro", "status": "completed", "policy": {"maximum_iterations": 5, "usd_cap": 50},
        "pricing": {"standard_rates": {"input": 2}}, "state": {"status": "frozen",
            "selected": {"sha256": "selected"}}, "final_test_metrics": {"aggregate": {"error": 1}}}
    atomic_write_json(root / "run_manifest.json", manifest)
    atomic_write_json(root / "selection_frozen.json", {
        "run_id": "pro-001", "source_sha256": "selected", "frozen_at": 1})
    return manifest


def test_single_run_reporting_view_is_read_only_and_owns_freeze(tmp_path):
    individual(tmp_path)
    original = (tmp_path / "run_manifest.json").read_bytes()
    view = reporting_view(tmp_path)
    assert set(view["models"]) == set(view["arms"]) == set(view["final_test_metrics"]) == {"pro"}
    assert view["run_ids"] == {"pro": "pro-001"}
    assert view["run_directories"] == {"pro": tmp_path}
    assert frozen_selections(view)["pro"]["source_sha256"] == "selected"
    assert (tmp_path / "run_manifest.json").read_bytes() == original
    atomic_write_json(tmp_path / "selection_frozen.json", {
        "run_id": "flash-001", "source_sha256": "selected", "frozen_at": 1})
    with pytest.raises(ValueError, match="different run"):
        frozen_selections(view)


def test_single_run_cannot_contain_joint_state(tmp_path):
    manifest = individual(tmp_path)
    manifest["arms"] = {"pro": {}, "flash": {}}
    atomic_write_json(tmp_path / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="multiple-model"):
        reporting_view(tmp_path)


def test_legacy_policy_and_paths_adapt_without_mutating_evidence(tmp_path):
    manifest = individual(tmp_path)
    manifest.pop("record_type")
    manifest["policy"] = {"iterations_per_arm": 5, "usd_cap_per_arm": 50}
    manifest["models"] = {"pro": "gemini-pro", "flash": "gemini-flash"}
    manifest["arms"] = {backend: copy.deepcopy(manifest["state"]) for backend in manifest["models"]}
    atomic_write_json(tmp_path / "run_manifest.json", manifest)
    original = (tmp_path / "run_manifest.json").read_bytes()
    view = reporting_view(tmp_path)
    assert view["policy"] == {"maximum_iterations": 5, "usd_cap": 50}
    assert view["run_directories"]["flash"] == tmp_path / "arms/flash"
    assert view["run_ids"]["flash"] != view["run_ids"]["pro"]
    assert (tmp_path / "run_manifest.json").read_bytes() == original


def test_finalization_preserves_single_run_schema(tmp_path, monkeypatch):
    from tools.chia_loop import finalize_artifacts as F
    individual(tmp_path)
    monkeypatch.setattr(F.real_eval, "verify_run_archives", lambda root: 2)
    monkeypatch.setattr(F.artifacts, "compress", lambda *args, **kwargs: None)
    monkeypatch.setattr(F.artifacts, "verify", lambda *args: None)
    assert F.finalize(tmp_path) == 2
    result = json.loads((tmp_path / "run_manifest.json").read_text())
    assert result["record_type"] == "optimization_run" and result["run_id"] == "pro-001"
    assert "arms" not in result and "models" not in result
    assert result["archives_verified"] is True
    assert result["post_run_artifact_finalization"]["additional_generation_calls"] == 0
