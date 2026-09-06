import copy
import json

import pytest

from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.run_records import reporting_view, frozen_selections, validate_limits, execution_limits


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


def test_extended_limits_survive_worker_imports_and_cannot_change_after_start(tmp_path):
    from tools.chia_loop import gemini_loop as G
    G.L.install(tmp_path)
    limits = validate_limits(25, 100, 6)
    atomic_write_json(tmp_path / "preparation_manifest.json", {"limits": limits})
    assert execution_limits(tmp_path) == limits
    policy = G.configured_policy(tmp_path)
    assert policy["maximum_iterations"] == 25 and policy["usd_cap"] == 100
    atomic_write_json(tmp_path / "run_manifest.json", {"record_type": "optimization_run", "policy": policy})
    assert G.run_ledger(tmp_path, "pro").cap_usd == 100
    atomic_write_json(tmp_path / "preparation_manifest.json", {"limits": validate_limits(30, 100, 6)})
    with pytest.raises(ValueError, match="changed after run start"):
        execution_limits(tmp_path)


def test_evolution_uses_all_25_configured_iterations_without_provider_calls(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.chia_loop import gemini_loop as G
    atomic_write_json(tmp_path / "preparation_manifest.json", {"limits": validate_limits(25, 100, 6)})
    (tmp_path / "seed").mkdir()
    (tmp_path / "seed/atomic_controller.cpp").write_text("seed")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts/system_v1.md").write_text("fixture")
    (tmp_path / "training/reports").mkdir(parents=True)
    def metrics(i):
        return {"aggregate": {"cycle_macro_mae_pct": 100 - i, "request_macro_mae_over_L": 1 - i / 100}}
    atomic_write_json(tmp_path / "training/reports/seed.json", {"models": {"seed": metrics(0)}})
    seen = []
    def propose(root, backend, iteration, *args):
        seen.append(iteration)
        return {"status": "evaluated", "candidate": {"source_path": str(tmp_path / "seed/atomic_controller.cpp"),
            "sha256": "fixture", "metrics": metrics(iteration), "label": f"pro_{iteration:03d}"}}
    monkeypatch.setattr(G, "make_prompt", lambda *args: "fixture")
    monkeypatch.setattr(G, "propose", SimpleNamespace(chia_remote=propose))
    monkeypatch.setattr(G, "get", lambda value: value)
    monkeypatch.setattr(G, "event", lambda *args, **kwargs: None)
    result = G.run_model(tmp_path, "pro")
    assert seen == list(range(1, 26))
    assert len(result["history"]) == 25 and result["incumbent"] == "pro_025"
    assert result["status"] == "frozen" and result["budget"]["api_attempts"] == 0


@pytest.mark.parametrize("limits", [(0, 100, 6), (25.0, 100, 6), (True, 100, 6),
    (25, 0, 6), (25, float("nan"), 6), (25, float("inf"), 6), (25, 100, 2), (25, 100, 13)])
def test_reject_invalid_limits(limits):
    with pytest.raises(ValueError):
        validate_limits(*limits)
