"""Offline feature ablations: defaults, immutable profiles and both agent paths."""
import json

import pytest

from tools.chia_loop import loop_config as L, gemini_loop as G
from tools.chia_loop.codex_cli import runner as B, queue as Q
from tools.chia_loop.core import atomic_write_json


def install(root, changes):
    profile = root / "operator_profile.json"
    atomic_write_json(profile, changes)
    return L.install(root, profile)


def test_fresh_defaults_and_legacy_are_separate(tmp_path):
    assert not L.enabled(tmp_path, "synthetic_diagnostics")
    config = L.install(tmp_path)
    assert L.enabled(tmp_path, "synthetic_diagnostics")
    assert config["synthetic"]["default_requests_per_stream"] == 50000
    assert L.load(tmp_path) == L.resolve(json.loads(L.DEFAULT.read_text()))
    assert json.loads(L.DEFAULT.read_text()) == L.DEFAULTS
    with pytest.raises(RuntimeError, match="replace"):
        L.install(tmp_path)


@pytest.mark.parametrize("change", [
    {"oracle_access": True}, {"features": {"synthetic_diagnostics": 1}},
    {"features": {"disable_compliance": True}}, {"schema_version": True},
    {"limits": {"diagnostic_calls_per_proposal": -1}},
    {"limits": {"model_turns_per_proposal": 2}},
    {"synthetic": {"max_total_reads_per_case": 500}},
    {"synthetic": {"max_total_reads_per_case": 1000}},
    {"search": {"promotion": "weighted_sum"}}, {"search": {"parent_selection": "test_score"}},
])
def test_profile_rejects_unknown_or_unsafe_settings(change):
    with pytest.raises(ValueError):
        L.resolve(change)


def test_resolved_configuration_tampering_is_rejected(tmp_path):
    config = L.install(tmp_path)
    config["features"]["synthetic_diagnostics"] = False
    atomic_write_json(tmp_path / "loop_config.json", config)
    with pytest.raises(RuntimeError, match="changed"):
        L.load(tmp_path)
    (tmp_path / "loop_config.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        L.load(tmp_path)
    (tmp_path / "loop_config_identity.json").unlink()
    atomic_write_json(tmp_path / "preparation_manifest.json", {"loop_configuration": config})
    with pytest.raises(RuntimeError, match="missing"):
        L.load(tmp_path)


def test_both_backends_use_the_same_resolved_limits(tmp_path):
    install(tmp_path, {"limits": {"model_turns_per_proposal": 9,
        "diagnostic_calls_per_proposal": 7, "drafts_per_iteration": 2}})
    for policy in (G.configured_policy(tmp_path), B.configured_policy(tmp_path)):
        assert policy["model_turns_per_proposal"] == 9
        assert policy["diagnostic_calls_per_proposal"] == 7
        assert policy["drafts_per_iteration"] == 2
        assert policy["loop_configuration_sha256"] == L.identity(L.load(tmp_path))
        assert "Pareto" in policy["promotion"]


def test_dispatch_enforces_disabled_features_before_access(tmp_path, monkeypatch):
    install(tmp_path, {"features": {k: False for k in L.DEFAULTS["features"]}})
    monkeypatch.setattr(G.E, "training_diagnostics", lambda *a, **kw: pytest.fail("disabled tool ran"))
    for request in ({"tool": "synthetic_diagnostics", "cases": [{}]},
                    {"tool": "training_diagnostics", "kind": "stats"}):
        with pytest.raises(ValueError, match="disabled"):
            G.inspect_tool(tmp_path, {}, request)
    for source in (*L.ORACLE_FILES, *L.COMPARISON_FILES):
        with pytest.raises(ValueError, match="manifest"):
            G.inspect_tool(tmp_path, {}, {"tool": "read_file", "path": source})
    assert L.tools(tmp_path) == ["read_file", "search_file"]
    assert "synthetic_diagnostics" not in L.tool_manifest(tmp_path, G.VISIBLE)
    assert L.initial_diagnostics(tmp_path, {}) == {}


def test_individual_diagnostic_kinds_are_gated(tmp_path, monkeypatch):
    install(tmp_path, {"features": {"request_extremes": False}})
    monkeypatch.setattr(G.E, "training_diagnostics", lambda *a, **kw: "training evidence")
    assert G.inspect_tool(tmp_path, {"label": "seed"}, {"tool": "training_diagnostics", "kind": "stats"}) == "training evidence"
    with pytest.raises(ValueError, match="disabled"):
        G.inspect_tool(tmp_path, {}, {"tool": "training_diagnostics", "kind": "extremes"})
    assert L.initial_diagnostics(tmp_path, {}) == {}


def test_feedback_off_is_enforced_in_both_initial_prompts(tmp_path, monkeypatch):
    install(tmp_path, {"features": {"per_workload_feedback": False, "comparison_feedback": False,
        "evolution_history": False, "request_extremes": False, "oracle_source": False, "comparison_source": False}})
    source = tmp_path / "seed/atomic_controller.cpp"
    source.parent.mkdir()
    source.write_text("seed fixture")
    atomic_write_json(tmp_path / "codex_config.json", {"maximum_iterations": 10, "usd_cap": 100})
    templates = tmp_path / "prompts"
    templates.mkdir()
    (templates / "iteration_v1.md").write_bytes((G.REPO / "tools/chia_loop/prompts/iteration_v1.md").read_bytes())
    monkeypatch.setattr(G.E, "training_diagnostics", lambda *a, **kw: pytest.fail("bootstrap leak"))
    metric = {"aggregate": {"cycle_macro_mae_pct": 10, "request_macro_mae_over_L": .4},
              "per_workload": {"PRIVATE_WORKLOAD_SCORE": 123}}
    parent = {"source_path": str(source), "sha256": G.P.sha(source.read_bytes()), "metrics": metric, "label": "seed"}
    state = {"candidates": {"seed": parent}, "incumbent": "seed", "history": [{"PRIVATE_HISTORY": True}]}
    responses = [B.prompt(tmp_path, state, "seed", 1)["content"],
                 G.make_prompt(tmp_path, "pro", 1, state["candidates"], "seed", "seed", state["history"])]
    for result in responses:
        assert "PRIVATE_WORKLOAD_SCORE" not in result and "PRIVATE_HISTORY" not in result
        assert "cycle_macro_mae_pct" in result
        for source_path in L.ORACLE_FILES | L.COMPARISON_FILES:
            assert source_path not in result
    assert metric["per_workload"]["PRIVATE_WORKLOAD_SCORE"] == 123


def test_parent_selection_can_be_ablated_without_changing_promotion(tmp_path):
    def c(x, y):
        return {"metrics": {"aggregate": {"cycle_macro_mae_pct": x, "request_macro_mae_over_L": y}}}
    candidates = {"a": c(1, 2), "b": c(2, 1)}
    assert L.parent(tmp_path, candidates, "a", 2) == "b"
    install(tmp_path, {"search": {"parent_selection": "incumbent"}})
    assert L.parent(tmp_path, candidates, "a", 2) == "a"
    assert not G.P.dominates((2, 1), (1, 2))


def test_queue_rejects_loop_profile_changes_before_preparation(tmp_path, monkeypatch):
    from tests.unit_tests.test_chia_codex_queue import args_for, forbidden
    args = args_for(tmp_path)
    args.loop_config = tmp_path / "operator_loop.json"
    atomic_write_json(args.loop_config, {"name": "before"})
    monkeypatch.setattr(Q, "fingerprint", lambda binary: {})
    observations = iter([[{"ready": False}], [{"ready": False}], [{"ready": True}]])
    monkeypatch.setattr(Q, "dependencies", lambda roots: next(observations))
    monkeypatch.setattr(Q.time, "sleep", lambda seconds: atomic_write_json(args.loop_config, {"name": "after"}))
    monkeypatch.setattr(Q.B, "prepare_with_wait", forbidden)
    monkeypatch.setattr(Q.B, "supervise", forbidden)
    with pytest.raises(RuntimeError, match="loop profile changed"):
        Q.execute(args)
    assert not args.root.exists()


def test_training_freeze_blocks_every_inspection(tmp_path):
    L.install(tmp_path)
    atomic_write_json(tmp_path / "selection_frozen.json", {})
    with pytest.raises(RuntimeError, match="freeze"):
        G.inspect_tool(tmp_path, {}, {"tool": "synthetic_diagnostics", "cases": [{}]})
