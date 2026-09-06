"""A completion report is read-only and never makes partial test scores eligible."""
from tools.chia_loop import completion_report as W
from tools.chia_loop.core import atomic_write_json


def fixture_run(root, completed):
    root.mkdir()
    atomic_write_json(root / "run_manifest.json", {"model": root.name,
        "status": "completed" if completed else "needs_attention", "archives_verified": completed,
        "human_intervention": False, "final_test_metrics": {"aggregate": {
            "cycle_macro_mae_pct": 12.5, "request_macro_mae_over_L": .25}}})
    atomic_write_json(root / "supervisor_state.json", {"status": "completed" if completed else "needs_attention"})
    atomic_write_json(root / "state.json", {"incumbent": "candidate_001", "history": [
        {"iteration": 1, "status": "valid", "promoted": True}]})
    return root


def test_comparison_preserves_runs_and_hides_incomplete_test_scores(tmp_path):
    roots = [fixture_run(tmp_path / "model_a", True), fixture_run(tmp_path / "model_b", False)]
    before = {p: p.read_bytes() for root in roots for p in root.iterdir()}
    output = tmp_path / "comparison"
    result = W.report(roots, output)
    assert result["all_terminal"] and result["optimization_feedback"] is False
    assert result["runs"][0]["test_metrics"]["aggregate"]["cycle_macro_mae_pct"] == 12.5
    assert result["runs"][1]["test_metrics"] is None
    assert before == {p: p.read_bytes() for root in roots for p in root.iterdir()}
    assert W.R.read_json(output / "status.json")["chat_wakeup_configured"] is False
    summary = (output / "summary.md").read_text()
    assert "12.5000" in summary and "not available" in summary


def test_archive_verification_is_required_for_eligible_scores(tmp_path):
    root = fixture_run(tmp_path / "model", True)
    manifest = W.R.read_json(root / "run_manifest.json")
    manifest["archives_verified"] = False
    atomic_write_json(root / "run_manifest.json", manifest)
    snapshot = W.snapshot(root)
    assert snapshot["terminal"] and snapshot["status"] == "stopped_incomplete"
    assert snapshot["test_metrics"] is None
