import hashlib
import copy
import csv
import json

import pytest

from tools.chia_loop.export_review import AUDIT_FIELDS, FIGURES_AND_TABLES, export_review, model_table
from tools.chia_loop.compare_reviews import compare_reviews


def campaign(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "analysis").mkdir()
    for name in FIGURES_AND_TABLES:
        (root / "analysis" / name).write_text(
            "split,model,value\ntraining,seed,3\ntraining,pro,1\ntraining,flash,2\n"
            if name.endswith(".csv") else "compact result")
    (root / "setup_discussion.md").write_text("PRIVATE SETUP TRANSCRIPT")
    source = root / "selected.cpp"
    source.write_text("// model source\n")
    state = {"status": "frozen", "model": "test-model", "incumbent": "design_001",
        "selected": {"source_path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "metrics": {"aggregate": {"cycle_macro_mae_pct": 1.0}}},
        "history": [{"id": "design_001", "parent": "seed", "iteration": 1, "status": "valid",
                     "promoted": True, "metrics": {"cycle_macro_mae_pct": 1.0},
                     "explanation": "PRIVATE MODEL DISCUSSION", "drafts": [{"directory": "/private"}]}],
        "budget": {"api_attempts": 2, "estimated_standard_usd": 1.5, "private_path": "/private"}}
    manifest = {"status": "completed", "archives_verified": True,
        "policy": {"iterations_per_arm": 5, "thinking_level": "HIGH"},
        "instructions_per_core": 20_000_000, "training": ["train"], "final_test": ["test"],
        "preparation": {"seed_sha256": "seed"},
        "models": {"pro": "test-pro", "flash": "test-flash"},
        "arms": {arm: {**copy.deepcopy(state), "model": "test-" + arm} for arm in ("pro", "flash")},
        "final_test_metrics": {arm: {"aggregate": {"cycle_macro_mae_pct": 2.0}}
                               for arm in ("pro", "flash")}}
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    audit = {key: True for key in AUDIT_FIELDS}
    audit["status"] = "pass"
    (root / "final_integrity_audit.json").write_text(json.dumps(audit))
    return root


def test_review_export_excludes_discussions_and_preserves_sources(tmp_path):
    root = campaign(tmp_path)
    destination = tmp_path / "review"
    result = export_review(root, destination)
    assert result["record_type"] == "run_comparison"
    assert "arms" not in result and "budget" not in result and "trajectory" not in result
    assert set(p.name for p in destination.iterdir()) == {*FIGURES_AND_TABLES, "runs", "summary.json"}
    assert len(result["runs"]) == 2
    for reference in result["runs"]:
        path = destination / reference["summary"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["summary_sha256"]
        single = json.loads(path.read_text())
        assert "PRIVATE" not in json.dumps(single) and "/private" not in json.dumps(single)
        assert single["record_type"] == "optimization_run_summary"
        assert single["model"] == reference["model"] and single["run_id"] == reference["run_id"]
        assert "arms" not in single and "models" not in single
        assert single["policy"]["maximum_iterations"] == 5
        assert (path.parent / single["source_file"]).read_bytes() == (root / "selected.cpp").read_bytes()
        for name in ("headline.csv", "per_workload.csv"):
            with (path.parent / name).open() as stream:
                assert {r["model"] for r in csv.DictReader(stream)} == {"seed", single["backend"]}
    with pytest.raises(FileExistsError):
        export_review(root, destination)


def test_individual_run_export_is_not_a_comparison(tmp_path):
    root = campaign(tmp_path)
    path = root / "run_manifest.json"
    raw = json.loads(path.read_text())
    raw.update(record_type="optimization_run", run_id="pro-001", backend="pro", model="test-pro",
               state=raw.pop("arms")["pro"], pricing={"standard_rates": {}})
    raw.pop("models")
    raw["final_test_metrics"] = raw["final_test_metrics"]["pro"]
    path.write_text(json.dumps(raw))
    audit_path = root / "final_integrity_audit.json"
    audit = json.loads(audit_path.read_text())
    audit.update(run_id="pro-001", model="test-pro", generation={"finish_reasons": {"STOP": 2}})
    audit_path.write_text(json.dumps(audit))
    for name in ("headline.csv", "per_workload.csv"):
        table = root / "analysis" / name
        table.write_bytes(model_table(table.read_bytes(), "pro"))
    result = export_review(root, tmp_path / "review")
    assert result["record_type"] == "optimization_run_summary" and result["run_id"] == "pro-001"
    assert "runs" not in result and "arms" not in result
    assert result["generation_finish_reasons"] == {"STOP": 2}
    assert (tmp_path / "review/selected.cpp").read_bytes() == (root / "selected.cpp").read_bytes()


def test_comparison_references_runs_without_merging_accounting(tmp_path):
    root = campaign(tmp_path)
    original = (root / "run_manifest.json").read_bytes()
    exported = tmp_path / "exports"
    legacy = export_review(root, exported)
    paths = [exported / ref["summary"] for ref in legacy["runs"]]
    result = compare_reviews(paths, tmp_path / "comparison")
    assert result["record_type"] == "run_comparison" and len(result["runs"]) == 2
    assert "budget" not in result and "trajectory" not in result and "arms" not in result
    assert (root / "run_manifest.json").read_bytes() == original
    with pytest.raises(ValueError, match="duplicate run ID"):
        compare_reviews([paths[0], paths[0]], tmp_path / "duplicate")
    changed = json.loads(paths[1].read_text())
    changed["instructions_per_core"] = 40_000_000
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="unmatched comparison setup"):
        compare_reviews(paths, tmp_path / "unmatched")
    assert not (tmp_path / "unmatched").exists()


def test_export_refuses_unsafe_run_id_and_changed_comparison_artifact(tmp_path):
    root = campaign(tmp_path)
    exported = tmp_path / "exports"
    legacy = export_review(root, exported)
    paths = [exported / ref["summary"] for ref in legacy["runs"]]
    (paths[0].parent / "selected.cpp").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        compare_reviews(paths, tmp_path / "changed")
    path = root / "run_manifest.json"
    record = json.loads(path.read_text())
    record["models"]["pro"] = "../../escape"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unsafe run identifier"):
        export_review(root, tmp_path / "unsafe")
    assert not (tmp_path / "unsafe").exists()


def test_review_export_requires_completed_audited_unchanged_campaign(tmp_path):
    root = campaign(tmp_path)
    destination = tmp_path / "review"
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = "running"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed"):
        export_review(root, destination)
    assert not destination.exists()
    manifest["status"] = "completed"
    path.write_text(json.dumps(manifest))
    (root / "selected.cpp").write_text("changed model")
    with pytest.raises(ValueError, match="source hash"):
        export_review(root, destination)
    assert not destination.exists()


def test_review_export_rejects_external_source(tmp_path):
    root = campaign(tmp_path)
    external = tmp_path / "external.cpp"
    external.write_bytes((root / "selected.cpp").read_bytes())
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["arms"]["pro"]["selected"]["source_path"] = str(external)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside campaign"):
        export_review(root, tmp_path / "review")
