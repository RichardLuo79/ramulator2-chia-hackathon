import hashlib
import json

import pytest

from tools.chia_loop.export_review import AUDIT_FIELDS, FIGURES_AND_TABLES, export_review


def campaign(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "analysis").mkdir()
    for name in FIGURES_AND_TABLES:
        (root / "analysis" / name).write_text("compact result")
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
        "preparation": {"seed_sha256": "seed"}, "arms": {"pro": state, "flash": state},
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
    assert set(p.name for p in destination.iterdir()) == {
        *FIGURES_AND_TABLES, "pro_selected.cpp", "flash_selected.cpp", "summary.json"}
    assert "PRIVATE" not in json.dumps(result) and "/private" not in json.dumps(result)
    assert (destination / "pro_selected.cpp").read_bytes() == (root / "selected.cpp").read_bytes()
    with pytest.raises(FileExistsError):
        export_review(root, destination)


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
