import gzip
import json
import pathlib

import pytest

from tools.eval import archive_results as AR
from tools.chia_loop import artifacts, real_eval as E


def test_incomplete_archival_is_explicit_and_handles_empty_files(tmp_path):
    raw = tmp_path / ".incomplete-example/trace.csv.ch0"
    raw.parent.mkdir()
    raw.write_bytes(b"")
    with pytest.raises(RuntimeError, match="no uncompressed"):
        AR.compress([raw], tmp_path / "normal.json", 3)
    (tmp_path / "failure.json").write_text(json.dumps({"complete": False, "eligible_for_metrics": False}))
    E.archive_failed_run(tmp_path)
    assert not raw.exists()
    AR.verify(tmp_path / "failed_archive_manifest.json")
    assert not (tmp_path / "manifest.json").exists()
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="completed"):
        E.archive_failed_run(tmp_path)


def test_archival_failure_does_not_mask_simulation_failure(tmp_path, monkeypatch):
    directory = tmp_path / "training/simpleo3/DDR5/workload/candidate"
    partial = directory / ".incomplete-example/trace.csv.ch0"
    partial.parent.mkdir(parents=True)
    partial.write_text("partial")
    def simulation(*args, **kwargs):
        raise ValueError("original simulator failure")
    def storage(*args, **kwargs):
        raise OSError("archive storage failure")
    monkeypatch.setattr(E, "_run_one", simulation)
    monkeypatch.setattr(E, "archive_failed_run", storage)
    with pytest.raises(ValueError, match="original simulator failure"):
        E.run_one(tmp_path, "workload", "candidate", "candidate", None, split="training")
    assert json.loads((directory / "archive_failure.json").read_text())["eligible_for_metrics"] is False


def test_exact_duplicate_recovery_preserves_bad_bytes_and_original_manifest(tmp_path):
    raw = tmp_path / "trace.csv.ch0"
    payload = b"type,arrive,depart\n0,10,30\n" * 100
    raw.write_bytes(payload)
    manifest = tmp_path / "archive_manifest.json"
    AR.compress([raw], manifest, 3)
    before = manifest.read_bytes()
    target = raw.with_name(raw.name + ".gz")
    duplicate = tmp_path / "duplicate.gz"
    duplicate.write_bytes(target.read_bytes())
    damaged = b"damaged gzip bytes"
    target.write_bytes(damaged)
    wrong = tmp_path / "wrong.gz"
    wrong.write_bytes(gzip.compress(b"different contents"))
    with pytest.raises(RuntimeError, match="original archive checksum"):
        AR.recover_duplicate(manifest, raw.name, wrong)
    assert target.read_bytes() == damaged
    record = AR.recover_duplicate(manifest, raw.name, duplicate)
    assert pathlib.Path(record["quarantine"]).read_bytes() == damaged
    assert manifest.read_bytes() == before
    assert gzip.decompress(target.read_bytes()) == payload
    AR.verify(manifest)
    with pytest.raises(RuntimeError, match="already matches"):
        AR.recover_duplicate(manifest, raw.name, duplicate)


def test_auxiliary_compression_keeps_earlier_manifest_entries(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    manifest = tmp_path / "aux_archive_manifest.json"
    for name in ("first.log", "second.log"):
        (logs / name).write_text("evidence " * 100)
        artifacts.compress(tmp_path, manifest, min_bytes=1)
    assert len(json.loads(manifest.read_text())["artifacts"]) == 2
    artifacts.verify(manifest)


def test_artifact_finalization_never_accepts_an_active_campaign(tmp_path):
    from tools.chia_loop.finalize_artifacts import finalize
    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="completed evaluation"):
        finalize(tmp_path)
    assert not (tmp_path / "archive_manifest.json").exists()
