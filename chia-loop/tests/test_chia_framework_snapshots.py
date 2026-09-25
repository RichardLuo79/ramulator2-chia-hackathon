"""Actual-file submissions, without model calls, compilers or repository history."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ramulator_chia.framework.identity import digest_json
from ramulator_chia.framework.snapshots import (
    InvalidSnapshot,
    ModelFiles,
    materialize,
    parse_parameters,
    publish_bytes,
    read_file,
    snapshot,
    verify_snapshot,
)

CONTRACT = ModelFiles(("model.cpp", "include/model.h"), "parameters.json")
LIMIT = 1024 * 1024


def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "include").mkdir()
    (root / "model.cpp").write_text("int predict() { return 1; }\n")
    (root / "include/model.h").write_text("int predict();\n")
    (root / "parameters.json").write_text('{"delay": 1.0}')
    (root / "private-unlisted.txt").write_text("not part of the submission")
    return root


def test_snapshot_is_actual_source_and_separate_parameters(tmp_path):
    root = workspace(tmp_path)
    store = tmp_path / "candidates"
    first = snapshot(root, store, CONTRACT, maximum_bytes=LIMIT)
    assert first == verify_snapshot(store, first["candidate_id"], CONTRACT, maximum_bytes=LIMIT)
    assert set(first["files"]) == set(CONTRACT.paths)
    assert not (store / first["candidate_id"] / "private-unlisted.txt").exists()
    assert first["configuration_sha256"] == digest_json({"delay": 1.0})
    (root / "model.cpp").write_text("int predict() { return 2; }\n")
    draft = snapshot(root, store, CONTRACT, maximum_bytes=LIMIT)
    assert draft["source_sha256"] != first["source_sha256"]
    assert draft["configuration_sha256"] == first["configuration_sha256"]
    assert (store / first["candidate_id"] / "model.cpp").read_text().endswith("1; }\n")
    (root / "parameters.json").write_text('{"delay": 2}')
    configured = snapshot(root, store, CONTRACT, maximum_bytes=LIMIT)
    assert configured["configuration_sha256"] != draft["configuration_sha256"]
    assert configured["source_sha256"] == draft["source_sha256"]


def test_concurrent_identical_publication_is_idempotent(tmp_path):
    root = workspace(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as workers:
        receipts = list(
            workers.map(
                lambda _: snapshot(root, tmp_path / "store", CONTRACT, maximum_bytes=LIMIT),
                range(8),
            )
        )
    assert all(receipt == receipts[0] for receipt in receipts)
    assert len(list((tmp_path / "store").iterdir())) == 1


def test_duplicate_publication_waits_for_temporary_alias_removal(tmp_path, monkeypatch):
    target = tmp_path / "settled.json"
    linked, release, started, reused = (threading.Event() for _ in range(4))
    original_link = os.link

    def held_link(*args, **kwargs):
        original_link(*args, **kwargs)
        linked.set()
        assert release.wait(5), "fixture publisher was not released"

    monkeypatch.setattr(os, "link", held_link)

    def second():
        started.set()
        publish_bytes(target, b"same immutable bytes")
        reused.set()

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(publish_bytes, target, b"same immutable bytes")
        assert linked.wait(5)
        duplicate = workers.submit(second)
        try:
            assert started.wait(5)
            assert not reused.wait(0.1), (
                "duplicate verified a not-yet-settled hard-link publication"
            )
        finally:
            release.set()
        first.result(timeout=5)
        duplicate.result(timeout=5)
    assert reused.is_set() and target.stat().st_nlink == 1
    assert (
        read_file(tmp_path, target.name, maximum_bytes=20, require_single_link=True)
        == b"same immutable bytes"
    )


def test_materialize_starts_independent_writable_episode(tmp_path):
    root = workspace(tmp_path)
    store = tmp_path / "store"
    receipt = snapshot(root, store, CONTRACT, maximum_bytes=LIMIT)
    next_root = tmp_path / "episode-2"
    assert (
        materialize(store, receipt["candidate_id"], next_root, CONTRACT, maximum_bytes=LIMIT)
        == receipt
    )
    (next_root / "model.cpp").write_text("edited only here")
    assert verify_snapshot(store, receipt["candidate_id"], CONTRACT, maximum_bytes=LIMIT) == receipt
    with pytest.raises(FileExistsError):
        materialize(store, receipt["candidate_id"], next_root, CONTRACT, maximum_bytes=LIMIT)


@pytest.mark.parametrize("entry", ["model.cpp", "include"])
def test_leaf_and_intermediate_symlinks_are_rejected(tmp_path, entry):
    root = workspace(tmp_path)
    target = root / entry
    saved = root / (entry + "-saved")
    target.rename(saved)
    target.symlink_to(saved, target_is_directory=saved.is_dir())
    with pytest.raises(OSError):
        snapshot(root, tmp_path / "store", CONTRACT, maximum_bytes=LIMIT)


def test_fifo_does_not_block_reader(tmp_path):
    root = workspace(tmp_path)
    (root / "model.cpp").unlink()
    os.mkfifo(root / "model.cpp")
    with pytest.raises(InvalidSnapshot, match="regular"):
        snapshot(root, tmp_path / "store", CONTRACT, maximum_bytes=LIMIT)


@pytest.mark.parametrize(
    "data",
    [
        b"[]",
        b'{"x":true}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1,"x":2}',
        b'{"x":{}}',
        b'{"../x":2}',
        b'{"x":"2"}',
        b"\xff",
        b'{"x":' + b"9" * 1000 + b"}",
    ],
)
def test_parameters_do_not_smuggle_hardware_config_or_non_numbers(data):
    with pytest.raises(InvalidSnapshot):
        parse_parameters(data)


@pytest.mark.parametrize(
    "paths, parameters",
    [
        (("../outside",), "p.json"),
        (("/absolute",), "p.json"),
        (("model.cpp", "model.cpp"), "p.json"),
        (("a", "a/b"), "p.json"),
        (("snapshot.json",), "p.json"),
        (("model.cpp",), "model.cpp"),
    ],
)
def test_explicit_contract_rejects_escapes_and_conflicts(paths, parameters):
    with pytest.raises(ValueError):
        ModelFiles(paths, parameters)


def test_declared_byte_guard_is_whole_submission(tmp_path):
    root = workspace(tmp_path)
    size = sum((root / path).stat().st_size for path in CONTRACT.paths)
    with pytest.raises(InvalidSnapshot, match="byte guard"):
        snapshot(root, tmp_path / "small", CONTRACT, maximum_bytes=size - 1)
    assert snapshot(root, tmp_path / "fits", CONTRACT, maximum_bytes=size)


def test_snapshot_integrity_and_contract_are_checked(tmp_path):
    root = workspace(tmp_path)
    store = tmp_path / "store"
    receipt = snapshot(root, store, CONTRACT, maximum_bytes=LIMIT)
    other = ModelFiles(tuple(reversed(CONTRACT.sources)), CONTRACT.parameters)
    with pytest.raises(InvalidSnapshot, match="contract changed"):
        verify_snapshot(store, receipt["candidate_id"], other, maximum_bytes=LIMIT)
    file = store / receipt["candidate_id"] / "model.cpp"
    file.chmod(0o600)
    file.write_text("corrupted")
    with pytest.raises(InvalidSnapshot, match="contents"):
        verify_snapshot(store, receipt["candidate_id"], CONTRACT, maximum_bytes=LIMIT)


def test_publication_never_overwrites_different_bytes(tmp_path):
    target = tmp_path / "receipt.json"
    publish_bytes(target, b"old")
    with pytest.raises(InvalidSnapshot, match="different bytes"):
        publish_bytes(target, b"new")
    assert target.read_bytes() == b"old"


def test_manifest_is_canonical_and_hashed(tmp_path):
    root = workspace(tmp_path)
    store = tmp_path / "store"
    receipt = snapshot(root, store, CONTRACT, maximum_bytes=LIMIT)
    path = store / receipt["candidate_id"] / "snapshot.json"
    path.chmod(0o600)
    path.write_text(json.dumps(json.loads(path.read_text()), indent=2))
    with pytest.raises(InvalidSnapshot, match="manifest identity"):
        verify_snapshot(store, receipt["candidate_id"], CONTRACT, maximum_bytes=LIMIT)


def test_read_limit_is_reported_not_silent_truncation(tmp_path):
    root = workspace(tmp_path)
    with pytest.raises(InvalidSnapshot, match="byte guard"):
        read_file(root, "model.cpp", maximum_bytes=1)
