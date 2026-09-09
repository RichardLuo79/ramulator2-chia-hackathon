"""Standard-library archive verification, corruption and publication tests."""

import errno
import gzip
import hashlib
import json
import lzma
import os
import sqlite3
import stat
import subprocess
import sys
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tools.chia_loop.framework import archive as A


def payload(
    tmp_path, name="evidence/request.csv.gz", *, data=b"id,latency\n1,42\n", compressed=True
):
    source = tmp_path / "settled-input"
    source.write_bytes(gzip.compress(data, mtime=0) if compressed else data)
    return A.describe_payload(source, name, codec="gzip" if compressed else "none")


def malformed(tmp_path, names, *, inventory=None, attributes=None, extra_manifest=None):
    path = tmp_path / "bad.zip"
    data = b"a"
    member = A.Member(
        "data", hashlib.sha256(data).hexdigest(), 1, hashlib.sha256(data).hexdigest(), 1, "none"
    )
    manifest = {
        "schema_version": 1,
        "format": "chia-campaign-zip64",
        "metadata": {},
        "compression": {},
        "members": inventory if inventory is not None else [asdict(member)],
    }
    manifest.update(extra_manifest or {})
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name in names:
            info = zipfile.ZipInfo(name)
            if attributes is not None:
                info.external_attr = attributes
            archive.writestr(info, data)
    return path


def test_seal_verify_relocate_and_extract_without_provider_imports(tmp_path):
    item = payload(tmp_path)
    receipt = A.seal([item], tmp_path / "sealed", metadata={"simulation_replay": "fixture_only"})
    moved = tmp_path / "relocated.zip"
    receipt.path.rename(moved)
    # Verification does not need the original file or checkout. Removing this
    # fixture input does not remove user data.
    item.source.unlink()
    checked = A.verify(moved, expected_sha256=receipt.sha256)
    assert checked["manifest"]["members"] == [asdict(item.member)]
    assert receipt.bytes == moved.stat().st_size
    with zipfile.ZipFile(moved) as archive:
        assert archive.getinfo(item.member.name).compress_type == zipfile.ZIP_STORED
    destination = tmp_path / "restored"
    A.extract(moved, destination, expected_sha256=receipt.sha256)
    assert gzip.decompress((destination / item.member.name).read_bytes()) == b"id,latency\n1,42\n"
    with pytest.raises(FileExistsError):
        A.extract(moved, destination)
    script = """import sys
from pathlib import Path
from tools.chia_loop.framework.archive import verify
verify(Path(sys.argv[1]))
for name in sys.modules:
    assert name != 'chia' and not name.startswith(('chia.', 'ray', 'google', 'openai', 'anthropic'))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(moved)], timeout=30, capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()


def test_inner_compression_and_outer_payload_both_have_sha256(tmp_path):
    item = payload(tmp_path, data=b"x" * 100_000)
    assert item.member.stored_bytes < item.member.logical_bytes
    assert item.member.stored_sha256 != item.member.logical_sha256
    receipt = A.seal([item], tmp_path / "sealed", metadata={})
    with pytest.raises(A.InvalidArchive, match="outer archive SHA-256"):
        A.verify(receipt.path, expected_sha256="0" * 64)
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(receipt.path.read_bytes()[:-80])
    with pytest.raises(A.InvalidArchive, match="integrity"):
        A.verify(corrupt)


def test_xz_payload_is_stream_verified_and_not_recompressed(tmp_path):
    # ChampSim consumes xz directly; archive extraction must retain those bytes.
    original = b"trace-record" * 100_000
    source = tmp_path / "input.champsimtrace.xz"
    source.write_bytes(lzma.compress(original, format=lzma.FORMAT_XZ))
    item = A.describe_payload(source, "inputs/trace.xz", codec="xz")
    assert item.member.logical_sha256 == hashlib.sha256(original).hexdigest()
    assert item.member.logical_bytes == len(original)
    receipt = A.seal([item], tmp_path / "sealed", metadata={})
    with zipfile.ZipFile(receipt.path) as zipped:
        assert zipped.getinfo(item.member.name).compress_type == zipfile.ZIP_STORED
    destination = tmp_path / "restored"
    A.extract(receipt.path, destination, expected_sha256=receipt.sha256)
    assert (destination / item.member.name).read_bytes() == source.read_bytes()
    A.stage_payload(item, tmp_path / "expanded")
    assert (tmp_path / "expanded").read_bytes() == original
    with pytest.raises(A.InvalidArchive, match="logical payload"):
        A.describe_payload(
            source, item.member.name, codec="xz", limits=A.ReadLimits(maximum_logical_bytes=1)
        )
    source.write_bytes(source.read_bytes()[:-10])
    with pytest.raises((EOFError, lzma.LZMAError)):
        A.describe_payload(source, item.member.name, codec="xz")


def test_opaque_dependency_archive_can_be_stored_without_recompression(tmp_path):
    item = replace(payload(tmp_path, compressed=False), store_verbatim=True)
    receipt = A.seal([item], tmp_path / "sealed", metadata={})
    with zipfile.ZipFile(receipt.path) as zipped:
        assert zipped.getinfo(item.member.name).compress_type == zipfile.ZIP_STORED
    assert A.verify(receipt.path)["manifest"]["members"] == [asdict(item.member)]
    with pytest.raises(ValueError, match="boolean"):
        replace(item, store_verbatim=1)


def test_changed_payload_does_not_publish(tmp_path):
    item = payload(tmp_path)
    item.source.write_bytes(b"changed")
    with pytest.raises(A.InvalidArchive, match="changed after inventory"):
        A.seal([item], tmp_path / "sealed", metadata={})
    assert not list((tmp_path / "sealed").glob("*.zip"))
    assert list((tmp_path / "sealed").glob("*.partial"))


def test_no_overwrite_even_when_two_revisions_have_identical_bytes(tmp_path, monkeypatch):
    item = payload(tmp_path)
    first = A.seal([item], tmp_path / "sealed", metadata={})
    original = first.path.read_bytes()
    assert A.seal([item], tmp_path / "sealed", metadata={}) == first
    assert first.path.read_bytes() == original
    first.path.write_bytes(b"damaged published archive")
    with pytest.raises(A.InvalidArchive):
        A.seal([item], tmp_path / "sealed", metadata={})
    assert first.path.read_bytes() == b"damaged published archive"


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../b", "./b", "a//b", "a\\b", "C:/b", "a/", "a\x00b"]
)
def test_unsafe_names_are_rejected(name):
    with pytest.raises(A.InvalidArchive, match="unsafe"):
        A.safe_name(name)


@pytest.mark.parametrize(
    "names,pattern",
    [
        (["data", "data"], "duplicate archive"),
        (["data", "unexpected"], "unlisted"),
        ([], "missing payload"),
        (["data", "data/child"], "parent directory"),
        (["../escape"], "unsafe"),
    ],
)
def test_bad_member_inventory_is_rejected(tmp_path, names, pattern):
    with pytest.warns(UserWarning) if len(names) != len(set(names)) else nullcontext():
        path = malformed(tmp_path, names)
    with pytest.raises(A.InvalidArchive, match=pattern):
        A.verify(path)


@pytest.mark.parametrize(
    "file_type", [stat.S_IFLNK, stat.S_IFDIR, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR]
)
def test_unexpected_member_types_are_rejected(tmp_path, file_type):
    path = malformed(tmp_path, ["data"], attributes=(file_type | 0o777) << 16)
    with pytest.raises(A.InvalidArchive, match="regular file"):
        A.verify(path)


def test_duplicate_and_inconsistent_manifest_entries_are_rejected(tmp_path):
    data_hash = hashlib.sha256(b"a").hexdigest()
    entry = asdict(A.Member("data", data_hash, 1, data_hash, 1, "none"))
    path = malformed(tmp_path, ["data"], inventory=[entry, entry])
    with pytest.raises(A.InvalidArchive, match="duplicate inventory"):
        A.verify(path)
    path = malformed(
        tmp_path,
        ["data"],
        inventory=[{**entry, "stored_sha256": "f" * 64, "logical_sha256": "f" * 64}],
    )
    with pytest.raises(A.InvalidArchive, match="payload identity mismatch"):
        A.verify(path)
    path = malformed(tmp_path, ["data"], inventory=[{**entry, "logical_bytes": 2}])
    with pytest.raises(A.InvalidArchive, match="inconsistent logical"):
        A.verify(path)


def test_declared_and_actual_expansion_are_bounded(tmp_path):
    item = payload(tmp_path, data=b"x" * 10_000)
    receipt = A.seal([item], tmp_path / "sealed", metadata={})
    with pytest.raises(A.InvalidArchive, match="logical inventory"):
        A.verify(receipt.path, limits=A.ReadLimits(maximum_logical_bytes=100))
    # A liar changes its declared uncompressed size. It still cannot expand past
    # that size before verification rejects it.
    path = tmp_path / "liar.zip"
    with zipfile.ZipFile(receipt.path) as source, zipfile.ZipFile(path, "w") as target:
        manifest = json.loads(source.read("manifest.json"))
        manifest["members"][0]["logical_bytes"] = 1
        target.writestr("manifest.json", json.dumps(manifest))
        target.writestr(
            item.member.name, source.read(item.member.name), compress_type=zipfile.ZIP_STORED
        )
    with pytest.raises(A.InvalidArchive, match="logical payload"):
        A.verify(path)


def test_full_gzip_stream_crc_is_checked(tmp_path):
    source = tmp_path / "bad.gz"
    data = bytearray(gzip.compress(b"evidence", mtime=0))
    data[-8] ^= 1
    source.write_bytes(data)
    with pytest.raises(gzip.BadGzipFile, match="CRC check failed"):
        A.describe_payload(source, "bad.gz", codec="gzip")


@pytest.mark.parametrize("compressed", [False, True])
def test_stage_verified_input_without_replacing_original(tmp_path, compressed):
    item = payload(tmp_path, compressed=compressed)
    original = item.source.read_bytes()
    destination = tmp_path / "staged.trace"
    A.stage_payload(item, destination)
    assert destination.read_bytes() == b"id,latency\n1,42\n"
    assert item.source.read_bytes() == original
    assert destination.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        A.stage_payload(item, destination)
    assert destination.read_bytes() == b"id,latency\n1,42\n"


@pytest.mark.parametrize("corruption", ["crc", "changed", "expansion"])
def test_invalid_staging_never_publishes_an_input(tmp_path, corruption):
    item = payload(tmp_path)
    if corruption == "crc":
        data = bytearray(item.source.read_bytes())
        data[-8] ^= 1
        item.source.write_bytes(data)
    elif corruption == "changed":
        item.source.write_bytes(gzip.compress(b"changed input\n", mtime=0))
    else:
        item.source.write_bytes(gzip.compress(b"a" * 100_000, mtime=0))
    destination = tmp_path / "staged.trace"
    with pytest.raises((A.InvalidArchive, gzip.BadGzipFile)):
        A.stage_payload(item, destination)
    assert not destination.exists()
    assert list(tmp_path.glob(".input-*.partial"))


def test_symlink_and_fifo_sources_are_not_opened(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"private")
    linked = tmp_path / "link"
    linked.symlink_to(target)
    with pytest.raises(OSError):
        A.describe_payload(linked, "leak")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(A.InvalidArchive, match="regular file"):
        A.describe_payload(fifo, "fifo")


def test_disk_full_cannot_publish_a_bundle(tmp_path, monkeypatch):
    item = payload(tmp_path)

    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "fixture disk full")

    monkeypatch.setattr(zipfile._ZipWriteFile, "write", no_space)
    with pytest.raises(OSError) as failure:
        A.seal([item], tmp_path / "sealed", metadata={})
    assert failure.value.errno == errno.ENOSPC
    assert not list((tmp_path / "sealed").glob("*.zip"))
    assert item.source.exists()


def test_sqlite_export_includes_committed_uncheckpointed_wal(tmp_path):
    source, backup = tmp_path / "live.sqlite", tmp_path / "snapshot.sqlite"
    with sqlite3.connect(source) as database:
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA wal_autocheckpoint=0")
        database.execute("CREATE TABLE evidence (value TEXT)")
        database.execute("INSERT INTO evidence VALUES ('committed in WAL')")
        database.commit()
        assert Path(str(source) + "-wal").stat().st_size > 0
        A.backup_sqlite(source, backup)
        database.execute("INSERT INTO evidence VALUES ('after export')")
        database.commit()
    with sqlite3.connect(backup) as restored:
        assert restored.execute("SELECT value FROM evidence").fetchall() == [("committed in WAL",)]
    with pytest.raises(FileExistsError):
        A.backup_sqlite(source, backup)


def test_inspection_cli_is_read_only_unless_extraction_is_requested(tmp_path):
    item = payload(tmp_path)
    receipt = A.seal([item], tmp_path / "sealed", metadata={"fixture": True})
    command = [sys.executable, "-m", "tools.chia_loop.framework.inspect"]
    verified = subprocess.run(
        [*command, "verify", str(receipt.path), "--sha256", receipt.sha256],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(verified.stdout)["verified"] is True
    bad = subprocess.run(
        [*command, "verify", str(receipt.path), "--output", str(tmp_path / "unexpected")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert bad.returncode != 0
    assert not (tmp_path / "unexpected").exists()
    restored = tmp_path / "cli-restored"
    subprocess.run(
        [*command, "extract", str(receipt.path), "--output", str(restored)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert (restored / item.member.name).is_file()


def test_inventory_schema_cannot_omit_permission_fields(tmp_path):
    data_hash = hashlib.sha256(b"a").hexdigest()
    entry = asdict(A.Member("data", data_hash, 1, data_hash, 1, "none"))
    del entry["executable"]
    path = malformed(tmp_path, ["data"], inventory=[entry])
    with pytest.raises(A.InvalidArchive, match="inventory fields"):
        A.verify(path)


@pytest.mark.skipif(
    os.environ.get("CHIA_RUN_LARGE_ARCHIVE_TEST") != "1",
    reason="explicit >4 GiB streaming qualification",
)
def test_real_zip64_member_larger_than_four_gib(tmp_path):
    source = tmp_path / "large-sparse-input"
    with source.open("wb") as stream:
        stream.truncate(2**32 + 17)
    item = A.describe_payload(source, "evidence/large", codec="none")
    receipt = A.seal([item], tmp_path / "sealed", metadata={"fixture": "actual >4 GiB member"})
    assert item.member.stored_bytes > 2**32
    assert A.verify(receipt.path)["manifest"]["members"][0]["logical_bytes"] == 2**32 + 17
    with zipfile.ZipFile(receipt.path) as archive:
        assert archive.getinfo("evidence/large").extract_version >= 45
