"""Immutable ZIP64 evidence bundles, readable with only the Python standard library.

This module packages an explicit payload inventory. It does not discover files in
an operator's checkout, decide whether an experiment is scientifically complete,
load provider SDKs, or execute archived code. Closure status belongs in metadata
and must be established by the campaign's input/runtime preflight.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import os
import re
import sqlite3
import stat
import tempfile
import zipfile
import zlib
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

CHUNK_BYTES = 1024 * 1024
MANIFEST_NAME = "manifest.json"


class InvalidArchive(ValueError):
    """A bundle is malformed, exceeds its reading limits, or fails verification."""


@dataclass(frozen=True)
class ReadLimits:
    """Operator safety limits, not compression-ratio heuristics or scientific caps."""

    maximum_members: int = 100_000
    maximum_manifest_bytes: int = 64 * 1024 * 1024
    maximum_stored_bytes: int = 8 * 1024**4
    maximum_logical_bytes: int = 8 * 1024**4

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("archive reading limits must be positive integers")


@dataclass(frozen=True)
class Member:
    name: str
    stored_sha256: str
    stored_bytes: int
    logical_sha256: str
    logical_bytes: int
    codec: str
    executable: bool = False

    def __post_init__(self):
        safe_name(self.name)
        if self.name == MANIFEST_NAME:
            raise InvalidArchive("the manifest cannot inventory itself")
        for digest in (self.stored_sha256, self.logical_sha256):
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise InvalidArchive("invalid member SHA-256")
        for count in (self.stored_bytes, self.logical_bytes):
            if type(count) is not int or count < 0:
                raise InvalidArchive("invalid member byte count")
        if self.codec not in {"none", "gzip", "xz"} or type(self.executable) is not bool:
            raise InvalidArchive("unsupported member encoding or permissions")
        if self.codec == "none" and (self.stored_sha256, self.stored_bytes) != (
            self.logical_sha256,
            self.logical_bytes,
        ):
            raise InvalidArchive("uncompressed member has inconsistent logical identity")


@dataclass(frozen=True)
class Payload:
    source: Path
    member: Member
    # Already-compressed data can be stored verbatim. Payload selection belongs
    # to the source/evidence exporter; this primitive does not collect software
    # environments or external project executables.
    store_verbatim: bool = False

    def __post_init__(self):
        if type(self.store_verbatim) is not bool:
            raise ValueError("verbatim storage must be an explicit boolean")


@dataclass(frozen=True)
class SealReceipt:
    path: Path
    sha256: str
    bytes: int


def safe_name(name: str) -> str:
    # Check the original spelling: PurePosixPath alone would normalize a/../b,
    # repeated separators and dot components before we could reject them.
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or any(ord(character) < 32 for character in name)
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise InvalidArchive(f"unsafe archive member name: {name!r}")
    return name


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidArchive(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise InvalidArchive(f"non-finite JSON number: {value}")


@contextmanager
def _regular_input(path: Path) -> Iterator[BinaryIO]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise InvalidArchive(f"payload is not a regular file: {path}")
        yield stream


class _HashingReader:
    """Count compressed bytes without staging an expansion."""

    def __init__(self, stream: BinaryIO, maximum: int):
        self.stream, self.maximum = stream, maximum
        self.digest, self.count = hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.count += len(data)
        if self.count > self.maximum:
            raise InvalidArchive("stored payload exceeds the declared reading limit")
        self.digest.update(data)
        return data


def _identity(
    stream: BinaryIO,
    *,
    name: str,
    codec: str,
    executable: bool,
    maximum_stored: int,
    maximum_logical: int,
    logical_output: BinaryIO | None = None,
) -> Member:
    counted = _HashingReader(stream, maximum_stored)
    logical_digest, logical_bytes = hashlib.sha256(), 0
    if codec not in {"none", "gzip", "xz"}:
        raise InvalidArchive(f"unsupported inner codec: {codec}")
    if codec == "gzip":
        reader = gzip.GzipFile(fileobj=counted, mode="rb")
    elif codec == "xz":
        reader = lzma.LZMAFile(counted, mode="rb", format=lzma.FORMAT_XZ)
    else:
        reader = counted
    try:
        while data := reader.read(CHUNK_BYTES):
            logical_bytes += len(data)
            if logical_bytes > maximum_logical:
                raise InvalidArchive("logical payload exceeds the declared reading limit")
            logical_digest.update(data)
            if logical_output is not None:
                logical_output.write(data)
    finally:
        if codec != "none":
            reader.close()
    return Member(
        name,
        counted.digest.hexdigest(),
        counted.count,
        logical_digest.hexdigest(),
        logical_bytes,
        codec,
        executable,
    )


def describe_payload(
    source: Path,
    name: str,
    *,
    codec: str = "none",
    executable: bool = False,
    limits: ReadLimits = ReadLimits(),
) -> Payload:
    """Snapshot a settled file's identities; sealing checks the bytes again."""
    with _regular_input(source) as stream:
        member = _identity(
            stream,
            name=name,
            codec=codec,
            executable=executable,
            maximum_stored=limits.maximum_stored_bytes,
            maximum_logical=limits.maximum_logical_bytes,
        )
    return Payload(source, member)


def stage_payload(payload: Payload, destination: Path) -> None:
    """Verify and expand one catalogued input for a reader that needs a plain file.

    The trusted caller supplies a private staging directory. Stored and logical
    identities, declared sizes and gzip integrity are checked on the same stream
    used for the copy. The original is never changed. Failed copies retain only
    a clearly named partial file; an existing destination is never overwritten.
    """
    expected = payload.member
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=destination.parent, prefix=".input-", suffix=".partial", delete=False
    ) as output:
        partial = Path(output.name)
        with _regular_input(payload.source) as source:
            actual = _identity(
                source,
                name=expected.name,
                codec=expected.codec,
                executable=expected.executable,
                maximum_stored=expected.stored_bytes,
                maximum_logical=expected.logical_bytes,
                logical_output=output,
            )
        if actual != expected:
            raise InvalidArchive(f"input identity changed: {expected.name}")
        output.flush()
        os.fsync(output.fileno())
        os.fchmod(output.fileno(), 0o400)
    os.link(partial, destination, follow_symlinks=False)
    partial.unlink()
    _sync_directory(destination.parent)


def _index(archive: zipfile.ZipFile, limits: ReadLimits) -> tuple[dict, dict[str, Member]]:
    infos = archive.infolist()
    if len(infos) > limits.maximum_members + 1:
        raise InvalidArchive("too many archive members")
    names = [safe_name(info.filename) for info in infos]
    if len(set(names)) != len(names):
        raise InvalidArchive("duplicate archive member names")
    # Files cannot also be parent directories (a together with a/b).
    name_set = set(names)
    if any(
        str(parent) in name_set
        for name in names
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    ):
        raise InvalidArchive("archive member conflicts with a parent directory")
    for info in infos:
        file_type = stat.S_IFMT(info.external_attr >> 16)
        if file_type not in {0, stat.S_IFREG} or info.is_dir():
            raise InvalidArchive("only regular file members are permitted")
        if info.flag_bits & 1 or info.compress_type not in {
            zipfile.ZIP_STORED,
            zipfile.ZIP_DEFLATED,
        }:
            raise InvalidArchive("encrypted or unsupported ZIP member")
    if MANIFEST_NAME not in name_set:
        raise InvalidArchive("missing manifest")
    if archive.getinfo(MANIFEST_NAME).file_size > limits.maximum_manifest_bytes:
        raise InvalidArchive("manifest exceeds its reading limit")
    with archive.open(MANIFEST_NAME) as stream:
        raw_manifest = stream.read(limits.maximum_manifest_bytes + 1)
    if len(raw_manifest) > limits.maximum_manifest_bytes:
        raise InvalidArchive("manifest exceeds its reading limit")
    try:
        manifest = json.loads(
            raw_manifest, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "format",
            "metadata",
            "compression",
            "members",
        }:
            raise InvalidArchive("unexpected manifest fields")
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["format"] != "chia-campaign-zip64"
        ):
            raise InvalidArchive("unsupported artifact manifest")
        if (
            not isinstance(manifest["metadata"], dict)
            or not isinstance(manifest["compression"], dict)
            or not isinstance(manifest["members"], list)
        ):
            raise InvalidArchive("invalid manifest metadata or inventory")
        if len(manifest["members"]) > limits.maximum_members:
            raise InvalidArchive("too many inventory members")
        expected_fields = set(Member.__dataclass_fields__)
        if any(
            not isinstance(item, dict) or set(item) != expected_fields
            for item in manifest["members"]
        ):
            raise InvalidArchive("unexpected payload inventory fields")
        members = [Member(**item) for item in manifest["members"]]
    except (TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidArchive("malformed artifact manifest") from error
    indexed = {member.name: member for member in members}
    if len(indexed) != len(members):
        raise InvalidArchive("duplicate inventory entries")
    if set(indexed) != name_set - {MANIFEST_NAME}:
        raise InvalidArchive("unlisted or missing payload members")
    if sum(member.stored_bytes for member in members) > limits.maximum_stored_bytes:
        raise InvalidArchive("stored inventory exceeds its reading limit")
    if sum(member.logical_bytes for member in members) > limits.maximum_logical_bytes:
        raise InvalidArchive("logical inventory exceeds its reading limit")
    for name, member in indexed.items():
        info = archive.getinfo(name)
        if info.file_size != member.stored_bytes:
            raise InvalidArchive(f"ZIP and inventory size mismatch: {name}")
        if member.codec != "none" and info.compress_type != zipfile.ZIP_STORED:
            raise InvalidArchive("inner compressed evidence must not be recompressed")
    return manifest, indexed


def verify(
    path: Path, *, expected_sha256: str | None = None, limits: ReadLimits = ReadLimits()
) -> dict:
    """Verify inventory, ZIP integrity and all stored/logical hashes without execution."""
    try:
        with _regular_input(path) as raw:
            digest = hashlib.file_digest(raw, "sha256").hexdigest()
            if expected_sha256 is not None and digest != expected_sha256:
                raise InvalidArchive("outer archive SHA-256 mismatch")
            raw.seek(0)
            with zipfile.ZipFile(raw, "r", allowZip64=True) as archive:
                manifest, members = _index(archive, limits)
                for name, expected in members.items():
                    with archive.open(name) as stream:
                        actual = _identity(
                            stream,
                            name=name,
                            codec=expected.codec,
                            executable=expected.executable,
                            maximum_stored=expected.stored_bytes,
                            maximum_logical=expected.logical_bytes,
                        )
                    if actual != expected:
                        raise InvalidArchive(f"payload identity mismatch: {name}")
        return {"archive_sha256": digest, "manifest": manifest}
    except (zipfile.BadZipFile, gzip.BadGzipFile, EOFError, zlib.error, lzma.LZMAError) as error:
        raise InvalidArchive(f"compressed evidence failed integrity checks: {error}") from error


def _sync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal(
    payloads: list[Payload],
    output_directory: Path,
    *,
    metadata: dict,
    limits: ReadLimits = ReadLimits(),
) -> SealReceipt:
    """Publish one verified bundle without replacing any existing evidence.

    Atomic no-replace publication uses a hard link followed by removal of our
    temporary name. Both are on the same filesystem. A failed temporary remains
    clearly marked .partial for inspection; it is never a published revision.
    """
    members = sorted((payload.member for payload in payloads), key=lambda member: member.name)
    if len({member.name for member in members}) != len(members):
        raise InvalidArchive("duplicate payload names")
    manifest = {
        "schema_version": 1,
        "format": "chia-campaign-zip64",
        "metadata": metadata,
        "compression": {
            "outer": "zip64",
            "text": "deflate",
            "deflate_level": "zlib_default",
            "inner_gzip": "stored",
            "inner_xz": "stored",
            "zlib_version": zlib.ZLIB_RUNTIME_VERSION,
        },
        "members": [asdict(member) for member in members],
    }
    encoded_manifest = _json_bytes(manifest)
    if len(encoded_manifest) > limits.maximum_manifest_bytes:
        raise InvalidArchive("manifest exceeds its reading limit")
    if (
        len(members) > limits.maximum_members
        or sum(m.stored_bytes for m in members) > limits.maximum_stored_bytes
        or sum(m.logical_bytes for m in members) > limits.maximum_logical_bytes
    ):
        raise InvalidArchive("payload inventory exceeds archive reading limits")
    output_directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="campaign-", suffix=".partial", dir=output_directory
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w+b") as raw:
        with zipfile.ZipFile(
            raw, "w", allowZip64=True, compression=zipfile.ZIP_DEFLATED
        ) as archive:
            # Content-addressed revisions must not change merely because an
            # interrupted publication was retried at another wall-clock time.
            info = zipfile.ZipInfo(MANIFEST_NAME)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, encoded_manifest)
            for payload in sorted(payloads, key=lambda item: item.member.name):
                member = payload.member
                info = zipfile.ZipInfo(member.name)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | (0o700 if member.executable else 0o600)) << 16
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if payload.store_verbatim or member.codec != "none"
                    else zipfile.ZIP_DEFLATED
                )
                # zipfile's streaming writer supports >4 GiB even when the file
                # size is not known by the local-header writer yet.
                info.file_size = member.stored_bytes
                digest, count = hashlib.sha256(), 0
                with (
                    _regular_input(payload.source) as source,
                    archive.open(info, "w", force_zip64=True) as target,
                ):
                    while data := source.read(CHUNK_BYTES):
                        count += len(data)
                        if count > member.stored_bytes:
                            raise InvalidArchive(f"payload grew after inventory: {member.name}")
                        digest.update(data)
                        target.write(data)
                if count != member.stored_bytes or digest.hexdigest() != member.stored_sha256:
                    raise InvalidArchive(f"payload changed after inventory: {member.name}")
        raw.flush()
        os.fsync(raw.fileno())
    checked = verify(temporary, limits=limits)
    destination = output_directory / f"campaign-{checked['archive_sha256']}.zip"
    # link() fails if the name already exists; no last-good bundle can be lost.
    try:
        os.link(temporary, destination)
    except FileExistsError:
        # Never overwrite, and never accept an existing filename as proof of
        # contents. This also closes the link-before-receipt recovery window.
        verify(destination, expected_sha256=checked["archive_sha256"], limits=limits)
    _sync_directory(output_directory)
    temporary.unlink()
    _sync_directory(output_directory)
    return SealReceipt(destination, checked["archive_sha256"], destination.stat().st_size)


def extract(
    path: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    limits: ReadLimits = ReadLimits(),
) -> dict:
    """Verify first, then extract stored payloads to a new private directory.

    Inner gzip stays compressed. No member is executed, and recorded executable
    bits do not grant automatic execution. The caller must request replay through
    the separate sandboxed evaluator.
    """
    checked = verify(path, expected_sha256=expected_sha256, limits=limits)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    with _regular_input(path) as raw, zipfile.ZipFile(raw, "r") as archive:
        manifest, members = _index(archive, limits)
        # Recheck the digest on this descriptor. A path replacement between
        # verification and extraction cannot introduce different contents.
        raw.seek(0)
        if hashlib.file_digest(raw, "sha256").hexdigest() != checked["archive_sha256"]:
            raise InvalidArchive("archive changed before extraction")
        for name, expected in members.items():
            target = destination / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            with archive.open(name) as source, target.open("xb") as output:
                while data := source.read(CHUNK_BYTES):
                    count += len(data)
                    if count > expected.stored_bytes:
                        raise InvalidArchive("extracted payload exceeds inventory size")
                    digest.update(data)
                    output.write(data)
                output.flush()
                os.fsync(output.fileno())
            if count != expected.stored_bytes or digest.hexdigest() != expected.stored_sha256:
                raise InvalidArchive(f"extracted payload identity mismatch: {name}")
            target.chmod(0o700 if expected.executable else 0o600)
        with (destination / MANIFEST_NAME).open("xb") as output:
            output.write(_json_bytes(manifest))
            output.flush()
            os.fsync(output.fileno())
    _sync_directory(destination)
    return checked


def backup_sqlite(source: Path, destination: Path) -> None:
    """Take a consistent SQLite backup, including committed WAL transactions."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    # Creating exclusively also protects against accidental overwrite by another
    # trusted export. The file is not an archive member until this returns.
    with destination.open("xb"):
        pass
    source_uri = source.resolve(strict=True).as_uri() + "?mode=ro"
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as original,
        closing(sqlite3.connect(destination)) as snapshot,
    ):
        original.backup(snapshot)
        if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise InvalidArchive("SQLite backup failed its integrity check")
    with destination.open("rb") as stream:
        os.fsync(stream.fileno())
    _sync_directory(destination.parent)
