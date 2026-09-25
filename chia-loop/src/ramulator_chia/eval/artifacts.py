"""Transparent access to raw or gzip-archived evaluation artifacts.

Evaluation manifests identify the uncompressed bytes of an artifact.  Once a
completed campaign is archived, the logical path recorded by that manifest may
be absent while ``<path>.gz`` contains the same bytes.  The helpers here keep
that storage decision out of metric and validation code.
"""

from __future__ import annotations

import glob as _glob
import gzip
import hashlib
import pathlib
from collections.abc import Iterator

GZIP_SUFFIX = ".gz"


def logical_path(path) -> pathlib.Path:
    """Return the manifest-facing path for a raw or ``.gz`` artifact."""
    path = pathlib.Path(path)
    if path.name.endswith(GZIP_SUFFIX):
        return path.with_name(path.name[:-len(GZIP_SUFFIX)])
    return path


def resolve(path) -> pathlib.Path:
    """Resolve a logical artifact path to its raw or gzip storage file.

    A raw file wins when both forms exist.  This makes an interrupted archive
    operation harmless: readers continue to consume the original until it is
    removed after the archive manifest has been durably updated.
    """
    path = pathlib.Path(path)
    if path.is_file():
        return path
    if not path.name.endswith(GZIP_SUFFIX):
        archived = path.with_name(path.name + GZIP_SUFFIX)
        if archived.is_file():
            return archived
    return path


def exists(path) -> bool:
    """Return whether either the raw or gzip representation exists."""
    return resolve(path).is_file()


def open_binary(path):
    """Open the uncompressed byte stream for a raw or gzip artifact."""
    stored = resolve(path)
    if stored.name.endswith(GZIP_SUFFIX):
        return gzip.open(stored, mode="rb")
    return stored.open(mode="rb")


def open_text(path, *, encoding="utf-8", newline=None):
    """Open the uncompressed text stream for a raw or gzip artifact."""
    stored = resolve(path)
    if stored.name.endswith(GZIP_SUFFIX):
        return gzip.open(
            stored, mode="rt", encoding=encoding, newline=newline,
        )
    return stored.open(mode="r", encoding=encoding, newline=newline)


def raw_provenance(path) -> dict[str, object]:
    """Hash the uncompressed bytes using the historical manifest schema."""
    supplied = pathlib.Path(path)
    requested = logical_path(supplied)
    # An explicitly supplied gzip path means that exact storage object, even
    # during the short crash-safe interval when its raw sibling also exists.
    stored = supplied if supplied.name.endswith(GZIP_SUFFIX) else resolve(requested)
    stat_before = stored.stat()
    digest = hashlib.sha256()
    size = 0
    with open_binary(stored) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    stat = stored.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
            stat_before.st_size, stat_before.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {stored}")
    return {
        "path": str(requested.resolve()),
        "size": size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def raw_provenance_matches(path, recorded) -> bool:
    """Compare durable uncompressed identity fields with a manifest record."""
    if not isinstance(recorded, dict):
        return False
    current = raw_provenance(path)
    return all(
        current[key] == recorded.get(key) for key in ("path", "size", "sha256")
    )


def glob_logical(pattern) -> Iterator[pathlib.Path]:
    """Yield logical paths matching raw files or their ``.gz`` counterparts."""
    matches = {
        logical_path(pathlib.Path(match))
        for candidate in (str(pattern), str(pattern) + GZIP_SUFFIX)
        for match in _glob.glob(candidate)
    }
    yield from sorted(matches)
