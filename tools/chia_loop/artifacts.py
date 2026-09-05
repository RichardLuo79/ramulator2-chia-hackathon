#!/usr/bin/env python3
"""Verified per-file gzip storage for finalized CHIA text artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import tempfile

SCHEMA_VERSION = 1


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(payload, stream, indent=1, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _gzip_verified(raw: pathlib.Path, level: int) -> tuple[pathlib.Path, dict]:
    archive = raw.with_name(raw.name + ".gz")
    if archive.exists():
        raise RuntimeError(f"refusing to replace existing archive: {archive}")
    raw_stat = raw.stat()
    raw_sha = _sha256(raw)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w+b", dir=raw.parent, prefix=f".{raw.name}.",
                suffix=".tmp.gz", delete=False) as output:
            temporary = pathlib.Path(output.name)
            with raw.open("rb") as source, gzip.GzipFile(
                    filename="", mode="wb", compresslevel=level,
                    fileobj=output, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        digest = hashlib.sha256()
        size = 0
        with gzip.open(temporary, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != raw_stat.st_size or digest.hexdigest() != raw_sha:
            raise RuntimeError(f"gzip verification failed for {raw}")
        os.replace(temporary, archive)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    os.utime(archive, ns=(raw_stat.st_atime_ns, raw_stat.st_mtime_ns))
    archive_stat = archive.stat()
    entry = {
        "raw_size": raw_stat.st_size,
        "raw_mtime_ns": raw_stat.st_mtime_ns,
        "raw_sha256": raw_sha,
        "archive_size": archive_stat.st_size,
        "archive_sha256": _sha256(archive),
    }
    raw.unlink()
    return archive, entry


def compress(root: pathlib.Path, manifest: pathlib.Path, *,
             min_bytes: int = 1_048_576, level: int = 6) -> dict:
    """Compress finalized large logs/transcripts, leaving small files readable."""
    root = root.resolve()
    manifest = manifest.resolve()
    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path != manifest and not path.name.endswith(".gz")
        and path.stat().st_size >= min_bytes
        and any(part in {"logs", "interactions", "profiles"} for part in path.parts)
    )
    payload = json.loads(manifest.read_text()) if manifest.exists() else {
        "schema_version": SCHEMA_VERSION,
        "codec": "gzip",
        "gzip_header_mtime": 0,
        "root": str(root),
        "minimum_raw_bytes": min_bytes,
        "artifacts": {},
    }
    if payload.get("schema_version") != SCHEMA_VERSION or pathlib.Path(payload["root"]).resolve() != root:
        raise RuntimeError("incompatible existing auxiliary archive manifest")
    for raw in candidates:
        archive, entry = _gzip_verified(raw, level)
        relative = str(raw.relative_to(root))
        payload["artifacts"][relative] = {
            **entry,
            "archive_path": str(archive.relative_to(root)),
        }
        _write_json(manifest, payload)
    if not manifest.exists():
        _write_json(manifest, payload)
    return payload


def verify(manifest: pathlib.Path) -> None:
    payload = json.loads(manifest.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported auxiliary archive schema")
    root = pathlib.Path(payload["root"])
    for raw_relative, entry in payload["artifacts"].items():
        archive = root / entry["archive_path"]
        if (not archive.is_file() or archive.stat().st_size != entry["archive_size"]
                or _sha256(archive) != entry["archive_sha256"]):
            raise RuntimeError(f"stored archive identity mismatch: {archive}")
        digest = hashlib.sha256()
        size = 0
        with gzip.open(archive, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != entry["raw_size"] or digest.hexdigest() != entry["raw_sha256"]:
            raise RuntimeError(f"uncompressed identity mismatch: {raw_relative}")


def restore(manifest: pathlib.Path, requested: list[str]) -> None:
    payload = json.loads(manifest.read_text())
    root = pathlib.Path(payload["root"])
    selected = set(requested) if requested else set(payload["artifacts"])
    missing = selected - set(payload["artifacts"])
    if missing:
        raise RuntimeError(f"artifacts absent from manifest: {sorted(missing)}")
    for relative in sorted(selected):
        entry = payload["artifacts"][relative]
        raw = root / relative
        archive = root / entry["archive_path"]
        if raw.exists():
            if raw.stat().st_size != entry["raw_size"] or _sha256(raw) != entry["raw_sha256"]:
                raise RuntimeError(f"refusing to replace mismatched raw file: {raw}")
            continue
        raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    "w+b", dir=raw.parent, prefix=f".{raw.name}.",
                    suffix=".restore.tmp", delete=False) as output:
                temporary = pathlib.Path(output.name)
                with gzip.open(archive, "rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if (temporary.stat().st_size != entry["raw_size"] or
                    _sha256(temporary) != entry["raw_sha256"]):
                raise RuntimeError(f"restored identity mismatch: {relative}")
            os.replace(temporary, raw)
            temporary = None
            os.utime(raw, ns=(entry["raw_mtime_ns"], entry["raw_mtime_ns"]))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("root", type=pathlib.Path)
    compress_parser.add_argument("--manifest", type=pathlib.Path, required=True)
    compress_parser.add_argument("--min-bytes", type=int, default=1_048_576)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("manifest", type=pathlib.Path)
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("manifest", type=pathlib.Path)
    restore_parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.command == "compress":
        compress(args.root, args.manifest, min_bytes=args.min_bytes)
    elif args.command == "verify":
        verify(args.manifest)
    else:
        restore(args.manifest, args.paths)


if __name__ == "__main__":
    main()
