#!/usr/bin/env python3
"""Compress, verify, and restore completed evaluation traces.

Archives are one gzip file per trace so a single damaged file cannot make an
entire campaign inaccessible.  The original run manifests remain unchanged:
``archive_manifest.json`` records both the original byte identity and the
stored gzip identity.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import os
import pathlib
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval import artifacts as A  # noqa: E402
from eval import config as C  # noqa: E402

ARCHIVE_SCHEMA_VERSION = 1
_TRACE_NAME = re.compile(r".+\.ch[0-9]+\Z")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _physical_provenance(path: pathlib.Path) -> dict[str, object]:
    stat_before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
            stat_before.st_size, stat_before.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _discover_raw(paths) -> list[pathlib.Path]:
    discovered = set()
    for item in paths:
        path = pathlib.Path(item)
        if path.is_dir():
            candidates = path.rglob("*")
        else:
            candidates = (path,)
        for candidate in candidates:
            if candidate.is_symlink() and _TRACE_NAME.fullmatch(candidate.name):
                raise RuntimeError(f"refusing to archive a symbolic link: {candidate}")
            if (candidate.is_file() and _TRACE_NAME.fullmatch(candidate.name)
                    and not any(part.startswith(".incomplete-")
                                for part in candidate.parts)):
                discovered.add(candidate.resolve())
    return sorted(discovered)


def _new_manifest() -> dict[str, object]:
    now = _utc_now()
    return {
        "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
        "codec": "gzip",
        "gzip_header_mtime": 0,
        "created_utc": now,
        "updated_utc": now,
        "artifacts": {},
    }


def _load_manifest(path: pathlib.Path) -> dict[str, object]:
    if not path.exists():
        return _new_manifest()
    import json

    payload = json.loads(path.read_text())
    if payload.get("archive_schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported archive manifest schema: {path}")
    if payload.get("codec") != "gzip" or not isinstance(
            payload.get("artifacts"), dict):
        raise RuntimeError(f"invalid archive manifest: {path}")
    return payload


def _relative(path: pathlib.Path, base: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _manifest_entry(raw: pathlib.Path, archived: pathlib.Path,
                    raw_provenance, archive_provenance,
                    manifest_path: pathlib.Path) -> dict[str, object]:
    base = manifest_path.parent
    return {
        "raw_path": _relative(raw, base),
        "archive_path": _relative(archived, base),
        "raw_size": raw_provenance["size"],
        "raw_mtime_ns": raw_provenance["mtime_ns"],
        "raw_sha256": raw_provenance["sha256"],
        "archive_size": archive_provenance["size"],
        "archive_sha256": archive_provenance["sha256"],
    }


def _archive_one(raw: pathlib.Path, level: int) -> tuple[pathlib.Path, dict, dict]:
    archived = raw.with_name(raw.name + A.GZIP_SUFFIX)
    raw_provenance = _physical_provenance(raw)
    raw_stat = raw.stat()

    if archived.exists():
        archived_raw = A.raw_provenance(archived)
        if (archived_raw["size"], archived_raw["sha256"]) != (
                raw_provenance["size"], raw_provenance["sha256"]):
            raise RuntimeError(
                f"refusing to replace mismatched existing archive: {archived}"
            )
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w+b", dir=raw.parent, prefix=f".{raw.name}.",
                    suffix=".tmp.gz", delete=False) as output:
                temporary = pathlib.Path(output.name)
                with raw.open("rb") as source, gzip.GzipFile(
                        filename="", mode="wb", compresslevel=level,
                        fileobj=output, mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            archived_raw = A.raw_provenance(temporary)
            if (archived_raw["size"], archived_raw["sha256"]) != (
                    raw_provenance["size"], raw_provenance["sha256"]):
                raise RuntimeError(f"gzip verification failed for {raw}")
            os.replace(temporary, archived)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    # Preserve cache-age semantics even though the storage representation is
    # new.  Content identity remains the original uncompressed SHA-256.
    os.utime(archived, ns=(raw_stat.st_atime_ns, raw_stat.st_mtime_ns))
    return archived, raw_provenance, _physical_provenance(archived)


def compress(paths, manifest_path: pathlib.Path, level: int, *, keep_raw=False) -> None:
    raw_paths = _discover_raw(paths)
    if not raw_paths:
        raise RuntimeError("no uncompressed *.chN trace artifacts found")
    manifest = _load_manifest(manifest_path)
    artifacts = manifest["artifacts"]
    total_raw = sum(path.stat().st_size for path in raw_paths)
    print(f"archiving {len(raw_paths)} traces ({total_raw} raw bytes)", flush=True)
    for index, raw in enumerate(raw_paths, start=1):
        archived, raw_provenance, archive_provenance = _archive_one(raw, level)
        key = _relative(raw, manifest_path.parent)
        artifacts[key] = _manifest_entry(
            raw, archived, raw_provenance, archive_provenance, manifest_path,
        )
        manifest["updated_utc"] = _utc_now()
        C.atomic_write_json(manifest_path, manifest)
        if not keep_raw:
            raw.unlink()
        reduction = 100.0 * (1.0 - archive_provenance["size"] /
                             raw_provenance["size"])
        print(
            f"[{index}/{len(raw_paths)}] {raw}: "
            f"{raw_provenance['size']} -> {archive_provenance['size']} "
            f"bytes ({reduction:.1f}% smaller)",
            flush=True,
        )


def _entry_paths(entry, manifest_path: pathlib.Path):
    def expand(value):
        path = pathlib.Path(value)
        return path if path.is_absolute() else manifest_path.parent / path

    return expand(entry["raw_path"]), expand(entry["archive_path"])


def verify(manifest_path: pathlib.Path) -> None:
    manifest = _load_manifest(manifest_path)
    entries = manifest["artifacts"]
    for index, entry in enumerate(entries.values(), start=1):
        raw, archived = _entry_paths(entry, manifest_path)
        if not archived.is_file():
            raise RuntimeError(f"missing archive: {archived}")
        raw_identity = A.raw_provenance(archived)
        archive_identity = _physical_provenance(archived)
        if (raw_identity["size"], raw_identity["sha256"]) != (
                entry["raw_size"], entry["raw_sha256"]):
            raise RuntimeError(f"uncompressed identity mismatch: {archived}")
        if (archive_identity["size"], archive_identity["sha256"]) != (
                entry["archive_size"], entry["archive_sha256"]):
            raise RuntimeError(f"stored identity mismatch: {archived}")
        print(f"[{index}/{len(entries)}] verified {raw}", flush=True)


def restore(manifest_path: pathlib.Path, requested) -> None:
    manifest = _load_manifest(manifest_path)
    entries = manifest["artifacts"]
    wanted = {str(pathlib.Path(path).resolve()) for path in requested}
    selected = []
    for entry in entries.values():
        raw, archived = _entry_paths(entry, manifest_path)
        if not wanted or str(raw.resolve()) in wanted or str(archived.resolve()) in wanted:
            selected.append((entry, raw, archived))
    if wanted and len(selected) != len(wanted):
        found = {str(path.resolve()) for _, raw, archived in selected
                 for path in (raw, archived)}
        missing = sorted(wanted - found)
        raise RuntimeError(f"paths are absent from archive manifest: {missing}")
    for index, (entry, raw, archived) in enumerate(selected, start=1):
        if raw.exists():
            if not A.raw_provenance_matches(raw, {
                    "path": str(raw.resolve()),
                    "size": entry["raw_size"],
                    "sha256": entry["raw_sha256"],
                }):
                raise RuntimeError(f"refusing to replace mismatched raw file: {raw}")
            print(f"[{index}/{len(selected)}] already restored {raw}", flush=True)
            continue
        raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w+b", dir=raw.parent, prefix=f".{raw.name}.",
                    suffix=".restore.tmp", delete=False) as output:
                temporary = pathlib.Path(output.name)
                with gzip.open(archived, "rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            restored = _physical_provenance(temporary)
            if (restored["size"], restored["sha256"]) != (
                    entry["raw_size"], entry["raw_sha256"]):
                raise RuntimeError(f"restored identity mismatch: {raw}")
            os.replace(temporary, raw)
            temporary = None
            os.utime(raw, ns=(entry["raw_mtime_ns"], entry["raw_mtime_ns"]))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"[{index}/{len(selected)}] restored {raw}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("paths", nargs="+")
    compress_parser.add_argument("--manifest", type=pathlib.Path, required=True)
    compress_parser.add_argument("--level", type=int, choices=range(1, 10), default=6)
    compress_parser.add_argument("--keep-raw", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("manifest", type=pathlib.Path)

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("manifest", type=pathlib.Path)
    restore_parser.add_argument("paths", nargs="*")

    args = parser.parse_args()
    if args.command == "compress":
        compress(args.paths, args.manifest, args.level, keep_raw=args.keep_raw)
    elif args.command == "verify":
        verify(args.manifest)
    else:
        restore(args.manifest, args.paths)


if __name__ == "__main__":
    main()
