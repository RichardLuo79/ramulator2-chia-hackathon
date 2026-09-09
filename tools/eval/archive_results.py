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


def _discover_raw(paths, *, allow_incomplete=False) -> list[pathlib.Path]:
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
                    and (allow_incomplete or not any(part.startswith(".incomplete-")
                                                    for part in candidate.parts))):
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


def compress(paths, manifest_path: pathlib.Path, level: int, *, keep_raw=False,
             allow_incomplete=False) -> None:
    # Opt-in is only for reaped, failed simulations with separate failure evidence.
    raw_paths = _discover_raw(paths, allow_incomplete=allow_incomplete)
    if not raw_paths:
        raise RuntimeError("no uncompressed *.chN trace artifacts found")
    compress_files(raw_paths, manifest_path, level, keep_raw=keep_raw)


def compress_files(paths, manifest_path: pathlib.Path, level: int, *, keep_raw=False) -> None:
    """Archive an explicit inventory without assuming channel-suffixed names.

    The caller establishes completion/eligibility. This shares the existing gzip,
    identity, manifest and removal workflow; it performs no directory discovery.
    The legacy command's *.chN selection and incomplete-run policy are unchanged.
    """
    raw_paths = []
    for item in paths:
        path = pathlib.Path(item)
        if path.is_symlink() or not path.is_file() or path.suffix == A.GZIP_SUFFIX:
            raise ValueError(f"explicit archive input must be an uncompressed regular file: {path}")
        raw_paths.append(path.resolve())
    if not raw_paths or len(raw_paths) != len(set(raw_paths)):
        raise ValueError("explicit archive inventory must be nonempty and unique")
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
        reduction = (100.0 * (1.0 - archive_provenance["size"] / raw_provenance["size"])
                     if raw_provenance["size"] else 0.0)
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


def recover_duplicate(manifest_path: pathlib.Path, raw_name: str, source: pathlib.Path):
    """Restore only byte-identical archived data; preserve the damaged original."""
    entry = _load_manifest(manifest_path)["artifacts"][raw_name]
    _, archived = _entry_paths(entry, manifest_path)
    source = pathlib.Path(source).resolve()
    if archived.is_symlink() or not archived.is_file() or source == archived.resolve():
        raise RuntimeError("recovery requires a regular target and a distinct duplicate")
    expected = (entry["archive_size"], entry["archive_sha256"])
    original = _physical_provenance(archived)
    if (original["size"], original["sha256"]) == expected:
        raise RuntimeError("archive already matches its original checksum")
    replacement = _physical_provenance(source)
    if (replacement["size"], replacement["sha256"]) != expected:
        raise RuntimeError("duplicate does not match the original archive checksum")
    unpacked = A.raw_provenance(source)
    if (unpacked["size"], unpacked["sha256"]) != (entry["raw_size"], entry["raw_sha256"]):
        raise RuntimeError("duplicate does not match the original raw checksum")
    quarantine = archived.with_name(archived.name + ".corrupt-" + original["sha256"][:16])
    record_path = archived.with_name(archived.name + ".recovery.json")
    if quarantine.exists() or record_path.exists():
        raise RuntimeError("existing recovery evidence must not be overwritten")
    record = {"status": "prepared", "utc": _utc_now(), "target": str(archived.resolve()),
              "duplicate": str(source), "quarantine": str(quarantine.resolve()),
              "original_damaged_sha256": original["sha256"],
              "restored_archive_sha256": entry["archive_sha256"],
              "unchanged_raw_sha256": entry["raw_sha256"]}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=archived.parent,
                prefix=".verified-recovery-", suffix=".gz", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        copied = _physical_provenance(temporary)
        if (copied["size"], copied["sha256"]) != expected:
            raise RuntimeError("duplicate changed while copying")
        if _physical_provenance(archived)["sha256"] != original["sha256"]:
            raise RuntimeError("target changed during recovery")
        C.atomic_write_json(record_path, record)
        previous_stat = archived.stat()
        os.replace(archived, quarantine)
        os.replace(temporary, archived)
        temporary = None
        os.utime(archived, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
        record.update(status="restored", finished_utc=_utc_now())
        C.atomic_write_json(record_path, record)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


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

    recovery_parser = subparsers.add_parser("recover-duplicate")
    recovery_parser.add_argument("manifest", type=pathlib.Path)
    recovery_parser.add_argument("raw_name", help="exact raw artifact key in the manifest")
    recovery_parser.add_argument("source", type=pathlib.Path)

    args = parser.parse_args()
    if args.command == "compress":
        compress(args.paths, args.manifest, args.level, keep_raw=args.keep_raw)
    elif args.command == "verify":
        verify(args.manifest)
    elif args.command == "recover-duplicate":
        print(recover_duplicate(args.manifest, args.raw_name, args.source))
    else:
        restore(args.manifest, args.paths)


if __name__ == "__main__":
    main()
