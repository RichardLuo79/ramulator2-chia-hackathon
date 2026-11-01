"""Read actual submitted files and publish immutable, content-bound candidates.

This is a task-level source contract, not a patch language or a C++ parser.
Only named model files and a model-parameter file are copied. The trusted store
must be outside the agent's writable mounts. Read-only modes prevent accidents;
the process boundary, not Unix ownership alone, protects snapshots from agents.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .archive import safe_name
from .identity import canonical_json, digest_json


class InvalidSnapshot(ValueError):
    """A submission violates its file contract or recorded identity."""


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise InvalidSnapshot("expected a lowercase SHA-256 identity")
    return value


@dataclass(frozen=True)
class ModelFiles:
    """Positive model-file grant. Byte limits are supplied by recorded launch policy."""

    sources: tuple[str, ...]
    parameters: str

    def __post_init__(self):
        if type(self.sources) is not tuple or not self.sources:
            raise InvalidSnapshot("source files must be a nonempty ordered tuple")
        paths = (*self.sources, self.parameters)
        if len(set(paths)) != len(paths):
            raise InvalidSnapshot("model file names must be unique")
        for path in paths:
            safe_name(path)
            if path == "snapshot.json" or any(other.startswith(path + "/") for other in paths):
                raise InvalidSnapshot("model file names conflict with the snapshot layout")

    @property
    def paths(self) -> tuple[str, ...]:
        return (*self.sources, self.parameters)

    def identity(self) -> str:
        return digest_json(asdict(self))


def parse_parameters(data: bytes) -> dict[str, int | float]:
    """Only model-owned numeric overrides; no frontend or hardware configuration."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidSnapshot(f"duplicate model parameter: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidSnapshot("model parameters must be a JSON object") from exc
    if type(value) is not dict:
        raise InvalidSnapshot("model parameters must be a JSON object")
    for name, number in value.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise InvalidSnapshot(f"invalid model parameter name: {name!r}")
        try:
            finite = type(number) in {int, float} and math.isfinite(number)
        except OverflowError:
            finite = False
        if not finite:
            raise InvalidSnapshot(f"model parameter must be finite and numeric: {name}")
    return value


@contextmanager
def _parent(root: Path, relative: str, *, create_parents: bool = False):
    """Walk with openat/O_NOFOLLOW; agent-controlled intermediate links cannot escape."""
    safe_name(relative)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            if create_parents:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor, parts[-1]
    finally:
        os.close(descriptor)


def regular_path(root: Path, relative: str) -> Path:
    """Check a settled trusted file without following symlink components."""
    with _parent(root, relative) as (directory, name):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise InvalidSnapshot(f"not a regular file: {relative}")
    return root / relative


def replace_file(root: Path, relative: str, data: bytes):
    """Atomic native-session/model edit through the existing no-follow reader boundary."""
    with _parent(root, relative, create_parents=True) as (directory, name):
        try:
            old = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            old = None
        if old is not None and (not stat.S_ISREG(old.st_mode) or old.st_nlink != 1):
            raise InvalidSnapshot("editable file must be a regular, unshared file")
        temporary = ".edit-" + secrets.token_hex(16)
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)


def read_file(
    root: Path, relative: str, *, maximum_bytes: int, require_single_link: bool = False
) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise ValueError("maximum_bytes must be a nonnegative integer")
    with _parent(root, relative) as (directory, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise InvalidSnapshot(f"model input is not a regular file: {relative}")
            if require_single_link and before.st_nlink != 1:
                raise InvalidSnapshot(f"workspace input has another filesystem alias: {relative}")
            data = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
    if len(data) > maximum_bytes:
        raise InvalidSnapshot("submission exceeds its declared byte guard")
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise InvalidSnapshot(f"model input changed while being read: {relative}")
    return data


def _fsync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes(path: Path, data: bytes):
    """Publish one settled trusted-store object, never overwrite another identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(temporary)
    # On write/disk failure leave the partial file for incident inspection.
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    with _parent(path.parent, path.name) as (directory, name):
        # Complete publication before another publisher verifies the same name.
        # Removing the temporary hard link changes inode ctime even though the
        # bytes are immutable; an overlapping duplicate read must not mistake
        # that transition for a source mutation. Lock the existing directory,
        # not an extra lock file inside the candidate's exact file inventory.
        fcntl.flock(directory, fcntl.LOCK_EX)
        try:
            os.link(
                temporary.name,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError:
            if read_file(path.parent, path.name, maximum_bytes=len(data)) != data:
                raise InvalidSnapshot(f"published object already has different bytes: {path.name}")
        os.unlink(temporary.name, dir_fd=directory)
        os.fsync(directory)


def _manifest(files: dict[str, bytes], contract: ModelFiles) -> dict:
    parameters = parse_parameters(files[contract.parameters])
    inventory = {
        name: {"sha256": _hash(data), "bytes": len(data)} for name, data in sorted(files.items())
    }
    return {
        "schema_version": 1,
        "contract_sha256": contract.identity(),
        "source_sha256": digest_json({name: inventory[name] for name in contract.sources}),
        "configuration_sha256": digest_json(parameters),
        "parameters": parameters,
        "files": inventory,
    }


def snapshot(workspace: Path, store: Path, contract: ModelFiles, *, maximum_bytes: int) -> dict:
    """Snapshot a submitted draft, independent of the incumbent or any answer text."""
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("the source byte guard must be a recorded positive integer")
    files = {}
    remaining = maximum_bytes
    for name in contract.paths:
        data = read_file(workspace, name, maximum_bytes=remaining, require_single_link=True)
        files[name] = data
        remaining -= len(data)
    manifest = _manifest(files, contract)
    candidate_id = digest_json(manifest)
    destination = store / candidate_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        publish_bytes(destination / name, data)
    # The manifest is the commit marker. No reader accepts a directory alone.
    publish_bytes(destination / "snapshot.json", canonical_json(manifest).encode())
    _fsync_directory(destination)
    _fsync_directory(store)
    return {"candidate_id": candidate_id, **manifest}


def verify_snapshot(
    store: Path, candidate_id: str, contract: ModelFiles, *, maximum_bytes: int
) -> dict:
    _digest(candidate_id)
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("the source byte guard must be a recorded positive integer")
    root = store / candidate_id
    # The inventory itself is proportional to the explicit file grant; its
    # independent bound is not an artificial limit on generated model code.
    raw = read_file(
        root,
        "snapshot.json",
        maximum_bytes=maximum_bytes + 4096 + sum(len(p) + 256 for p in contract.paths),
    )
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidSnapshot("invalid snapshot manifest") from exc
    if _hash(raw) != candidate_id or raw != canonical_json(manifest).encode():
        raise InvalidSnapshot("snapshot manifest identity or canonical encoding changed")
    files, remaining = {}, maximum_bytes
    for name in contract.paths:
        data = read_file(root, name, maximum_bytes=remaining)
        files[name] = data
        remaining -= len(data)
    if _manifest(files, contract) != manifest:
        raise InvalidSnapshot("snapshot contents or file contract changed")
    return {"candidate_id": candidate_id, **manifest}


def materialize(
    store: Path, candidate_id: str, workspace: Path, contract: ModelFiles, *, maximum_bytes: int
) -> dict:
    """Start a new writable episode workspace from a verified immutable parent."""
    manifest = verify_snapshot(store, candidate_id, contract, maximum_bytes=maximum_bytes)
    workspace.mkdir(parents=True, exist_ok=False)
    for name in contract.paths:
        data = read_file(store / candidate_id, name, maximum_bytes=maximum_bytes)
        if _hash(data) != manifest["files"][name]["sha256"]:
            raise InvalidSnapshot("parent changed after verification")
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return manifest
