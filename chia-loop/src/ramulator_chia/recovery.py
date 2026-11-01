"""Operational state and local supervision, independent of scientific selection."""
from __future__ import annotations

import contextlib
import fcntl
import gzip
import json
import pathlib
import shutil
import time


class OperationalPause(RuntimeError):
    def __init__(self, reason, *, retryable=True, retry_at=0):
        super().__init__(reason)
        self.retryable, self.retry_at = retryable, retry_at


class OperatorStop(OperationalPause):
    def __init__(self, reason="operator stop requested"):
        super().__init__(reason, retryable=False)


class SearchLimit(RuntimeError):
    """A declared finite search guard, not an infrastructure failure."""


def read_json(path):
    """Completed interaction records may already be compressed on resume."""
    path = pathlib.Path(path)
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def exists(path):
    path = pathlib.Path(path)
    return path.exists() or pathlib.Path(str(path) + ".gz").exists()


@contextlib.contextmanager
def exclusive_lock(path):
    """Nonblocking advisory lease; a dead process cannot leave a stale PID lock."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OperationalPause("another worker still owns " + str(path),
                                   retry_at=time.time() + 10) from exc
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextlib.contextmanager
def cpu_lease(directory, count):
    """All new runners on this checkout share twelve process-lifetime slots."""
    if not isinstance(count, int) or isinstance(count, bool) or not 3 <= count <= 12:
        raise ValueError("CPU lease must be in [3, 12]")
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    held = []
    try:
        for index in range(12):
            stream = (directory / f"cpu-{index:02d}.lock").open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            held.append(stream)
            if len(held) == count:
                break
        if len(held) != count:
            raise OperationalPause("not enough of the shared twelve CPU slots available",
                                   retry_at=time.time() + 10)
        yield
    finally:
        for stream in held:
            stream.close()


def check_stop(root):
    if (pathlib.Path(root) / "STOP").exists():
        raise OperatorStop()


def check_run_deadline(root):
    path = pathlib.Path(root) / "run_manifest.json"
    if path.exists():
        manifest = read_json(path)
        duration = manifest.get("policy", {}).get("maximum_run_wall_seconds")
        if duration and time.time() >= manifest["started_at"] + duration:
            raise OperationalPause("run wall-time guard reached", retryable=False)


def check_storage(root, minimum_bytes=8 * 1024**3):
    if shutil.disk_usage(root).free < minimum_bytes:
        raise OperationalPause("insufficient free disk space; no automatic deletion is permitted", retryable=False)


def wait_until(root, deadline):
    # Interruptible even during an extended service cooldown.
    while time.time() < deadline:
        check_stop(root)
        time.sleep(min(1, max(0, deadline - time.time())))
    check_stop(root)


def has_test_started(root):
    root = pathlib.Path(root)
    if (root / "test_started.json").exists() or (root / "test").exists():
        return True
    events = root / "events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            if line.strip() and json.loads(line).get("event") == "frozen_test_started":
                return True
    return False


def assert_training_open(root):
    root = pathlib.Path(root)
    if has_test_started(root) or (root / "selection_frozen.json").exists():
        raise RuntimeError("optimization cannot resume after selection freeze/test exposure")
    state = root / "state.json"
    if state.exists() and read_json(state).get("status") == "frozen":
        raise RuntimeError("optimization cannot resume a frozen selection")

