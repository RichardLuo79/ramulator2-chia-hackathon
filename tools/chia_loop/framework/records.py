"""Small campaign checkpoints on CHIA's SQLite primitives.

The driver is the only writer (under the campaign lease). Native transports
retain their own session/evidence files. This store neither schedules work nor
tries to reconstruct missing provider replies.
"""

from __future__ import annotations

import json
import time
import traceback
from inspect import unwrap
from pathlib import Path
from typing import Callable

from chia.database.sqlite_node import SQLiteNode

from .identity import canonical_json, digest_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attempts (
    step TEXT NOT NULL,
    number INTEGER NOT NULL,
    input_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('started', 'complete', 'failed')),
    result TEXT,
    error TEXT,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    finished_at TEXT,
    PRIMARY KEY(step, number)
);
"""


class RecordConflict(RuntimeError):
    pass


class UnresolvedStep(RuntimeError):
    """An operation may have completed; repeating it is not known to be safe."""


class TransientFailure(RuntimeError):
    """An adapter has established that this failure is safe to retry."""

    def __init__(self, message, *, retry_after_seconds=None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class CampaignState:
    def __init__(self, path: Path):
        self.path = str(path.absolute())
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reuse CHIA's SQLite operations locally. Its profiling decorator looks
        # up a Ray actor even for local calls, which auto-starts a cluster during
        # offline inspection/export. Standard unwrap bypasses only that wrapper.
        unwrap(SQLiteNode.init_schema)(self.path, SCHEMA, connect_opts={"synchronous": "FULL"})

    def _query(self, sql, args=()):
        return unwrap(SQLiteNode.query)(self.path, sql, args)

    def _write(self, sql, args=()):
        return unwrap(SQLiteNode.execute)(
            self.path, sql, args, connect_opts={"synchronous": "FULL"}
        )

    def get(self, key: str):
        rows = self._query("SELECT value FROM records WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else None

    def completed_step(self, key: str):
        rows = self._query(
            "SELECT result FROM attempts WHERE step=? AND status='complete' "
            "ORDER BY number DESC LIMIT 1",
            (key,),
        )
        return json.loads(rows[0]["result"]) if rows else None

    def save(self, key: str, value: dict):
        encoded = canonical_json(value)
        self._write("INSERT OR IGNORE INTO records VALUES (?, ?)", (key, encoded))
        if self.get(key) != value:
            raise RecordConflict(f"cannot change committed campaign record: {key}")
        return value

    def step(
        self,
        key: str,
        inputs: dict,
        action: Callable[[], dict],
        *,
        maximum_attempts: int,
        retry_delay_seconds: int,
        repeat_after_interruption: bool = False,
    ) -> dict:
        """Reuse a completed step; never silently repeat a lost model reply.

        Only adapters may classify a failure as transient/safe. Ordinary
        exceptions, interruptions and invalid output remain visible. Retries
        count across process restarts, not just within the current Python call.
        """
        identity = digest_json(inputs)
        while True:
            rows = self._query("SELECT * FROM attempts WHERE step=? ORDER BY number", (key,))
            if any(row["input_sha256"] != identity for row in rows):
                raise RecordConflict(f"step inputs changed: {key}")
            previous = rows[-1] if rows else None
            if previous and previous["status"] == "complete":
                return json.loads(previous["result"])
            if previous and previous["status"] == "started" and not repeat_after_interruption:
                raise UnresolvedStep(f"reconcile interrupted operation before continuing: {key}")
            if len(rows) >= maximum_attempts:
                raise UnresolvedStep(f"attempt limit reached: {key}")
            if previous and previous["status"] == "failed":
                try:
                    retry = json.loads(previous["error"] or "null")
                except ValueError:
                    retry = None  # Older receipts contain a plain error message.
                if isinstance(retry, dict) and "retry_not_before" in retry:
                    deadline = retry["retry_not_before"]
                    if deadline > time.time():
                        print(
                            canonical_json(
                                {
                                    "event": "retry_cooldown",
                                    "step": key,
                                    "after_attempt": previous["number"],
                                    "retry_not_before": deadline,
                                }
                            ),
                            flush=True,
                        )
                    while (remaining := deadline - time.time()) > 0:
                        time.sleep(min(remaining, 55))
            number = len(rows) + 1
            self._write(
                "INSERT INTO attempts(step, number, input_sha256, status) "
                "VALUES (?, ?, ?, 'started')",
                (key, number, identity),
            )
            try:
                result = action()
                if not isinstance(result, dict):
                    raise TypeError(f"step result must be an object: {key}")
                encoded = canonical_json(result)
            except TransientFailure as exc:
                delay = (
                    retry_delay_seconds
                    if exc.retry_after_seconds is None
                    else exc.retry_after_seconds
                )
                retry = {"message": str(exc), "retry_not_before": time.time() + delay}
                self._write(
                    "UPDATE attempts SET status='failed', error=?, finished_at="
                    "strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE step=? AND number=?",
                    (canonical_json(retry), key, number),
                )
                if number >= maximum_attempts:
                    raise
                continue
            except BaseException:
                # Preserve the failure but leave 'started': an exception alone
                # cannot prove whether the provider or a tool completed work.
                self._write(
                    "UPDATE attempts SET error=? WHERE step=? AND number=?",
                    (traceback.format_exc(), key, number),
                )
                raise
            self._write(
                "UPDATE attempts SET status='complete', result=?, finished_at="
                "strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE step=? AND number=?",
                (encoded, key, number),
            )
            return result
