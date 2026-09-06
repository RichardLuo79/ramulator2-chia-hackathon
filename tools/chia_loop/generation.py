"""Journaled, budgeted Gemini requests shared by proposer and isolated reviewer.

The operation key is stable across process restarts. A durable complete response
is replayed, never generated twice. A request with an unknown outcome retains
its full reservation; retrying is a NEW billable attempt, not free recovery.
"""
from __future__ import annotations

import json
import pathlib
import random
import time

import httpx

from tools.chia_loop import real_core as P
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.recovery import (OperationalPause, SearchLimit, check_stop, check_run_deadline, check_storage,
                                     exclusive_lock, exists, read_json, wait_until)


def transient_generation_error(exc):
    return (getattr(exc, "code", None) in (408, 429, 499, 500, 502, 503, 504)
            or isinstance(exc, (httpx.NetworkError, httpx.TimeoutException,
                                httpx.RemoteProtocolError)))


def generate(root, ledger, policy, *, project, backend, purpose, iteration, turn,
             operation_key, system, contents):
    from google import genai
    from google.genai import types
    root = pathlib.Path(root)
    check_stop(root)
    # Keys are runner-created, not provider-supplied paths.
    if not operation_key.replace("_", "").isalnum():
        raise ValueError("unsafe generation operation key")
    checkpoint = root / "checkpoints" / (operation_key + ".json")
    with exclusive_lock(checkpoint.with_suffix(".lock")):
        config = types.GenerateContentConfig(system_instruction=system, temperature=1.0,
            max_output_tokens=P.MAX_OUTPUT,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH", include_thoughts=True),
            response_mime_type="application/json")
        payload = {"model": P.MODELS[backend], "config": config.model_dump(mode="json", exclude_none=True),
                   "contents": [c.model_dump(mode="json", exclude_none=True) for c in contents]}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        identity = {"run_id": root.name, "operation_key": operation_key, "backend": backend,
                    "purpose": purpose, "iteration": iteration, "turn": turn,
                    "payload_sha256": P.sha(encoded)}
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if saved["identity"] != identity:
                raise RuntimeError("generation checkpoint identity/context changed")
        else:
            # The proposal checkpoint reconstructs this request. Keep its hash,
            # not another full copy of an ever-growing conversation per turn.
            saved = {"identity": identity, "failures": 0}
            atomic_write_json(checkpoint, saved)
        if saved.get("blocked"):
            raise OperationalPause(saved["blocked"], retryable=False)

        directory = root / "interactions" / f"iter_{iteration:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        # Reconcile even a crash between reserve(), saving request, and dispatch.
        for row in ledger.calls():
            if row.get("operation_key") != operation_key:
                continue
            if row["payload_sha256"] != identity["payload_sha256"]:
                raise RuntimeError("ledger operation context mismatch")
            prefix = directory / f"call_{row['id']:03d}"
            if not exists(prefix.with_suffix(".request.json")):
                atomic_write_json(prefix.with_suffix(".request.json"), payload)
            if exists(prefix.with_suffix(".response.json")):
                raw = read_json(prefix.with_suffix(".response.json"))
                evidence_path = prefix.with_suffix(".call.json")
                if exists(evidence_path) and read_json(evidence_path)["response_sha256"] != P.sha(json.dumps(raw, sort_keys=True)):
                    raise RuntimeError("saved response disagrees with its durable receipt")
                if row["state"] == "usage_recorded" and row["usage"] != raw.get("usage_metadata"):
                    raise RuntimeError("saved response disagrees with ledger usage")
                ledger.settle(row["id"], raw.get("usage_metadata"))
                if not exists(evidence_path):
                    atomic_write_json(evidence_path, {"wall_s": None, "recovered_after_crash": True,
                        "iteration": iteration, "turn": turn, "purpose": purpose,
                        "response_sha256": P.sha(json.dumps(raw, sort_keys=True)), "usage": raw.get("usage_metadata")})
                saved["complete_call_id"] = row["id"]
                atomic_write_json(checkpoint, saved)
                return raw
            if row["state"] == "reserved":
                ledger.settle(row["id"], None, "process interrupted; response outcome unknown")
                atomic_write_json(prefix.with_suffix(".error.json"), {
                    "error": "process interrupted; response outcome unknown", "recovered_after_crash": True})
                saved["failures"] += 1
                saved.setdefault("outage_started", time.time())
                atomic_write_json(checkpoint, saved)

        if saved.get("retry_at", 0) > time.time():
            raise OperationalPause("provider recovery cooldown", retry_at=saved["retry_at"])
        for retry in range(policy["transient_retries_per_turn"] + 1):
            check_stop(root)
            check_run_deadline(root)
            check_storage(root, policy["minimum_free_disk_bytes"])
            if saved.get("outage_started") and time.time() - saved["outage_started"] >= policy["maximum_outage_seconds"]:
                raise OperationalPause("provider outage recovery window exhausted", retryable=False)
            if saved["failures"] >= policy["maximum_transport_failures_per_operation"]:
                raise OperationalPause("provider transport recovery attempt limit exhausted", retryable=False)
            if sum(r["iteration"] == iteration for r in ledger.calls()) >= policy["api_attempts_per_proposal"]:
                raise SearchLimit("iteration API-attempt safety limit (proposal and review)")
            client = genai.Client(vertexai=True, project=project, location="global",
                http_options=types.HttpOptions(timeout=policy["provider_timeout_seconds"] * 1000,
                    retry_options=types.HttpRetryOptions(attempts=1)))
            call_id = None
            started = time.time()
            try:
                try:
                    counted = client.models.count_tokens(model=P.MODELS[backend], contents=contents,
                        config=types.CountTokensConfig(system_instruction=system))
                except Exception as exc:
                    _failure(saved, checkpoint, policy, exc, retry, root)
                    continue
                check_stop(root)
                call_id = ledger.reserve(encoded, iteration, turn, input_tokens=counted.total_tokens,
                    purpose=purpose, backend=backend, operation_key=operation_key)
                prefix = directory / f"call_{call_id:03d}"
                atomic_write_json(prefix.with_suffix(".request.json"), payload)
                # Catch transport/provider failures ONLY around the API call.
                # Disk, accounting and local serialization faults must fail closed.
                try:
                    response = client.models.generate_content(model=P.MODELS[backend], contents=contents, config=config)
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {str(exc)[:2000]}"
                    ledger.settle(call_id, None, detail, http_status=getattr(exc, "code", None))
                    atomic_write_json(prefix.with_suffix(".error.json"), {
                        "error": detail, "retry": retry, "purpose": purpose, "wall_s": time.time() - started})
                    _failure(saved, checkpoint, policy, exc, retry, root)
                    continue
                raw = response.model_dump(mode="json", exclude_none=True)
                atomic_write_json(prefix.with_suffix(".response.json"), raw)
                ledger.settle(call_id, raw.get("usage_metadata"))
                atomic_write_json(prefix.with_suffix(".call.json"), {
                    "wall_s": time.time() - started, "iteration": iteration, "turn": turn,
                    "retry": retry, "purpose": purpose,
                    "response_sha256": P.sha(json.dumps(raw, sort_keys=True)), "usage": raw.get("usage_metadata")})
                saved["complete_call_id"] = call_id
                atomic_write_json(checkpoint, saved)
                return raw
            finally:
                client.close()
        raise AssertionError("request retry loop ended without an outcome")


def _failure(saved, checkpoint, policy, exc, retry, root):
    detail = f"{type(exc).__name__}: {str(exc)[:2000]}"
    saved["last_error"] = detail
    if not transient_generation_error(exc):
        saved["blocked"] = detail
        atomic_write_json(checkpoint, saved)
        raise OperationalPause(detail, retryable=False) from exc
    saved["failures"] += 1
    saved.setdefault("outage_started", time.time())
    exhausted = saved["failures"] >= policy["maximum_transport_failures_per_operation"]
    if retry == policy["transient_retries_per_turn"]:
        delay = min(policy["recovery_cooldown_max_seconds"],
                    policy["recovery_cooldown_seconds"] * 2 ** min(saved["failures"] // 3, 8))
    else:
        delay = 2 ** (retry + 1)
    saved["retry_at"] = time.time() + delay * random.uniform(1, 1.2)
    atomic_write_json(checkpoint, saved)
    check_stop(root)
    if exhausted or retry == policy["transient_retries_per_turn"]:
        raise OperationalPause(detail, retryable=not exhausted, retry_at=saved["retry_at"]) from exc
    wait_until(root, saved["retry_at"])
