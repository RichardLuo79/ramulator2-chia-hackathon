"""Trusted, auth-only access to the operator's existing Codex login.

This helper is NOT an experimental agent. It never creates, resumes, lists or
reads threads. Only Codex manages OAuth refresh; credentials never enter the
experimental CLI home, environment, request journal or model context.
"""
from __future__ import annotations

import base64
import json
import math
import os
import pathlib
import selectors
import subprocess
import tempfile
import time

from tools.chia_loop import recovery as R


def read_tokens(auth_file):
    try:
        data = json.loads(pathlib.Path(auth_file).read_text())
        tokens = data.get("tokens") or {}
        if data.get("auth_mode") != "chatgpt" or not all(
                isinstance(tokens.get(k), str) and tokens[k]
                for k in ("access_token", "refresh_token", "account_id")):
            raise ValueError("not a managed ChatGPT login")
        return tokens
    except (OSError, ValueError, TypeError, AttributeError):
        raise RuntimeError("existing managed ChatGPT login is unavailable; use codex login") from None


def expires_soon(access, now=None):
    # Decode only the expiry hint. This is not token/signature verification;
    # the provider authenticates the request. Unknown formats require refresh.
    try:
        payload = access.split(".")[1]
        expiry = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"]
        return (isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or not math.isfinite(expiry)
                or expiry <= (time.time() if now is None else now) + 300)
    except (ValueError, KeyError, IndexError, TypeError):
        return True


def account_read(binary, auth_file, *, refresh=False, timeout=60):
    """Use a fresh stdio app-server solely for initialize/account/read.

The trusted helper uses the existing credential store, not a copied OAuth
refresh token. No generation endpoint or thread/turn method is invoked.
Account details and server diagnostics are discarded rather than logged.
"""
    auth_file = pathlib.Path(auth_file).resolve(strict=True)
    if auth_file.name != "auth.json":
        raise ValueError("Codex-managed refresh requires its existing auth.json")
    from .transport import DISABLED
    args = [str(binary), "app-server", "--stdio", "-c", 'model_provider="openai"',
            "-c", 'cli_auth_credentials_store="file"', "-c", "analytics.enabled=false"]
    for feature in DISABLED:
        args += ["--disable", feature]
    with tempfile.TemporaryDirectory(prefix="chia-codex-auth-") as scratch:
        # These child-only home settings do not change the orchestrator's home.
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": scratch,
               "CODEX_HOME": str(auth_file.parent), "TMPDIR": scratch,
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        with subprocess.Popen(args, cwd=scratch, env=env, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              bufsize=0, close_fds=True, start_new_session=True) as process:
            deadline = time.monotonic() + timeout
            pending = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                def send(message):
                    process.stdin.write((json.dumps(message) + "\n").encode())

                def receive(request_id):
                    while time.monotonic() < deadline:
                        while b"\n" in pending:
                            line, _, rest = pending.partition(b"\n")
                            pending[:] = rest
                            event = json.loads(line)
                            if event.get("id") == request_id:
                                if "error" in event or "result" not in event:
                                    raise RuntimeError("Codex auth-only request failed")
                                return event["result"]
                            if "method" in event and "id" in event:
                                raise RuntimeError("unexpected server request during auth-only check")
                        if not selector.select(max(0, deadline - time.monotonic())):
                            break
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            raise RuntimeError("Codex auth-only helper exited early")
                        pending.extend(chunk)
                        if len(pending) > 1_000_000:
                            raise RuntimeError("auth-only protocol size guard")
                    raise TimeoutError("Codex auth-only check timed out")

                try:
                    send({"id": 0, "method": "initialize", "params": {"clientInfo": {
                        "name": "chia_auth_check", "title": "CHIA auth-only helper", "version": "1.0"}}})
                    receive(0)
                    send({"method": "initialized", "params": {}})
                    send({"id": 1, "method": "account/read", "params": {"refreshToken": refresh}})
                    result = receive(1)
                    if (result.get("account") or {}).get("type") != "chatgpt":
                        raise RuntimeError("Codex did not report an existing ChatGPT login")
                    return {"auth_mode": "chatgpt", "account_check": "passed", "refresh_requested": refresh,
                            "thread_methods_called": 0, "generation_calls": 0}
                except Exception:
                    # Native error strings/account payloads can contain PII or
                    # credentials. Preserve neither in exception chains/logs.
                    raise RuntimeError("Codex auth-only check failed; recheck the existing login") from None
                finally:
                    process.stdin.close()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


def headers(binary, auth_file):
    auth_file = pathlib.Path(auth_file).resolve(strict=True)
    if auth_file.name != "auth.json":
        raise ValueError("use the existing Codex auth.json")
    # Serialize CHIA refreshes across both independent effort runs. Codex owns
    # its native credential persistence; never implement a competing OAuth flow.
    with R.exclusive_lock(auth_file.parent / ".chia-auth-refresh.lock"):
        tokens = read_tokens(auth_file)
        if expires_soon(tokens["access_token"]):
            account_read(binary, auth_file, refresh=True)
            tokens = read_tokens(auth_file)
            if expires_soon(tokens["access_token"]):
                raise RuntimeError("Codex did not persist usable refreshed credentials")
        return {"Authorization": "Bearer " + tokens["access_token"],
                "ChatGPT-Account-Id": tokens["account_id"], "originator": "codex_exec"}
