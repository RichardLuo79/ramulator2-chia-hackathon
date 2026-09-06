"""One fresh Codex CLI process per CHIA action, through an audited broker.

Codex has no native tools here. JSON inspection actions are executed by CHIA,
not by a model-generated shell. Credentials and the complete experiment root
remain outside its filesystem domain. Only the broker can call the provider.
"""
from __future__ import annotations

import http.server
import asyncio
import gzip
import json
import math
import os
import pathlib
import secrets
import subprocess
import sys
import threading
import time
import uuid

from tools.chia_loop import real_core as P, recovery as R
from tools.chia_loop.core import atomic_write_json
from . import usage as U
from . import budget as F
from .stream import terminal_response

MODEL = "gpt-6-astra"
EFFORTS = ("xhigh", "max")
REASONING_SUMMARY = "auto"
MAX_OUTPUT = 128_000
MAX_INPUT_BYTES = 900_000
HERE = pathlib.Path(__file__).resolve().parent
DISABLED = ("apps", "plugins", "remote_plugin", "memories", "multi_agent", "multi_agent_v2",
            "shell_tool", "unified_exec", "shell_snapshot", "code_mode", "code_mode_host",
            "computer_use", "browser_use", "browser_use_external", "in_app_browser",
            "image_generation", "view_image", "goals", "hooks", "in_app_local_automation",
            "skill_search", "auth_elicitation", "unbounded_connection_retries",
            "enable_request_compression")


class Ledger:
    """API-tariff accounting, including unknown responses, before every dispatch.

ChatGPT-auth usage is NOT an API invoice. For that mode these are conservative
API-equivalent experiment guards; subscription quotas remain provider-owned.
"""
    def __init__(self, root, cap):
        if cap is None:
            F.check_iteration_authorization(root)
        elif isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("a finite positive per-run cap is required")
        self.root, self.path, self.cap = pathlib.Path(root), pathlib.Path(root) / "ledger.json", cap

    def transaction(self, update):
        with R.exclusive_lock(self.path.with_suffix(".lock")):
            if self.cap is None:
                F.check_iteration_authorization(self.root)
            carryover = F.load(self.root, MODEL, self.cap)
            data = R.read_json(self.path) if self.path.exists() else {
                "run_id": self.root.name, "model": MODEL, "cap_usd": self.cap, "calls": [],
                "schema_version": U.SCHEMA, "tariff": U.TARIFF, "currency": "USD", "invoice": False,
                "carryover": carryover}
            if (data["run_id"] != self.root.name or data["cap_usd"] != self.cap
                    or data.get("model") != MODEL or data.get("tariff") != U.TARIFF
                    or data.get("carryover") != carryover):
                raise RuntimeError("cannot replace a run's ledger/authorization")
            result = update(data)
            atomic_write_json(self.path, data)
            return result

    def reserve(self, request, role, operation, *, attempt_path=None, attempt_number=None, auth_mode=None):
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True).encode()
        if len(encoded) > MAX_INPUT_BYTES:
            raise P.BudgetExhausted("input byte safety limit; no silent compaction")
        # A byte is a conservative token bound, plus framing allowance; use
        # Long-context input/cache-write and output tariffs. The input guard
        # covers the higher cache-write rate, not just uncached input.
        reserve = ((len(encoded) + 8192) * U.TARIFF["guard_input"] + MAX_OUTPUT * U.TARIFF["guard_output"]) / 1e6
        def update(data):
            prior_charge = (data.get("carryover") or {}).get("totals", {}).get("conservative_guard_usd", 0)
            if self.cap is not None and prior_charge + sum(c["cap_charge_usd"] for c in data["calls"]) + reserve > self.cap:
                raise P.BudgetExhausted("next request would exceed the run's conservative cap")
            row = {"id": len(data["calls"]), "operation": operation, "role": role,
                   "request_sha256": P.sha(encoded), "reserved_at": time.time(),
                   "cap_charge_usd": reserve, "reserved_usd": reserve,
                   "state": "unknown_reservation_retained", "known_standard_usd": None,
                   "model": request.get("model"), "effort": (request.get("reasoning") or {}).get("effort"),
                   "reasoning_summary": (request.get("reasoning") or {}).get("summary"),
                   "auth_mode": auth_mode, "attempt_path": attempt_path, "attempt_number": attempt_number,
                   "input_bytes": len(encoded), **U.dimensions(operation)}
            data["calls"].append(row)
            return row["id"]
        return self.transaction(update)

    def settle(self, index, response, *, response_sha256=None):
        usage = response.get("usage")
        tokens = U.normalized(usage)
        metadata = {"response_id": response.get("id"), "response_model": response.get("model"),
                    "response_effort": (response.get("reasoning") or {}).get("effort"),
                    "response_status": response.get("status"), "service_tier": response.get("service_tier"),
                    "incomplete_reason": (response.get("incomplete_details") or {}).get("reason")}
        if response_sha256:
            metadata["response_sha256"] = response_sha256
        if response.get("status") == "completed":
            parsed = action(response)
            metadata["action_status"] = parsed.get("status", parsed.get("verdict"))
        def update(data):
            row = data["calls"][index]
            if row["state"] == "usage_recorded":
                if row["usage"] != usage or any(row.get(k) != v for k, v in metadata.items()):
                    raise RuntimeError("cannot change settled usage")
                return
            row.update(metadata)
            if tokens is None:
                row.update(cost_assumptions=["usage_not_reported"])
                return  # Unknown usage is never free.
            inp, out = tokens["input_tokens"], tokens["output_tokens"]
            charge = (inp * U.TARIFF["guard_input"] + out * U.TARIFF["guard_output"]) / 1e6
            if charge > row["reserved_usd"]:
                raise RuntimeError("usage exceeds conservative reservation; stop")
            estimate, assumptions = U.priced(tokens)
            row.update(state="usage_recorded", usage=usage, tokens=tokens, cap_charge_usd=charge,
                       known_standard_usd=estimate, cost_assumptions=assumptions, settled_at=time.time())
        self.transaction(update)

    def totals(self):
        data = R.read_json(self.path) if self.path.exists() else {"calls": []}
        previous = (data.get("carryover") or {}).get("totals", U.summarize([]))
        return {"attempts": len(data["calls"]), "cap_usd": self.cap,
                "guard_mode": "iterations" if self.cap is None else "usd",
                "cap_charge_usd": previous["conservative_guard_usd"] + sum(c["cap_charge_usd"] for c in data["calls"]),
                "known_standard_usd": previous["known_standard_usd"] + sum(c["known_standard_usd"] or 0 for c in data["calls"]),
                "unknown_usage_calls": previous["usage_unknown_calls"] + sum(c["state"] != "usage_recorded" for c in data["calls"]),
                "carryover_attempts": previous["generation_attempts"],
                "carryover_cap_charge_usd": previous["conservative_guard_usd"],
                "carryover_known_standard_usd": previous["known_standard_usd"],
                "known_token_totals": U.summarize(data["calls"])["token_totals"]}


class Upstream:
    def __init__(self, auth_mode, auth_file=None, binary=None):
        if auth_mode not in {"api", "chatgpt"}:
            raise ValueError("choose api or chatgpt authentication explicitly")
        self.mode, self.auth_file = auth_mode, pathlib.Path(auth_file) if auth_file else None
        self.binary = binary
        self._headers = None
        self._journal = None

    def journal_to(self, directory):
        self._journal = pathlib.Path(directory)

    def prepare_auth(self):
        """Non-generation auth check BEFORE making a financial reservation."""
        if self.mode == "api":
            key = os.environ.get("CHIA_OPENAI_API_KEY")
            if not key:
                raise RuntimeError("CHIA_OPENAI_API_KEY is required in the trusted runner, not the child")
            self.url, headers = "https://api.openai.com/v1/responses", {"Authorization": "Bearer " + key}
        else:
            if self.auth_file is None:
                raise RuntimeError("ChatGPT mode requires an explicit operator-owned auth file")
            from . import auth
            headers = auth.headers(self.binary, self.auth_file)
            self.url = "https://chatgpt.com/backend-api/codex/responses"
        headers.update({"Content-Type": "application/json", "Accept": "text/event-stream"})
        self._headers = headers

    def __call__(self, request):
        import httpx
        if self._headers is None:
            raise RuntimeError("trusted auth preparation must precede the reserved dispatch")
        # No implicit network retries. Bound the entire response, not just an
        # individual socket read: an endless stream must not hang shutdown.
        metadata = {"started_at": time.time(), "client_request_id": str(uuid.uuid4()),
                    "auth_mode": self.mode, "model": request.get("model"),
                    "effort": (request.get("reasoning") or {}).get("effort"), "status": "dispatched"}
        started = time.monotonic()
        raw = bytearray()
        def save():
            if self._journal:
                atomic_write_json(self._journal / "provider_exchange.json", metadata)
        async def dispatch():
            stream_log = None
            save()
            try:
                async with asyncio.timeout(1800):
                    async with httpx.AsyncClient(timeout=1800, follow_redirects=False, trust_env=False) as client:
                        headers = {**self._headers, "X-Client-Request-ID": metadata["client_request_id"]}
                        async with client.stream("POST", self.url, headers=headers, json=request) as response:
                            # Log only explicitly safe metadata, never auth or
                            # arbitrary response headers/HTTP error bodies.
                            metadata.update(http_status=response.status_code, response_headers_seconds=time.monotonic() - started,
                                            provider_request_id=response.headers.get("x-request-id"))
                            save()
                            response.raise_for_status()
                            if self._journal:
                                stream_log = (self._journal / "provider_response.sse").open("xb")
                            async for chunk in response.aiter_bytes():
                                if "first_byte_seconds" not in metadata:
                                    metadata["first_byte_seconds"] = time.monotonic() - started
                                raw.extend(chunk)
                                if len(raw) > 64 * 1024**2:
                                    raise RuntimeError("provider response size guard")
                                if stream_log:
                                    stream_log.write(chunk)
                                    stream_log.flush()
                            metadata["status"] = "stream_received"
                            return bytes(raw)
            except BaseException as exc:
                metadata.update(status="failed", error_kind=type(exc).__name__)
                raise
            finally:
                if stream_log:
                    stream_log.flush()
                    os.fsync(stream_log.fileno())
                    stream_log.close()
                metadata.update(finished_at=time.time(), wall_seconds=time.monotonic() - started,
                                received_bytes=len(raw), received_sha256=P.sha(bytes(raw)))
                save()
        return asyncio.run(dispatch())


def completed_response(raw):
    result = terminal_response(raw)
    if result.get("status") != "completed":
        raise RuntimeError("incomplete provider response")
    if any(item.get("type") not in {"message", "reasoning"} for item in result.get("output", [])):
        raise RuntimeError("native tool output is forbidden; use CHIA JSON actions")
    if not any(part.get("type") in {"output_text", "refusal"}
               for item in result["output"] if item.get("type") == "message"
               for part in item.get("content", [])):
        raise RuntimeError("completed provider response has no finalized answer")
    return result


def check_response_identity(response, effort):
    if response.get("model") != MODEL:
        raise RuntimeError("provider response model differs from requested Astra")
    reported = (response.get("reasoning") or {}).get("effort")
    if reported is not None and reported != effort:
        raise RuntimeError("provider response reasoning effort differs from request")


def cli_usage(path, provider_usage):
    """Independent CLI receipt cross-check; provider usage stays authoritative."""
    events = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    receipts = [event["usage"] for event in events if event.get("type") == "turn.completed" and "usage" in event]
    if not receipts:
        return {"cli_usage_check": "not_reported", "cli_usage": None}
    if len(receipts) != 1:
        return {"cli_usage_check": "multiple_receipts", "cli_usage": receipts}
    expected = U.normalized(provider_usage)
    if expected is None:
        return {"cli_usage_check": "provider_usage_not_reported", "cli_usage": receipts[0]}
    keys = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
    compared = [k for k in keys if expected.get(k) is not None and k in receipts[0]]
    matches = all(expected[k] == receipts[0][k] for k in compared)
    return {"cli_usage_check": "matched" if matches and compared else "mismatch" if compared else "no_comparable_fields",
            "cli_usage": receipts[0], "cli_usage_fields_compared": compared}


def response_bytes(path):
    path = pathlib.Path(path)
    if path.exists():
        return path.read_bytes()
    with gzip.open(str(path) + ".gz", "rb") as stream:
        return stream.read()


def action(response):
    text = "".join(part.get("text", "") for item in response.get("output", [])
                   if item.get("type") == "message" for part in item.get("content", [])
                   if part.get("type") == "output_text")
    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("not an object")
        return result
    except ValueError:
        return {"status": "invalid_response", "reason": "Return a complete JSON object, without fences or prose."}


def clean_environment(work):
    # These are per-child, documented home locations, not assignments to the
    # orchestrator's HOME/CODEX_HOME or inherited session environment.
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(work / "home"),
            "CODEX_HOME": str(work / "home"), "XDG_CONFIG_HOME": str(work / "home"),
            "XDG_CACHE_HOME": str(work / "home/cache"), "TMPDIR": str(work / "tmp"),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def command(binary, work, port, token, effort):
    if effort not in EFFORTS:
        raise ValueError("Astra comparison supports exactly xhigh and max; no substitution")
    config = {"model_provider": "chia_broker", "model_reasoning_effort": effort,
              "model_reasoning_summary": REASONING_SUMMARY, "model_supports_reasoning_summaries": True,
              "approval_policy": "never", "web_search": "disabled", "agents.enabled": False,
              "memories.use_memories": False, "memories.generate_memories": False,
              "history.persistence": "none", "project_doc_max_bytes": 0,
              "model_instructions_file": str(work / "system.md"),
              "shell_environment_policy.inherit": "none", "analytics.enabled": False,
              "model_auto_compact_token_limit": 1_000_000,
              "model_providers.chia_broker.name": "CHIA isolated broker",
              "model_providers.chia_broker.base_url": f"http://127.0.0.1:{port}/v1",
              "model_providers.chia_broker.wire_api": "responses",
              "model_providers.chia_broker.requires_openai_auth": False,
              "model_providers.chia_broker.http_headers.X-CHIA-Token": token,
              "model_providers.chia_broker.request_max_retries": 0,
              "model_providers.chia_broker.stream_max_retries": 0,
              "model_providers.chia_broker.stream_idle_timeout_ms": 1_850_000}
    args = [str(binary), "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
            "--skip-git-repo-check", "--sandbox", "read-only", "--json", "--color", "never",
            "--model", MODEL, "--cd", str(work)]
    for key, value in config.items():
        args += ["-c", key + "=" + json.dumps(value)]
    for feature in DISABLED:
        args += ["--disable", feature]
    return args + ["-"]


def check_stop(root):
    root = pathlib.Path(root).resolve()
    R.check_stop(root)
    if root.with_name(root.name + ".preparation_STOP").exists():
        raise R.OperatorStop("Astra preparation/launch is on operator hold")


def invoke(root, operation, system, conversation, *, effort, role, cap, upstream, binary):
    """Durably replay completed calls; retry interrupted work only when affordable."""
    root = pathlib.Path(root).resolve()
    if not operation.replace("_", "").isalnum() or role not in {"proposal", "review"}:
        raise ValueError("invalid operation/role")
    check_stop(root)
    R.check_storage(root)
    R.check_run_deadline(root)
    directory = root / "interactions" / operation
    directory.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(root, cap)
    identity = {"model": MODEL, "effort": effort, "reasoning_summary": REASONING_SUMMARY,
                "role": role, "run_id": root.name,
                "system_sha256": P.sha(system), "conversation_sha256": P.sha(json.dumps(conversation, sort_keys=True))}
    with R.exclusive_lock(directory / ".lock"):
        binding = directory / "identity.json"
        if R.exists(binding) and R.read_json(binding) != identity:
            raise RuntimeError("cannot change a checkpoint's model, effort, input or owner")
        if not R.exists(binding):
            atomic_write_json(binding, identity)
            atomic_write_json(directory / "input.json", {"system": system, "conversation": conversation})
        if R.exists(directory / "result.json"):
            result = R.read_json(directory / "result.json")
            raw = response_bytes(directory / result["response_file"])
            if P.sha(raw) != result["response_sha256"]:
                raise RuntimeError("saved response changed")
            response = completed_response(raw)
            check_response_identity(response, effort)
            ledger.settle(result["call_id"], response, response_sha256=P.sha(raw))
            return result["answer"]
        attempts = sorted(directory.glob("attempt_*"))
        for old in attempts:
            if R.exists(old / "receipt.json") and R.read_json(old / "receipt.json").get("cli_usage_check") in {"mismatch", "multiple_receipts"}:
                raise R.OperationalPause("saved CLI/provider usage mismatch requires attention", retryable=False)
            if R.exists(old / "provider_response.sse") and R.exists(old / "reservation.json"):
                raw = response_bytes(old / "provider_response.sse")
                try:
                    response = terminal_response(raw)
                except (RuntimeError, ValueError):
                    continue  # Its reservation remains charged in full.
                call_id = R.read_json(old / "reservation.json")["call_id"]
                ledger.settle(call_id, response, response_sha256=P.sha(raw))
                check_response_identity(response, effort)
                try:
                    completed_response(raw)
                except RuntimeError:
                    continue  # Known incomplete usage still counts, never an accepted design.
                result = {"call_id": call_id, "response_sha256": P.sha(raw),
                          "response_file": str((old / "provider_response.sse").relative_to(directory)),
                          "answer": action(response), "recovered_complete_response": True}
                atomic_write_json(directory / "result.json", result)
                return result["answer"]
        if len(attempts) >= 3:
            raise R.OperationalPause("three CLI attempts exhausted; no unbounded retry", retryable=False)
        attempt = directory / f"attempt_{len(attempts) + 1:03d}"
        work = attempt / "workspace"
        for p in (work, work / "home", work / "home/cache", work / "tmp"):
            p.mkdir(parents=True, mode=0o700, exist_ok=True)
        (work / "system.md").write_text(system + "\nReturn only the requested CHIA JSON object. Native tools are unavailable.\n")
        token = secrets.token_hex(32)
        outcome = {"started_at": time.time(), "operation": operation, "attempt_number": len(attempts) + 1,
                   "model": MODEL, "effort": effort, "reasoning_summary": REASONING_SUMMARY,
                   "role": role, "auth_mode": getattr(upstream, "mode", "offline_fixture")}
        atomic_write_json(attempt / "receipt.json", outcome)
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Never print headers or credentials.
            def do_POST(self):
                if (self.path != "/v1/responses" or self.headers.get("X-CHIA-Token") != token
                        or outcome.get("received")):
                    self.send_error(403)
                    return
                outcome["received"] = True
                outcome["stage"] = "request_validation"
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_INPUT_BYTES:
                        raise ValueError("request length guard")
                    if self.headers.get("Content-Encoding") not in (None, "identity"):
                        raise ValueError("compressed request is not audited")
                    request = json.loads(self.rfile.read(length))
                    atomic_write_json(attempt / "cli_request.json", request)
                    if request.get("model") != MODEL or request.get("reasoning", {}).get("effort") != effort:
                        raise ValueError("CLI substituted model/reasoning effort")
                    if request.get("reasoning", {}).get("summary") != REASONING_SUMMARY:
                        raise ValueError("CLI omitted or changed the required readable reasoning-summary request")
                    if request.get("previous_response_id") or request.get("conversation"):
                        raise ValueError("server-side conversation inheritance forbidden")
                    # Same JSON-action interface as Gemini: native Codex tools
                    # cannot run, even if a future CLI registers additional tools.
                    request.update(tools=[], store=False, stream=True)
                    request.pop("tool_choice", None)
                    request.pop("service_tier", None)  # Never inherit priority-price routing.
                    if getattr(upstream, "mode", "api") == "api":
                        request["max_output_tokens"] = MAX_OUTPUT
                    atomic_write_json(attempt / "provider_request.json", request)
                    check_stop(root)
                    R.check_run_deadline(root)
                    if hasattr(upstream, "prepare_auth"):
                        outcome["stage"] = "auth_preparation"
                        upstream.prepare_auth()
                        outcome["auth_prepared_at"] = time.time()
                    outcome["stage"] = "budget_reservation"
                    call_id = ledger.reserve(request, role, operation,
                        attempt_path=str(attempt.relative_to(root)), attempt_number=len(attempts) + 1,
                        auth_mode=getattr(upstream, "mode", "offline_fixture"))
                    outcome["call_id"] = call_id
                    atomic_write_json(attempt / "reservation.json", {"call_id": call_id})
                    if hasattr(upstream, "journal_to"):
                        upstream.journal_to(attempt)
                    outcome["stage"] = "provider_dispatch"
                    atomic_write_json(attempt / "receipt.json", outcome)
                    raw = upstream(request)
                    # Persist provider bytes before parsing/settling; never
                    # forward a native tool call to the CLI executor.
                    response_path = attempt / "provider_response.sse"
                    if not response_path.exists():
                        response_path.write_bytes(raw)
                    elif response_path.read_bytes() != raw:
                        raise RuntimeError("captured provider stream differs from returned bytes")
                    outcome["stage"] = "response_validation"
                    response = terminal_response(raw)
                    ledger.settle(call_id, response, response_sha256=P.sha(raw))
                    check_response_identity(response, effort)
                    completed_response(raw)  # Usage is accounted even if the answer is unusable.
                    outcome.update(answer=action(response), response_sha256=P.sha(raw),
                                   response_file=str((attempt / "provider_response.sse").relative_to(directory)))
                    outcome["stage"] = "response_accepted"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except Exception as exc:
                    # Credentials, request bodies, and provider exception text
                    # do not belong in model-visible error messages.
                    outcome["error_kind"] = type(exc).__name__
                    outcome["http_status"] = getattr(getattr(exc, "response", None), "status_code", None)
                    outcome["retryable"] = isinstance(exc, R.OperationalPause) and exc.retryable
                    self.send_error(502, "CHIA broker stopped; inspect the trusted journal")
                finally:
                    outcome["broker_finished_at"] = time.time()
                    atomic_write_json(attempt / "receipt.json", outcome)
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_port
        args = command(binary, work, port, token, effort)
        atomic_write_json(attempt / "cli_command.json", {"argv": [arg.replace(token, "<ephemeral_broker_token>") for arg in args],
            "model": MODEL, "effort": effort, "reasoning_summary": REASONING_SUMMARY,
            "new_process": True, "ephemeral": True})
        policy = {"read": ["/usr", "/lib", "/lib64", "/bin", "/dev/null", "/dev/urandom", str(binary), str(work)],
                  "write": [str(work)], "port": port, "cwd": str(work), "environment": clean_environment(work)}
        atomic_write_json(attempt / "boundary.json", policy)
        cli_started = time.monotonic()
        try:
            with (attempt / "cli_events.jsonl").open("x") as stdout, (attempt / "cli_stderr.log").open("x") as stderr:
                process = subprocess.run([sys.executable, str(HERE / "boundary.py"),
                    str(attempt / "boundary.json"), "--", *args], input=json.dumps(conversation),
                    text=True, stdout=stdout, stderr=stderr, cwd=work,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, timeout=1900, close_fds=True)
            outcome["cli_exit_code"] = process.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            outcome.setdefault("error_kind", type(exc).__name__)
            outcome["cli_timeout"] = isinstance(exc, subprocess.TimeoutExpired)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            outcome.update(finished_at=time.time(), cli_wall_seconds=time.monotonic() - cli_started)
            if R.exists(attempt / "provider_response.sse") and (attempt / "cli_events.jsonl").exists():
                try:
                    response = terminal_response(response_bytes(attempt / "provider_response.sse"))
                    outcome.update(cli_usage(attempt / "cli_events.jsonl", response.get("usage")))
                except (RuntimeError, ValueError):
                    outcome["cli_usage_check"] = "unavailable_or_interrupted"
            atomic_write_json(attempt / "receipt.json", outcome)
        if outcome.get("cli_usage_check") in {"mismatch", "multiple_receipts"}:
            raise R.OperationalPause("CLI/provider usage receipts disagree", retryable=False)
        if "answer" in outcome:
            # A completed paid response remains recoverable even if the CLI
            # disconnects while printing it. The answer is already in the journal.
            atomic_write_json(directory / "result.json", outcome)
            return outcome["answer"]
        if outcome.get("error_kind") == "BudgetExhausted":
            raise P.BudgetExhausted("CLI broker budget guard")
        transient = (outcome.get("retryable") is True or outcome.get("http_status") in (408, 429, 500, 502, 503, 504)
                     or outcome.get("error_kind") in {"ConnectError", "ReadError", "WriteError",
                         "RemoteProtocolError", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "TimeoutError"})
        raise R.OperationalPause("Codex invocation failed: " + outcome.get("error_kind", "startup_or_transport"),
                                 retryable=transient, retry_at=time.time() + 60)
