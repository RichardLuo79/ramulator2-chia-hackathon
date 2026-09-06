"""Fresh, tool-free native Claude CLI behind an audited Messages-only gateway.

Native Claude Code owns authentication. This is a fixed-destination transport
for that unmodified binary, not an alternative OAuth client. Credentials never
enter prompts or archived workspaces. Native tools/workflows are prohibited in
the actual request and response, as well as by the CLI configuration.
"""
from __future__ import annotations

import asyncio
import datetime
import gzip
import http.server
import json
import os
import pathlib
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

from tools.chia_loop import real_core as P, recovery as R
from tools.chia_loop.core import atomic_write_json
from . import auth, usage as U
from .stream import terminal_response, action, readable_thinking

MODEL = U.MODEL
EFFORTS = ("xhigh", "max")
MAX_OUTPUT = 128_000
MAX_INPUT_BYTES = 900_000
HERE = pathlib.Path(__file__).resolve().parent
Ledger = U.Ledger


def limits(maximum_iterations, cap, cpus, *, iteration_guard=False, auth_mode="claude_subscription"):
    from tools.chia_loop.run_records import validate_limits
    if not iteration_guard or cap is not None or auth_mode != "claude_subscription":
        raise ValueError("explicit subscription iteration guard required")
    value = validate_limits(maximum_iterations, 1.0, cpus)
    value["usd_cap"] = None
    return value


def check_stop(root):
    root = pathlib.Path(root)
    R.check_stop(root)
    if root.with_name(root.name + ".preparation_STOP").exists():
        raise R.OperatorStop("Fable preparation is on hold")


def clean_environment(work, port, token, effort):
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(work / "home"),
            "CLAUDE_CONFIG_DIR": str(work / "home/.claude"), "TMPDIR": str(work / "tmp"),
            "XDG_CACHE_HOME": str(work / "home/cache"), "NO_COLOR": "1", "TERM": "dumb",
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "ANTHROPIC_CUSTOM_HEADERS": "X-CHIA-Token: " + token,
            "CLAUDE_CODE_EFFORT_LEVEL": effort, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(MAX_OUTPUT),
            "CLAUDE_CODE_MAX_RETRIES": "0", "CLAUDE_CODE_MAX_TURNS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_WORKFLOWS": "1", "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1", "CLAUDE_CODE_GZIP_REQUEST_BODIES": "0",
            "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1"}


def command(binary, work, effort):
    if effort not in EFFORTS:
        raise ValueError("only exact xhigh/max Fable comparison is authorized")
    return [str(binary), "-p", "--model", MODEL, "--effort", effort,
            "--safe-mode", "--no-session-persistence", "--setting-sources", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--tools", "", "--disable-slash-commands", "--no-chrome",
            "--permission-mode", "dontAsk", "--permission-prompts", "none",
            "--output-format", "stream-json", "--verbose", "--include-partial-messages", "--max-turns", "1",
            "--system-prompt-file", str(work / "system.md"), "--autocompact", "1M"]


def validate_request(request, effort, conversation, system):
    if request.get("model") != MODEL or request.get("output_config", {}).get("effort") != effort:
        raise RuntimeError("native CLI changed model or effort")
    if request.get("max_tokens") != MAX_OUTPUT or request.get("stream") is not True:
        raise RuntimeError("native CLI changed output/stream configuration")
    if request.get("tools") or request.get("tool_choice") or request.get("container"):
        raise RuntimeError("native tools or server-side context forbidden")
    thinking = request.get("thinking") or {}
    if thinking.get("type") != "adaptive" or thinking.get("display") == "omitted":
        raise RuntimeError("adaptive thinking with exposed content requested")
    messages = request.get("messages")
    if not isinstance(messages, list) or len(messages) != 2 or messages[0].get("role") != "user":
        raise RuntimeError("implicit native conversation history forbidden")
    tail = messages[1]
    if (tail.get("role") != "system" or tail.get("output_config") != {"effort": effort}
            or len(tail.get("content", [])) != 1 or tail["content"][0].get("type") != "text"
            or tail["content"][0].get("text") != "<total_tokens>15000000 tokens left</total_tokens>"):
        raise RuntimeError("unexpected native turn-scoped context")
    content = messages[0].get("content")
    if isinstance(content, str):
        contents = [content]
    elif isinstance(content, list) and all(b.get("type") == "text" for b in content):
        contents = [b["text"] for b in content]
    else:
        raise RuntimeError("unexpected native message content")
    expected = json.dumps(conversation, sort_keys=True)
    date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    reminder = ("<system-reminder>\nAs you answer the user's questions, you can use the following context:\n"
                "# currentDate\nToday's date is " + date + ".\n\n      IMPORTANT: this context may or may not be relevant "
                "to your tasks. You should not respond to this context unless it is highly relevant to your task.\n"
                "</system-reminder>\n\n")
    if contents != [reminder, expected]:
        raise RuntimeError("native CLI imported extra user context")
    instruction = request.get("system", [])
    if isinstance(instruction, str):
        instruction = [{"type": "text", "text": instruction}]
    if not any(b.get("text") == system for b in instruction):
        raise RuntimeError("explicit CHIA system prompt not preserved")
    # Pinned CLI 2.1.263 attribution and outcome-reporting boilerplate; reviewed
    # against the actual no-network CLI fixture. A change requires re-audit,
    # never a wildcard allowing host instructions into an experiment.
    boilerplate = {"0d7062851dd7bd7e66d4be4f12ac4951e3d2f587ec408295333a49963bd3f6b7",
                   "a11f50f6b3db6a8398147c47e0705204c4edeecb443aa0bb282d659bb9170fae"}
    if len(instruction) != 4:
        raise RuntimeError("unexpected system block count")
    for block in instruction:
        value = block.get("text", "")
        if block.get("type") != "text" or not (value == system or
                re.fullmatch(r"x-anthropic-billing-header: cc_version=2\.1\.263\.[0-9a-f]{3}; cc_entrypoint=sdk-cli;", value) or
                P.sha(value) in boilerplate):
            raise RuntimeError("unexpected native system context")


def safe_request(request):
    value = dict(request)
    # Account/session routing metadata is not model input. Do not archive it.
    if "metadata" in value:
        value["metadata"] = {"redacted_nonprompt_routing_metadata": True}
    return value


class Upstream:
    mode = "claude_subscription"

    def __call__(self, request, headers, directory):
        import httpx
        permitted = {"authorization", "anthropic-version", "anthropic-beta", "user-agent", "x-app"}
        forwarded = {key: value for key, value in headers.items() if key.lower() in permitted}
        if not any(k.lower() == "authorization" and v.startswith("Bearer ") for k, v in forwarded.items()):
            raise RuntimeError("native subscription authentication missing; no API fallback")
        forwarded.update({"Content-Type": "application/json", "Accept": "text/event-stream"})
        metadata = {"started_at": time.time(), "status": "dispatched", "model": MODEL}
        async def dispatch():
            raw = bytearray()
            try:
                async with asyncio.timeout(1800):
                    async with httpx.AsyncClient(timeout=1800, trust_env=False, follow_redirects=False) as client:
                        async with client.stream("POST", "https://api.anthropic.com/v1/messages?beta=true",
                                                 headers=forwarded, json=request) as response:
                            metadata.update(http_status=response.status_code,
                                            provider_request_id=response.headers.get("request-id"))
                            atomic_write_json(directory / "provider_exchange.json", metadata)
                            response.raise_for_status()
                            with (directory / "provider_response.sse").open("xb") as output:
                                async for chunk in response.aiter_bytes():
                                    raw.extend(chunk)
                                    if len(raw) > 64 * 1024**2:
                                        raise RuntimeError("provider response size guard")
                                    output.write(chunk)
                                    output.flush()
                                os.fsync(output.fileno())
                            return bytes(raw)
            finally:
                metadata.update(finished_at=time.time(), received_bytes=len(raw), received_sha256=P.sha(bytes(raw)))
                atomic_write_json(directory / "provider_exchange.json", metadata)
        return asyncio.run(dispatch())


def response_bytes(path):
    return pathlib.Path(path).read_bytes() if pathlib.Path(path).exists() else gzip.decompress(pathlib.Path(str(path) + ".gz").read_bytes())


def response_identity(response):
    if response.get("model") != MODEL:
        raise RuntimeError("provider response model mismatch")


def cli_receipt(path, response):
    """Provider stream is authoritative; contradictory native receipts stop."""
    try:
        events = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    except ValueError:
        return {"cli_usage_check": "unavailable_or_interrupted"}
    initial = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    for item in initial:
        if (item.get("model") != MODEL or any(item.get(k) for k in ("tools", "mcp_servers", "skills", "plugins", "slash_commands"))
                or item.get("apiKeySource") != "none"):
            raise RuntimeError("CLI reports unexpected model/tools/context/authentication")
    terminal = [e for e in events if e.get("type") == "result"]
    if len(terminal) > 1:
        raise RuntimeError("multiple CLI terminal receipts")
    if not terminal:
        return {"cli_usage_check": "unavailable_or_interrupted"}
    native, expected = U.normalized(terminal[0].get("usage")), U.normalized(response.get("usage"))
    if native != expected:
        raise RuntimeError("CLI/provider token receipts disagree")
    if set(terminal[0].get("modelUsage", {})) != {MODEL} or terminal[0].get("subagent_stats", {}).get("spawned", 0):
        raise RuntimeError("native model substitution/delegation occurred")
    if not terminal[0].get("is_error"):
        try:
            final = json.loads(terminal[0]["result"])
        except ValueError:
            final = action({**response, "content": [{"type": "text", "text": terminal[0]["result"]}]})
        if final != action(response):
            raise RuntimeError("CLI/provider finalized answers disagree")
    return {"cli_usage_check": "matched", "cli_is_error": terminal[0].get("is_error"),
            "cli_cost_estimate_usd": terminal[0].get("total_cost_usd"),
            "cli_native_turns": terminal[0].get("num_turns")}


def invoke(root, operation, system, conversation, *, effort, role, cap, upstream, binary, auth_file):
    root = pathlib.Path(root).resolve()
    if effort not in EFFORTS or role not in {"proposal", "review"} or not re.fullmatch(r"[a-zA-Z0-9_]+", operation):
        raise ValueError("invalid model action")
    check_stop(root)
    R.check_storage(root)
    R.check_run_deadline(root)
    directory = root / "interactions" / operation
    directory.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(root, cap)
    identity = {"run_id": root.name, "model": MODEL, "effort": effort, "role": role,
                "system_sha256": P.sha(system), "conversation_sha256": P.sha(json.dumps(conversation, sort_keys=True))}
    with R.exclusive_lock(directory / ".lock"):
        if R.exists(directory / "identity.json"):
            if R.read_json(directory / "identity.json") != identity:
                raise RuntimeError("operation owner/input changed")
        else:
            atomic_write_json(directory / "identity.json", identity)
            atomic_write_json(directory / "input.json", {"system": system, "conversation": conversation})
        attempts = sorted(directory.glob("attempt_*"))
        for attempt in attempts:
            if R.exists(attempt / "receipt.json") and R.read_json(attempt / "receipt.json").get("cli_receipt_conflict"):
                raise R.OperationalPause("saved CLI/provider receipt conflict; no automatic retry", retryable=False)
            if R.exists(attempt / "provider_response.sse") and R.exists(attempt / "reservation.json"):
                raw = response_bytes(attempt / "provider_response.sse")
                try:
                    response = terminal_response(raw)
                except (RuntimeError, ValueError):
                    continue
                index = R.read_json(attempt / "reservation.json")["call_id"]
                row = R.read_json(root / "ledger.json")["calls"][index]
                if (row["operation"] != operation or row["role"] != role or row["effort"] != effort
                        or row["attempt_path"] != str(attempt.relative_to(root))
                        or P.sha(json.dumps(R.read_json(attempt / "provider_request.json"), ensure_ascii=False, sort_keys=True).encode())
                           != row["request_sha256"]):
                    raise RuntimeError("saved request/usage binding changed")
                ledger.settle(index, response, response_sha256=P.sha(raw))
                response_identity(response)
                if response.get("stop_reason") != "end_turn":
                    raise R.OperationalPause("completed but truncated/nonfinal response; usage retained", retryable=False)
                result = {"answer": action(response), "response_sha256": P.sha(raw),
                          "response_file": str((attempt / "provider_response.sse").relative_to(directory)), "call_id": index}
                if R.exists(directory / "result.json"):
                    if R.read_json(directory / "result.json") != result:
                        raise RuntimeError("stored action contradicts provider output")
                else:
                    atomic_write_json(directory / "result.json", result)
                return result["answer"]
        if len(attempts) >= 3:
            raise R.OperationalPause("three native CLI attempts exhausted", retryable=False)
        auth.check(auth_file)  # no financial reservation until valid native request
        attempt = directory / f"attempt_{len(attempts) + 1:03d}"
        attempt.mkdir()
        outcome = {"started_at": time.time(), "attempt": len(attempts) + 1,
                   "model": MODEL, "effort": effort, "auth_mode": "claude_subscription"}
        token = secrets.token_hex(32)

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_error(403, "Only one audited Messages operation is allowed")

            def do_POST(self):
                if (urllib.parse.urlsplit(self.path).path != "/v1/messages"
                        or self.headers.get("X-CHIA-Token") != token or outcome.get("received")):
                    self.send_error(403, "Only one audited Messages operation is allowed")
                    return
                outcome["received"] = True
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_INPUT_BYTES or self.headers.get("Content-Encoding") not in (None, "identity"):
                        raise RuntimeError("native request byte/encoding guard")
                    request_bytes = self.rfile.read(length)
                    request = json.loads(request_bytes)
                    outcome["wire_request_sha256"] = P.sha(request_bytes)
                    # Save only noncredential request data; fail before dispatch
                    # on inherited context, tools, effort or output-limit drift.
                    atomic_write_json(attempt / "cli_request.json", safe_request(request))
                    validate_request(request, effort, conversation, system)
                    logged = safe_request(request)
                    atomic_write_json(attempt / "provider_request.json", logged)
                    call_id = ledger.reserve(logged, role, operation, attempt_path=str(attempt.relative_to(root)),
                                             attempt_number=len(attempts) + 1)
                    atomic_write_json(attempt / "reservation.json", {"call_id": call_id})
                    outcome["call_id"] = call_id
                    raw = upstream(request, self.headers, attempt)
                    if not (attempt / "provider_response.sse").exists():
                        with (attempt / "provider_response.sse").open("xb") as output:
                            output.write(raw)
                            output.flush()
                            os.fsync(output.fileno())
                    response = terminal_response(raw)
                    ledger.settle(call_id, response, response_sha256=P.sha(raw))
                    response_identity(response)
                    atomic_write_json(attempt / "exposed_thinking.json", readable_thinking(response))
                    if response.get("stop_reason") != "end_turn":
                        raise RuntimeError("provider response truncated or nonfinal")
                    outcome.update(answer=action(response), response_sha256=P.sha(raw))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except Exception as exc:
                    # Never echo HTTP headers, credential values or raw errors.
                    outcome["error_kind"] = type(exc).__name__
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    outcome["http_status"] = status
                    outcome["retryable"] = status in (408, 429, 499, 500, 502, 503, 504) or type(exc).__name__ in {
                        "ConnectError", "ReadError", "WriteError", "ReadTimeout", "ConnectTimeout", "TimeoutError", "RemoteProtocolError"}
                    self.send_error(502, "CHIA native CLI transport stopped; inspect trusted receipts")
                finally:
                    atomic_write_json(attempt / "receipt.json", outcome)

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Never keep a credential binding inside archived interaction paths.
            with tempfile.TemporaryDirectory(prefix="chia-claude-") as temporary:
                work = pathlib.Path(temporary)
                for p in (work / "home/cache", work / "tmp"):
                    p.mkdir(parents=True)
                credential = auth.bind(work, auth_file)
                (work / "system.md").write_text(system)
                environment = clean_environment(work, server.server_port, token, effort)
                policy = {"read": ["/usr", "/lib", "/lib64", "/bin", "/dev/null", "/dev/urandom",
                                     str(binary), str(work), str(credential)], "write": [str(work)],
                          "port": server.server_port, "cwd": str(work), "environment": environment}
                atomic_write_json(work / "boundary.json", policy)
                argv = command(binary, work, effort)
                atomic_write_json(attempt / "cli_command.json", {"argv": argv, "version_pinned": True,
                    "native_tools": False, "fresh_process": True, "persistent_session": False})
                # The private auth binding is readable only by the trusted CLI,
                # never by candidate C++ or a native model tool (none exist).
                process = subprocess.run([sys.executable, str(HERE / "boundary.py"),
                    str(work / "boundary.json"), "--", *argv], input=json.dumps(conversation, sort_keys=True),
                    text=True, capture_output=True, close_fds=True, cwd=work,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, timeout=1900)
                outcome["cli_exit_code"] = process.returncode
                # Native output has no auth headers; additionally redact token
                # shapes as defense in depth. No debug/credential files copied.
                redact = lambda s: re.sub(r"(?:sk-ant-[A-Za-z0-9_-]+|Bearer\s+\S+)", "<redacted>", s).replace(token, "<broker-token>")
                (attempt / "cli_events.jsonl").write_text(redact(process.stdout))
                (attempt / "cli_stderr.log").write_text(redact(process.stderr))
                if R.exists(attempt / "provider_response.sse"):
                    try:
                        response = terminal_response(response_bytes(attempt / "provider_response.sse"))
                        outcome.update(cli_receipt(attempt / "cli_events.jsonl", response))
                    except (RuntimeError, ValueError) as exc:
                        outcome.pop("answer", None)
                        outcome.update(error_kind=type(exc).__name__, cli_receipt_conflict=True, retryable=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            outcome.update(error_kind=type(exc).__name__, retryable=isinstance(exc, subprocess.TimeoutExpired))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            outcome["finished_at"] = time.time()
            atomic_write_json(attempt / "receipt.json", outcome)
        if "answer" in outcome:
            result = {"answer": outcome["answer"], "response_sha256": outcome["response_sha256"],
                      "response_file": str((attempt / "provider_response.sse").relative_to(directory)), "call_id": outcome["call_id"]}
            atomic_write_json(directory / "result.json", result)
            return result["answer"]
        raise R.OperationalPause("Claude invocation stopped: " + outcome.get("error_kind", "native_cli_startup"),
                                 retryable=outcome.get("retryable", False), retry_at=time.time() + 60)
