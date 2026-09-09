"""Native CHIA transports with fresh profiles and the existing process boundary.

The small relay forwards native HTTP bodies unchanged to a fixed provider.
It does not implement conversations, tools, retries, OAuth, or selection.
Credentials stay in the trusted relay, never in the agent's profile/environment.
"""

import gzip
import http.server
import json
import os
import re
import secrets
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx

from .identity import canonical_json
from .snapshots import publish_bytes

CLI = {
    "codex_cli": {
        "binary": "/home/dev/.local/bin/codex",
        "companions": ("codex-code-mode-host",),
        "path": "/v1/responses",
        "upstream": "https://chatgpt.com/backend-api/codex/responses",
    },
    "claude_cli": {
        "binary": "/home/dev/.local/bin/claude",
        "companions": (),
        "path": "/v1/messages",
        "upstream": "https://api.anthropic.com/v1/messages",
    },
}
SYSTEM_READS = (
    "/usr",
    "/lib",
    "/lib64",
    "/bin",
    "/etc/ssl/certs",
    "/etc/ld.so.cache",
    "/etc/localtime",
    "/dev/null",
    "/dev/urandom",
    "/dev/random",
)


@contextmanager
def relay(backend, root, headers, timeout_seconds=3600):
    """One role's authenticated fixed-destination HTTP stream, without retries."""
    token = secrets.token_urlsafe(32)
    profile = CLI[backend.kind]

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            allowed = {profile["path"]}
            allowed.add(
                profile["path"] + ("/compact" if backend.kind == "codex_cli" else "/count_tokens")
            )
            if self.headers.get("X-CHIA-Token") != token or path not in allowed:
                self.send_error(403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64 * 1024**2:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            if self.headers.get("Content-Encoding") not in (None, "identity"):
                self.send_error(415)
                return
            request = json.loads(body)
            if request.get("model") not in (None, backend.model):
                self.send_error(400, "Configured model must not change")
                return
            directory = root / uuid4().hex
            directory.mkdir(parents=True)
            publish_bytes(directory / "request.json.gz", gzip.compress(body, mtime=0))
            record = {"started": time.time(), "path": path, "status": "dispatched"}
            publish_bytes(directory / "dispatch.json", canonical_json(record).encode())
            try:
                forwarded = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() in {"anthropic-version", "anthropic-beta", "user-agent", "x-app"}
                }
                forwarded.update(headers())
                forwarded.update(
                    {"Content-Type": "application/json", "Accept": "text/event-stream"}
                )
                suffix = path.removeprefix(profile["path"])
                url = profile["upstream"] + suffix
                if backend.kind == "claude_cli":
                    url += "?beta=true"
                with httpx.stream(
                    "POST",
                    url,
                    content=body,
                    headers=forwarded,
                    timeout=timeout_seconds,
                    trust_env=False,
                    follow_redirects=False,
                ) as response:
                    record["http_status"] = response.status_code
                    self.send_response(response.status_code)
                    self.send_header(
                        "Content-Type", response.headers.get("Content-Type", "text/event-stream")
                    )
                    self.send_header("Connection", "close")
                    self.end_headers()
                    with gzip.open(directory / "response.raw.gz", "wb") as saved:
                        for chunk in response.iter_bytes():
                            saved.write(chunk)
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    record["status"] = "received"
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__)
                self.close_connection = True
            finally:
                record["finished"] = time.time()
                publish_bytes(directory / "receipt.json", canonical_json(record).encode())

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, token
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@dataclass
class NativeTransport:
    codex_auth: Path = Path("/home/dev/.codex/auth.json")
    claude_auth: Path = Path("/home/dev/.claude/chia-oauth-token")

    @contextmanager
    def __call__(self, view, private, tool_environment):
        backend = view.research.configuration.backend
        if backend.kind == "vertex_gemini":
            # SDK executes no model-supplied local code. Its only agent actions
            # are the same authenticated, training-only CHIA tools as the CLIs.
            import google.auth

            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            yield {"vertex_client_kwargs": {"credentials": credentials}}
            return
        profile = CLI[backend.kind]
        binary = Path(profile["binary"]).resolve(strict=True)
        executables = [binary, *(binary.with_name(name) for name in profile["companions"])]
        if backend.kind == "codex_cli":
            from tools.chia_loop.codex_cli.auth import headers as codex_headers

            def auth_headers():
                return codex_headers(binary, self.codex_auth)
        else:
            from tools.chia_loop.claude_cli.auth import read_setup_token

            def auth_headers():
                return {"Authorization": "Bearer " + read_setup_token(self.claude_auth)}

        evidence = view.research.root / "native-evidence" / str(view.iteration) / view.role / "http"
        with relay(
            backend, evidence, auth_headers, view.research.configuration.run.model_timeout_seconds
        ) as (port, token):
            env = {
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "HOME": str(private),
                "TMPDIR": str(private / "tmp"),
                "NO_COLOR": "1",
                "TERM": "dumb",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                **tool_environment,
            }
            extra = []
            if backend.kind == "codex_cli":
                env["CODEX_HOME"] = str(private)
                values = {
                    "model_provider": "chia_native",
                    "web_search": "disabled",
                    "analytics.enabled": False,
                    "shell_environment_policy.inherit": "none",
                    "model_providers.chia_native.name": "CHIA native subscription relay",
                    "model_providers.chia_native.base_url": f"http://127.0.0.1:{port}/v1",
                    "model_providers.chia_native.wire_api": "responses",
                    "model_providers.chia_native.requires_openai_auth": False,
                    "model_providers.chia_native.http_headers.X-CHIA-Token": token,
                    "model_providers.chia_native.request_max_retries": 0,
                    "model_providers.chia_native.stream_max_retries": 0,
                    "model_providers.chia_native.stream_idle_timeout_ms": 3_600_000,
                    "features.enable_request_compression": False,
                }
                for key, value in values.items():
                    extra += ["-c", key + "=" + json.dumps(value)]
            else:
                env.update(
                    {
                        "CLAUDE_CONFIG_DIR": str(private),
                        "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-" + "x" * 64,
                        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                        "ANTHROPIC_CUSTOM_HEADERS": "X-CHIA-Token: " + token,
                        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                        "CLAUDE_CODE_GZIP_REQUEST_BODIES": "0",
                        "CLAUDE_CODE_MAX_RETRIES": "0",
                        "DISABLE_AUTOUPDATER": "1",
                        "DISABLE_TELEMETRY": "1",
                    }
                )
                extra = ["--permission-mode", "dontAsk", "--disable-slash-commands", "--no-chrome"]

            def run(command, **options):
                arguments = list(command)
                # Append only native configuration, before the terminal prompt/session args.
                insertion = -2 if backend.kind == "codex_cli" and "resume" in arguments else -1
                arguments[insertion:insertion] = extra
                text = " ".join(arguments)
                if backend.kind == "claude_cli" and "--mcp-config" in arguments:
                    text += Path(arguments[arguments.index("--mcp-config") + 1]).read_text()
                ports = {
                    port,
                    *(int(p) for p in re.findall(r"http://[0-9.]+:(\d+)/ramulator/mcp", text)),
                }
                grants = view.grants()
                policy = {
                    "kind": backend.kind,
                    "cwd": str(view.root),
                    "ports": sorted(ports),
                    "read": [p for p in SYSTEM_READS if Path(p).exists()]
                    + [*map(str, executables), str(private), *map(str, grants["read"])],
                    "write": [str(private), *map(str, grants["write"])],
                }
                policy_file = private / "tmp" / ("boundary-" + uuid4().hex + ".json")
                publish_bytes(policy_file, canonical_json(policy).encode())
                child_env = {**env, "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
                command = [
                    "/usr/bin/python3",
                    "-m",
                    "tools.chia_loop.framework.native_process",
                    str(policy_file),
                    *arguments,
                ]
                from tools.eval.archive_results import compress_files
                from tools.eval.artifacts import open_text

                logs = evidence.parent / "process" / (view.phase + "-" + uuid4().hex)
                logs.mkdir(parents=True)
                paths = [logs / "stdout.jsonl", logs / "stderr.txt"]
                try:
                    with paths[0].open("x") as out, paths[1].open("x") as err:
                        with subprocess.Popen(
                            command,
                            cwd=view.root,
                            env=child_env,
                            stdin=subprocess.PIPE,
                            stdout=out,
                            stderr=err,
                            text=True,
                            start_new_session=True,
                        ) as process:
                            try:
                                process.communicate(
                                    options.get("input"), timeout=options["timeout"]
                                )
                            except BaseException:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                                raise
                        for stream in (out, err):
                            stream.flush()
                            os.fsync(stream.fileno())
                finally:
                    # Keep partial output on timeout too. The shared compressor
                    # verifies gzip bytes before removing each closed raw file.
                    compress_files(paths, logs / "archive_manifest.json", 3)
                with open_text(paths[0]) as stream:
                    stdout = stream.read()
                with open_text(paths[1]) as stream:
                    stderr = stream.read()
                return subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)

            yield {
                "process_runner": run,
                "cli_executable": binary,
                "external_process_boundary": True,
            }
