"""One native-session driver shared by all model backends.

CHIA owns the provider conversation and tool loop. This class supplies the
phase prompt and saves its original result/state. A scoped transport is injected
when constructing the model; there is deliberately no unrestricted CLI fallback.
"""

from __future__ import annotations

import gzip
import json
import secrets
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Callable, ContextManager
from uuid import uuid4

import httpx
from pydantic_core import to_jsonable_python

from .archive import safe_name
from .identity import canonical_json, digest_json, file_sha256
from .model_sessions import ModelSession, NativeCheckpoint, create_session
from .records import CampaignState, TransientFailure
from .snapshots import publish_bytes, replace_file
from .tool_server import phase_tools
from .usage import Tariff, at_path, normalize, protocols, quote
from .workspace import Workspace

TOOL_TOKEN_ENV = "CHIA_WORKSPACE_TOKEN"


def codex_capacity_failure(raw_stdout: str) -> bool:
    """Match the CLI's terminal error, never text written by the model or a tool."""
    for line in raw_stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "turn.failed":
            continue
        error = event.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and message.startswith("Selected model is at capacity."):
            return True
    return False


INSTRUCTIONS = {
    "explore": (
        "Read references/task.md and references/api.h. Inspect your own earlier summaries "
        "if present. Develop a principled immediate-response model in draft/. Use training "
        "evaluations and enabled diagnostics to investigate it. Poll every started job to "
        "completion. End this turn when your draft is ready for final evaluation."
    ),
    "review": (
        "Independently inspect the proposed model against references/task.md and api.h. "
        "Give one concise critique of semantic compliance, citing source locations and "
        "uncertainty. Do not edit files or suggest workload-specific answers. Your review "
        "is advisory, not a promotion decision."
    ),
    "revise": (
        "Consider this one independent critique in the same conversation. You may inspect, "
        "revise and evaluate your draft. Poll all started jobs before ending. This is the "
        "last editing phase; the harness will evaluate the source you leave in draft/."
    ),
    "reflect": (
        "The submitted source and scores are frozen. Write notes/summary.md with what you "
        "changed, the physical reasoning, measured results, failed ideas, remaining issues "
        "and next questions. Distinguish evidence from conjecture. The next iteration will "
        "receive this summary and your campaign history. Do not alter the model."
    ),
}


@dataclass
class NativeSessions:
    """Attach one native-session lifecycle to the common DRAM service.

    ``transport`` is a trusted context manager supplied by the execution host.
    It receives (workspace, private_directory, tool_environment) and returns the
    scoped process/SDK options accepted by ``create_session``. CLI runners must
    enforce workspace.grants() on every phase and bind only the supplied tool
    environment. This factory does not grant process access or read a login.

    There is deliberately no inherited-environment or unrestricted fallback.
    The same factory serves all three backends, including the independent review.
    """

    transport: Callable[[Workspace, Path, dict[str, str]], ContextManager[dict]] | None
    tariff: Tariff | None = None
    capacity_cooldown_seconds: int = 3600

    def __post_init__(self):
        if type(self.capacity_cooldown_seconds) is not int or self.capacity_cooldown_seconds < 0:
            raise ValueError("capacity cooldown must be a nonnegative integer")

    @contextmanager
    def __call__(self, research, iteration, role, candidate, history):
        if not callable(self.transport):
            raise ValueError("native sessions require an explicit scoped transport")
        state = CampaignState(research.root / "campaign.sqlite")
        identity = state.get("native_identity")
        if identity is None:
            identity = state.save("native_identity", {"id": uuid4().hex})
        view = Workspace(research, iteration, role, candidate, history)
        private = research.root / "private" / str(iteration) / role
        private.mkdir(mode=0o700, parents=True, exist_ok=True)
        token = secrets.token_urlsafe(32)
        timeout = research.configuration.run.model_timeout_seconds

        @contextmanager
        def tools(workspace):
            with phase_tools(workspace, token) as offered:
                yield offered

        with self.transport(view, private, {TOOL_TOKEN_ENV: token}) as options:
            model = create_session(
                research.configuration.backend,
                # A reused human-readable campaign name is not permission to
                # import another campaign's native conversation/checkpoint.
                session_id=f"{identity['id']}:{iteration}:{role}",
                private_directory=private,
                workspace=view.root,
                system_message=(view.root / "references/task.md").read_text(),
                timeout_seconds=timeout,
                tool_token_env=lambda tool: TOOL_TOKEN_ENV,
                tool_http_client_factory=lambda tool: httpx.AsyncClient(
                    headers={"Authorization": "Bearer " + token}, trust_env=False, timeout=60
                ),
                # Construction makes no request. NativeAgent installs its
                # durable SDK evidence callback before the first turn.
                vertex_event_callback=lambda event, state: None,
                **options,
            )
            yield NativeAgent(
                view,
                model,
                tools,
                tariff=self.tariff,
                capacity_cooldown_seconds=self.capacity_cooldown_seconds,
            )


def cli_usage(raw_stdout: str, backend: str, tariff) -> list[dict]:
    """Read native terminal counters once; do not sum nested and aggregate totals."""
    rule = protocols()["backends"][backend]
    terminals = []
    for line in raw_stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get("type") in rule["terminal_types"]:
            terminals.append(event)
    if not terminals:
        return [
            {
                "scope": rule["scope"],
                "usage": None,
                "cost": quote(normalize(None, rule["protocol"]), tariff, scope=rule["scope"]),
            }
        ]
    rows = []
    for event in terminals:
        per_model = at_path(event, rule["per_model_path"]) if "per_model_path" in rule else None
        if per_model:
            entries = [
                (model, raw, rule["per_model_protocol"], rule["per_model_scope"])
                for model, raw in per_model.items()
            ]
        else:
            entries = [(None, at_path(event, rule["usage_path"]), rule["protocol"], rule["scope"])]
        for model, raw, protocol, scope in entries:
            usage = normalize(raw, protocol)
            accepted = tariff is not None and (
                model is None or model in (tariff.model, *tariff.accepted_response_models)
            )
            rows.append(
                {
                    "model": model,
                    "scope": scope,
                    "raw": raw,
                    "usage": usage,
                    "cost": quote(usage, tariff if accepted else None, scope=scope),
                }
            )
    return rows


class NativeAgent:
    def __init__(
        self, workspace, model: ModelSession, tools, *, tariff=None, capacity_cooldown_seconds=3600
    ):
        self.workspace, self.model, self.tools, self.tariff = workspace, model, tools, tariff
        self.capacity_cooldown_seconds = capacity_cooldown_seconds
        self.root = (
            workspace.research.root / "native-evidence" / str(workspace.iteration) / workspace.role
        )
        self.root.mkdir(parents=True, exist_ok=True)
        publish_bytes(self.root / "settings.json", canonical_json(model.settings).encode())
        if tariff is not None and tariff.model != model.backend.model:
            raise ValueError("usage tariff belongs to another configured model")
        publish_bytes(
            self.root / "tariff.json",
            canonical_json(None if tariff is None else tariff.model_dump(mode="json")).encode(),
        )
        self.active_attempt = None
        self.sdk_usage = []
        self.sdk_event_number = 0
        if model.backend.kind == "vertex_gemini":
            model.model.event_callback = self.vertex_event
        pointer = self.root / "continuation.json"
        if pointer.exists():
            receipt = json.loads(pointer.read_text())
            directory = self.root / receipt["directory"]
            if file_sha256(directory / "checkpoint.json") != receipt["sha256"]:
                raise ValueError("native continuation checkpoint changed")
            self.restore(directory)

    def snapshot(self):
        return self.workspace.snapshot()

    def summary(self):
        return self.workspace.summary()

    def restore(self, directory: Path):
        """Resume this role's native bytes; never turn answer text into history."""
        metadata = json.loads((directory / "checkpoint.json").read_text())
        for name in metadata["files"]:
            safe_name(name)
        files = {
            name: gzip.decompress((directory / "state" / (name + ".gz")).read_bytes())
            for name in metadata["files"]
        }
        self.model.restore(NativeCheckpoint(metadata, files))

    def vertex_event(self, event, state):
        """Persist exposed SDK events before CHIA proceeds to another request."""
        if self.active_attempt is None:
            raise RuntimeError("native request arrived outside an active campaign turn")
        self.sdk_event_number += 1
        value = {"event": event}
        rule = protocols()["backends"]["vertex_gemini"]
        if event["kind"] == "request":
            usage = normalize(None, rule["protocol"])
            self.sdk_usage.append(
                {
                    "scope": rule["scope"],
                    "model": self.model.backend.model,
                    "raw": None,
                    "usage": usage,
                    "cost": quote(usage, self.tariff, scope=rule["scope"]),
                    "status": "awaiting_response_usage_unknown",
                }
            )
            value["accounting"] = self.sdk_usage[-1]
        if event["kind"] == "request_error" and self.sdk_usage:
            self.sdk_usage[-1]["status"] = "failed_usage_unknown"
            value["accounting"] = self.sdk_usage[-1]
        if event["kind"] == "response":
            if (
                not self.sdk_usage
                or self.sdk_usage[-1]["status"] != "awaiting_response_usage_unknown"
            ):
                raise ValueError("SDK response has no matching recorded request")
            raw = at_path(event["payload"], rule["usage_path"])
            usage = normalize(raw, rule["protocol"])
            reported = at_path(event["payload"], rule["model_path"])
            accepted = self.tariff is not None and reported in (
                self.tariff.model,
                *self.tariff.accepted_response_models,
            )
            row = {
                "scope": rule["scope"],
                "model": reported,
                "raw": raw,
                "usage": usage,
                "cost": quote(usage, self.tariff if accepted else None, scope=rule["scope"]),
            }
            row["status"] = "response_recorded"
            self.sdk_usage[-1] = row
            value["accounting"] = row
        publish_bytes(
            self.active_attempt / "sdk" / f"{self.sdk_event_number:06d}.json.gz",
            gzip.compress(canonical_json(value).encode(), mtime=0),
        )
        if state is not None:
            # Only the most recent native state is needed for crash diagnosis;
            # the individual request/response/tool events remain immutable.
            replace_file(
                self.active_attempt,
                "sdk-latest-state.json.gz",
                gzip.compress(canonical_json(state).encode(), mtime=0),
            )

    def turn(self, phase: str, inputs: dict) -> dict:
        self.workspace.set_phase(phase)
        prompt = INSTRUCTIONS[phase] + "\n\n" + canonical_json(inputs)
        attempt = self.root / (phase + "-" + uuid4().hex)
        attempt.mkdir()
        self.active_attempt, self.sdk_usage, self.sdk_event_number = attempt, [], 0
        publish_bytes(attempt / "prompt.txt", prompt.encode())
        self.workspace.record(
            {
                "event": "native_turn_started",
                "phase": phase,
                "prompt_sha256": digest_json(prompt),
                "attempt": str(attempt.relative_to(self.workspace.research.root)),
            }
        )
        # The caller binds this one role's token/transport to this endpoint.
        # A transport crash remains an unresolved campaign step, not a free retry.
        with self.tools(self.workspace) as offered:
            result = self.model.prompt_once(prompt, offered)
            receipt = self.save_result(result, attempt)
        self.active_attempt = None
        self.workspace.record({"event": "native_turn_finished", "phase": phase, **receipt})
        if not result.success:
            message = f"native turn failed; evidence retained at {receipt['attempt']}"
            if (
                self.model.backend.kind == "codex_cli"
                and (attempt / "checkpoint.json").is_file()
                and codex_capacity_failure(result.raw_stdout)
            ):
                raise TransientFailure(
                    message + "; selected model is at capacity",
                    retry_after_seconds=self.capacity_cooldown_seconds,
                )
            # A completed, drained native attempt with resumable state can
            # continue after CHIA-classified service failure or truncation.
            # Unknown transport loss/capture failure never reaches this branch.
            if (attempt / "checkpoint.json").is_file() and getattr(
                result.error, "error_type", None
            ) in {"server_error", "rate_limit", "max_output_tokens"}:
                raise TransientFailure(message)
            raise RuntimeError(message)
        return receipt

    def save_result(self, result, attempt):
        """Save provider output before endpoint cleanup can itself fail."""
        raw = {}
        if not is_dataclass(result):
            raise TypeError("CHIA returned an unsupported native result type")
        for field in fields(result):
            if field.name not in {"session_state", "session_transcript", "error"}:
                raw[field.name] = to_jsonable_python(getattr(result, field.name))
        raw["error"] = (
            None
            if result.error is None
            else {"type": type(result.error).__name__, "message": str(result.error)}
        )
        publish_bytes(
            attempt / "result.json.gz", gzip.compress(canonical_json(raw).encode(), mtime=0)
        )
        # Failure state is valuable too. Save it before propagating failure.
        try:
            checkpoint = self.model.checkpoint(result)
        except ValueError as exc:
            publish_bytes(attempt / "checkpoint-error.txt", str(exc).encode())
            if result.success:
                raise
        else:
            for name, data in checkpoint.files.items():
                publish_bytes(attempt / "state" / (name + ".gz"), gzip.compress(data, mtime=0))
            publish_bytes(attempt / "checkpoint.json", canonical_json(checkpoint.metadata).encode())
            replace_file(
                self.root,
                "continuation.json",
                canonical_json(
                    {
                        "directory": attempt.name,
                        "sha256": file_sha256(attempt / "checkpoint.json"),
                    }
                ).encode(),
            )
        usage = (
            self.sdk_usage
            if self.model.backend.kind == "vertex_gemini"
            else cli_usage(result.raw_stdout, self.model.backend.kind, self.tariff)
        )
        receipt = {
            "text": result.result,
            "success": result.success,
            "attempt": str(attempt.relative_to(self.workspace.research.root)),
            "result_sha256": file_sha256(attempt / "result.json.gz"),
            "usage": usage,
        }
        publish_bytes(attempt / "receipt.json", canonical_json(receipt).encode())
        return receipt
