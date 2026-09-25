"""Bind strict backend settings to CHIA's native, single-attempt session adapters.

This module does not launch a campaign, retry requests, choose candidates or
implement a model/tool protocol. Each instance belongs to one episode role.
The caller supplies the scoped CLI launcher or SDK transport and persists the
returned native evidence before another turn. Full process isolation and the
campaign's accounting remain requirements of that caller, not claims made here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable
from uuid import NAMESPACE_URL, uuid5

from .archive import safe_name
from .config import (ClaudeBackend, CodexBackend, ContextCompaction, DeepSeekBackend,
                     VertexBackend, VertexClaudeBackend)
from .identity import digest_json
from .snapshots import InvalidSnapshot, _parent, read_file


@dataclass(frozen=True)
class NativeSessionFiles:
    """Use the existing no-follow file operations at the native capture boundary.

    The agent controls its native files. A trusted adapter must not follow their
    links while reading a transcript, or follow a planted link during restore.
    This grants explicit roots only; it does not interpret the native contents.
    """

    roots: tuple[Path, ...]

    def __post_init__(self):
        for root in self.roots:
            if (
                not root.is_absolute()
                or root == Path("/")
                or not root.is_dir()
                or root.resolve(strict=True) != root
            ):
                raise ValueError("native file roots must be explicit canonical directories")

    def _location(self, path: str):
        target = Path(path)
        if not target.is_absolute() or str(target) != path:
            raise InvalidSnapshot("native file path must be an explicit absolute path")
        for root in self.roots:
            if target.is_relative_to(root) and target != root:
                relative = target.relative_to(root).as_posix()
                safe_name(relative)
                return root, relative
        raise InvalidSnapshot("native file is outside this session's storage roots")

    def read(self, path: str) -> bytes:
        root, relative = self._location(path)
        with _parent(root, relative) as (directory, name):
            size = os.stat(name, dir_fd=directory, follow_symlinks=False).st_size
        return read_file(root, relative, maximum_bytes=size, require_single_link=True)

    def write(self, path: str, data: bytes):
        from .snapshots import replace_file

        root, relative = self._location(path)
        with _parent(root, relative, create_parents=True):
            pass
        replace_file(root, relative, data)


@dataclass(frozen=True)
class NativeCheckpoint:
    """An identity envelope plus original native files, not a reconstructed chat.

    Archive assembly stores the files with its existing compression/inventory
    primitives. Only this role's session files belong here; login profiles and
    credentials do not. The metadata is ordinary JSON for safe offline inspection.
    """

    metadata: dict
    files: dict[str, bytes]

    def verify(self):
        expected = self.metadata["files"]
        if set(expected) != set(self.files):
            raise ValueError("native continuation inventory changed")
        for name, data in self.files.items():
            safe_name(name)
            if type(data) is not bytes or hashlib.sha256(data).hexdigest() != expected[name]:
                raise ValueError("native continuation bytes changed")


def upstream_identity() -> dict:
    """Require the reviewed source bytes, not just a matching package version."""
    from chia.base import llm_call
    from chia.models import claude, codex, openai_compat, vertex

    pin = json.loads((Path(__file__).parent / "upstream/session-hooks.json").read_text())
    modules = {
        "chia/base/llm_call.py": llm_call,
        "chia/models/claude.py": claude,
        "chia/models/codex.py": codex,
        "chia/models/openai_compat.py": openai_compat,
        "chia/models/vertex.py": vertex,
    }
    hashes = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in modules.items()
    }
    if hashes != pin["patched_files"]:
        raise RuntimeError("CHIA adapter source differs from the reviewed session extension")
    return {"revision": pin["revision"], "patch_sha256": pin["patch_sha256"], "files": hashes}


@dataclass
class ModelSession:
    """One private native session; the common episode owns its phase sequence."""

    backend: CodexBackend | ClaudeBackend | VertexBackend | VertexClaudeBackend | DeepSeekBackend
    session_id: str
    model: object
    settings: dict

    def prompt_once(self, text: str, tools=()):
        """Return CHIA's native result, including failure evidence, without retries."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("a native turn needs explicit instructions")
        if (
            tools
            and self.backend.kind in {"codex_cli", "claude_cli"}
            and self.settings.get("mcp_authentication") != "native_environment_reference"
        ):
            raise ValueError("native CLI tools need an explicit phase authentication binding")
        return self.model.prompt_once(text, list(tools))

    def checkpoint(self, result) -> NativeCheckpoint:
        """Retain failure continuations too; never serialize the whole adapter."""
        kind = self.backend.kind
        if kind == "codex_cli":
            if not result.session_id or not result.session_state:
                raise ValueError("Codex returned no resumable native session")
            state = {"session_id": result.session_id}
            files = dict(result.session_state)
            for name in files:
                if not (
                    name.startswith("sessions/") and name.endswith(".jsonl")
                ) and not re.fullmatch(
                    r"(?:state|thread_history|goals|memories|queue)_\d+\.sqlite(?:-wal|-shm)?", name
                ):
                    raise ValueError("unexpected Codex continuation file")
        elif kind == "claude_cli":
            if not result.session_id or not result.session_transcript:
                raise ValueError("Claude returned no resumable native transcript")
            state = {"session_id": result.session_id}
            files = {"transcript.jsonl": result.session_transcript}
        else:
            if result.session_state is None:
                raise ValueError("SDK returned no native continuation")
            state = {
                "sdk": "google.genai.types.Content"
                if kind == "vertex_gemini"
                else "anthropic.messages" if kind == "vertex_claude"
                else "chat_completions.messages"
            }
            # This is the SDK's own JSON representation, including opaque
            # signatures and function IDs. No answer text becomes history.
            files = {"contents.json": json.dumps(result.session_state, ensure_ascii=False).encode()}
        metadata = {
            "schema_version": 1,
            "session_id": self.session_id,
            "settings_sha256": digest_json(self.settings),
            "backend": kind,
            "state": state,
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }
        checkpoint = NativeCheckpoint(metadata, files)
        checkpoint.verify()
        return checkpoint

    def restore(self, checkpoint: NativeCheckpoint):
        """Restore only this role/settings identity through CHIA's native methods."""
        checkpoint.verify()
        meta = checkpoint.metadata
        if any(
            meta.get(key) != expected
            for key, expected in {
                "schema_version": 1,
                "session_id": self.session_id,
                "settings_sha256": digest_json(self.settings),
                "backend": self.backend.kind,
            }.items()
        ):
            raise ValueError("native continuation belongs to different settings or session")
        if self.backend.kind == "codex_cli":
            from chia.models.codex import CodexQueryResult

            self.model._sync_session(
                CodexQueryResult(
                    "",
                    0,
                    "",
                    "",
                    session_id=meta["state"]["session_id"],
                    session_state=dict(checkpoint.files),
                    session_state_paths=tuple(sorted(checkpoint.files)),
                )
            )
        elif self.backend.kind == "claude_cli":
            from chia.models.claude import ClaudeCodeQueryResult

            if meta["state"]["session_id"] != self.model._session_id:
                raise ValueError("Claude native session identity changed")
            self.model._sync_transcript(
                ClaudeCodeQueryResult(
                    "",
                    0,
                    "",
                    "",
                    session_transcript=checkpoint.files["transcript.jsonl"],
                    session_id=meta["state"]["session_id"],
                )
            )
        else:
            self.model.restore_session(json.loads(checkpoint.files["contents.json"]))


def create_session(
    backend: CodexBackend | ClaudeBackend | VertexBackend | VertexClaudeBackend | DeepSeekBackend,
    *,
    session_id: str,
    private_directory: Path,
    workspace: Path,
    system_message: str,
    timeout_seconds: int,
    process_runner: Callable | None = None,
    cli_executable: Path | None = None,
    temporary_directory: Path | None = None,
    vertex_client_kwargs: dict | None = None,
    vertex_execution_project: str | None = None,
    vertex_event_callback: Callable | None = None,
    chat_client_kwargs: dict | None = None,
    chat_event_callback: Callable | None = None,
    anthropic_client_kwargs: dict | None = None,
    anthropic_event_callback: Callable | None = None,
    tool_http_client_factory: Callable | None = None,
    tool_token_env: Callable | None = None,
    external_process_boundary: bool = False,
    context_compaction: ContextCompaction | None = None,
    context_anchor: Callable | None = None,
) -> ModelSession:
    """Translate settings, using one common turn contract for every backend.

    CLI process hooks are mandatory: CHIA's default inherited home/environment
    and subprocess execution do not establish our boundary. The Vertex transport
    and per-request evidence hook are mandatory for the same reason. Supplying
    hooks is not itself qualification; the launcher must test their enforcement.
    No credentials are read here and construction never makes a model call.
    """
    from chia.models.claude import ClaudeCodeLLM
    from chia.models.codex import CodexLLM
    from chia.models.openai_compat import OpenAICompatLLM
    from chia.models.vertex import VertexGeminiLLM

    if not all(
        hasattr(cls, "prompt_once")
        for cls in (CodexLLM, ClaudeCodeLLM, VertexGeminiLLM, OpenAICompatLLM)
    ):
        raise RuntimeError(
            "install the pinned CHIA session extension before binding a native adapter"
        )
    dependency = upstream_identity()
    if not isinstance(backend, (CodexBackend, ClaudeBackend, VertexBackend, VertexClaudeBackend, DeepSeekBackend)):
        raise TypeError("fixture and unknown backends are not native adapters")
    if vertex_execution_project is not None:
        if not isinstance(backend, VertexBackend) or not re.fullmatch(
            r"[a-z][a-z0-9-]{4,28}[a-z0-9]", vertex_execution_project
        ):
            raise ValueError("invalid Vertex execution project override")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("native request deadline must be a positive integer")
    if not session_id or not private_directory.is_absolute() or not workspace.is_absolute():
        raise ValueError("native session needs an identity and explicit absolute directories")
    settings = {
        "backend": backend.model_dump(mode="json"),
        "timeout_seconds": timeout_seconds,
        "system_sha256": hashlib.sha256(system_message.encode()).hexdigest(),
        "adapter_retry_attempts": 1,
        "resume_session": True,
        "chia": dependency,
    }
    common = {
        "model": backend.model,
        "system_message": system_message,
        "timeout_seconds": timeout_seconds,
        "retries": 1,
        "resume_session": True,
    }
    if context_compaction is not None:
        if not isinstance(backend, (VertexBackend, DeepSeekBackend)):
            raise ValueError("ADK compaction is for API backends; CLI compaction stays native")
        from .compaction import AdkCompactor

        common["context_compactor"] = AdkCompactor(
            context_compaction, vertex=isinstance(backend, VertexBackend), anchor=context_anchor
        )
        settings["context_compaction"] = {
            **context_compaction.model_dump(mode="json"),
            "implementation": "google-adk==2.3.0:LlmEventSummarizer",
        }
    file_options = {}
    if isinstance(backend, (CodexBackend, ClaudeBackend)) and process_runner is not None:
        storage = NativeSessionFiles((private_directory,))
        temporary_directory = temporary_directory or private_directory / "tmp"
        if not temporary_directory.is_relative_to(private_directory):
            raise ValueError("native temporary files must stay within the private session root")
        relative = (temporary_directory / "unused").relative_to(private_directory).as_posix()
        with _parent(private_directory, relative, create_parents=True):
            pass
        file_options = {"native_file_reader": storage.read, "native_file_writer": storage.write}
        settings["native_file_access"] = "no_follow_role_storage_v1"
    if isinstance(backend, CodexBackend):
        if process_runner is None:
            raise ValueError("Codex requires the scoped native process launcher")
        extra = ["--ignore-user-config", "--strict-config"]
        if backend.service_tier is not None:
            extra += ["-c", "service_tier=" + json.dumps(backend.service_tier)]
        model = CodexLLM(
            **common,
            **file_options,
            process_runner=process_runner,
            codex_bin="codex" if cli_executable is None else str(cli_executable),
            temporary_directory=None if temporary_directory is None else str(temporary_directory),
            work_dir=str(workspace),
            codex_home=str(private_directory),
            reasoning_effort=backend.reasoning_effort,
            auto_compact_token_limit=None
            if backend.auto_compact_tokens is None
            else backend.auto_compact_tokens.value,
            dangerously_bypass_approvals_and_sandbox=False,
            approval_policy="never",
            sandbox="danger-full-access" if external_process_boundary else "workspace-write",
            ignore_rules=True,
            extra_cli_args=extra,
            tool_token_env=tool_token_env,
            mcp_approval_policy="approve",
            capture_all_sessions=True,
        )
        settings.update(
            instruction_mode="native_guidance_with_user_task_prefix",
            compaction=(
                "native_default"
                if backend.auto_compact_tokens is None
                else backend.auto_compact_tokens.model_dump()
            ),
            native_permission_mode=(
                "external_landlock" if external_process_boundary else "workspace-write"
            ),
            approval_policy="never",
            mcp_approval_policy="approve_registered_phase_tools",
            native_state_inventory="codex_private_profile_v2",
            # The installed CLI stores absolute rollout paths in its database.
            # Restoring native bytes on another worker therefore requires this
            # same role path; do not silently rewrite or discard native state.
            native_home=str(private_directory),
        )
    elif isinstance(backend, ClaudeBackend):
        if process_runner is None:
            raise ValueError("Claude requires the scoped native process launcher")
        if backend.output_tokens is not None:
            # The CLI's output-token limit is an environment setting, not CHIA's
            # ignored max_tokens constructor argument. Do not claim it applied
            # until the scoped launcher has an explicit environment binding.
            raise ValueError(
                "explicit Claude output limits require the qualified "
                "launch-profile environment binding"
            )
        model = ClaudeCodeLLM(
            **common,
            **file_options,
            process_runner=process_runner,
            claude_bin="claude" if cli_executable is None else str(cli_executable),
            work_dir=str(workspace),
            temporary_directory=None if temporary_directory is None else str(temporary_directory),
            projects_cwd=str(
                private_directory / "projects" / re.sub(r"[^a-zA-Z0-9]", "-", str(workspace))
            ),
            session_id=str(uuid5(NAMESPACE_URL, "ramulator-chia:" + session_id)),
            append_system_message=True,
            log_stream=True,
            dangerously_skip_permissions=False,
            extra_cli_args=[
                "--effort",
                backend.reasoning_effort,
                "--setting-sources",
                "",
                "--strict-mcp-config",
            ],
            tool_token_env=tool_token_env,
        )
        settings.update(
            instruction_mode="append_to_native_system",
            output_limit="native_default",
            native_permission_mode="scoped_launcher_required",
            native_session_layout="claude_config_dir_projects_v1",
        )
    elif isinstance(backend, VertexBackend):
        if vertex_client_kwargs is None or vertex_event_callback is None:
            raise ValueError(
                "Vertex requires explicit SDK transport and per-request evidence hooks"
            )
        from google.genai import types

        # Reuse the SDK's validated timeout/retry knobs. A supplied HttpOptions
        # object can carry approved HTTP hooks/credentials without exposing them
        # in the scientific settings or native continuation.
        options = dict(vertex_client_kwargs)
        http = types.HttpOptions.model_validate(options.get("http_options", {}))
        if http.extra_body:
            raise ValueError("transport extra_body cannot override the scientific request settings")
        if http.timeout not in (None, timeout_seconds * 1000):
            raise ValueError("Vertex transport deadline differs from the requested settings")
        if http.retry_options is not None and http.retry_options.attempts not in (None, 1):
            raise ValueError("SDK retries would bypass per-attempt accounting")
        options["http_options"] = http.model_copy(
            update={
                "timeout": timeout_seconds * 1000,
                "retry_options": types.HttpRetryOptions(attempts=1),
            }
        )
        thinking = types.ThinkingConfig(
            thinking_level=backend.reasoning_effort.upper(),
            include_thoughts=backend.include_thoughts,
        )
        model = VertexGeminiLLM(
            **common,
            project=vertex_execution_project or backend.project,
            location=backend.location,
            client_kwargs=options,
            max_tokens=(None if backend.output_tokens is None else backend.output_tokens.value),
            max_tool_iterations=None,
            event_callback=vertex_event_callback,
            tool_http_client_factory=tool_http_client_factory,
            generation_config={
                "thinking_config": thinking.model_dump(mode="json", exclude_none=True)
            },
        )
        settings.update(
            instruction_mode="system_instruction",
            sdk_retry_attempts=1,
            sdk_timeout_ms=timeout_seconds * 1000,
            tool_iteration_limit=None,
        )
    elif isinstance(backend, VertexClaudeBackend):
        if anthropic_client_kwargs is None or anthropic_event_callback is None:
            raise ValueError("Vertex Claude requires explicit SDK transport and evidence hooks")
        if set(anthropic_client_kwargs) - {"credentials", "http_client"} or \
                anthropic_client_kwargs.get("credentials") is None:
            raise ValueError("Vertex Claude transport must supply credentials, not request overrides")
        import anthropic

        if anthropic.__version__ != "1.7.0":
            raise RuntimeError("Vertex Claude requires the qualified anthropic==1.7.0 SDK")
        options = dict(anthropic_client_kwargs)
        model = ClaudeCodeLLM(
            **common,
            backend="api",
            api_client_factory=partial(anthropic.AsyncAnthropicVertex,
                **options, project_id=backend.project, region=backend.location,
                base_url="https://aiplatform.googleapis.com/v1",
                max_retries=0, timeout=timeout_seconds,
            ),
            max_tokens=backend.output_tokens.value,
            thinking="adaptive",
            max_tool_iterations=None,
            api_stream=True,
            generation_config={
                "output_config": {"effort": backend.reasoning_effort},
                "cache_control": {"type": "ephemeral"},
                "betas": ["compact-2026-01-12"],
                "context_management": {"edits": [{
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": backend.compaction_input_tokens},
                }]},
            },
            event_callback=anthropic_event_callback,
            tool_http_client_factory=tool_http_client_factory,
        )
        settings.update(
            instruction_mode="system_message", sdk_retry_attempts=1,
            sdk_timeout_ms=timeout_seconds * 1000, tool_iteration_limit=None,
            thinking="adaptive", sdk="anthropic==1.7.0:AsyncAnthropicVertex",
            prompt_cache="stable_system_prefix_and_automatic_history_5m",
            compaction={"implementation": "provider_native_threshold", "instructions": "provider_default",
                        "input_tokens": backend.compaction_input_tokens,
                        "beta": "compact-2026-01-12", "original_history": "retained"},
        )
    elif isinstance(backend, DeepSeekBackend):
        if chat_client_kwargs is None or chat_event_callback is None:
            raise ValueError(
                "DeepSeek requires explicit SDK transport and per-request evidence hooks"
            )
        if set(chat_client_kwargs) - {"api_key", "http_client"} or not chat_client_kwargs.get(
            "api_key"
        ):
            raise ValueError("DeepSeek transport must supply a key, not override request settings")
        model = OpenAICompatLLM(
            **common,
            base_url="https://api.deepseek.com",
            client_kwargs={**chat_client_kwargs, "max_retries": 0, "timeout": timeout_seconds},
            max_tokens=None if backend.output_tokens is None else backend.output_tokens.value,
            max_tool_iterations=None,
            generation_config={
                "reasoning_effort": backend.reasoning_effort,
                "extra_body": {"thinking": {"type": "enabled"}},
            },
            event_callback=chat_event_callback,
            tool_http_client_factory=tool_http_client_factory,
        )
        settings.update(
            instruction_mode="system_message",
            endpoint="https://api.deepseek.com/chat/completions",
            sdk_retry_attempts=1,
            sdk_timeout_ms=timeout_seconds * 1000,
            tool_iteration_limit=None,
            thinking="enabled",
        )
    else:
        raise TypeError("fixture and unknown backends are not native adapters")
    if isinstance(backend, (CodexBackend, ClaudeBackend)):
        settings["mcp_authentication"] = (
            "native_environment_reference" if tool_token_env is not None else "not_bound"
        )
    return ModelSession(backend, session_id, model, settings)
