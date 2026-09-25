"""Offline qualification of the pinned CHIA adapter extension, not live models.

Run with the patched CHIA checkout first on PYTHONPATH. Native CLI subprocesses
and provider transport are replaced with explicit fixtures; SDK Content/schema
objects and CHIA's parsing, capture, restore and tool loop are real. No fixture
contains a previous DRAM design, credentials or research discussion.
"""

import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from chia.models.claude import ClaudeCodeLLM
from chia.models.codex import CodexLLM
from chia.models.vertex import VertexGeminiLLM
from google.genai import types
from ray import cloudpickle

pytestmark = pytest.mark.skipif(
    not all(hasattr(cls, "prompt_once") for cls in (CodexLLM, ClaudeCodeLLM, VertexGeminiLLM)),
    reason="requires the qualified CHIA session patch; see framework/upstream/README.md",
)
SID = "123e4567-e89b-12d3-a456-426614174000"
LONG = "native output kept in full\n" * 500


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("adapter qualification cannot call a service or launch a model")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(subprocess.Popen, "__init__", denied)
    from chia.trace import profiler

    monkeypatch.setattr(profiler, "get_collector", lambda namespace=None: None)
    profiler.reset_profiler()
    yield
    profiler.reset_profiler()


def codex(tmp_path, run):
    return CodexLLM(
        model="gpt-6-astra",
        reasoning_effort="xhigh",
        resume_session=True,
        codex_home=str(tmp_path),
        work_dir=str(tmp_path),
        process_runner=run,
        auto_compact_token_limit=None,
        dangerously_bypass_approvals_and_sandbox=False,
        retries=1,
    )


def codex_stdout(*, failed=False):
    return "\n".join(
        json.dumps(value)
        for value in [
            {"type": "thread.started", "thread_id": SID},
            {"type": "item.completed", "item": {"type": "reasoning", "text": LONG}},
            {
                "type": "turn.failed" if failed else "turn.completed",
                "usage": {"input_tokens": 100, "cached_input_tokens": 70, "output_tokens": 8},
            },
        ]
    )


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            {
                "message": (
                    "stream disconnected before completion: stream closed before response.completed"
                )
            },
            "server_error",
        ),
        ({"message": "Selected model is at capacity. Please try again later."}, "server_error"),
        ({"message": "unexpected status 400 Bad Request"}, "invalid_request"),
        ({"message": "HTTP/1.1 401"}, "authentication_failed"),
        ({"message": "HTTP 429 Too Many Requests"}, "rate_limit"),
        ({"message": "HTTP 503"}, "server_error"),
        ({"message": "context window exceeded"}, "max_output_tokens"),
        (
            {"message": "unknown failure", "codexErrorInfo": "ResponseStreamDisconnected"},
            "server_error",
        ),
        ({"message": "unrecognized terminal failure"}, "unknown"),
    ],
)
def test_codex_classifies_native_errors_not_generated_transcript(tmp_path, error, expected):
    from chia.models.codex import CodexError, CodexQueryResult

    # The saved max failure contained 400 in generated text. Include other
    # classifier vocabulary too: none of it identifies a provider failure.
    generated = "400 401 402 429 500 503 rate limit billing authentication max output"
    raw = "\n".join(
        json.dumps(event)
        for event in [
            {"type": "item.completed", "item": {"type": "agent_message", "text": generated}},
            {"type": "turn.failed", "error": error},
        ]
    )
    result = CodexQueryResult(generated, 1, "", generated, raw_stdout=raw)
    with pytest.raises(CodexError) as failed:
        codex(tmp_path, None)._classify_error(result)
    assert failed.value.error_type == expected


def test_codex_success_and_unknown_exit_ignore_error_words_in_model_output(tmp_path):
    from chia.models.codex import CodexQueryResult, UnknownCodexError

    generated = "400 unauthorized billing rate limit server error"
    model = codex(tmp_path, None)
    raw = json.dumps(
        {"type": "item.completed", "item": {"type": "agent_message", "text": generated}}
    )
    model._classify_error(CodexQueryResult(generated, 0, "", generated, raw_stdout=raw))
    with pytest.raises(UnknownCodexError):
        model._classify_error(CodexQueryResult(generated, 1, "", generated, raw_stdout=raw))
    recovered = raw + "\n" + json.dumps({"type": "error", "message": "HTTP 429"})
    recovered += "\n" + json.dumps({"type": "turn.completed"})
    model._classify_error(CodexQueryResult(generated, 0, "", generated, raw_stdout=recovered))


def test_codex_cli_startup_failure_uses_stderr_without_a_native_event(tmp_path):
    from chia.models.codex import CodexQueryResult, InvalidRequestError

    with pytest.raises(InvalidRequestError):
        codex(tmp_path, None)._classify_error(
            CodexQueryResult("400", 1, "unrecognized option --broken", "")
        )


def test_cli_mcp_auth_reuses_native_environment_configuration(tmp_path):
    tool = SimpleNamespace(hostname="127.0.0.1", port=8000, name="workspace")

    def callback(offered):
        return "CHIA_MCP_FIXTURE" if offered is tool else None

    codex_model = codex(tmp_path, None)
    codex_model.tool_token_env = callback
    codex_model.mcp_approval_policy = "approve"
    args = codex_model._mcp_config_args([tool])
    assert 'mcp_servers.workspace.bearer_token_env_var="CHIA_MCP_FIXTURE"' in args
    assert "mcp_servers.workspace.required=true" in args
    assert 'mcp_servers.workspace.default_tools_approval_mode="approve"' in args
    claude_model = ClaudeCodeLLM(tool_token_env=callback)
    entry = claude_model._build_mcp_config([tool])["mcpServers"]["workspace"]
    assert entry == {
        "type": "http",
        "url": "http://127.0.0.1:8000/workspace/mcp",
        "headers": {"Authorization": "Bearer ${CHIA_MCP_FIXTURE}"},
    }


@pytest.mark.parametrize("invalid", [None, "", "Bearer credential", "bad${TOKEN}", 7])
def test_cli_mcp_auth_rejects_values_that_are_not_environment_names(tmp_path, invalid):
    tool = SimpleNamespace(hostname="127.0.0.1", port=8000, name="workspace")
    codex_model = codex(tmp_path, None)
    codex_model.tool_token_env = lambda _: invalid
    claude_model = ClaudeCodeLLM(tool_token_env=lambda _: invalid)
    with pytest.raises(ValueError, match="environment-variable"):
        codex_model._mcp_config_args([tool])
    with pytest.raises(ValueError, match="environment-variable"):
        claude_model._build_mcp_config([tool])


def rollout(root, value):
    target = root / "sessions" / "2099" / f"rollout-fixture-{SID}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)
    return target


def test_codex_private_inventory_preserves_runtime_databases_and_child_rollouts(tmp_path):
    # Native bytes are opaque. No schema rewriting or history reconstruction.
    database_names = [
        f"{name}_1.sqlite" for name in ("state", "thread_history", "goals", "memories", "queue")
    ]
    for name in database_names:
        (tmp_path / name).write_bytes((name + " original native bytes").encode())
    (tmp_path / "auth.json").write_text("not a continuation asset")
    (tmp_path / "logs_2.sqlite").write_bytes(b"separate runtime log, not conversation state")
    child = tmp_path / "sessions/2099/rollout-child-other-session.jsonl"
    child.parent.mkdir(parents=True)
    child.write_bytes(b"child native conversation")

    def run(cmd, **kw):
        rollout(tmp_path, b"parent native conversation")
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text("done")
        return SimpleNamespace(stdout=codex_stdout(), stderr="", returncode=0)

    model = codex(tmp_path, run)
    # An upstream caller that has not declared a private profile retains CHIA's
    # narrower legacy inventory; do not scoop up another thread's native state.
    default_result = model.prompt_once("fixture")
    assert set(default_result.session_state) == {
        "state_1.sqlite",
        f"sessions/2099/rollout-fixture-{SID}.jsonl",
    }
    model.capture_all_sessions = True
    result = model.prompt_once("fixture")
    assert result.success
    assert all(name in result.session_state for name in database_names)
    assert result.session_state[child.relative_to(tmp_path).as_posix()] == child.read_bytes()
    assert "auth.json" not in result.session_state and "logs_2.sqlite" not in result.session_state


@pytest.mark.parametrize("failure", ["server", "authentication", "timeout"])
def test_codex_failed_attempt_captures_latest_state_before_retry(tmp_path, failure):
    calls = []
    latest = b'{"fixture": "newest continuation after a tool", "opaque": "unchanged"}\n'

    def run(cmd, **kw):
        calls.append(cmd)
        rollout(tmp_path, latest)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1, output=codex_stdout(failed=True).encode())
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text("partial")
        return SimpleNamespace(
            stdout=codex_stdout(failed=True),
            stderr=("503 service unavailable" if failure == "server" else "401 unauthorized"),
            returncode=1,
        )

    model = codex(tmp_path, run)
    result = model.prompt_once("explore the fixture", [])
    assert not result.success and result.error is not None
    assert len(calls) == 1
    assert result.session_id == SID
    assert latest in result.session_state.values()
    assert result.raw_stdout == codex_stdout(failed=True)
    assert LONG in result.raw_stdout.replace("\\n", "\n")

    # Simulate a worker replacement, using CHIA's own result synchronization and
    # restoration. No construction from a model's answer text is involved.
    replacement_root = tmp_path / "replacement"

    def retry(cmd, **kw):
        assert cmd[-2:] == [SID, "-"] and "resume" in cmd
        assert next(replacement_root.rglob("rollout-*.jsonl")).read_bytes() == latest
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text("completed")
        return SimpleNamespace(stdout=codex_stdout(), stderr="", returncode=0)

    replacement = codex(replacement_root, retry)
    replacement._sync_session(cloudpickle.loads(cloudpickle.dumps(result)))
    completed = replacement.prompt_once("continue after the recorded failure", [])
    assert completed.success and completed.session_id == SID


@pytest.mark.parametrize("failure", ["server", "authentication", "timeout"])
def test_claude_failed_attempt_preserves_transcript_and_append_prompt(tmp_path, failure):
    calls = []
    transcript = b'{"type":"fixture","signature":"native-value","text":"tool acknowledged"}\n'
    raw = json.dumps(
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": LONG}]}}
    )

    def run(cmd, **kw):
        calls.append(cmd)
        assert "--append-system-prompt" in cmd and "--system-prompt" not in cmd
        assert cmd[cmd.index("--session-id") + 1] == SID
        (tmp_path / (SID + ".jsonl")).write_bytes(transcript)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1, output=raw.encode())
        return SimpleNamespace(
            stdout=raw,
            stderr=("503 service unavailable" if failure == "server" else "401 unauthorized"),
            returncode=1,
        )

    model = ClaudeCodeLLM(
        model="fixture-fable",
        system_message="fixture contract",
        resume_session=True,
        session_id=SID,
        projects_cwd=str(tmp_path),
        process_runner=run,
        append_system_message=True,
        retries=1,
        extra_cli_args=["--effort", "xhigh"],
    )
    result = model.prompt_once("explore", [])
    assert not result.success and result.error is not None and len(calls) == 1
    assert result.session_transcript == transcript and result.session_id == SID
    assert result.raw_stdout == raw

    other = tmp_path / "replacement"

    def retry(cmd, **kw):
        assert "--session-id" not in cmd and cmd[cmd.index("--resume") + 1] == SID
        assert (other / (SID + ".jsonl")).read_bytes() == transcript
        return SimpleNamespace(
            stdout=json.dumps({"type": "result", "result": "done"}), stderr="", returncode=0
        )

    replacement = ClaudeCodeLLM(
        model="fixture-fable",
        resume_session=True,
        session_id=result.session_id,
        projects_cwd=str(other),
        process_runner=retry,
    )
    replacement._sync_transcript(cloudpickle.loads(cloudpickle.dumps(result)))
    assert replacement.prompt_once("continue", []).success


def response(parts=None, *, text="done", finish="STOP"):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=([types.Part(text=text)] if parts is None else parts)
                ),
                finish_reason=finish,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            cached_content_token_count=70,
            candidates_token_count=10,
            thoughts_token_count=20,
            total_token_count=130,
        ),
    )


def vertex_fixture(monkeypatch, responses, *, schema=None, callback=None, mock_sdk=True):
    from contextlib import asynccontextmanager

    import mcp
    import mcp.client.streamable_http as transport
    from google import genai
    from mcp import types as mcp_types

    calls, tool_calls, snapshots, events = [], [], [], []

    class Models:
        def generate_content(self, **kw):
            calls.append(
                {
                    "contents": [
                        c.model_dump(mode="json", exclude_none=True) for c in kw["contents"]
                    ],
                    "config": kw["config"],
                    "model": kw["model"],
                }
            )
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    if mock_sdk:
        monkeypatch.setattr(
            genai, "Client", lambda **kw: SimpleNamespace(models=Models(), close=lambda: None)
        )

    @asynccontextmanager
    async def connect(*args, **kw):
        yield object(), object(), None

    class Session:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def list_tools(self):
            return mcp_types.ListToolsResult(
                tools=[
                    mcp_types.Tool(
                        name="inspect",
                        description="fixture instrument",
                        inputSchema=schema or {"type": "object"},
                    )
                ]
            )

        async def call_tool(self, name, args):
            tool_calls.append((name, args))
            return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=LONG)])

    monkeypatch.setattr(mcp, "ClientSession", Session)
    monkeypatch.setattr(transport, "streamable_http_client", connect)

    def observe(event, state):
        events.append(event)
        snapshots.append(state)
        if callback:
            callback(event, state)

    def model():
        return VertexGeminiLLM(
            model="gemini-3.8-flash",
            project="offline-project",
            location="global",
            resume_session=True,
            retries=1,
            max_tokens=None,
            max_tool_iterations=None,
            generation_config={
                "thinking_config": {"thinking_level": "HIGH", "include_thoughts": True}
            },
            event_callback=observe,
        )

    return SimpleNamespace(
        model=model,
        calls=calls,
        tools=tool_calls,
        events=events,
        snapshots=snapshots,
        tool=SimpleNamespace(name="dram", hostname="127.0.0.1", port=1),
    )


def test_vertex_sdk_contents_signatures_and_call_ids_survive_failure_and_replacement(monkeypatch):
    signature = bytes(range(256))
    tool_part = types.Part(
        thought_signature=signature,
        function_call=types.FunctionCall(
            name="dram__inspect", id="fixture-native-call-id", args={"draft": "current"}
        ),
    )
    fixture = vertex_fixture(
        monkeypatch, [response([tool_part]), RuntimeError("fixture connection failure"), response()]
    )
    model = fixture.model()
    result = model.prompt_once("explore", [fixture.tool])
    assert not result.success and len(fixture.calls) == 2 and len(fixture.tools) == 1
    assert [e["kind"] for e in result.native_events] == [
        "request",
        "response",
        "tool_request",
        "tool_response",
        "tools_complete",
        "request",
        "request_error",
    ]
    assert len(result.native_events[1]["payload"]["candidates"]) == 1
    assert result.native_events[1]["payload"]["usage_metadata"]["thoughts_token_count"] == 20
    assert result.session_state["open_user_message"] == "explore"
    assert result.session_state["pending_tools"] is False
    assert (
        result.session_state["contents"][-1]["parts"][0]["function_response"]["id"]
        == "fixture-native-call-id"
    )

    replacement = fixture.model()
    # JSON is the actual SDK serialization, not an assistant-text reconstruction.
    replacement.restore_session(json.loads(json.dumps(result.session_state)))
    assert replacement._session_contents[1].parts[0].thought_signature == signature
    completed = replacement.prompt_once("explore", [fixture.tool])
    assert completed.success and len(fixture.tools) == 1
    assert len(fixture.calls[-1]["contents"]) == 3  # no duplicated user prompt/tool effects
    assert completed.session_state["open_user_message"] is None


def test_vertex_one_session_spans_explore_revision_and_readonly_reflection(monkeypatch):
    fixture = vertex_fixture(
        monkeypatch,
        [response(text="draft"), response(text="revision"), response(text="reflection")],
    )
    model = fixture.model()
    for phase in ("explore", "revise", "reflect"):
        result = model.prompt_once(phase, [])
        assert result.success
        replacement = fixture.model()
        replacement.restore_session(result.session_state)
        model = replacement
    assert [len(call["contents"]) for call in fixture.calls] == [1, 3, 5]
    assert [c["parts"][0]["text"] for c in result.session_state["contents"]] == [
        "explore",
        "draft",
        "revise",
        "revision",
        "reflect",
        "reflection",
    ]
    assert fixture.calls[0]["config"].thinking_config.thinking_level == "HIGH"
    assert fixture.calls[0]["config"].max_output_tokens is None


def test_vertex_mcp_nested_schema_is_not_stripped(monkeypatch):
    from mcp.server.fastmcp import FastMCP

    from ramulator_chia.framework.diagnostic_cases import SyntheticRequest

    def diagnostic(case: SyntheticRequest) -> dict:
        return {}

    server = FastMCP("schema")
    server.add_tool(diagnostic)
    schema = server._tool_manager.list_tools()[0].parameters
    assert "$defs" in schema
    fixture = vertex_fixture(monkeypatch, [response()], schema=schema)
    assert fixture.model().prompt_once("inspect schema", [fixture.tool]).success
    declaration = fixture.calls[0]["config"].tools[0].function_declarations[0]
    assert declaration.parameters is None and declaration.parameters_json_schema == schema


def test_vertex_checkpoint_failure_stops_without_swallowing_or_repeating_tool(monkeypatch):
    def fail(event, state):
        if event["kind"] == "tool_response":
            raise OSError("fixture evidence storage failed")

    fixture = vertex_fixture(
        monkeypatch,
        [response([types.Part(function_call=types.FunctionCall(name="dram__inspect", args={}))])],
        callback=fail,
    )
    result = fixture.model().prompt_once("explore", [fixture.tool])
    assert not result.success and len(fixture.calls) == 1 and len(fixture.tools) == 1
    assert result.session_state["pending_tools"] is True
    replacement = fixture.model()
    replacement.restore_session(result.session_state)
    again = replacement.prompt_once("explore", [fixture.tool])
    assert not again.success and "unresolved tool" in str(again.error)
    assert len(fixture.calls) == 1 and len(fixture.tools) == 1


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY"])
def test_vertex_unusable_response_retains_native_evidence(monkeypatch, finish):
    fixture = vertex_fixture(monkeypatch, [response(text=LONG, finish=finish)])
    result = fixture.model().prompt_once("explore", [])
    assert not result.success and len(fixture.calls) == 1
    assert LONG in json.dumps(result.session_state).replace("\\n", "\n")
    assert result.native_events[-1]["kind"] == "response"


def test_single_attempt_capture_failure_is_not_a_reason_to_retry():
    from chia.base.llm_call import QueryResult, single_attempt

    calls = []

    def run():
        calls.append("run")
        return QueryResult("done", 0, "", "")

    def capture(result):
        calls.append("capture")
        raise OSError("fixture disk failure")

    with pytest.raises(OSError, match="disk failure"):
        single_attempt(run, lambda: None, capture, QueryResult)
    assert calls == ["run", "capture"]


def test_single_attempt_preserves_cancellation_evidence_without_swallowing_interrupt():
    from chia.base.llm_call import QueryResult, single_attempt

    captured = []

    def run():
        raise KeyboardInterrupt("fixture cancellation")

    with pytest.raises(KeyboardInterrupt, match="fixture cancellation"):
        single_attempt(run, lambda: None, captured.append, QueryResult)
    assert len(captured) == 1 and isinstance(captured[0].error, KeyboardInterrupt)


def test_codex_resumed_command_uses_supported_sandbox_configuration(tmp_path):
    model = codex(tmp_path, lambda *a, **k: None)
    command = model._build_cmd(resume_session_id=SID)
    assert "--sandbox" not in command
    assert 'sandbox_mode="workspace-write"' in command
    assert "--dangerously-bypass-approvals-and-sandbox" not in command


@pytest.mark.parametrize("kind", ["codex_cli", "claude_cli"])
@pytest.mark.parametrize("alias", ["file_symlink", "parent_symlink", "hardlink"])
def test_native_capture_cannot_follow_an_agent_planted_file_alias(tmp_path, kind, alias):
    import os

    from ramulator_chia.framework.config import ClaudeBackend, CodexBackend
    from ramulator_chia.framework.model_sessions import create_session

    private = tmp_path / "role"
    private.mkdir()
    canary = tmp_path / "canary"
    canary.write_bytes(b"outside session; fixture only")
    constructor = CodexBackend if kind == "codex_cli" else ClaudeBackend
    backend = constructor(kind=kind, model="fixture", reasoning_effort="xhigh")

    def run(command, **kw):
        if kind == "codex_cli":
            target = private / "sessions/2099" / ("rollout-fixture-" + SID + ".jsonl")
            Path(command[command.index("--output-last-message") + 1]).write_text("fixture")
            stdout = codex_stdout()
        else:
            sid = command[command.index("--session-id") + 1]
            target = Path(model.model._transcript_path())
            assert target.name == sid + ".jsonl"
            stdout = json.dumps({"type": "result", "result": "fixture"})
        if alias == "parent_symlink":
            if kind == "codex_cli":
                outside = tmp_path / "outside"
                (outside / "2099").mkdir(parents=True)
                (outside / "2099" / target.name).write_bytes(canary.read_bytes())
                (private / "sessions").symlink_to(outside, target_is_directory=True)
            else:
                # The root is itself agent-controlled in this adversarial fixture.
                outside = tmp_path / "outside"
                private.rename(outside)
                outside_target = outside / target.relative_to(private)
                outside_target.parent.mkdir(parents=True, exist_ok=True)
                outside_target.write_bytes(canary.read_bytes())
                private.symlink_to(outside, target_is_directory=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if alias == "hardlink":
                os.link(canary, target)
            else:
                target.symlink_to(canary)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    model = create_session(
        backend,
        session_id="fixture:proposer",
        private_directory=private,
        workspace=tmp_path,
        system_message="fixture",
        timeout_seconds=30,
        process_runner=run,
    )
    with pytest.raises((OSError, ValueError)):
        model.prompt_once("fixture")
    assert canary.read_bytes() == b"outside session; fixture only"


@pytest.mark.parametrize("kind", ["codex_cli", "claude_cli"])
def test_native_restore_uses_nofollow_writes_and_does_not_touch_a_canary(tmp_path, kind):
    from ramulator_chia.framework.config import ClaudeBackend, CodexBackend
    from ramulator_chia.framework.model_sessions import create_session

    private = tmp_path / "role"
    private.mkdir()
    canary = tmp_path / "canary"
    canary.write_bytes(b"preserve fixture canary")
    constructor = CodexBackend if kind == "codex_cli" else ClaudeBackend
    called = []
    session = create_session(
        constructor(kind=kind, model="fixture", reasoning_effort="xhigh"),
        session_id="fixture:proposer",
        private_directory=private,
        workspace=tmp_path,
        system_message="fixture",
        timeout_seconds=30,
        process_runner=lambda *args, **kw: called.append(1),
    )
    if kind == "codex_cli":
        session.model._session_state = {"sessions/rollout-fixture.jsonl": b"native state"}
        (private / "sessions").mkdir()
        (private / "sessions/rollout-fixture.jsonl").symlink_to(canary)
    else:
        session.model._session_transcript = b"native state"
        target = Path(session.model._transcript_path())
        target.parent.mkdir(parents=True)
        target.symlink_to(canary)
    with pytest.raises(ValueError, match="regular, unshared"):
        session.prompt_once("fixture")
    assert canary.read_bytes() == b"preserve fixture canary" and not called


def test_shared_binding_rejects_implicit_native_transports(tmp_path):
    from ramulator_chia.framework.config import ClaudeBackend, CodexBackend, VertexBackend
    from ramulator_chia.framework.model_sessions import create_session

    profiles = [
        CodexBackend(kind="codex_cli", model="gpt-6-astra", reasoning_effort="xhigh"),
        ClaudeBackend(kind="claude_cli", model="fixture-fable", reasoning_effort="xhigh"),
        VertexBackend(
            kind="vertex_gemini",
            model="gemini-3.8-flash",
            reasoning_effort="high",
            project="offline-project",
            location="global",
        ),
    ]
    for profile in profiles:
        with pytest.raises(ValueError, match="requires"):
            create_session(
                profile,
                session_id="fixture:iteration-1:proposer",
                private_directory=tmp_path,
                workspace=tmp_path,
                system_message="fixture contract",
                timeout_seconds=30,
            )


@pytest.mark.parametrize("kind", ["codex_cli", "claude_cli", "vertex_gemini"])
def test_common_settings_and_source_bound_native_checkpoint(tmp_path, monkeypatch, kind):
    from ramulator_chia.framework.config import ClaudeBackend, CodexBackend, VertexBackend
    from ramulator_chia.framework.model_sessions import NativeCheckpoint, create_session

    profile = {
        "codex_cli": CodexBackend(kind="codex_cli", model="gpt-6-astra", reasoning_effort="max"),
        "claude_cli": ClaudeBackend(
            kind="claude_cli", model="fixture-fable", reasoning_effort="xhigh"
        ),
        "vertex_gemini": VertexBackend(
            kind="vertex_gemini",
            model="gemini-3.8-flash",
            reasoning_effort="high",
            project="offline-project",
            location="global",
        ),
    }[kind]
    private = tmp_path / "private"
    private.mkdir()
    native_calls = []

    def run(cmd, **kw):
        native_calls.append((cmd, kw))
        if kind == "codex_cli":
            rollout(private, b'{"fixture":"native conversation"}\n')
            Path(cmd[cmd.index("--output-last-message") + 1]).write_text("done")
            raw = codex_stdout()
        else:
            sid = cmd[cmd.index("--session-id") + 1]
            target = Path(session.model._transcript_path())
            assert target.name == sid + ".jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'{"fixture":"native conversation"}\n')
            raw = json.dumps({"type": "result", "result": "done"})
        return SimpleNamespace(stdout=raw, stderr="", returncode=0)

    vertex = vertex_fixture(monkeypatch, [response()])

    def make(session_id="fixture:iteration-1:proposer", **overrides):
        kwargs = dict(
            session_id=session_id,
            private_directory=private,
            workspace=tmp_path,
            system_message="fixture contract",
            timeout_seconds=30,
            process_runner=run,
            vertex_client_kwargs={},
            vertex_event_callback=lambda event, state: None,
        )
        kwargs.update(overrides)
        return create_session(profile, **kwargs)

    session = make()
    assert session.settings["adapter_retry_attempts"] == 1
    assert session.model.retries == 1
    if kind == "codex_cli":
        assert session.model.auto_compact_token_limit is None
        assert "--ignore-user-config" in session.model._build_cmd()
        assert session.model.capture_all_sessions
    elif kind == "claude_cli":
        command = session.model._build_cmd()
        assert "--append-system-prompt" in command and "--system-prompt" not in command
        assert command[command.index("--setting-sources") + 1] == ""
        assert "--strict-mcp-config" in command
        import re

        assert Path(session.model._resolve_projects_dir()) == (
            private / "projects" / re.sub(r"[^a-zA-Z0-9]", "-", str(tmp_path))
        )
    else:
        http = session.model.client_kwargs["http_options"]
        assert http.timeout == 30000 and http.retry_options.attempts == 1
        assert session.model.max_tokens is None and session.model.max_tool_iterations is None

    result = session.prompt_once("explore", [])
    assert result.success
    checkpoint = session.checkpoint(result)
    if kind in {"codex_cli", "claude_cli"}:
        with pytest.raises(ValueError, match="phase authentication"):
            session.prompt_once("cannot send unauthenticated tools", [object()])
    if kind == "codex_cli":
        relocated = tmp_path / "other-private"
        relocated.mkdir()
        with pytest.raises(ValueError, match="different settings or session"):
            make(private_directory=relocated).restore(checkpoint)
    replacement = make()
    replacement.restore(checkpoint)
    with pytest.raises(ValueError, match="different settings or session"):
        make("fixture:iteration-1:reviewer").restore(checkpoint)
    with pytest.raises(ValueError, match="different settings or session"):
        make(timeout_seconds=40).restore(checkpoint)
    damaged = NativeCheckpoint(
        checkpoint.metadata, {name: value + b"changed" for name, value in checkpoint.files.items()}
    )
    with pytest.raises(ValueError, match="bytes changed"):
        replacement.restore(damaged)
    assert len(vertex.calls) + len(native_calls) == 1


def test_vertex_hidden_transport_retries_and_deadlines_are_rejected(tmp_path):
    from ramulator_chia.framework.config import VertexBackend
    from ramulator_chia.framework.model_sessions import create_session

    profile = VertexBackend(
        kind="vertex_gemini",
        model="gemini-3.8-flash",
        reasoning_effort="high",
        project="offline-project",
        location="global",
    )
    for options in (
        {"timeout": 50},
        {"retry_options": {"attempts": 3}},
        {"extra_body": {"generationConfig": {"thinkingConfig": {"thinkingLevel": "LOW"}}}},
    ):
        with pytest.raises(ValueError):
            create_session(
                profile,
                session_id="fixture",
                private_directory=tmp_path,
                workspace=tmp_path,
                system_message="fixture",
                timeout_seconds=30,
                vertex_event_callback=lambda *a: None,
                vertex_client_kwargs={"http_options": options},
            )


def test_actual_google_sdk_request_schema_native_content_and_no_hidden_retry(tmp_path, monkeypatch):
    import base64

    import httpx
    from google.auth.credentials import AnonymousCredentials
    from mcp.server.fastmcp import FastMCP

    from ramulator_chia.framework.config import VertexBackend
    from ramulator_chia.framework.diagnostic_cases import SyntheticRequest
    from ramulator_chia.framework.model_sessions import create_session

    def diagnostic(case: SyntheticRequest) -> dict:
        return {}

    server = FastMCP("schema")
    server.add_tool(diagnostic)
    schema = server._tool_manager.list_tools()[0].parameters
    fixture = vertex_fixture(monkeypatch, [], schema=schema, mock_sdk=False)
    requests, events = [], []
    signature = bytes(range(256))
    first = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "thoughtSignature": base64.b64encode(signature).decode(),
                            "functionCall": {
                                "name": "dram__inspect",
                                "id": "native-call-id",
                                "args": {"case": {"name": "fixture"}},
                            },
                        }
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 20,
            "cachedContentTokenCount": 70,
        },
    }

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=first)
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": 503,
                    "message": "offline fixture failure",
                    "status": "UNAVAILABLE",
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    # The SDK requires a nonempty token even for an in-memory HTTP transport.
    # This literal has no account, is never sent over a socket, and cannot refresh.
    fixture_credentials = AnonymousCredentials()
    fixture_credentials.token = "offline-fixture-not-a-valid-credential"
    model = create_session(
        VertexBackend(
            kind="vertex_gemini",
            model="gemini-3.8-flash",
            reasoning_effort="high",
            project="offline-project",
            location="global",
        ),
        session_id="fixture:iteration-1:proposer",
        private_directory=tmp_path,
        workspace=tmp_path,
        system_message="fixture contract",
        timeout_seconds=30,
        vertex_client_kwargs={
            "credentials": fixture_credentials,
            "http_options": {"httpx_client": client},
        },
        vertex_event_callback=lambda event, state: events.append(event),
    )
    result = model.prompt_once("explore", [fixture.tool])
    assert not result.success and len(requests) == 2  # exactly one attempt at the 503
    assert result.error.error_type == "server_error"
    # ProtoJSON permits both original and lowerCamelCase field names. The SDK
    # emits original names inside this nested message; do not add a converter.
    thinking = types.ThinkingConfig.model_validate(
        requests[0]["generationConfig"]["thinkingConfig"]
    )
    assert thinking.include_thoughts is True and thinking.thinking_level == "HIGH"
    assert "maxOutputTokens" not in requests[0]["generationConfig"]
    declaration = types.FunctionDeclaration.model_validate(
        requests[0]["tools"][0]["functionDeclarations"][0]
    )
    assert declaration.parameters_json_schema == schema and declaration.parameters is None
    expected = first["candidates"][0]["content"]
    # The SDK may choose URL-safe base64 when re-encoding a bytes field. Compare
    # the decoded native content, including every opaque signature byte.
    resumed_content = types.Content.model_validate(requests[1]["contents"][1])
    assert resumed_content == types.Content.model_validate(expected)
    assert resumed_content.parts[0].thought_signature == signature
    assert requests[1]["contents"][-1]["parts"][0]["functionResponse"]["id"] == "native-call-id"
    assert len(fixture.tools) == 1 and len([e for e in events if e["kind"] == "request"]) == 2
    model.checkpoint(result).verify()


def test_native_binding_rejects_unreviewed_adapter_source(tmp_path, monkeypatch):
    from chia.models import codex as upstream

    from ramulator_chia.framework.model_sessions import upstream_identity

    assert upstream_identity()["revision"] == "16c35e92aaaf9511c6453bf94cd5cf589698f4e3"
    unrelated = tmp_path / "unreviewed.py"
    unrelated.write_text("# different adapter code\n")
    monkeypatch.setattr(upstream, "__file__", str(unrelated))
    with pytest.raises(RuntimeError, match="differs from the reviewed"):
        upstream_identity()
