"""The same campaign through the actual CHIA adapters, with inert transports.

No model subprocess or network connection is allowed. Numerical measurements
are protocol fixtures, not DRAM accuracy claims. The native adapters still own
their command parsing and continuation formats.
"""

import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from google import genai
from test_chia_framework_campaign import FixtureResearch, configuration
from test_chia_framework_campaign import no_services as no_services
from test_chia_framework_model_adapters import codex_stdout, response, rollout

from ramulator_chia.framework import agent as agent_module
from ramulator_chia.framework.agent import TOOL_TOKEN_ENV, NativeAgent, NativeSessions, cli_usage
from ramulator_chia.framework.build import MODEL_API_HEADER
from ramulator_chia.framework.campaign import Campaign
from ramulator_chia.framework.config import CampaignConfig
from ramulator_chia.framework.model_sessions import create_session
from ramulator_chia.framework.snapshots import publish_bytes

pytestmark = pytest.mark.usefixtures("no_services")


@pytest.mark.parametrize("kind", ["codex_cli", "claude_cli", "vertex_gemini", "deepseek_api", "vertex_claude"])
def test_same_campaign_native_continuity_and_summary_memory(tmp_path, monkeypatch, kind):
    config = configuration().model_dump(mode="json")
    config["run"]["model_timeout_seconds"] = 123
    config["backend"] = {"kind": kind, "model": "fixture", "reasoning_effort": "high"}
    if kind == "vertex_gemini":
        config["backend"].update(project="offline", location="global")
    if kind == "deepseek_api":
        config["backend"]["model"] = "deepseek-v4-flash"
    if kind == "vertex_claude":
        config["backend"]["model"] = "claude-fable-5-1"
        config["backend"]["project"] = "fixture-project"
    research = FixtureResearch(tmp_path, CampaignConfig.model_validate(config))
    research.resources = research.configuration.run.resources
    research.runtime = tmp_path / "runtime"
    publish_bytes(research.runtime / "export" / MODEL_API_HEADER, b"// fixture API\n")
    sessions, calls, native_models, tokens = [], [], {}, []

    @contextmanager
    def transport(view, private, tool_environment):
        iteration = view.iteration
        assert set(tool_environment) == {TOOL_TOKEN_ENV}
        tokens.append(tool_environment[TOOL_TOKEN_ENV])
        role_calls = []
        calls.append(role_calls)

        def perform_fixture_actions():
            methods = view.methods()
            role_calls.append(view.phase)
            if view.phase == "explore":
                methods["write"]("draft/parameters.json", json.dumps({"error": 20 // iteration}))
                if iteration == 2:
                    assert (
                        "fixture summary"
                        in methods["read"]("references/history/1/summary.md")["text"]
                    )
            elif view.phase == "review":
                assert set(methods) == {"files", "read"}
                assert "references/history/1/summary.md" not in methods["files"]()["files"]
            elif view.phase == "reflect":
                assert set(methods) == {"files", "read", "write"}
                methods["write"]("notes/summary.md", "Native-adapter fixture summary.\n")

        def run(command, **kwargs):
            native = native_models[private]
            perform_fixture_actions()
            if kind == "codex_cli":
                if len(role_calls) > 1:
                    assert "resume" in command
                    assert native.model._session_state
                rollout(private, ("opaque reasoning " + str(role_calls)).encode())
                Path(command[command.index("--output-last-message") + 1]).write_text("done")
                stdout = codex_stdout()
            else:
                if len(role_calls) > 1:
                    assert "--resume" in command and native.model._session_transcript
                target = Path(native.model._transcript_path())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("opaque native state " + str(role_calls))
                stdout = json.dumps(
                    {
                        "type": "result",
                        "result": "done",
                        "usage": {
                            "input_tokens": 10,
                            "cache_read_input_tokens": 20,
                            "cache_creation_input_tokens": 0,
                            "output_tokens": 5,
                        },
                    }
                )
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

        def generate_content(**kwargs):
            perform_fixture_actions()
            assert len(kwargs["contents"]) == 2 * len(role_calls) - 1
            return response()

        chat_client = None
        if kind == "vertex_claude":
            import anthropic
            from contextlib import asynccontextmanager
            from anthropic.types.beta import BetaMessage

            @asynccontextmanager
            async def stream(**kwargs):
                perform_fixture_actions()
                assert len(kwargs["messages"]) == 2 * len(role_calls) - 1

                async def final():
                    return BetaMessage.model_validate(dict(
                        id="msg_fixture", type="message", model="claude-fable-5-1", role="assistant",
                        content=[dict(type="text", text="done")], stop_reason="end_turn",
                        usage=dict(input_tokens=10, output_tokens=5, cache_read_input_tokens=20,
                                   cache_creation_input_tokens=0),
                    ))

                yield SimpleNamespace(get_final_message=final)

            monkeypatch.setattr(anthropic, "AsyncAnthropicVertex", lambda **_: SimpleNamespace(
                beta=SimpleNamespace(messages=SimpleNamespace(stream=stream))))
        if kind == "deepseek_api":
            from chia.models.openai_compat import OpenAICompatLLM
            from openai.types.chat import ChatCompletion

            async def complete(**kwargs):
                perform_fixture_actions()
                assert len(kwargs["messages"]) == 2 * len(role_calls)
                return ChatCompletion.model_validate(
                    {
                        "id": "fixture",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "deepseek-v4-flash",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "done",
                                    "reasoning_content": "fixture provider reasoning",
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 30,
                            "total_tokens": 130,
                            "prompt_cache_hit_tokens": 70,
                        },
                    }
                )

            chat_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
            )
            monkeypatch.setattr(
                OpenAICompatLLM, "_make_client", lambda self: self.client_kwargs["http_client"]
            )

        monkeypatch.setattr(
            genai,
            "Client",
            lambda **kw: SimpleNamespace(
                models=SimpleNamespace(generate_content=kw["fixture_generate"]), close=lambda: None
            ),
        )
        yield {
            "process_runner": run,
            "vertex_client_kwargs": {"fixture_generate": generate_content},
            "chat_client_kwargs": {"api_key": "fixture-not-a-key", "http_client": chat_client},
            "anthropic_client_kwargs": {"credentials": object()},
        }

    @contextmanager
    def offered_tools(workspace, token):
        # Actual MCP schema/auth/dispatch is qualified separately. This fixture
        # exercises the common factory, native sessions and campaign contract.
        assert token in tokens
        yield []

    monkeypatch.setattr(agent_module, "phase_tools", offered_tools)
    factory = NativeSessions(transport)

    @contextmanager
    def session(iteration, role, candidate, history):
        with factory(research, iteration, role, candidate, history) as agent:
            private = tmp_path / "private" / str(iteration) / role
            native_models[private] = agent.model
            assert agent.model.settings["timeout_seconds"] == config["run"]["model_timeout_seconds"]
            sessions.append(agent)
            yield agent

    research.session = session
    campaign = Campaign(tmp_path, research.configuration, research)
    selected = campaign.run_search()
    assert selected["candidate"]["parameters"]["error"] == 10
    assert calls == [["explore", "revise", "reflect"], ["review"]] * 2
    assert len(set(tokens)) == 4  # Each iteration/role gets independent authentication.
    assert campaign.evaluate()["execution"] == "offline_fixture"
    assert len(list(tmp_path.glob("native-evidence/*/*/*/receipt.json"))) == 8
    assert len(list(tmp_path.glob("native-evidence/*/*/*/checkpoint.json"))) == 8
    for path in tmp_path.glob("native-evidence/*/*/*/result.json.gz"):
        assert json.loads(gzip.decompress(path.read_bytes()))["success"]
    if kind in {"vertex_gemini", "deepseek_api", "vertex_claude"}:
        assert len(list(tmp_path.glob("native-evidence/*/*/*/sdk/*.gz"))) == 24
    count = len(sessions)
    assert campaign.run_search() == selected and len(sessions) == count


def test_saved_response_survives_cleanup_failure_and_reflection_gets_late_diagnostic(tmp_path):
    from test_chia_framework_workspace import workspace
    from ramulator_chia.framework.config import CodexBackend
    from ramulator_chia.framework.identity import digest_json
    from ramulator_chia.framework.records import CampaignState

    view = workspace(tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    calls = []
    failure = {"reason": "runtime_limit", "diagnostic": "open_loop", "training_case": "train"}

    def run(command, **kwargs):
        calls.append(command)
        rollout(private, b"opaque saved native conversation")
        Path(command[command.index("--output-last-message") + 1]).write_text("done")
        return SimpleNamespace(stdout=codex_stdout(), stderr="", returncode=0)

    @contextmanager
    def tools(workspace):
        yield []
        if workspace.phase == "explore":
            workspace.record({"tool": "open_loop", "diagnostic_failure": failure})
            raise RuntimeError("fixture cleanup interruption after saved response")

    def make_agent():
        return NativeAgent(view, create_session(
            CodexBackend(kind="codex_cli", model="fixture", reasoning_effort="high"),
            session_id="fixture:recovery", private_directory=private, workspace=view.root,
            system_message="fixture", timeout_seconds=30, process_runner=run,
        ), tools)

    agent = make_agent()
    state = CampaignState(tmp_path / "campaign.sqlite")
    inputs = {"round": 1}
    with pytest.raises(RuntimeError, match="cleanup interruption"):
        state.step("iteration:1:explore", inputs, lambda: agent.turn("explore", inputs),
                   maximum_attempts=3, retry_delay_seconds=0)
    saved = json.loads(next(agent.root.glob("explore-*/receipt.json")).read_text())
    assert saved["success"] and len(calls) == 1
    state.reconcile_saved_result("iteration:1:explore", digest_json(inputs), saved, {"fixture": True})
    restored = make_agent()
    assert state.step("iteration:1:explore", inputs, lambda: pytest.fail("repeated inference"),
                      maximum_attempts=3, retry_delay_seconds=0) == saved
    assert restored.turn("reflect", {"frozen": True})["success"]
    assert len(calls) == 2 and "resume" in calls[1]
    prompt = next(agent.root.glob("reflect-*/prompt.txt")).read_text()
    assert '"diagnostic_failures"' in prompt and '"runtime_limit"' in prompt
    assert len(list(agent.root.glob("explore-*/receipt.json"))) == 1


def test_native_factory_has_no_unrestricted_transport_fallback():
    with pytest.raises(ValueError, match="explicit scoped transport"):
        with NativeSessions(None)(None, 1, "proposer", {}, []):
            pytest.fail("an unscoped native session was constructed")


def test_reusing_a_campaign_name_does_not_reuse_its_native_identity(tmp_path, monkeypatch):
    from test_chia_framework_workspace import workspace

    views = [workspace(tmp_path / name) for name in ("first", "second")]
    assert (
        views[0].research.configuration.campaign_id == views[1].research.configuration.campaign_id
    )
    identities = []

    def create(backend, **options):
        identities.append(options["session_id"])
        return SimpleNamespace(backend=backend, settings={})

    @contextmanager
    def transport(view, private, environment):
        yield {}

    monkeypatch.setattr(agent_module, "create_session", create)
    factory = NativeSessions(transport)
    for view in (views[0], views[0], views[1]):
        with factory(view.research, 1, "proposer", view.snapshot(), []):
            pass
    assert identities[0] == identities[1]
    assert identities[0] != identities[2]


def test_cli_accounting_uses_terminal_totals_without_double_counting():
    rows = cli_usage(
        json.dumps(
            {
                "type": "result",
                "usage": {"input_tokens": 999},
                "modelUsage": {
                    "fixture": {
                        "inputTokens": 20,
                        "cacheReadInputTokens": 30,
                        "cacheCreationInputTokens": 0,
                        "outputTokens": 5,
                    }
                },
            }
        ),
        "claude_cli",
        None,
    )
    assert len(rows) == 1 and rows[0]["usage"]["tokens"]["input_tokens"] == 50
    assert rows[0]["scope"] == "native_cli_turn_including_subagents"
    assert rows[0]["cost"]["upper_micro_usd"] is None
    unknown = cli_usage("lost terminal output", "codex_cli", None)
    assert unknown[0]["usage"] is None


def test_claude_resume_uses_turn_usage_not_cumulative_model_usage():
    event = dict(type="result", usage=dict(input_tokens=2, cache_read_input_tokens=28006,
        cache_creation_input_tokens=97, output_tokens=11), modelUsage={
        "claude-opus-5-5":dict(inputTokens=6, cacheReadInputTokens=55889,
                              cacheCreationInputTokens=28103, outputTokens=99)})
    rows = cli_usage(json.dumps(event), "claude_cli", None)
    assert len(rows) == 1
    assert rows[0]["usage"]["tokens"]["input_tokens"] == 28105
    assert rows[0]["usage"]["tokens"]["output_tokens"] == 11
    assert rows[0]["scope"] == "main_agent_cli_turn"
    assert rows[0]["cumulative_model_usage_not_added"] == event["modelUsage"]


@pytest.mark.parametrize("failure", ["server", "capacity"])
def test_retry_retains_failed_native_state_and_accounting(tmp_path, failure):
    from test_chia_framework_workspace import workspace

    from ramulator_chia.framework.config import CodexBackend
    from ramulator_chia.framework.records import TransientFailure

    view = workspace(tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        rollout(private, b"opaque failed-then-resumed conversation")
        Path(command[command.index("--output-last-message") + 1]).write_text("partial then done")
        stdout = codex_stdout(failed=len(commands) == 1)
        if failure == "capacity" and len(commands) == 1:
            stdout += "\n" + json.dumps(
                {
                    "type": "turn.failed",
                    "error": {
                        "message": "Selected model is at capacity. Please try a different model."
                    },
                }
            )
        return SimpleNamespace(
            stdout=stdout,
            stderr="503 service unavailable" if len(commands) == 1 and failure == "server" else "",
            returncode=1 if len(commands) == 1 else 0,
        )

    @contextmanager
    def tools(view):
        yield []

    model = create_session(
        CodexBackend(kind="codex_cli", model="fixture", reasoning_effort="high"),
        session_id="fixture:retry",
        private_directory=private,
        workspace=view.root,
        system_message="fixture",
        timeout_seconds=30,
        process_runner=run,
    )
    agent = NativeAgent(view, model, tools)
    with pytest.raises(TransientFailure) as error:
        agent.turn("explore", {"fixture": True})
    assert error.value.retry_after_seconds == (3600 if failure == "capacity" else None)
    assert agent.turn("explore", {"fixture": True})["success"]
    assert len(commands) == 2 and "resume" in commands[1]
    receipts = [json.loads(p.read_text()) for p in agent.root.glob("explore-*/receipt.json")]
    assert sorted(p["success"] for p in receipts) == [False, True]
    assert all(p["usage"][0]["usage"]["tokens"]["input_tokens"] == 100 for p in receipts)


def test_claude_subscription_exhaustion_preserves_state_without_retry(tmp_path):
    from test_chia_framework_workspace import workspace
    from ramulator_chia.framework.config import ClaudeBackend
    from ramulator_chia.framework.records import TransientFailure

    view = workspace(tmp_path)
    private = tmp_path / "private"
    private.mkdir()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        transcript = Path(model.model._transcript_path())
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text("preserved interrupted subscription conversation")
        return SimpleNamespace(stdout=json.dumps(dict(
            type="result", is_error=True,
            result="You've hit your limit · resets 4pm (America/Los_Angeles)",
            usage=dict(input_tokens=0, output_tokens=0, cache_read_input_tokens=0,
                       cache_creation_input_tokens=0))), stderr="", returncode=0)

    @contextmanager
    def tools(view):
        yield []

    model = create_session(
        ClaudeBackend(kind="claude_cli", model="fixture", reasoning_effort="high"),
        session_id="fixture:subscription-limit", private_directory=private,
        workspace=view.root, system_message="fixture", timeout_seconds=30, process_runner=run,
    )
    agent = NativeAgent(view, model, tools)
    with pytest.raises(RuntimeError, match="subscription quota exhausted") as error:
        agent.turn("explore", {"fixture": True})
    assert not isinstance(error.value, TransientFailure)
    assert len(commands) == 1
    assert len(list(agent.root.glob("explore-*/checkpoint.json"))) == 1
    receipts = list(agent.root.glob("explore-*/receipt.json"))
    assert len(receipts) == 1 and not json.loads(receipts[0].read_text())["success"]


@pytest.mark.parametrize(
    "event",
    [
        {"type": "item.completed", "item": {"text": "Selected model is at capacity."}},
        {"type": "turn.completed", "error": {"message": "Selected model is at capacity."}},
        {"type": "turn.failed", "error": {"message": "stream disconnected before completion"}},
        {"type": "turn.failed", "error": "Selected model is at capacity."},
        None,
    ],
)
def test_capacity_classification_does_not_match_model_text_or_other_failures(event):
    assert not agent_module.codex_capacity_failure(json.dumps(event))


def test_missing_sdk_reply_is_unknown_usage_not_zero(tmp_path):
    from test_chia_framework_workspace import workspace

    from ramulator_chia.framework.config import VertexBackend

    view = workspace(tmp_path)
    model = create_session(
        VertexBackend(
            kind="vertex_gemini",
            model="fixture",
            reasoning_effort="high",
            project="offline",
            location="global",
        ),
        session_id="fixture:missing-reply",
        private_directory=tmp_path,
        workspace=view.root,
        system_message="fixture",
        timeout_seconds=30,
        vertex_client_kwargs={},
        vertex_event_callback=lambda event, state: None,
    )
    agent = NativeAgent(view, model, None)
    agent.active_attempt = agent.root / "fixture-attempt"
    agent.active_attempt.mkdir()
    agent.sdk_event({"kind": "request", "payload": {"fixture": True}}, None)
    agent.sdk_event({"kind": "request_error", "payload": {"message": "lost reply"}}, None)
    assert len(agent.sdk_usage) == 1
    assert agent.sdk_usage[0]["status"] == "failed_usage_unknown"
    assert agent.sdk_usage[0]["usage"]["tokens"] is None
    assert agent.sdk_usage[0]["cost"]["upper_micro_usd"] is None


def test_compaction_calls_share_accounting_but_keep_their_purpose(tmp_path):
    from test_chia_framework_workspace import workspace

    from ramulator_chia.framework.config import VertexBackend

    view = workspace(tmp_path)
    model = create_session(
        VertexBackend(kind="vertex_gemini", model="fixture", reasoning_effort="high",
                      project="offline", location="global"),
        session_id="fixture:compaction-usage", private_directory=tmp_path,
        workspace=view.root, system_message="fixture", timeout_seconds=30,
        vertex_client_kwargs={}, vertex_event_callback=lambda event, state: None,
    )
    agent = NativeAgent(view, model, None)
    agent.active_attempt = agent.root / "compaction-fixture"
    agent.active_attempt.mkdir()
    agent.sdk_event({"kind": "context_count", "payload": {"tokens": 100}}, None)
    agent.sdk_event({"kind": "request", "payload": {"purpose": "context_compaction"}}, None)
    agent.sdk_event({"kind": "response", "payload": response().model_dump(
        mode="json", exclude_none=True)}, None)
    assert len(agent.sdk_usage) == 1
    assert agent.sdk_usage[0]["purpose"] == "context_compaction"
    assert agent.sdk_usage[0]["status"] == "response_recorded"
    assert agent.sdk_usage[0]["usage"]["tokens"] is not None
