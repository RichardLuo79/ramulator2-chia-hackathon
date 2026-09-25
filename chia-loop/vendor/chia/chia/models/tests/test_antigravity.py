"""Offline tests for :class:`chia.models.antigravity.AntigravityLLM`.

Set ``ANTIGRAVITY_LIVE_TEST=1`` to run the opt-in live smoke test against an
authenticated local ``agy`` CLI.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import timezone
from types import SimpleNamespace

import pytest

from chia.models import antigravity as agy_mod
from chia.models.antigravity import (
    AntigravityLLM,
    AntigravityQueryResult,
    AuthenticationError,
    BillingError,
    InvalidRequestError,
    MaxOutputTokensError,
    QueryResult,
    RateLimitError,
    ServerError,
    UnknownAntigravityError,
    parse_rate_limit_reset,
)


def _cli(returncode=1, stderr="", result="", stream_result=""):
    return QueryResult(result, returncode, stderr, stream_result)


def _fake_subprocess(monkeypatch, capture, *, stdout="PONG", stderr="", returncode=0):
    def fake_run(cmd, **kwargs):
        capture.update(cmd=cmd, kwargs=kwargs)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(agy_mod.subprocess, "run", fake_run)


def test_constructor_and_chia_surface(caplog):
    with caplog.at_level("INFO", logger="antigravity"):
        llm = AntigravityLLM()
    assert llm.model is None
    assert llm.agy_bin == "agy"
    assert "experimental" in caplog.text
    assert "default model" in caplog.text
    assert hasattr(AntigravityLLM.prompt, "chia_remote")
    assert AntigravityLLM.prompt._chia_options["resources"] == {"antigravity_creds": 0.01}


def test_prompt_formatting():
    assert AntigravityLLM()._format_prompt("hi") == "hi"
    formatted = AntigravityLLM(system_message="be terse")._format_prompt("say pong")
    assert "[System Instructions]" in formatted
    assert "be terse" in formatted
    assert "[User Request]" in formatted


def test_build_cmd_flags():
    llm = AntigravityLLM(
        model="gemini-2.5-pro",
        add_dirs=["/tmp/work"],
        timeout_seconds=120,
    )
    cmd = llm._build_cmd("say pong")
    assert cmd[0] == "agy"
    assert "--dangerously-skip-permissions" in cmd
    assert cmd[cmd.index("--model") + 1] == "gemini-2.5-pro"
    assert cmd[cmd.index("--add-dir") + 1] == "/tmp/work"
    assert cmd[cmd.index("--print-timeout") + 1] == "120s"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--conversation" not in cmd
    # The prompt is the value of --print and comes last.
    assert cmd[-2] == "--print"
    assert cmd[-1] == "say pong"


def test_build_cmd_safe_defaults_omit_optional_flags():
    cmd = AntigravityLLM(dangerously_skip_permissions=False)._build_cmd("hi")
    assert "--dangerously-skip-permissions" not in cmd
    assert "--sandbox" not in cmd
    assert "--model" not in cmd


@pytest.mark.parametrize(
    "tool_name, expected_url",
    [("calc", "http://localhost:9001/calc/mcp"), ("bash_42", "http://10.0.0.5:8000/bash_42/mcp")],
)
def test_run_home_has_only_this_calls_tools(tmp_path, tool_name, expected_url):
    llm = AntigravityLLM(gemini_dir=str(tmp_path))
    host, port = expected_url.split("//")[1].split("/")[0].split(":")
    tool = SimpleNamespace(name=tool_name, hostname=host, port=int(port))
    home = llm._prepare_run_home([tool])
    try:
        with open(os.path.join(home, ".gemini", "config", "mcp_config.json")) as f:
            servers = json.load(f)["mcpServers"]
        assert list(servers) == [tool_name]
        assert servers[tool_name]["serverUrl"] == expected_url
        # Shared state is reached through the symlink; the user's config is untouched.
        link = os.path.join(home, ".gemini", "antigravity-cli")
        assert os.path.islink(link) and os.path.realpath(link) == os.path.realpath(tmp_path / "antigravity-cli")
        assert not os.path.exists(llm._mcp_config_path)
    finally:
        shutil.rmtree(home)


def test_run_home_isolates_concurrent_tool_sets_and_copies_user_config(tmp_path):
    llm = AntigravityLLM(gemini_dir=str(tmp_path))
    os.makedirs(tmp_path / "config")
    (tmp_path / "config" / "config.json").write_text('{"userSettings": {"x": 1}}')
    (tmp_path / "config" / "mcp_config.json").write_text('{"mcpServers": {"users_own": {}}}')
    a = llm._prepare_run_home([SimpleNamespace(name="bash_1", hostname="h", port=8000)])
    b = llm._prepare_run_home([SimpleNamespace(name="chipyard_bash", hostname="h2", port=8000)])
    try:
        for home, expected in ((a, ["bash_1"]), (b, ["chipyard_bash"])):
            with open(os.path.join(home, ".gemini", "config", "mcp_config.json")) as f:
                assert list(json.load(f)["mcpServers"]) == expected
            with open(os.path.join(home, ".gemini", "config", "config.json")) as f:
                assert json.load(f) == {"userSettings": {"x": 1}}
        # No-tools run: empty server list, nothing inherited from the user's file.
        c = llm._prepare_run_home([])
        with open(os.path.join(c, ".gemini", "config", "mcp_config.json")) as f:
            assert json.load(f) == {"mcpServers": {}}
        shutil.rmtree(c)
        assert json.loads((tmp_path / "config" / "mcp_config.json").read_text()) == {"mcpServers": {"users_own": {}}}
    finally:
        shutil.rmtree(a); shutil.rmtree(b)


def test_run_antigravity_uses_private_home_and_cleans_up(monkeypatch, tmp_path):
    capture = {}
    _fake_subprocess(monkeypatch, capture, stdout="ok")
    tool = SimpleNamespace(name="calc", hostname="localhost", port=9001)
    AntigravityLLM(gemini_dir=str(tmp_path))._run_antigravity("use calc", tools=[tool])
    home = capture["kwargs"]["env"]["HOME"]
    assert os.path.basename(home).startswith("agy_home_")
    assert not os.path.exists(home)                      # removed after the run
    assert not os.path.exists(tmp_path / "config" / "mcp_config.json")   # user's file untouched


def test_gemini_dir_resolves_home_lazily(monkeypatch, tmp_path):
    # Built under one HOME (e.g. a circt worker, HOME=/root), run under another
    # (the antigravity_creds node): the default must follow the RUNTIME home.
    monkeypatch.setenv("HOME", "/nonexistent/build-home")
    llm = AntigravityLLM()
    monkeypatch.setenv("HOME", str(tmp_path))
    assert llm.gemini_dir == str(tmp_path / ".gemini")
    assert llm._mcp_config_path == str(tmp_path / ".gemini" / "config" / "mcp_config.json")
    # An explicit gemini_dir is still honored verbatim.
    assert AntigravityLLM(gemini_dir="/x").gemini_dir == "/x"


def test_parse_rate_limit_reset():
    reset = parse_rate_limit_reset("usage limit - resets 4pm (America/Los_Angeles)")
    assert reset is not None
    assert reset.tzinfo == timezone.utc


# --------------------------------------------------------------------------- #
# stream-json parsing + sessions (resume_session=True)
# --------------------------------------------------------------------------- #

# Captured from `agy --output-format stream-json` (agy 1.0.x): one tool call
# then a text answer. Abbreviated but structurally faithful.
_CONV = "17ebbc36-25ba-4dc4-8996-a0320ed18c49"
_STREAM_JSON = "\n".join([
    json.dumps({"event": "init", "conversation_id": _CONV,
                "init": {"model": "gemini-3.7-flash-low", "cwd": "/home/ubuntu", "tools": ["run_command"]}}),
    json.dumps({"event": "step_update", "step_update": {"conversation_id": _CONV, "step_index": 0,
                "state": "DONE", "step_type": "user_input"}}),
    json.dumps({"event": "step_update", "step_update": {"conversation_id": _CONV, "step_index": 1,
                "state": "DONE", "step_type": "agent_response", "usage": {"total_tokens": 12752}}}),
    json.dumps({"event": "step_update", "step_update": {"conversation_id": _CONV, "step_index": 2,
                "state": "ACTIVE", "step_type": "tool", "tool_name": "run_command",
                "tool_info": {"name": "run_command", "parameters": {"CommandLine": "echo hello-from-tool"}}}}),
    json.dumps({"event": "step_update", "step_update": {"conversation_id": _CONV, "step_index": 2,
                "state": "DONE", "step_type": "tool", "tool_name": "run_command",
                "tool_info": {"name": "run_command", "parameters": {"CommandLine": "echo hello-from-tool"},
                              "output": "hello-from-tool\r\n"}}}),
    json.dumps({"event": "step_update", "step_update": {"conversation_id": _CONV, "step_index": 3,
                "state": "ACTIVE", "step_type": "agent_response", "text_delta": "PONG"}}),
    json.dumps({"event": "step_update", "step_update": {"conversation_id": _CONV, "step_index": 3,
                "state": "DONE", "step_type": "agent_response", "text_delta": "\n",
                "usage": {"total_tokens": 1922}}}),
    json.dumps({"event": "result", "result": {"conversation_id": _CONV, "status": "SUCCESS",
                "response": "PONG\n", "num_turns": 1, "duration_seconds": 5.6,
                "usage": {"input_tokens": 14262, "output_tokens": 412, "total_tokens": 14674}}}),
])


def test_stream_json_parsed_into_result_and_transcript(monkeypatch, tmp_path):
    capture = {}
    _fake_subprocess(monkeypatch, capture, stdout=_STREAM_JSON)
    cli = AntigravityLLM(gemini_dir=str(tmp_path))._run_antigravity("go", tools=[])
    assert isinstance(cli, AntigravityQueryResult)
    assert cli.result == "PONG"
    assert cli.conversation_id == _CONV
    assert cli.usage["total_tokens"] == 14674
    assert len(cli.events) == 8
    # Readable turn-by-turn transcript, Claude-style section headers.
    assert "[Init]\nmodel=gemini-3.7-flash-low" in cli.stream_result
    assert '[Tool Call: run_command]\nArgs: {"CommandLine": "echo hello-from-tool"}' in cli.stream_result
    assert "[Tool Result]\nhello-from-tool" in cli.stream_result
    assert "[Response]\nPONG" in cli.stream_result
    assert "[Result]\nstatus=SUCCESS" in cli.stream_result
    assert cli.returncode == 0
    # Not resuming -> nothing carried.
    assert cli.session_transcript is None


def test_successful_run_is_not_classified_by_its_own_content(monkeypatch, tmp_path):
    # Regression: a debug turn whose tool output / answer mention "429",
    # "rate limit", "timeout", "login" etc. must NOT be classified as an error.
    conv = "aaaaaaaa-0000-0000-0000-000000000000"
    noisy = "\n".join([
        json.dumps({"event": "init", "conversation_id": conv, "init": {"model": "m", "cwd": "/"}}),
        json.dumps({"event": "step_update", "step_update": {"conversation_id": conv, "step_index": 1,
                    "state": "DONE", "step_type": "tool", "tool_name": "chipyard_bash_run",
                    "tool_info": {"name": "chipyard_bash_run", "parameters": {"cmd": "make"},
                                  "output": "sim: 429 cycles; rate limit reached on mem port; "
                                            "timeout after 1000; please sign in to view"}}}),
        json.dumps({"event": "step_update", "step_update": {"conversation_id": conv, "step_index": 2,
                    "state": "DONE", "step_type": "agent_response",
                    "text_delta": "Fixed the RESOURCE_EXHAUSTED handling; too many requests were queued."}}),
        json.dumps({"event": "result", "result": {"conversation_id": conv, "status": "SUCCESS",
                    "response": "Fixed the RESOURCE_EXHAUSTED handling; too many requests were queued.",
                    "num_turns": 1, "usage": {"total_tokens": 1}}}),
    ])
    _fake_subprocess(monkeypatch, {}, stdout=noisy, returncode=0)
    llm = AntigravityLLM(gemini_dir=str(tmp_path))
    cli = llm._run_antigravity("debug it", tools=[])
    assert "429" in cli.stream_result and "rate limit" in cli.stream_result   # transcript intact
    assert cli.diagnostics == ""                                              # nothing diagnostic
    llm._classify_error(cli)                                                  # must not raise
    # ...while the same words on stderr still classify.
    cli.diagnostics = "Error: 429 Too Many Requests"
    with pytest.raises(RateLimitError):
        llm._classify_error(cli)


def test_stream_json_error_result_is_classified(monkeypatch, tmp_path):
    # agy exits 0 but reports status=ERROR in the result event.
    bad = json.dumps({"event": "result", "result": {
        "conversation_id": "", "status": "ERROR", "response": "",
        "error": 'invalid model selection (--model "gemini-nope"): model gemini-nope is not recognized'}})
    _fake_subprocess(monkeypatch, {}, stdout=bad, returncode=0)
    llm = AntigravityLLM(gemini_dir=str(tmp_path))
    cli = llm._run_antigravity("hi", tools=[])
    assert cli.returncode == 1 and "invalid model selection" in cli.stderr
    assert "invalid model selection" in cli.diagnostics
    with pytest.raises(InvalidRequestError):
        llm._classify_error(cli)


def test_resume_session_adds_conversation_flag_only_once_known(tmp_path):
    llm = AntigravityLLM(gemini_dir=str(tmp_path), resume_session=True)
    assert llm._session_id is not None
    assert "--conversation" not in llm._build_cmd("first")      # id not known yet
    llm._conversation_id = _CONV
    cmd = llm._build_cmd("second")
    assert cmd[cmd.index("--conversation") + 1] == _CONV


def _make_conversation_db(path, marker: bytes):
    import sqlite3
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE steps (id INTEGER PRIMARY KEY, blob BLOB)")
    con.execute("INSERT INTO steps (blob) VALUES (?)", (marker,))
    con.commit()
    con.close()


def test_capture_and_restore_transcript_round_trip(tmp_path):
    # Worker A: agy wrote conversations/<id>.db (+ WAL); capture checkpoints
    # it into one byte string on the result.
    a = AntigravityLLM(gemini_dir=str(tmp_path / "A"), resume_session=True)
    conv_dir = tmp_path / "A" / "antigravity-cli" / "conversations"
    conv_dir.mkdir(parents=True)
    _make_conversation_db(str(conv_dir / f"{_CONV}.db"), b"MARKER-BYTES")
    cli = AntigravityQueryResult(result="OK", returncode=0, stderr="", stream_result="",
                                 conversation_id=_CONV)
    a._capture_transcript(cli)
    assert cli.session_transcript and b"MARKER-BYTES" in cli.session_transcript
    assert cli.session_transcript_path == str(conv_dir / f"{_CONV}.db")
    assert not (conv_dir / f"{_CONV}.db-wal").exists() or \
        (conv_dir / f"{_CONV}.db-wal").stat().st_size == 0   # checkpointed

    # Driver: get() runs _sync_transcript on the (never-run) instance.
    driver = AntigravityLLM(gemini_dir=str(tmp_path / "B"), resume_session=True)
    driver._sync_transcript(cli)
    assert driver._conversation_id == _CONV and driver._call_counter == 1
    assert driver._session_transcript == cli.session_transcript

    # Worker B: restore pastes the bytes where agy looks, dropping stale side files.
    conv_dir_b = tmp_path / "B" / "antigravity-cli" / "conversations"
    conv_dir_b.mkdir(parents=True)
    (conv_dir_b / f"{_CONV}.db-wal").write_bytes(b"stale")
    driver._restore_transcript()
    assert (conv_dir_b / f"{_CONV}.db").read_bytes() == cli.session_transcript
    assert not (conv_dir_b / f"{_CONV}.db-wal").exists()
    cmd = driver._build_cmd("again")
    assert cmd[cmd.index("--conversation") + 1] == _CONV


def test_sync_transcript_noop_without_resume():
    llm = AntigravityLLM()
    cli = AntigravityQueryResult(result="", returncode=0, stderr="", stream_result="",
                                 conversation_id=_CONV, session_transcript=b"x")
    llm._sync_transcript(cli)
    assert llm._conversation_id is None and llm._session_transcript is None


def test_prompt_captures_transcript_when_resuming(monkeypatch, tmp_path):
    _fake_subprocess(monkeypatch, {}, stdout=_STREAM_JSON)
    llm = AntigravityLLM(gemini_dir=str(tmp_path), resume_session=True)
    conv_dir = tmp_path / "antigravity-cli" / "conversations"
    conv_dir.mkdir(parents=True)
    _make_conversation_db(str(conv_dir / f"{_CONV}.db"), b"TURN-1")
    cli = llm.prompt("hi", tools=[])
    assert cli.success and cli.conversation_id == _CONV
    assert cli.session_transcript and b"TURN-1" in cli.session_transcript
    assert llm._conversation_id == _CONV
    assert llm._last_metadata["conversation_id"] == _CONV
    assert llm._last_metadata["usage"]["total_tokens"] == 14674


def test_prompt_remote_returns_callback_only_when_resuming(monkeypatch):
    from chia.base.ChiaFunction import ObjectRefCallback
    sentinel = object()
    monkeypatch.setattr(AntigravityLLM.prompt, "chia_remote", lambda *a, **k: sentinel)
    plain = AntigravityLLM()
    assert plain.prompt.chia_remote(plain, "hi") is sentinel
    resuming = AntigravityLLM(resume_session=True)
    wrapped = resuming.prompt.chia_remote(resuming, "hi")
    assert isinstance(wrapped, ObjectRefCallback)


def test_prompt_routes_to_run_antigravity(monkeypatch):
    llm = AntigravityLLM()
    sentinel = AntigravityQueryResult("X", 0, "", "")
    monkeypatch.setattr(llm, "_run_antigravity", lambda user, tools: sentinel)
    out = llm.prompt("hi", tools=[])
    assert out is sentinel
    assert out.success is True
    assert llm._last_metadata["model"] == "antigravity-default"


def test_run_antigravity_subprocess_flow(monkeypatch, tmp_path):
    capture = {}
    _fake_subprocess(monkeypatch, capture, stdout="  PONG  ", stderr="")
    cli = AntigravityLLM(
        model="gemini-2.5-pro",
        system_message="be terse",
        work_dir="/tmp",
        gemini_dir=str(tmp_path),
        timeout_seconds=33,
    )._run_antigravity("say pong", tools=[])
    assert cli.result == "PONG"  # stdout is stripped
    assert "[Response]\nPONG" in cli.stream_result
    assert capture["cmd"][-1].startswith("[System Instructions]")
    assert capture["kwargs"]["timeout"] == 63  # timeout_seconds + 30
    assert capture["kwargs"]["cwd"] == "/tmp"


def test_classify_clean_success_no_raise():
    AntigravityLLM()._classify_error(_cli(returncode=0, result="PONG"))


def test_classify_benign_prose_on_clean_exit_no_raise():
    # Generic words like "timeout"/"login" in a real answer must NOT be flagged
    # when agy exited cleanly — only the precise CLI signatures count at exit 0.
    AntigravityLLM()._classify_error(
        _cli(returncode=0, result="To fix a connection timeout, check your login settings.")
    )


def test_classify_auth_failure_on_exit_zero():
    # agy prints the OAuth prompt to stdout and exits 0 when unauthenticated;
    # the hard signature must still raise AuthenticationError.
    with pytest.raises(AuthenticationError):
        AntigravityLLM()._classify_error(
            _cli(returncode=0, result="Authentication required. Please visit the URL to log in:")
        )


@pytest.mark.parametrize(
    ("message", "error_cls", "returncode"),
    [
        ("429 rate limit", RateLimitError, 0),
        ("Authentication required. Please sign in", AuthenticationError, 0),
        ("You are not logged into Antigravity", AuthenticationError, 0),
        ("Error: authentication timed out.", AuthenticationError, 0),
        ("payment required: out of credit", BillingError, 1),
        ("invalid model: nope", InvalidRequestError, 1),
        ("503 service unavailable", ServerError, 1),
        ("maximum output token limit reached", MaxOutputTokensError, 1),
        ("something surprising", UnknownAntigravityError, 1),
    ],
)
def test_classify_errors(message, error_cls, returncode):
    with pytest.raises(error_cls):
        AntigravityLLM()._classify_error(_cli(returncode=returncode, stderr=message))


live = pytest.mark.skipif(
    os.environ.get("ANTIGRAVITY_LIVE_TEST") != "1" or not shutil.which("agy"),
    reason="set ANTIGRAVITY_LIVE_TEST=1 and authenticate agy to run live tests",
)


@live
def test_live_antigravity_simple_prompt():
    llm = AntigravityLLM(
        system_message="You answer with a single word and nothing else.",
        timeout_seconds=180,
    )
    cli = llm.prompt("Reply with exactly the word: PONG", tools=[])
    assert cli.success is True
    assert "PONG" in cli.result.upper()


# ---------------------------------------------------------------------------
# Permission controls (live): agy honors --dangerously-skip-permissions; it has
# no opencode-style `permission` block. Gated by ANTIGRAVITY_LIVE_TEST=1 + agy.
# ---------------------------------------------------------------------------


@live
def test_live_antigravity_skip_permissions_runs_prompt():
    llm = AntigravityLLM(
        system_message="You answer with a single word and nothing else.",
        timeout_seconds=180,
        dangerously_skip_permissions=True,
    )
    assert llm.dangerously_skip_permissions is True
    cli = llm.prompt("Reply with exactly the word: PONG", tools=[])
    assert cli.success is True
    assert "PONG" in cli.result.upper()


@live
def test_live_antigravity_permission_arg_warns_but_still_runs():
    with pytest.warns(UserWarning, match="does not support a 'config'"):
        llm = AntigravityLLM(
            system_message="You answer with a single word and nothing else.",
            timeout_seconds=180,
            config={"x": "y"},
        )
    cli = llm.prompt("Reply with exactly the word: PONG", tools=[])
    assert cli.success is True
    assert "PONG" in cli.result.upper()


# ---------------------------------------------------------------------------
# Permission controls (live_remote): dispatch onto a real antigravity_creds
# worker so --dangerously-skip-permissions applies inside the worker container.
# ---------------------------------------------------------------------------


# Worker for this test: `chia up chia/models/tests/cluster/all_models.yaml`
# (advertises antigravity_creds); the remote_prompt fixture skips if it's absent.
@pytest.mark.live_remote
def test_live_remote_antigravity_skip_permissions(remote_prompt):
    llm = AntigravityLLM(
        system_message="You answer with a single word and nothing else.",
        timeout_seconds=180,
        dangerously_skip_permissions=True,
    )
    cli = remote_prompt(llm, "Reply with exactly the word: PONG", "antigravity_creds")
    assert cli.success is True
    assert "PONG" in cli.result.upper()
