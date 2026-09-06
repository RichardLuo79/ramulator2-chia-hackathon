"""Real Claude binary, local fake provider only. Never consumes model usage."""
import json
import os
import pathlib
import shutil
import time

import pytest

from tools.chia_loop.claude_cli import transport as T, usage as U, auth
from tools.chia_loop.claude_cli.stream import terminal_response, action, readable_thinking
from tools.chia_loop.core import atomic_write_json


def configure(root):
    root.mkdir(exist_ok=True)
    atomic_write_json(root / "claude_config.json", {"run_id": root.name, "model": T.MODEL,
        "effort": "xhigh", "guard_mode": "iterations", "usd_cap": None, "maximum_iterations": 20,
        "auth_mode": "claude_subscription", "cpu_budget": 3})


def credentials(path):
    atomic_write_json(path, {"claudeAiOauth": {"accessToken": "sk-ant-oat01-PRIVATE_AUTH_CANARY",
        "refreshToken": "PRIVATE_REFRESH_CANARY", "expiresAt": (time.time() + 86400) * 1000,
        "scopes": ["user:inference", "user:profile"], "subscriptionType": "max", "rateLimitTier": "default_claude_max_5x"}})
    path.chmod(0o600)


def fake_sse(answer, stop="end_turn", thinking=True):
    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": "Exposed fixture summary.", "signature": "OPAQUE_SIGNATURE"})
    content.append({"type": "text", "text": json.dumps(answer)})
    usage = {"input_tokens": 100, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 50,
             "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 0}, "output_tokens": 50}
    events = [{"type": "message_start", "message": {"id": "msg_fixture", "type": "message", "role": "assistant",
              "model": T.MODEL, "content": [], "stop_reason": None, "stop_sequence": None, "usage": {**usage, "output_tokens": 1}}}]
    for index, block in enumerate(content):
        field = "thinking" if block["type"] == "thinking" else "text"
        events += [{"type": "content_block_start", "index": index, "content_block": {"type": block["type"], field: ""}},
                   {"type": "content_block_delta", "index": index, "delta": {"type": field + "_delta", field: block[field]}}]
        if "signature" in block:
            events += [{"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": block["signature"]}}]
        events += [{"type": "content_block_stop", "index": index}]
    events += [{"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 50}},
               {"type": "message_stop"}]
    return "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events).encode()


def test_stream_completion_and_thinking():
    response = terminal_response(fake_sse({"status": "no_change"}))
    assert action(response)["status"] == "no_change"
    assert readable_thinking(response)["text"] == ["Exposed fixture summary."]
    with pytest.raises(RuntimeError):
        terminal_response(fake_sse({})[:-60])
    with pytest.raises(RuntimeError):
        action(terminal_response(fake_sse({}, stop="max_tokens")))
    with pytest.raises(RuntimeError, match="native tool"):
        terminal_response(fake_sse({}).replace(b'"type": "text"', b'"type": "tool_use"'))


def test_cache_pricing_and_unknown_usage(tmp_path):
    configure(tmp_path)
    tokens = U.normalized(terminal_response(fake_sse({}))["usage"])
    assert tokens["input_tokens"] == 170
    assert tokens["reasoning_output_tokens"] is None
    assert U.priced(tokens)[0] == pytest.approx((1000 + 250 + 12.5 + 2500) / 1e6)
    ledger = U.Ledger(tmp_path)
    request = {"model": T.MODEL, "output_config": {"effort": "xhigh"}}
    index = ledger.reserve(request, "proposal", "proposal_001_001", attempt_path="interactions/a", attempt_number=1)
    assert ledger.totals()["unknown_usage_calls"] == 1
    response = terminal_response(fake_sse({}))
    ledger.settle(index, response, response_sha256="a")
    ledger.settle(index, response, response_sha256="a")
    assert ledger.totals()["attempts"] == 1 and ledger.totals()["unknown_usage_calls"] == 0
    with pytest.raises(RuntimeError):
        ledger.settle(index, response, response_sha256="changed")
    assert U.priced(U.normalized({"input_tokens": 1, "output_tokens": 1}))[0] is None


def test_configuration_no_inherited_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "PRIVATE_API_CANARY")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "PRIVATE_LOGIN_CANARY")
    monkeypatch.setenv("CODEX_THREAD_ID", "PRIVATE_CONVERSATION_CANARY")
    for effort in T.EFFORTS:
        args = T.command(pathlib.Path("/binary"), tmp_path, effort)
        assert args[args.index("--effort") + 1] == effort
        assert "--no-session-persistence" in args and "--safe-mode" in args and "--bare" not in args
        assert "--resume" not in args and "--continue" not in args
        assert args[args.index("--tools") + 1] == ""
        assert "PRIVATE" not in json.dumps(T.clean_environment(tmp_path, 12345, "nonce", effort))
    with pytest.raises(ValueError):
        T.command(pathlib.Path("/binary"), tmp_path, "ultracode")


@pytest.mark.parametrize("effort", T.EFFORTS)
def test_real_cli_isolated_and_replayable(tmp_path, monkeypatch, effort):
    binary = os.environ.get("CHIA_CLAUDE_PREFLIGHT_BINARY") or shutil.which("claude")
    assert binary, "real Claude binary required; no silent skip of isolation preflight"
    root = tmp_path / "run"
    configure(root)
    config = json.loads((root / "claude_config.json").read_text())
    atomic_write_json(root / "claude_config.json", {**config, "effort": effort})
    credential = tmp_path / "auth.json"
    credentials(credential)
    before = credential.read_bytes()
    (tmp_path / "CLAUDE.md").write_text("PRIVATE_RULE_CANARY")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "PRIVATE_API_CANARY")
    requests = []
    def upstream(request, headers, attempt):
        requests.append(request)
        assert "CANARY" not in json.dumps(request)
        assert headers.get("Authorization") == "Bearer sk-ant-oat01-PRIVATE_AUTH_CANARY"
        assert request["output_config"]["effort"] == effort
        assert not request.get("tools")
        return fake_sse({"status": "no_change", "limitations": "offline fixture"})
    kwargs = {"root": root, "operation": "proposal_001_001", "system": "A fixture contract. Return a JSON object.",
              "conversation": [{"role": "user", "content": "Return no_change JSON."}], "effort": effort,
              "role": "proposal", "cap": None, "upstream": upstream, "binary": pathlib.Path(binary).resolve(), "auth_file": credential}
    try:
        assert T.invoke(**kwargs)["status"] == "no_change"
    except Exception:
        for log in (root / "interactions").rglob("*.log"):
            print(log.name, log.read_text()[-4000:])
        for path in (root / "interactions").rglob("cli_request.json"):
            request = json.loads(path.read_text())
            print("request shape", {k: v for k, v in request.items() if k != "metadata"})
        raise
    assert T.invoke(**kwargs)["status"] == "no_change"
    assert len(requests) == 1
    events = [json.loads(line) for line in (root / "interactions/proposal_001_001/attempt_001/cli_events.jsonl").read_text().splitlines()]
    terminal = [e for e in events if e["type"] == "result"]
    assert len(terminal) == 1 and terminal[0]["is_error"] is False
    assert terminal[0]["num_turns"] == 1
    assert json.loads(terminal[0]["result"])["status"] == "no_change"
    assert credential.read_bytes() == before
    assert not list(root.rglob(".credentials.json"))
    assert "PRIVATE_AUTH_CANARY" not in "".join(p.read_text() for p in root.rglob("*") if p.is_file())
    assert U.Ledger(root).totals()["attempts"] == 1
    assert not U.report(root)["audit_issues"]
    # Compression must not make a completed call billable again or hide an
    # altered source receipt. Its ledger and raw provider hash remain binding.
    from tools.chia_loop import artifacts
    artifacts.compress(root, root / "packed.json", min_bytes=1)
    artifacts.verify(root / "packed.json")
    assert T.invoke(**kwargs)["status"] == "no_change" and len(requests) == 1
    assert not U.report(root)["audit_issues"]
