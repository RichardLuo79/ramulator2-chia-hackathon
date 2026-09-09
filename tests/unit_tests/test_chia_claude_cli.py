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


def configure(root, credential_kind="native_login"):
    root.mkdir(exist_ok=True)
    atomic_write_json(root / "claude_config.json", {"run_id": root.name, "model": T.MODEL,
        "effort": "xhigh", "guard_mode": "iterations", "usd_cap": None, "maximum_iterations": 20,
        "auth_mode": "claude_subscription", "credential_kind": credential_kind, "cpu_budget": 3})


def credentials(path, credential_kind="native_login"):
    token = "sk-ant-oat01-" + "PRIVATE_SETUP_AUTH_CANARY_" * 3
    if credential_kind == "setup_token":
        path.write_text(token + "\n")
    else:
        token = "sk-ant-oat01-PRIVATE_AUTH_CANARY"
        atomic_write_json(path, {"claudeAiOauth": {"accessToken": token,
            "refreshToken": "PRIVATE_REFRESH_CANARY", "expiresAt": (time.time() + 86400) * 1000,
            "scopes": ["user:inference", "user:profile"], "subscriptionType": "max", "rateLimitTier": "default_claude_max_5x"}})
    path.chmod(0o600)
    return token


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
@pytest.mark.parametrize("credential_kind", auth.KINDS)
@pytest.mark.parametrize("streaming", (False, True))
def test_real_cli_isolated_and_replayable(tmp_path, monkeypatch, effort, credential_kind, streaming):
    binary = os.environ.get("CHIA_CLAUDE_PREFLIGHT_BINARY") or shutil.which("claude")
    assert binary, "real Claude binary required; no silent skip of isolation preflight"
    root = tmp_path / "run"
    configure(root, credential_kind)
    config = json.loads((root / "claude_config.json").read_text())
    atomic_write_json(root / "claude_config.json", {**config, "effort": effort})
    credential = tmp_path / "auth.json"
    expected_token = credentials(credential, credential_kind)
    before = credential.read_bytes()
    (tmp_path / "CLAUDE.md").write_text("PRIVATE_RULE_CANARY")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "PRIVATE_API_CANARY")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "PRIVATE_INHERITED_OAUTH_CANARY")
    original_run = T.subprocess.run
    boundary_policies = []
    def inspect_boundary(args, **options):
        policy = json.loads(pathlib.Path(args[2]).read_text())
        boundary_policies.append(policy)
        assert expected_token not in json.dumps(policy)
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in policy["environment"]
        assert expected_token not in json.dumps(args)
        assert expected_token not in json.dumps(options.get("env", {}))
        assert str(credential) in policy["read"]
        assert str(credential.parent) not in policy["read"]
        if credential_kind == "setup_token":
            assert policy["setup_token_file"] == str(credential)
            assert not (pathlib.Path(policy["cwd"]) / "home/.claude/.credentials.json").exists()
        else:
            assert "setup_token_file" not in policy
        return original_run(args, **options)
    monkeypatch.setattr(T.subprocess, "run", inspect_boundary)
    requests = []
    def upstream(request, headers, attempt):
        requests.append(request)
        assert "CANARY" not in json.dumps(request)
        assert headers.get("Authorization") == "Bearer " + expected_token
        assert request["output_config"]["effort"] == effort
        assert not request.get("tools")
        return fake_sse({"status": "no_change", "limitations": "offline fixture"})
    class StreamingFixture:
        def stream(self, request, headers, attempt, *, on_chunk):
            raw = upstream(request, headers, attempt)
            for part in raw.splitlines(keepends=True):
                on_chunk(part)
            return raw
    kwargs = {"root": root, "operation": "proposal_001_001", "system": "A fixture contract. Return a JSON object.",
              "conversation": [{"role": "user", "content": "Return no_change JSON."}], "effort": effort,
              "role": "proposal", "cap": None, "upstream": StreamingFixture() if streaming else upstream,
              "binary": pathlib.Path(binary).resolve(),
              "auth_file": credential, "credential_kind": credential_kind}
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
    assert len(boundary_policies) == 1
    events = [json.loads(line) for line in (root / "interactions/proposal_001_001/attempt_001/cli_events.jsonl").read_text().splitlines()]
    terminal = [e for e in events if e["type"] == "result"]
    assert len(terminal) == 1 and terminal[0]["is_error"] is False
    assert terminal[0]["num_turns"] == 1
    assert json.loads(terminal[0]["result"])["status"] == "no_change"
    assert credential.read_bytes() == before
    assert not list(root.rglob(".credentials.json"))
    archived = "".join(p.read_text() for p in root.rglob("*") if p.is_file())
    assert "PRIVATE_" not in archived and expected_token not in archived
    identity = json.loads((root / "interactions/proposal_001_001/identity.json").read_text())
    assert identity["credential_kind"] == credential_kind
    with pytest.raises(RuntimeError, match="owner/input changed"):
        T.invoke(**{**kwargs, "credential_kind": next(k for k in auth.KINDS if k != credential_kind)})
    assert U.Ledger(root).totals()["attempts"] == 1
    assert not U.report(root)["audit_issues"]
    # Compression must not make a completed call billable again or hide an
    # altered source receipt. Its ledger and raw provider hash remain binding.
    from tools.chia_loop import artifacts
    artifacts.compress(root, root / "packed.json", min_bytes=1)
    artifacts.verify(root / "packed.json")
    assert T.invoke(**kwargs)["status"] == "no_change" and len(requests) == 1
    assert not U.report(root)["audit_issues"]


def test_setup_token_metadata_and_binding_never_copy_credentials(tmp_path):
    credential = tmp_path / "private-token"
    expected = credentials(credential, "setup_token")
    # File age cannot establish an opaque token's expiry or subscription tier.
    os.utime(credential, (1, 1))
    status = auth.check(credential, now=time.time() + 10**9, credential_kind="setup_token")
    assert status["expires_at"] is None and status["subscription_type"] is None
    assert status["generation_calls"] == 0 and expected not in json.dumps(status)
    work = tmp_path / "work"
    work.mkdir()
    assert auth.bind(work, credential, credential_kind="setup_token") == credential.resolve()
    assert not list(work.rglob("*"))


@pytest.mark.parametrize("credential_kind", auth.KINDS)
@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_real_cli_cache_layout_preserves_history_and_replays_without_payment(tmp_path, monkeypatch, credential_kind, effort):
    from tools.chia_loop import prompt_cache as K, artifacts
    binary = os.environ.get("CHIA_CLAUDE_PREFLIGHT_BINARY") or shutil.which("claude")
    assert binary, "cache preflight requires the actual native CLI"
    root = tmp_path / "run"
    configure(root, credential_kind)
    config = T.R.read_json(root / "claude_config.json")
    atomic_write_json(root / "claude_config.json", {**config, "effort": effort})
    K.install(root)
    credential = tmp_path / "private-token"
    private = credentials(credential, credential_kind)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "PRIVATE_WRONG_LOGIN")
    (tmp_path / "CLAUDE.md").write_text("PRIVATE_BRANCH_CANARY")
    calls = []
    def upstream(request, headers, directory):
        assert private not in json.dumps(request) and "CANARY" not in json.dumps(request)
        assert any(k.lower() == "authorization" and private in v for k, v in headers.items())
        calls.append(request)
        return fake_sse({"status": "no_change"})
    conversation = [{"role": "user", "content": "stable fixture " * 4000}]
    args = dict(root=root, system="Test contract, no tools.", conversation=conversation,
                effort=effort, role="proposal", cap=None, upstream=upstream, binary=pathlib.Path(binary).resolve(),
                auth_file=credential, credential_kind=credential_kind)
    assert T.invoke(operation="proposal_001_001", **args)["status"] == "no_change"
    extended = conversation + [{"role": "assistant", "content": "inspect"}, {"role": "user", "content": "own new data"}]
    assert T.invoke(operation="proposal_001_002", **{**args, "conversation": extended})["status"] == "no_change"
    first, second = calls
    assert first["system"] == second["system"]  # Includes unchanged native attribution.
    left = [b["text"] for b in first["messages"][0]["content"]]
    right = [b["text"] for b in second["messages"][0]["content"]]
    assert right[:len(left) - 1] == left[:-1]
    assert "".join(left[1:]) == json.dumps(conversation, sort_keys=True)
    assert "".join(right[1:]) == json.dumps(extended, sort_keys=True)
    assert sum("cache_control" in b for b in second["system"] + [b for m in second["messages"] for b in m["content"]]) <= 4
    first_anchor = len(left) - 2
    assert "cache_control" in first["messages"][0]["content"][first_anchor]
    assert "cache_control" in second["messages"][0]["content"][first_anchor]
    artifacts.compress(root, root / "packed.json", min_bytes=1)
    assert T.invoke(operation="proposal_001_002", **{**args, "conversation": extended})["status"] == "no_change"
    assert len(calls) == 2 and U.Ledger(root).totals()["attempts"] == 2
    assert not U.report(root)["audit_issues"]


@pytest.mark.parametrize("invalid", ["permissions", "empty", "api_key", "two_tokens", "oversize", "unicode", "fifo", "in_repository"])
def test_setup_token_rejects_unsafe_files_without_disclosing_contents(tmp_path, monkeypatch, invalid):
    credential = tmp_path / "private-token"
    token = credentials(credential, "setup_token")
    if invalid == "permissions": credential.chmod(0o644)
    elif invalid == "empty": credential.write_text("")
    elif invalid == "api_key": credential.write_text("sk-ant-api03-" + "PRIVATE_" * 10)
    elif invalid == "two_tokens": credential.write_text(token + "\n" + token)
    elif invalid == "oversize": credential.write_text(token * 100)
    elif invalid == "unicode": credential.write_bytes(b"\xffPRIVATE_CANARY")
    elif invalid == "fifo":
        credential.unlink()
        os.mkfifo(credential, 0o600)
    elif invalid == "in_repository": monkeypatch.setattr(auth, "REPO", tmp_path)
    with pytest.raises(RuntimeError) as exc:
        auth.check(credential, credential_kind="setup_token")
    assert token not in str(exc.value) and "PRIVATE_" not in str(exc.value)


def test_setup_token_401_does_not_fall_back_or_retry_inside_cli(tmp_path, monkeypatch):
    import httpx
    binary = os.environ.get("CHIA_CLAUDE_PREFLIGHT_BINARY") or shutil.which("claude")
    assert binary
    root = tmp_path / "run"
    configure(root, "setup_token")
    credential = tmp_path / "private-token"
    expected = credentials(credential, "setup_token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "PRIVATE_API_FALLBACK_CANARY")
    requests = []
    def rejected(request, headers, attempt):
        assert headers.get("Authorization") == "Bearer " + expected
        assert headers.get("x-api-key") is None
        requests.append(request)
        response = httpx.Response(401, request=httpx.Request("POST", "https://provider.invalid/v1/messages"))
        response.raise_for_status()
    with pytest.raises(T.R.OperationalPause) as exc:
        T.invoke(root, "proposal_001_001", "fixture", [], effort="xhigh", role="proposal", cap=None,
                 upstream=rejected, binary=pathlib.Path(binary).resolve(), auth_file=credential, credential_kind="setup_token")
    assert exc.value.retryable is False and len(requests) == 1
    receipt = json.loads((root / "interactions/proposal_001_001/attempt_001/receipt.json").read_text())
    assert receipt["http_status"] == 401
    assert U.Ledger(root).totals()["attempts"] == 1
    assert U.Ledger(root).totals()["unknown_usage_calls"] == 1
    assert "PRIVATE_" not in "".join(p.read_text() for p in root.rglob("*") if p.is_file())
