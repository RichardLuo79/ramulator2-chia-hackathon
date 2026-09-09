"""No provider calls: real CLI uses a local fake Responses broker."""
import json
import copy
import base64
import os
import pathlib
import shutil
import socket
import subprocess
import sys

import pytest

from tools.chia_loop.codex_cli import transport as T
from tools.chia_loop import real_core as P
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.codex_cli import runner as B
from tools.chia_loop.codex_cli import auth as A


def fake_sse(answer, *, summary=None, terminal_output=True):
    item = {"id": "msg_mock", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(answer), "annotations": []}]}
    reasoning = {"id": "rs_mock", "type": "reasoning", "summary": [
        {"type": "summary_text", "text": summary}], "encrypted_content": "OPAQUE_NOT_READABLE"}
    output = [reasoning, item] if summary is not None else [item]
    response = {"id": "resp_mock", "object": "response", "model": T.MODEL, "status": "completed",
                "output": output, "usage": {"input_tokens": 100, "output_tokens": 50,
                    "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 17},
                    "total_tokens": 150}}
    events = [{"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}]
    events += [{"type": "response.output_item.done", "output_index": index, "item": value}
               for index, value in enumerate(output)]
    events += [{"type": "response.completed", "response": response if terminal_output else {**response, "output": []}}]
    return ("".join("data: " + json.dumps(e) + "\n\n" for e in events)).encode()


def test_efforts_are_exact_and_environment_has_no_host_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", "PRIVATE_CONVERSATION_CANARY")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_SECRET_CANARY")
    for effort in T.EFFORTS:
        cmd = T.command(pathlib.Path("/binary"), tmp_path, 12345, "dummy", effort)
        assert 'model_reasoning_effort="' + effort + '"' in cmd
        assert 'model_reasoning_summary="auto"' in cmd
        assert 'model_supports_reasoning_summaries=true' in cmd
        assert "--ephemeral" in cmd and "--ignore-user-config" in cmd
        assert "resume" not in cmd and "fork" not in cmd and "--remote" not in cmd
    assert "PRIVATE" not in json.dumps(T.clean_environment(tmp_path))
    with pytest.raises(ValueError):
        T.command(pathlib.Path("/binary"), tmp_path, 12345, "dummy", "high")


def test_budget_unknown_usage_cannot_be_replenished(tmp_path):
    ledger = T.Ledger(tmp_path, 20)
    request = {"model": T.MODEL, "input": "prompt"}
    first = ledger.reserve(request, "proposal", "proposal_001")
    ledger.reserve(request, "review", "review_001")
    with pytest.raises(P.BudgetExhausted):
        ledger.reserve(request, "proposal", "proposal_002")
    response = T.completed_response(fake_sse({"status": "no_change"}))
    ledger.settle(first, response)
    ledger.settle(first, response)
    assert ledger.totals()["unknown_usage_calls"] == 1
    assert ledger.totals()["known_standard_usd"] == pytest.approx(.0035)
    with pytest.raises(RuntimeError):
        T.Ledger(tmp_path, 100).reserve(request, "proposal", "different_cap")


@pytest.mark.parametrize("cap", [0, -1, float("nan"), float("inf"), True])
def test_budget_rejects_unbounded_or_invalid_authorization(tmp_path, cap):
    with pytest.raises(ValueError, match="finite positive"):
        T.Ledger(tmp_path, cap)


def jwt_exp(expiry):
    return "mock." + base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=") + ".sig"


def test_managed_login_refresh_is_auth_only_and_not_copied(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.json"
    tokens = {"access_token": jwt_exp(1), "refresh_token": "PRIVATE_REFRESH", "account_id": "PRIVATE_ACCOUNT"}
    atomic_write_json(auth_file, {"auth_mode": "chatgpt", "tokens": tokens})
    calls = []
    def refresh(binary, path, *, refresh):
        assert binary == "/pinned/codex" and path == auth_file and refresh is True
        calls.append("account/read")
        atomic_write_json(path, {"auth_mode": "chatgpt", "tokens": {**tokens, "access_token": jwt_exp(4_000_000_000)}})
    monkeypatch.setattr(A, "account_read", refresh)
    headers = A.headers("/pinned/codex", auth_file)
    assert headers["ChatGPT-Account-Id"] == "PRIVATE_ACCOUNT"
    assert "PRIVATE_REFRESH" not in json.dumps(headers)
    assert A.headers("/pinned/codex", auth_file) == headers
    assert calls == ["account/read"]
    assert A.expires_soon("opaque") and A.expires_soon(jwt_exp(float("nan")))
    assert A.expires_soon(jwt_exp(1299), now=1000)
    assert not A.expires_soon(jwt_exp(1301), now=1000)
    atomic_write_json(auth_file, {"auth_mode": "api", "OPENAI_API_KEY": "PRIVATE_KEY"})
    with pytest.raises(RuntimeError, match="managed ChatGPT") as error:
        A.read_tokens(auth_file)
    assert "PRIVATE" not in str(error.value)


def test_auth_helper_never_starts_or_reads_a_thread(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_THREAD_ID", "PRIVATE_CONVERSATION")
    binary = tmp_path / "mock_codex"
    binary.write_text('''#!/usr/bin/python3
import json, os, pathlib, sys
home = pathlib.Path(os.environ["CODEX_HOME"])
assert "CODEX_THREAD_ID" not in os.environ
assert os.environ["HOME"] != str(home)
with (home / "auth_requests.jsonl").open("w") as log:
    for line in sys.stdin:
        event = json.loads(line)
        log.write(line)
        log.flush()
        assert event["method"] in {"initialize", "initialized", "account/read"}
        if "id" in event:
            result = {} if event["method"] == "initialize" else {"account": {"type": "chatgpt", "email": "PRIVATE_EMAIL"}}
            print(json.dumps({"id": event["id"], "result": result}), flush=True)
''')
    binary.chmod(0o755)
    (tmp_path / "auth.json").write_text("{}")
    result = A.account_read(binary, tmp_path / "auth.json", refresh=True)
    requests = [json.loads(line) for line in (tmp_path / "auth_requests.jsonl").read_text().splitlines()]
    assert [row["method"] for row in requests] == ["initialize", "initialized", "account/read"]
    assert requests[-1]["params"] == {"refreshToken": True}
    assert result["generation_calls"] == 0 and result["thread_methods_called"] == 0
    assert "PRIVATE" not in json.dumps(result)


def test_chatgpt_upstream_uses_only_trusted_credentials(tmp_path, monkeypatch):
    import httpx
    native_client = httpx.AsyncClient
    private = {"Authorization": "Bearer PRIVATE_ACCESS", "ChatGPT-Account-Id": "PRIVATE_ACCOUNT"}
    monkeypatch.setattr(A, "headers", lambda binary, path: dict(private))
    calls = []
    def receive(request):
        assert request.url == "https://chatgpt.com/backend-api/codex/responses"
        assert request.headers["Authorization"] == private["Authorization"]
        assert "PRIVATE" not in request.content.decode()
        calls.append(request)
        return httpx.Response(200, content=fake_sse({"status": "no_change"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: native_client(transport=httpx.MockTransport(receive), **kwargs))
    upstream = T.Upstream("chatgpt", tmp_path / "auth.json", "/pinned/codex")
    with pytest.raises(RuntimeError, match="auth preparation"):
        upstream({"model": T.MODEL})
    upstream.prepare_auth()
    assert T.action(T.completed_response(upstream({"model": T.MODEL})))["status"] == "no_change"
    assert len(calls) == 1


def test_incomplete_and_native_tool_responses_never_reach_cli():
    with pytest.raises(RuntimeError):
        T.completed_response(b'data: {"type":"response.created"}\n\n')
    event = {"type": "response.completed", "response": {"status": "completed",
             "output": [{"type": "function_call", "name": "exec_command"}]}}
    with pytest.raises(RuntimeError, match="native tool"):
        T.completed_response(("data: " + json.dumps(event) + "\n\n").encode())


def test_whole_process_boundary_denies_host_files_and_sockets(tmp_path):
    work = tmp_path / "workspace"
    work.mkdir()
    secret = tmp_path / "host_private.txt"
    secret.write_text("HOST_CONVERSATION_CANARY")
    (work / "escape").symlink_to(secret)
    (work / "visible.txt").write_text("public interface")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        code = '''import pathlib, socket, json
assert pathlib.Path("visible.txt").read_text() == "public interface"
for path in (SECRET, "escape", "../host_private.txt", "/proc/self/environ", "/home/dev/.codex/auth.json"):
    try: pathlib.Path(path).read_bytes()
    except (PermissionError, FileNotFoundError): pass
    else: raise AssertionError("private path accessible: " + path)
try: socket.socket(socket.AF_UNIX)
except PermissionError: pass
else: raise AssertionError("Unix socket available")
try: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except PermissionError: pass
else: raise AssertionError("UDP available")
try: socket.create_connection(("127.0.0.1", 443), timeout=1)
except PermissionError: pass
else: raise AssertionError("non-broker port available")
with socket.create_connection(("127.0.0.1", PORT), timeout=1): pass
print("BOUNDARY_PASS")
'''.replace("SECRET", repr(str(secret))).replace("PORT", str(port))
        policy = {"cwd": str(work), "read": ["/usr", "/lib", "/lib64", "/bin", str(work)],
                  "write": [str(work)], "port": port, "environment": {"PATH": "/usr/bin:/bin"}}
        atomic_write_json(tmp_path / "boundary.json", policy)
        run = subprocess.run([sys.executable, str(T.HERE / "boundary.py"), str(tmp_path / "boundary.json"),
                              "--", str(pathlib.Path(sys.executable).resolve()), "-c", code], text=True, capture_output=True, timeout=10)
        assert run.returncode == 0, run.stderr
        assert "BOUNDARY_PASS" in run.stdout


@pytest.mark.parametrize("effort", T.EFFORTS)
@pytest.mark.parametrize("terminal_output", [True, False])
def test_real_cli_has_fresh_context_and_only_brokered_json_actions(tmp_path, monkeypatch, effort, terminal_output):
    binary = os.environ.get("CHIA_CODEX_PREFLIGHT_BINARY") or shutil.which("codex")
    if not binary:
        pytest.skip("Codex CLI required for local mocked-provider integration")
    # Both global environment and parent directories contain contamination
    # canaries. The provider must see none of them.
    monkeypatch.setenv("CODEX_THREAD_ID", "PARENT_THREAD_CANARY_78542")
    (tmp_path / "AGENTS.md").write_text("SECRET_PARENT_RULE_CANARY_48956")
    root = tmp_path / "run"
    root.mkdir()
    calls = []
    def fake_upstream(request):
        calls.append(request)
        encoded = json.dumps(request)
        assert "CANARY" not in encoded
        assert request["tools"] == []
        assert not request.get("previous_response_id") and not request.get("conversation")
        assert request["model"] == T.MODEL and request["reasoning"]["effort"] == effort
        assert request["reasoning"]["summary"] == "auto"
        return fake_sse({"status": "no_change", "reason": "offline fixture"},
                        summary="Offline reasoning-summary fixture: no evidence for a change.", terminal_output=terminal_output)
    args = dict(root=root, operation="proposal_001_001", system="Test contract, no tools.",
                conversation=[{"role": "user", "content": "Return a no_change JSON object."}],
                effort=effort, role="proposal", cap=100, upstream=fake_upstream,
                binary=pathlib.Path(binary).resolve())
    assert T.invoke(**args)["status"] == "no_change"
    assert T.invoke(**args)["status"] == "no_change"
    assert len(calls) == 1
    assert T.Ledger(root, 100).totals()["attempts"] == 1
    receipt = B.R.read_json(root / "interactions/proposal_001_001/attempt_001/receipt.json")
    assert receipt["cli_usage_check"] == "matched"
    assert receipt["cli_usage"]["reasoning_output_tokens"] == 17
    assert receipt["reasoning_summary"] == "auto"
    operation = root / "interactions/proposal_001_001"
    for name in ("cli_request.json", "provider_request.json"):
        assert B.R.read_json(operation / "attempt_001" / name)["reasoning"]["summary"] == "auto"
    raw = (operation / "attempt_001/provider_response.sse").read_text()
    assert "Offline reasoning-summary fixture" in raw
    events = (operation / "attempt_001/cli_events.jsonl").read_text()
    assert "Offline reasoning-summary fixture" in events
    report = B.U.write_report(root)
    assert report["audit_issues"] == []
    assert report["totals"]["token_totals"]["output_tokens"]["known_sum"] == 50
    assert report["totals"]["token_totals"]["reasoning_output_tokens"]["known_sum"] == 17
    assert "Test contract" not in json.dumps(report)
    with pytest.raises(RuntimeError, match="checkpoint"):
        T.invoke(**{**args, "effort": "max" if effort == "xhigh" else "xhigh"})


@pytest.mark.parametrize("auth_mode", ["chatgpt", "api"])
@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_real_cli_cache_layout_has_stable_prefix_without_shared_sessions(tmp_path, monkeypatch, auth_mode, effort):
    from tools.chia_loop import prompt_cache as K
    binary = os.environ.get("CHIA_CODEX_PREFLIGHT_BINARY") or shutil.which("codex")
    assert binary, "cache preflight requires the actual native CLI"
    root = tmp_path / "run"
    K.install(root)
    monkeypatch.setenv("CODEX_THREAD_ID", "PRIVATE_THREAD_CANARY")
    (tmp_path / "AGENTS.md").write_text("PRIVATE_BRANCH_CANARY")
    calls = []
    class Fake:
        mode = auth_mode
        def __call__(self, request):
            calls.append(copy.deepcopy(request))
            assert "CANARY" not in json.dumps(request)
            return fake_sse({"status": "no_change"}, summary="Offline cache test.")
    conversation = [{"role": "user", "content": "stable fixture " * 4000}]
    args = dict(root=root, system="Test contract, no tools.", conversation=conversation,
                effort=effort, role="proposal", cap=100, upstream=Fake(), binary=pathlib.Path(binary).resolve())
    assert T.invoke(operation="proposal_001_001", **args)["status"] == "no_change"
    next_conversation = conversation + [{"role": "assistant", "content": "inspect"}, {"role": "user", "content": "own new data"}]
    assert T.invoke(operation="proposal_001_002", **{**args, "conversation": next_conversation})["status"] == "no_change"
    first, second = calls
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["input"][:-1] == second["input"][:-1]
    first_text = [b["text"] for b in first["input"][-1]["content"]]
    second_text = [b["text"] for b in second["input"][-1]["content"]]
    assert second_text[:len(first_text) - 1] == first_text[:-1]
    assert "".join(first_text) == json.dumps(conversation)
    assert "".join(second_text) == json.dumps(next_conversation)
    assert "proposal_001_" not in json.dumps(first["input"])
    assert ("prompt_cache_options" in first) == (auth_mode == "api")
    assert ("prompt_cache_breakpoint" in json.dumps(first)) == (auth_mode == "api")
    assert not first.get("previous_response_id") and not first.get("conversation")
    native = B.R.read_json(root / "interactions/proposal_001_001/attempt_001/cli_request.json")
    # Framing drift, even inside a nominally allowed frame, must fail closed.
    native["input"][2]["content"][0]["text"] += "PRIVATE_UNEXPECTED_CONTEXT"
    with pytest.raises(RuntimeError, match="context changed"):
        K.codex_request(native, work=root / "interactions/proposal_001_001/attempt_001/workspace",
            conversation=conversation, system=K.scoped_system(args["system"], first["prompt_cache_key"]),
            namespace=first["prompt_cache_key"], api_mode=auth_mode == "api")
    B.artifacts.compress(root, root / "packed.json", min_bytes=1)
    assert T.invoke(operation="proposal_001_002", **{**args, "conversation": next_conversation})["status"] == "no_change"
    assert len(calls) == 2 and not B.U.report(root)["audit_issues"]


@pytest.mark.parametrize("terminal_output", [True, False])
def test_saved_provider_response_survives_missing_cli_receipt(tmp_path, monkeypatch, terminal_output):
    root = tmp_path / "run"
    root.mkdir()
    operation = "proposal_001_001"
    system, conversation = "contract", [{"role": "user", "content": "input"}]
    directory = root / "interactions" / operation
    attempt = directory / "attempt_001"
    attempt.mkdir(parents=True)
    atomic_write_json(directory / "identity.json", {"model": T.MODEL, "effort": "max", "role": "proposal",
        "reasoning_summary": T.REASONING_SUMMARY,
        "run_id": root.name, "system_sha256": P.sha(system),
        "conversation_sha256": P.sha(json.dumps(conversation, sort_keys=True))})
    index = T.Ledger(root, 100).reserve({"model": T.MODEL, "input": "input"}, "proposal", operation)
    atomic_write_json(attempt / "reservation.json", {"call_id": index})
    (attempt / "provider_response.sse").write_bytes(fake_sse({"status": "no_change"}, terminal_output=terminal_output))
    def forbidden(*args, **kwargs):
        raise AssertionError("recovery must not call the provider or CLI")
    monkeypatch.setattr(T.subprocess, "run", forbidden)
    assert T.invoke(root, operation, system, conversation, effort="max", role="proposal", cap=100,
                    upstream=forbidden, binary=pathlib.Path("/no-binary"))["status"] == "no_change"
    assert T.Ledger(root, 100).totals()["unknown_usage_calls"] == 0
    # Finalized provider receipts and large JSON can be archived. Replay must
    # use their decompressed bytes, not turn compressed evidence into a retry.
    B.artifacts.compress(root, root / "archive.json", min_bytes=1)
    B.artifacts.verify(root / "archive.json")
    assert (attempt / "provider_response.sse.gz").exists()
    assert T.invoke(root, operation, system, conversation, effort="max", role="proposal", cap=100,
                    upstream=forbidden, binary=pathlib.Path("/no-binary"))["status"] == "no_change"
    assert T.Ledger(root, 100).totals()["attempts"] == 1


def test_failed_auth_is_not_reserved_or_disclosed_to_cli(tmp_path):
    binary = os.environ.get("CHIA_CODEX_PREFLIGHT_BINARY") or shutil.which("codex")
    if not binary:
        pytest.skip("Codex CLI required for local mocked-provider integration")
    class FailedAuth:
        mode = "chatgpt"
        def prepare_auth(self):
            raise RuntimeError("PRIVATE_CREDENTIAL_CANARY")
        def __call__(self, request):
            raise AssertionError("generation must not start")
    with pytest.raises(B.R.OperationalPause):
        T.invoke(tmp_path, "proposal_001", "fixture", [{"role": "user", "content": "Return JSON"}],
                 effort="xhigh", role="proposal", cap=100, upstream=FailedAuth(), binary=pathlib.Path(binary).resolve())
    assert not (tmp_path / "ledger.json").exists()
    for path in (tmp_path / "interactions").rglob("*"):
        if path.is_file():
            assert b"PRIVATE_CREDENTIAL_CANARY" not in path.read_bytes()


@pytest.mark.parametrize("summary", [None, "none", "concise"])
def test_missing_or_changed_summary_request_stops_before_auth_or_usage(tmp_path, monkeypatch, summary):
    binary = os.environ.get("CHIA_CODEX_PREFLIGHT_BINARY") or shutil.which("codex")
    if not binary:
        pytest.skip("Codex CLI required for local mocked-provider integration")
    original = T.command
    def wrong_command(*args):
        command = original(*args)
        index = command.index('model_reasoning_summary="auto"')
        if summary is None:
            del command[index - 1:index + 1]
            command.insert(-1, "-c")
            command.insert(-1, "model_supports_reasoning_summaries=false")
        else:
            command[index] = "model_reasoning_summary=" + json.dumps(summary)
        return command
    monkeypatch.setattr(T, "command", wrong_command)
    class Forbidden:
        def prepare_auth(self):
            pytest.fail("invalid request must not refresh auth")
        def __call__(self, request):
            pytest.fail("invalid request must not reach the provider")
    with pytest.raises(B.R.OperationalPause) as stopped:
        T.invoke(tmp_path, "proposal_001_001", "fixture", [{"role": "user", "content": "Return JSON"}],
                 effort="max", role="proposal", cap=100, upstream=Forbidden(), binary=pathlib.Path(binary).resolve())
    assert not stopped.value.retryable
    assert not (tmp_path / "ledger.json").exists()
    attempt = tmp_path / "interactions/proposal_001_001/attempt_001"
    assert not (attempt / "provider_request.json").exists()
    assert B.R.read_json(attempt / "receipt.json")["stage"] == "request_validation"


def test_operator_preparation_hold_blocks_prepare_launch_and_generation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root = tmp_path / "held"
    root.with_name(root.name + ".preparation_STOP").write_text("Do not run until fixed")
    def forbidden(*args, **kwargs):
        pytest.fail("held operation must stop before subprocess/auth/evaluation")
    monkeypatch.setattr(B, "load_config", forbidden)
    args = SimpleNamespace(root=root, max_iterations=10, usd_cap=100, cpus=6)
    for function in (lambda: B.prepare(args), lambda: B.run(root), lambda: B.supervise(root),
                     lambda: T.invoke(root, "proposal_001_001", "fixture", [], effort="max", role="proposal",
                                      cap=100, upstream=forbidden, binary=pathlib.Path("/unused"))):
        with pytest.raises(B.R.OperatorStop, match="operator hold"):
            function()
    assert not root.exists()


def fake_root(tmp_path):
    root = tmp_path / "run"
    (root / "seed").mkdir(parents=True)
    (root / "prompts").mkdir()
    source = (B.REPO / P.MUTABLE).read_text()
    (root / "seed/atomic_controller.cpp").write_text(source)
    (root / "prompts/system_v1.md").write_text("fixed contract")
    (root / "prompts/compliance_v1.md").write_text("fixed review rubric")
    atomic_write_json(root / "codex_config.json", {"run_id": root.name, "usd_cap": 100, "maximum_iterations": 3,
        "cpu_budget": 6, "effort": "max", "policy": B.POLICY})
    atomic_write_json(root / "run_manifest.json", {"started_at": __import__("time").time()})
    metrics = {"aggregate": {"cycle_macro_mae_pct": 50, "request_macro_mae_over_L": .8}}
    atomic_write_json(root / "training/reports/seed.json", {"models": {"seed": metrics}})
    parent = {"id": "seed", "source_path": str(root / "seed/atomic_controller.cpp"), "sha256": P.sha(source)}
    return root, parent


def test_rejected_draft_is_repaired_by_model_within_same_iteration(tmp_path, monkeypatch):
    root, parent = fake_root(tmp_path)
    monkeypatch.setattr(B, "prompt", lambda *args: {"role": "user", "content": "own run only"})
    proposals = iter([{"status": "proposal", "n": 1}, {"status": "proposal", "n": 2}])
    conversations = []
    def call(*args, **kwargs):
        conversations.append(json.loads(json.dumps(args[3])))
        return next(proposals)
    monkeypatch.setattr(B, "call", call)
    def draft(root, parent, proposal, directory, label):
        if proposal["n"] == 1:
            raise ValueError("source violates contract")
        return {"sha256": "model_owned_source", "label": label}
    monkeypatch.setattr(B, "draft", draft)
    state = {"candidates": {"seed": parent}}
    result = B.evolve_one(root, state, "seed", 1)
    assert result["status"] == "evaluated"
    assert [d["status"] for d in result["drafts"]] == ["rejected", "valid"]
    assert "source violates contract" in json.dumps(conversations[1])
    assert B.evolve_one(root, state, "seed", 1) == result


def test_operational_failure_does_not_commit_iteration_or_test(tmp_path, monkeypatch):
    root, _ = fake_root(tmp_path)
    monkeypatch.setattr(B, "verify", lambda *args: {})
    monkeypatch.setattr(B, "event", lambda *args, **kwargs: None)
    def failed(*args):
        raise B.R.OperationalPause("offline transport outage", retryable=False)
    monkeypatch.setattr(B, "evolve_one", failed)
    with pytest.raises(B.R.OperationalPause):
        B.search(root)
    saved = B.R.read_json(root / "state.json")
    assert saved["history"] == [] and saved["active_parent"] == "seed"
    assert not (root / "selection_frozen.json").exists() and not (root / "test_started.json").exists()


def test_pareto_promotion_and_terminal_checkpoint(tmp_path, monkeypatch):
    root, parent = fake_root(tmp_path)
    monkeypatch.setattr(B, "verify", lambda *args: {})
    monkeypatch.setattr(B, "event", lambda *args, **kwargs: None)
    scores = [(40, .7), (30, .75)]  # First dominates; second is a tradeoff.
    def evolve(root, state, parent_id, iteration):
        if iteration == 3:
            return {"status": "no_change", "reason": "fixture", "drafts": []}
        a, b = scores[iteration - 1]
        candidate = {**parent, "metrics": {"aggregate": {"cycle_macro_mae_pct": a, "request_macro_mae_over_L": b}}}
        return {"status": "evaluated", "candidate": candidate, "proposal": {}, "drafts": []}
    monkeypatch.setattr(B, "evolve_one", evolve)
    result = B.search(root)
    assert result["incumbent"] == "astra_001"
    assert result["history"][1]["promoted"] is False
    assert result["status"] == "frozen" and result["termination"] == "no_change"
    with pytest.raises(RuntimeError, match="frozen"):
        B.search(root)


def test_final_test_opens_only_after_immutable_freeze(tmp_path, monkeypatch):
    root, parent = fake_root(tmp_path)
    calls = []
    def evaluate(*args, **kwargs):
        assert (root / "selection_frozen.json").exists() and (root / "test_started.json").exists()
        assert kwargs["split"] == "test" and kwargs["resume"]
        calls.append(kwargs)
        return {"selected_max": {"aggregate": {}}}
    monkeypatch.setattr(B.E, "evaluate", evaluate)
    monkeypatch.setattr(B, "event", lambda *args, **kwargs: None)
    state = {"termination": "infrastructure_error", "selected": {**parent, "plugin": "fixture"},
             "incumbent": "seed", "frozen_at": 123}
    with pytest.raises(RuntimeError, match="scientific"):
        B.finish(root, state)
    assert not calls
    state["termination"] = "iteration_limit"
    assert B.finish(root, state) == {"aggregate": {}}
    assert len(calls) == 2
    state["selected"]["sha256"] = "changed"
    with pytest.raises(RuntimeError, match="selection"):
        B.finish(root, state)


def test_review_is_source_bound_and_has_no_scores_or_history(tmp_path, monkeypatch):
    root, parent = fake_root(tmp_path)
    directory = root / "review_fixture"
    directory.mkdir()
    source = pathlib.Path(parent["source_path"]).read_text()
    proposal = {field: "technical explanation" for field in B.C.EXPLANATION_FIELDS}
    proposal.update(evidence="HIDDEN_SCORE_CANARY", history="HIDDEN_HISTORY_CANARY")
    calls = []
    def reviewer(*args):
        calls.append(args)
        assert args[-1] == "review"
        assert "CANARY" not in json.dumps(args[3])
        return {"source_sha256": P.sha(source), "verdict": "pass",
                "checks": {k: {"verdict": "pass", "reason": "fixture source check"} for k in B.C.CHECKS}}
    monkeypatch.setattr(B, "call", reviewer)
    result = B.review(root, source, proposal, directory, "review_fixture")
    assert result["approved"] and result["effort"] == "xhigh" and not result["human_intervention"]
    assert B.review(root, source, proposal, directory, "review_fixture") == result and len(calls) == 1
    with pytest.raises(RuntimeError, match="binding"):
        B.review(root, source + "\n", proposal, directory, "review_fixture")
