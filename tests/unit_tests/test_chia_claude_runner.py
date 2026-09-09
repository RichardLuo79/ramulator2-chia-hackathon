"""Offline scientific-contract and privacy regression checks for Fable."""
import ast
import copy
import json
import pathlib
import socket
import subprocess
import sys
import time

import pytest

from tools.chia_loop.claude_cli import runner as B, transport as T, usage as U
from tools.chia_loop.codex_cli import runner as Original
from tools.chia_loop.core import atomic_write_json
from tests.unit_tests.test_chia_claude_cli import configure, credentials, fake_sse


def test_same_scientific_engine_as_astra():
    """Provider wiring is independent; the experiment isn't silently redesigned."""
    first = ast.parse(pathlib.Path(Original.__file__).read_text().replace("astra_", "fable_").replace("isolated_codex_cli", "isolated_claude_cli"))
    second = ast.parse(pathlib.Path(B.__file__).read_text())
    names = ("prompt", "review", "draft", "evolve_one", "search", "finish")
    for name in names:
        left = next(n for n in first.body if isinstance(n, ast.FunctionDef) and n.name == name)
        right = next(n for n in second.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(left) == ast.dump(right), name
    assert B.POLICY["reviewer_effort"] == "xhigh"
    assert B.POLICY["promotion"] == "strict_two_objective_Pareto"
    assert B.POLICY["native_claude_tools"] is False
    assert B.POLICY["human_intervention"] is False


def test_rich_profiles_do_not_cross_training_test_or_transfer():
    profile = B.W.validate(json.loads((B.REPO / "tools/chia_loop/configs/ddr5_frontend_transfer_v1.json").read_text()))
    assert profile["standard"] == "DDR5"
    assert profile["simpleo3"]["instructions_per_core"] == 20_000_000
    assert len(profile["simpleo3"]["training"]) == len(profile["simpleo3"]["test"]) == 8
    assert set(profile["simpleo3"]["training"]).isdisjoint(profile["simpleo3"]["test"])
    assert set(profile["transfer"]) == {"champsim", "gem5"}
    assert B.E.COMPARISONS == ["fixedlat", "md1", "wmg1", "mess"]
    loop = B.L.read_source(B.L.DEFAULT)
    assert loop["features"]["synthetic_diagnostics"]


def test_public_cli_requires_explicit_paid_authorization(tmp_path):
    result = subprocess.run([sys.executable, "-m", "tools.chia_loop.claude_cli", "run", "--root", str(tmp_path / "not_created")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "--authorize-paid" in result.stderr
    assert not (tmp_path / "not_created").exists()


def test_frozen_campaign_cannot_return_to_search(tmp_path):
    configure(tmp_path)
    atomic_write_json(tmp_path / "selection_frozen.json", {"source_sha256": "fixture"})
    with pytest.raises(RuntimeError, match="freeze"):
        B.search(tmp_path)


def test_no_holdout_or_other_run_in_prompt(tmp_path, monkeypatch):
    configure(tmp_path)
    B.W.install(tmp_path, B.REPO / "tools/chia_loop/configs/ddr5_frontend_transfer_v1.json")
    B.L.install(tmp_path)
    source = tmp_path / "seed.cpp"
    source.write_text("clean atomic skeleton")
    monkeypatch.setattr(B.L, "initial_diagnostics", lambda *a: {})
    monkeypatch.setattr(B.L, "comparisons", lambda *a: {})
    parent = {"sha256": "abc", "source_path": str(source), "metrics": {"aggregate": {}}}
    monkeypatch.setattr(B.L, "feedback_metrics", lambda *a: {})
    state = {"candidates": {"seed": parent}, "incumbent": "seed", "history": []}
    prompt = B.prompt(tmp_path, state, "seed", 1)
    content = json.loads(prompt["content"])
    assert content["training_configuration"]["workloads"] == B.W.workloads(tmp_path, "training")
    assert content["maximum_iterations"] == 20
    assert not any(w in prompt["content"] for w in B.W.workloads(tmp_path, "test"))
    assert content["human_modeling_hint"] is None


@pytest.mark.parametrize("change", ["model", "effort", "tools", "history", "system", "output", "per_message_effort"])
def test_request_mutations_fail_before_dispatch(tmp_path, change):
    # The fixture captured by the real-CLI tests establishes legitimate native
    # framing. Build an equivalent fully explicit request here, no host files.
    import datetime
    system, conversation = "experiment", [{"role": "user", "content": "data"}]
    date = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    reminder = ("<system-reminder>\nAs you answer the user's questions, you can use the following context:\n"
        "# currentDate\nToday's date is " + date + ".\n\n      IMPORTANT: this context may or may not be relevant "
        "to your tasks. You should not respond to this context unless it is highly relevant to your task.\n</system-reminder>\n\n")
    # Four contract blocks, each explicitly allowed. The identity/hash of actual
    # native boilerplate is covered separately by the real-CLI preflight.
    request = {"model": T.MODEL, "output_config": {"effort": "xhigh"}, "max_tokens": T.MAX_OUTPUT,
        "stream": True, "thinking": {"type": "adaptive"}, "tools": [],
        "system": [{"type": "text", "text": system}] * 4,
        "messages": [{"role": "user", "content": [{"type": "text", "text": reminder},
                     {"type": "text", "text": json.dumps(conversation, sort_keys=True)}]},
                     {"role": "system", "output_config": {"effort": "xhigh"},
                      "content": [{"type": "text", "text": "<total_tokens>15000000 tokens left</total_tokens>"}]}]}
    T.validate_request(request, "xhigh", conversation, system)
    if change == "model": request["model"] = "another-model"
    elif change == "effort": request["output_config"]["effort"] = "max"
    elif change == "tools": request["tools"] = [{"name": "Bash"}]
    elif change == "history": request["messages"].append({"role": "assistant", "content": "old run"})
    elif change == "system": request["system"].append({"type": "text", "text": "PRIVATE_RULE"})
    elif change == "per_message_effort": request["messages"][1]["output_config"]["effort"] = "max"
    else: request["max_tokens"] = 4096
    with pytest.raises(RuntimeError):
        T.validate_request(request, "xhigh", conversation, system)


def test_auth_is_read_only_and_expiry_cannot_be_bypassed(tmp_path):
    credential = tmp_path / "auth.json"
    credentials(credential)
    with pytest.raises(RuntimeError, match="expires"):
        T.auth.check(credential, now=time.time() + 86300)
    credential.chmod(0o644)
    with pytest.raises(RuntimeError, match="private"):
        T.auth.check(credential)


def test_bun_boundary_denies_other_processes_files_and_network(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    private = tmp_path / "private.txt"
    private.write_text("PRIVATE_HOST_DATA")
    (work / "escape").symlink_to(private)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        program = '''import pathlib,socket,os
assert pathlib.Path('/proc/self/maps').read_text()
for p in (PRIVATE, 'escape', '../private.txt', '/proc/self/environ', '/proc/self/mem', '/proc/1/maps', '/home/dev/.claude/.credentials.json'):
    try: pathlib.Path(p).read_bytes()
    except (PermissionError,FileNotFoundError): pass
    else: raise AssertionError('private access permitted')
for family,kind in ((socket.AF_UNIX,socket.SOCK_STREAM),(socket.AF_INET,socket.SOCK_DGRAM)):
    try: socket.socket(family,kind)
    except PermissionError: pass
    else: raise AssertionError('forbidden socket')
try: socket.create_connection(('127.0.0.1',443),timeout=1)
except PermissionError: pass
else: raise AssertionError('non-gateway port allowed')
with socket.create_connection(('127.0.0.1',PORT),timeout=1): pass
print('ISOLATION_PASS')
'''.replace("PRIVATE", repr(str(private))).replace("PORT", str(port))
        policy = {"read": ["/usr", "/lib", "/lib64", "/bin", str(work)], "write": [str(work)],
                  "cwd": str(work), "port": port, "environment": {"PATH": "/usr/bin:/bin"}}
        atomic_write_json(tmp_path / "boundary.json", policy)
        result = subprocess.run(["/usr/bin/python3", str(T.HERE / "boundary.py"), str(tmp_path / "boundary.json"),
                                 "--", "/usr/bin/python3", "-c", program], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert "ISOLATION_PASS" in result.stdout


def test_no_shared_mutation_or_automatic_launch_in_prepare():
    source = pathlib.Path(B.__file__).read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    calls = [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]
    assert not set(calls) & {"run", "supervise", "call", "T.invoke"}
    assert "E.compile_candidate" in calls and "E.evaluate" in calls


@pytest.mark.parametrize("paid", [False, True])
@pytest.mark.parametrize("credential_kind", T.auth.KINDS)
def test_queue_separates_preparation_from_generation(tmp_path, monkeypatch, paid, credential_kind):
    from types import SimpleNamespace
    from tools.chia_loop.claude_cli import queue as Q
    credential = tmp_path / "auth.json"
    credentials(credential, credential_kind)
    binary = tmp_path / "claude"
    binary.write_text("fixture")
    args = SimpleNamespace(root=tmp_path / "independent", effort="max", max_iterations=20, cpus=3,
        prepare_only=not paid, authorize_paid=paid, claude_binary=str(binary), auth_file=credential,
        credential_kind=credential_kind,
        evaluation_config=B.REPO / "tools/chia_loop/configs/ddr5_frontend_transfer_v1.json", loop_config=B.L.DEFAULT)
    monkeypatch.setattr(Q, "fingerprint", lambda *a: {"fixture": "pinned"})
    actions = []
    def prepare(got):
        actions.append("prepare")
        assert got.auth_mode == "claude_subscription" and got.usd_cap is None and got.iteration_guard
        assert got.credential_kind == credential_kind
        got.root.mkdir()
        Q.B.K.install(got.root, getattr(got, "prompt_cache", Q.B.K.DEFAULT))
    monkeypatch.setattr(Q.B, "prepare_with_wait", prepare)
    monkeypatch.setattr(Q.B, "verify", lambda *a: actions.append("verify"))
    def supervise(root):
        actions.append("generation")
        atomic_write_json(root / "supervisor_state.json", {"status": "completed"})
    monkeypatch.setattr(Q.B, "supervise", supervise)
    Q.execute(args)
    assert actions == ["prepare", "verify"] + (["generation"] if paid else [])
    record = Q.R.read_json(args.root.with_name(args.root.name + ".queue.json"))
    assert record["generation_authorized"] == paid and not record["model_generation_started"]
    assert record["model"] == T.MODEL and record["effort"] == "max"
    assert record["credential_kind"] == credential_kind


def test_mismatched_cli_usage_is_not_replayed_as_success(tmp_path, monkeypatch):
    configure(tmp_path)
    directory = tmp_path / "interactions/proposal_001_001/attempt_001"
    directory.mkdir(parents=True)
    atomic_write_json(directory / "receipt.json", {"cli_receipt_conflict": True})
    (directory / "provider_response.sse").write_bytes(fake_sse({"status": "no_change"}))
    with pytest.raises(T.R.OperationalPause, match="conflict"):
        T.invoke(tmp_path, "proposal_001_001", "system", [], effort="xhigh", role="proposal", cap=None,
                 upstream=lambda *a: pytest.fail("must not dispatch"), binary="/unneeded", auth_file="/unneeded")
