import difflib
import json
import pathlib
import subprocess
import sys

import pytest

from tools.chia_loop import real_core as P
from tools.chia_loop.core import atomic_write_json

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_reservation_survives_restart_and_refuses_overspend(tmp_path):
    ledger = P.Ledger(tmp_path / "ledger.json", "pro")
    for _ in range(40):
        try:
            ledger.reserve("input", 1, 1)
        except P.BudgetExhausted:
            break
    again = P.Ledger(tmp_path / "ledger.json", "pro")
    assert again.totals()["cap_charge_usd"] <= 50
    with pytest.raises(P.BudgetExhausted):
        again.reserve("x" * P.MAX_CONTEXT_BYTES, 1, 1)
    assert again.totals()["unknown_usage_calls"] > 0


def test_usage_counts_thinking_once_and_keeps_unknown(tmp_path):
    ledger = P.Ledger(tmp_path / "ledger.json", "pro")
    call = ledger.reserve("hello", 1, 1)
    ledger.settle(call, {"prompt_token_count": 1000, "candidates_token_count": 200,
                        "thoughts_token_count": 800, "total_token_count": 2000})
    assert ledger.totals()["estimated_standard_usd"] == pytest.approx(0.014)
    assert ledger.totals()["output_tokens_including_thinking"] == 1000
    next_call = ledger.reserve("hello", 1, 2)
    before = ledger.totals()["cap_charge_usd"]
    ledger.settle(next_call, None, "timeout")
    assert ledger.totals()["cap_charge_usd"] == before


def test_provider_maximum_is_not_unlimited_and_reservation_covers_it(tmp_path):
    assert P.MAX_OUTPUT == 65_536
    ledger = P.Ledger(tmp_path / "ledger.json", "flash")
    call = ledger.reserve("request", 1, 1)
    ledger.settle(call, {"prompt_token_count": 1000, "candidates_token_count": 5536,
                        "thoughts_token_count": 60_000, "total_token_count": 66_536})
    assert ledger.totals()["output_tokens_including_thinking"] == 65_536
    assert ledger.totals()["estimated_standard_usd"] == pytest.approx(0.24651)


def test_token_count_bounds_and_http_error_accounting(tmp_path):
    ledger = P.Ledger(tmp_path / "ledger.json", "flash")
    call = ledger.reserve("hello", 1, 1, input_tokens=1000)
    ledger.settle(call, None, "429", http_status=429)
    assert ledger.totals()["cap_charge_usd"] == 0
    assert ledger.totals()["unknown_usage_calls"] == 0
    with pytest.raises(P.BudgetExhausted, match="token"):
        ledger.reserve("hello", 1, 2, input_tokens=P.MAX_INPUT_TOKENS + 1)
    with pytest.raises(P.BudgetExhausted):
        ledger.reserve("hello", 1, 2, input_tokens=True)


def test_budget_carryover_cannot_replenish_authorization(tmp_path):
    ledger = P.Ledger(tmp_path / "ledger.json", "pro")
    ledger.initialize_carryover({"cap_charge_usd": 49, "estimated_standard_usd": 25, "api_attempts": 10})
    with pytest.raises(P.BudgetExhausted):
        ledger.reserve("hello", 1, 1, input_tokens=10)
    assert ledger.totals()["cap_charge_usd"] == 49
    assert ledger.totals()["api_attempts"] == 0
    assert ledger.totals()["carryover_api_attempts"] == 10
    with pytest.raises(ValueError):
        ledger.initialize_carryover({"cap_charge_usd": 0})


def test_region_body_submission_preserves_scaffold_and_supports_parameters():
    seed = (REPO / P.MUTABLE).read_text()
    before, body = P.regions(seed)
    code = body["CODE"].replace("void init_model() {}", 'double delay; void init_model() { delay = model_param("delay", 7, 1, 20); }')
    source = P.assemble_regions(seed, {"includes": "#include <algorithm>\n", "code": code})
    assert P.regions(source)[0] == before
    assert P.validate_source(seed, source)["static_boundary_pass"]
    with pytest.raises(ValueError, match="boundary markers"):
        P.assemble_regions(seed, {"includes": "", "code": "// CHIA_MODEL_BEGIN\n"})
    with pytest.raises(ValueError, match="exactly"):
        P.assemble_regions(seed, {"includes": "", "code": code, "scaffold": "oops"})
    with pytest.raises(ValueError, match="forbidden"):
        P.validate_source(seed, source.replace('delay = model_param("delay", 7, 1, 20)',
            'RAMULATOR_PARSE_PARAM(delay, float, "trace_path").default_val(7)'))


def test_truncated_json_is_never_applied_even_if_syntactically_complete():
    raw = {"candidates": [{"finish_reason": "MAX_TOKENS", "content": {"parts": [
        {"text": '{"status":"proposal","patch":"partial"}'}]}}]}
    result = P.parse_provider_response(raw)
    assert result["failure_kind"] == "generation_truncated"
    assert "patch" not in result


def test_provider_stop_parses_answer_only_and_handles_blocked_content():
    raw = {"candidates": [{"finish_reason": "STOP", "content": {"parts": [
        {"text": "reasoning summary", "thought": True}, {"text": '{"status":"no_change"}'}]}}]}
    assert P.parse_provider_response(raw) == {"status": "no_change"}
    assert P.parse_provider_response({"candidates": [{"finish_reason": "SAFETY", "content": None}]})["failure_kind"] == "provider_finish"
    assert P.parse_provider_response({"candidates": []})["failure_kind"] == "missing_candidate"
    raw["candidates"][0]["content"]["parts"] = [{"text": "[]"}]
    assert P.parse_provider_response(raw)["failure_kind"] == "invalid_json"


def test_completed_trace_archives_preserve_gzip_diagnostics(tmp_path):
    from tools.chia_loop import real_eval as E
    for label in ("oracle", "seed"):
        directory = tmp_path / "training/simpleo3/DDR5/429.mcf" / label
        directory.mkdir(parents=True)
        atomic_write_json(directory / "manifest.json", {"insts_per_core": 20_000_000})
        for name in ("trace.csv.ch0", "controller_trace.csv.ch0"):
            (directory / name).write_text("type,arrive,depart\n0,10,30\n")
        E.archive_completed_run(directory)
        assert not (directory / "trace.csv.ch0").exists()
    assert E.verify_run_archives(tmp_path) == 4
    result = E.training_diagnostics(tmp_path, "seed", "429.mcf", "logical", 1)
    assert result["oracle"] == result["seed"] == [{"type": 0, "arrive": 10, "depart": 30}]
    with pytest.raises(RuntimeError, match="incomplete"):
        E.archive_completed_run(tmp_path / "incomplete")


def test_source_regions_protect_lifecycle_and_deny_io():
    seed = (REPO / P.MUTABLE).read_text()
    changed = seed.replace("return m_clk + m_latency;", "return m_clk + 2 * m_latency;")
    assert P.validate_source(seed, changed)["static_boundary_pass"]
    with pytest.raises(ValueError, match="protected lifecycle"):
        P.validate_source(seed, changed.replace("m_pending.pop();", ""))
    with pytest.raises(ValueError, match="forbidden"):
        P.validate_source(seed, changed.replace("return m_clk + 2 * m_latency;", 'std::ifstream f("input"); return 1;'))


def test_exact_patch_and_extra_paths():
    seed = (REPO / P.MUTABLE).read_text()
    changed = seed.replace("return m_clk + m_latency;", "return m_clk + 2 * m_latency;")
    diff = "".join(difflib.unified_diff(seed.splitlines(True), changed.splitlines(True),
                                      fromfile="a/"+P.MUTABLE, tofile="b/"+P.MUTABLE))
    assert P.apply_unified(seed, diff) == changed
    with pytest.raises(ValueError):
        P.apply_unified(seed, diff + "--- a/other\n+++ b/other\n@@ -1 +1 @@\n-x\n+y\n")


def test_landlock_and_network_fail_closed(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "private_sentinel"
    secret.write_text("must not be visible")
    script = allowed / "probe.py"
    script.write_text("import pathlib,socket\n"
        f"p=pathlib.Path({str(secret)!r})\n"
        "try: p.read_text()\n"
        "except PermissionError: pass\n"
        "else: raise AssertionError('read allowed')\n"
        "try: p.write_text('oops')\n"
        "except PermissionError: pass\n"
        "else: raise AssertionError('write allowed')\n"
        "try: socket.socket()\n"
        "except PermissionError: pass\n"
        "else: raise AssertionError('network allowed')\n"
        "print('isolated')\n")
    policy = tmp_path / "policy.json"
    atomic_write_json(policy, {"read": ["/usr", "/lib", "/lib64", str(allowed)],
        "write": [str(allowed)], "cwd": str(allowed), "cpu_seconds": 10,
        "memory_bytes": 256 * 1024**2})
    result = subprocess.run([sys.executable, str(REPO / "tools/chia_loop/sandbox.py"),
                             str(policy), "--", "/usr/bin/python3", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "isolated" in result.stdout
    assert secret.read_text() == "must not be visible"


def test_training_tool_rejects_heldout_and_path_traversal(tmp_path):
    from tools.chia_loop.gemini_loop import inspect_tool
    with pytest.raises(ValueError):
        inspect_tool(tmp_path, {}, {"tool": "read_file", "path": "../../doc/atomic_chia_hackathon_status.md"})
    with pytest.raises(ValueError):
        inspect_tool(tmp_path, {"label": "seed"}, {"tool": "training_diagnostics", "workload": "603.bwaves_s-1080B"})


def test_source_search_and_training_time_paging(tmp_path):
    from tools.chia_loop.gemini_loop import inspect_tool
    from tools.chia_loop import real_eval as E
    source = tmp_path / "atomic.cpp"
    source.write_text("one\nneedle\nthree\n")
    result = inspect_tool(tmp_path, {"source_path": str(source)},
        {"tool": "search_file", "path": P.MUTABLE, "query": "NEEDLE"})
    assert result["total_matches"] == 1
    for label in ("oracle", "seed"):
        directory = tmp_path / "training/simpleo3/DDR5/429.mcf" / label
        directory.mkdir(parents=True)
        (directory / "trace.csv.ch0").write_text("type,arrive,depart\n0,10,30\n1,20,60\n1,25,70\n0,100,120\n")
    result = E.training_diagnostics(tmp_path, "seed", "429.mcf", "logical", 10,
        arrival_min=15, arrival_max=50, request_type=1, start_row=1)
    assert result["oracle"] == result["seed"] == [{"type": 1, "arrive": 25, "depart": 70}]


def test_real_loop_repairs_drafts_inside_one_evaluated_iteration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from google import genai
    from google.genai import types
    from tools.chia_loop import gemini_loop as G
    (tmp_path / "seed").mkdir()
    (tmp_path / "seed/atomic_controller.cpp").write_text("seed")
    answers = iter([None, {"status": "proposal"}, {"status": "proposal"}])
    calls = []
    class Models:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(total_tokens=100)
        def generate_content(self, **kwargs):
            assert all(content.parts for content in kwargs["contents"])
            calls.append(kwargs)
            answer = next(answers)
            if answer is None:
                return types.GenerateContentResponse.model_validate({"candidates": [{
                    "finish_reason": "MALFORMED_FUNCTION_CALL", "content": {"role": "model"}}],
                    "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 50, "total_token_count": 150}})
            return types.GenerateContentResponse.model_validate({"candidates": [{
                "finish_reason": "STOP", "content": {"role": "model", "parts": [{"text": json.dumps(answer)}]}}],
                "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 50, "total_token_count": 150}})
    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(models=Models(), close=lambda: None))
    drafts = []
    def evaluate(*args, **kwargs):
        drafts.append(args)
        if len(drafts) == 1:
            raise ValueError("mock compiler diagnostic")
        return {"sha256": "valid", "metrics": {}}
    monkeypatch.setattr(G, "evaluate_draft", evaluate)
    result = G.propose._chia_original(str(tmp_path), "flash", 1, "contract", "initial prompt", {})
    assert result["status"] == "evaluated"
    assert [d["status"] for d in result["drafts"]] == ["rejected", "valid"]
    assert len(calls) == 3
    assert "mock compiler diagnostic" in calls[2]["contents"][-1].parts[0].text
    assert P.Ledger(tmp_path / "ledger.json", "flash", run_id=tmp_path.name).totals()["api_attempts"] == 3
    assert not (tmp_path / "arms").exists()


@pytest.mark.parametrize("mode,expected_status,expected_calls", [
    ("recover", "no_change", 2),
    ("repeat_failure", "failed", 2),
    ("budget", "budget_stop", 1),
    ("attempt_cap", "failed", 1),
    ("stop", "stopped", 1),
    ("local_error", "failed", 1),
])
def test_transport_retry_preserves_caps_context_and_stop(tmp_path, monkeypatch,
                                                       mode, expected_status, expected_calls):
    import httpx
    from types import SimpleNamespace
    from google import genai
    from google.genai import types
    from tools.chia_loop import gemini_loop as G
    from tools.chia_loop.run_records import validate_limits

    (tmp_path / "seed").mkdir()
    (tmp_path / "seed/atomic_controller.cpp").write_text("seed")
    atomic_write_json(tmp_path / "preparation_manifest.json", {"limits": validate_limits(25, 100, 6)})
    ledger = G.run_ledger(tmp_path, "flash")
    if mode == "budget":
        ledger.initialize_carryover({"cap_charge_usd": 98, "estimated_standard_usd": 40, "api_attempts": 10})
    if mode == "attempt_cap":
        monkeypatch.setitem(G.POLICY, "api_attempts_per_proposal", 1)
    calls = []
    closed = []
    class Models:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(total_tokens=100)
        def generate_content(self, **kwargs):
            calls.append(json.dumps([c.model_dump(mode="json") for c in kwargs["contents"]]))
            if mode == "local_error":
                raise ValueError("invalid local request")
            if mode == "stop":
                (tmp_path / "STOP").write_text("operator stop")
            if len(calls) == 1 or mode == "repeat_failure":
                raise httpx.RemoteProtocolError("incomplete chunked read")
            return types.GenerateContentResponse.model_validate({"candidates": [{
                "finish_reason": "STOP", "content": {"role": "model", "parts": [
                    {"text": '{"status":"no_change","reason":"mock completed response"}'}]}}],
                "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 50, "total_token_count": 150}})
    def client(**kwargs):
        assert kwargs["http_options"].retry_options.attempts == 1
        return SimpleNamespace(models=Models(), close=lambda: closed.append(True))
    monkeypatch.setattr(genai, "Client", client)
    monkeypatch.setattr(G.time, "sleep", lambda _: None)
    result = G.propose._chia_original(str(tmp_path), "flash", 1, "contract", "initial prompt", {})
    records = json.loads((tmp_path / "ledger.json").read_text())["calls"]
    assert result["status"] == expected_status
    assert len(calls) == len(records) == expected_calls
    assert closed == [True]
    assert all(r["iteration"] == r["turn"] == 1 for r in records)
    assert len(set(calls)) == 1  # retry the same context; never splice in partial output
    assert records[0]["state"] == "usage_unknown_reservation_retained"
    assert records[0]["cap_charge_usd"] == records[0]["reserved_usd"]
    assert ledger.totals()["cap_charge_usd"] <= 100
    if mode == "recover":
        assert records[1]["state"] == "usage_recorded"
        assert ledger.totals()["unknown_usage_calls"] == 1
    elif mode == "repeat_failure":
        assert ledger.totals()["unknown_usage_calls"] == 2


def test_only_transient_provider_failures_are_retryable():
    import httpx
    from tools.chia_loop.gemini_loop import transient_generation_error
    for error in (httpx.ConnectError("offline"), httpx.ReadTimeout("timeout"),
                  httpx.WriteError("connection lost"), httpx.RemoteProtocolError("truncated")):
        assert transient_generation_error(error)
    for error in (httpx.LocalProtocolError("bad request"), ValueError("invalid"), OSError("disk full")):
        assert not transient_generation_error(error)
    class ProviderError(Exception):
        def __init__(self, code):
            self.code = code
    assert transient_generation_error(ProviderError(503))
    assert not transient_generation_error(ProviderError(401))


def test_script_entrypoint_chia_functions_are_serializable():
    # Imported functions pickle by reference and hide this class of bug.
    # CLI-defined functions must serialize their referenced globals by value.
    import runpy
    import ray.cloudpickle as cloudpickle
    namespace = runpy.run_path(str(REPO / "tools/chia_loop/gemini_loop.py"), run_name="serialization_probe")
    for name in ("propose", "build", "score"):
        restored = cloudpickle.loads(cloudpickle.dumps(namespace[name]._chia_original))
        assert callable(restored)


def test_no_weighted_tradeoff_promotion():
    assert P.dominates((1, 2), (2, 3))
    assert not P.dominates((1, 4), (2, 3))
    assert not P.dominates((1, 2), (1, 2))


def test_real_windows_default_to_full_roi_and_reject_short_windows(tmp_path):
    from tools.chia_loop import real_eval as E
    assert set(E.TRAIN).isdisjoint(E.TEST)
    assert len(E.TEST) == 4
    assert E.evaluation_insts(tmp_path) == 20_000_000
    atomic_write_json(tmp_path / "window_policy.json", {"instructions_per_core": 40_000_000})
    assert E.evaluation_insts(tmp_path) == 40_000_000
    for value in (50_000, 19_999_999, True, 20_000_000.0):
        atomic_write_json(tmp_path / "window_policy.json", {"instructions_per_core": value})
        with pytest.raises(ValueError, match="20,000,000"):
            E.evaluation_insts(tmp_path)


def test_streamed_candidate_departure_audit(tmp_path):
    from tools.chia_loop.real_eval import audit_candidate_trace
    trace = tmp_path / "controller_trace.csv.ch0"
    trace.write_text("type,arrive,depart\n0,10,30\n1,20,60\n0,50,90\n")
    stats = {"num_read_reqs": 2, "num_write_reqs": 1, "read_latency": 60}
    audit_candidate_trace(trace, stats)
    with pytest.raises(RuntimeError, match="latency sum"):
        audit_candidate_trace(trace, {**stats, "read_latency": 59})
    with pytest.raises(RuntimeError, match="count mismatch"):
        audit_candidate_trace(trace, {**stats, "num_write_reqs": 2})
    trace.write_text("type,arrive,depart\n0,10,10\n")
    with pytest.raises(RuntimeError, match="non-positive"):
        audit_candidate_trace(trace, stats)


def test_traffic_adequacy_counts_dram_owners_not_llc_hits(tmp_path):
    from tools.chia_loop.traffic import traffic_population
    atomic_write_json(tmp_path / "manifest.json", {
        "per_core_cycles": [100], "workload": "fixture", "wall_s": 1,
        "controller_stats": {"num_read_reqs": 1, "num_write_reqs": 0}})
    (tmp_path / "trace.csv.ch0").write_text(
        "type,arrive,depart,llc_path\n0,1,2,0\n0,5,40,2\n1,3,4,2\n")
    result = traffic_population(tmp_path, 10_000)
    assert result["logical_reads"] == 2
    assert result["owners"] == 1
    assert not result["owner_count_pass"]
    assert not result["sustained_traffic_pass"]
    assert result["L"] == 18


def test_paid_runner_rejects_stale_short_training_before_calls(tmp_path, monkeypatch):
    from tools.chia_loop import gemini_loop as G
    atomic_write_json(tmp_path / "preflight_pass.json", {})
    directory = tmp_path / "training/simpleo3/DDR5/429.mcf/oracle"
    directory.mkdir(parents=True)
    atomic_write_json(directory / "manifest.json", {"insts_per_core": 50_000})
    monkeypatch.setattr(sys, "argv", ["gemini_loop.py", "--root", str(tmp_path), "--model", "pro"])
    with pytest.raises(RuntimeError, match="prepared training ROI"):
        G.main()


def test_ledger_cannot_mix_models_or_run_identities(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = P.Ledger(path, "pro", run_id="pro-trial-001")
    ledger.reserve("hello", 1, 1, input_tokens=10)
    original = path.read_bytes()
    data = json.loads(original)
    assert data["model"] == P.MODELS["pro"]
    assert P.MODELS["flash"] not in json.dumps(data)
    assert "flash" not in data["pricing"] and "pro" not in data["pricing"]
    for backend, run_id in (("flash", "pro-trial-001"), ("pro", "pro-trial-002")):
        other = P.Ledger(path, backend, run_id=run_id)
        with pytest.raises(ValueError, match="different"):
            other.totals()
        with pytest.raises(ValueError, match="different"):
            other.reserve("hello", 1, 1, input_tokens=10)
        assert path.read_bytes() == original


def test_explicit_higher_budget_is_pinned_and_cannot_reset(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = P.Ledger(path, "flash", run_id="extended", cap_usd=100)
    ledger.initialize_carryover({"cap_charge_usd": 99, "estimated_standard_usd": 40})
    with pytest.raises(P.BudgetExhausted, match="100"):
        ledger.reserve("hello", 1, 1, input_tokens=10)
    with pytest.raises(ValueError, match="cannot change"):
        P.Ledger(path, "flash", run_id="extended", cap_usd=200).reserve("hello", 1, 1)
    assert P.Ledger(path, "flash", run_id="extended", cap_usd=100).totals()["cap_charge_usd"] == 99


@pytest.mark.parametrize("backend", ["pro", "flash"])
def test_single_model_main_records_own_freeze_and_test_only(tmp_path, monkeypatch, backend):
    """Exercise orchestration with no provider, compiler, Ray worker, or simulator."""
    from types import SimpleNamespace
    from tools.chia_loop import gemini_loop as G
    atomic_write_json(tmp_path / "preflight_pass.json", {})
    atomic_write_json(tmp_path / "preparation_manifest.json", {
        "run_id": tmp_path.name, "model": P.MODELS[backend], "seed_sha256": "seed",
        "limits": {"maximum_iterations": 25, "usd_cap": 100, "cpu_budget": 6}})
    for workload in G.E.TRAIN:
        for label in ("oracle", "seed", *G.E.COMPARISONS):
            directory = tmp_path / "training/simpleo3/DDR5" / workload / label
            directory.mkdir(parents=True)
            atomic_write_json(directory / "manifest.json", {"insts_per_core": 20_000_000})
    state = {"run_id": tmp_path.name, "model": P.MODELS[backend], "status": "frozen",
        "incumbent": backend + "_001", "frozen_at": 1,
        "selected": {"sha256": "candidate", "plugin": "mock-plugin"}}
    evolved, evaluated = [], []
    def evolve(root, selected_backend):
        evolved.append(selected_backend)
        return state
    def evaluate(root, *args, **kwargs):
        frozen = json.loads((tmp_path / "selection_frozen.json").read_text())
        assert frozen["run_id"] == tmp_path.name and frozen["source_sha256"] == "candidate"
        evaluated.append(kwargs["split"])
        return {"aggregate": {"fixture": 1}}
    monkeypatch.setattr(G, "run_model", evolve)
    monkeypatch.setattr(G.E, "evaluate", evaluate)
    monkeypatch.setattr(G, "score", SimpleNamespace(chia_remote=lambda *args: evaluate(args[0], split=args[3])))
    monkeypatch.setattr(G, "get", lambda value: value)
    monkeypatch.setattr(G, "ray", SimpleNamespace(init=lambda **kwargs: None, shutdown=lambda: None))
    monkeypatch.setattr(G, "start_collector", lambda **kwargs: None)
    monkeypatch.setattr(G, "stop_collector", lambda: None)
    monkeypatch.setattr(G, "get_collector", lambda: None)
    monkeypatch.setattr(G.E, "verify_run_archives", lambda *args: 0)
    monkeypatch.setattr(G.artifacts, "compress", lambda *args, **kwargs: None)
    monkeypatch.setattr(G.artifacts, "verify", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["gemini_loop.py", "--root", str(tmp_path), "--model", backend])
    G.main()
    result = json.loads((tmp_path / "run_manifest.json").read_text())
    assert result["record_type"] == "optimization_run" and result["model"] == P.MODELS[backend]
    assert result["status"] == "completed" and result["state"] == state
    assert result["policy"]["maximum_iterations"] == 25 and result["policy"]["usd_cap"] == 100
    assert result["policy"]["cpu_budget"] == 6
    assert "arms" not in result and "models" not in result
    assert result["final_test_metrics"] == {"aggregate": {"fixture": 1}}
    assert evolved == [backend] and evaluated == ["test"] * 3
    assert not (tmp_path / "both_frozen.json").exists()
