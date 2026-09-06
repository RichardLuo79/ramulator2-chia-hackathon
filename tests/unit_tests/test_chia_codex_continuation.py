"""Offline coverage for iteration-only guards and training-only continuation."""
import copy
import json
import pathlib
import time

import pytest

from tools.chia_loop import recovery as R, real_core as P
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.codex_cli import budget as F, continuation as H, runner as B, transport as T, usage as U
from test_chia_codex_budget import previous
from test_chia_codex_cli import fake_root


def authorize(root, maximum=20):
    root.mkdir(exist_ok=True)
    config = {"run_id": root.name, "model": T.MODEL, "effort": "xhigh", "auth_mode": "chatgpt",
              "maximum_iterations": maximum, "cpu_budget": 6, "guard_mode": "iterations", "usd_cap": None}
    atomic_write_json(root / "codex_config.json", config)
    return config


def test_iteration_only_still_records_usage_and_unknown_reservations(tmp_path):
    authorize(tmp_path)
    ledger = T.Ledger(tmp_path, None)
    for i in range(15):
        assert ledger.reserve({"model": T.MODEL}, "proposal", f"proposal_{i+1:03d}_001") == i
    assert ledger.totals()["cap_charge_usd"] > 100
    assert ledger.totals()["unknown_usage_calls"] == 15
    report = U.write_report(tmp_path)
    assert report["guard_mode"] == "iterations" and report["guard_cap_usd"] is None
    assert report["totals"]["generation_attempts"] == 15
    assert "no USD stopping guard" in (tmp_path / "reports/llm_usage/summary.md").read_text()


@pytest.mark.parametrize("change", ["missing", "api", "mode", "iterations", "owner", "cap"])
def test_iteration_only_needs_explicit_bounded_chatgpt_authorization(tmp_path, change):
    if change != "missing":
        config = authorize(tmp_path)
        key, value = {"api": ("auth_mode", "api"), "mode": ("guard_mode", "usd"),
                      "iterations": ("maximum_iterations", 0), "owner": ("run_id", "another"),
                      "cap": ("usd_cap", 100)}[change]
        config[key] = value
        atomic_write_json(tmp_path / "codex_config.json", config)
    with pytest.raises(ValueError):
        T.Ledger(tmp_path, None)


def test_iteration_mode_cannot_be_removed_mid_ledger(tmp_path):
    config = authorize(tmp_path)
    ledger = T.Ledger(tmp_path, None)
    ledger.reserve({"model": T.MODEL}, "proposal", "proposal_001_001")
    config["guard_mode"] = "usd"
    atomic_write_json(tmp_path / "codex_config.json", config)
    with pytest.raises(ValueError):
        ledger.reserve({"model": T.MODEL}, "proposal", "proposal_001_002")


@pytest.mark.parametrize("maximum,cap,iteration_guard,auth_mode", [
    (20, None, False, "chatgpt"), (20, 100, True, "chatgpt"),
    (20, None, True, "api"), (0, None, True, "chatgpt"), (True, None, True, "chatgpt")])
def test_invalid_guard_combinations_fail(maximum, cap, iteration_guard, auth_mode):
    with pytest.raises(ValueError):
        F.limits(maximum, cap, 6, iteration_guard=iteration_guard, auth_mode=auth_mode)


def test_changed_guard_carries_usage_once_without_resetting_predecessor(tmp_path):
    old = previous(tmp_path)
    config = R.read_json(old / "codex_config.json")
    config["auth_mode"] = "chatgpt"
    atomic_write_json(old / "codex_config.json", config)
    before = {name: (old / name).read_bytes() for name in ("ledger.json", "state.json", "supervisor_state.json")}
    totals = T.Ledger(old, 100).totals()
    root = tmp_path / "continuation"
    config = authorize(root)
    digest = F.prepare(root, old, model=T.MODEL, effort="xhigh", cap=None, iteration_guard=True)
    config["budget_carryover_sha256"] = digest
    atomic_write_json(root / "codex_config.json", config)
    ledger = T.Ledger(root, None)
    ledger.transaction(lambda data: None)
    assert ledger.totals()["cap_charge_usd"] == totals["cap_charge_usd"]
    assert ledger.totals()["unknown_usage_calls"] == 1
    assert ledger.totals()["carryover_attempts"] == 2
    assert U.report(root)["authorization_totals"]["generation_attempts"] == 2
    assert all((old / name).read_bytes() == raw for name, raw in before.items())
    another = tmp_path / "another"
    authorize(another)
    with pytest.raises(RuntimeError, match="different successor"):
        F.prepare(another, old, model=T.MODEL, effort="xhigh", cap=None, iteration_guard=True)


def checkpoint():
    history = [{"iteration": n, "status": "evaluated", "parent": "seed", "drafts": []} for n in range(1, 8)]
    return {"history": history + [{"iteration": 8, "status": "budget_stop"}],
            "candidates": {k: {} for k in ["seed", *(f"astra_{n:03d}" for n in range(1, 8))]},
            "incumbent": "astra_004"}


def test_only_completed_prefix_is_carried_and_old_checkpoint_is_unchanged():
    state = checkpoint()
    original = copy.deepcopy(state)
    rows, retired = H.completed_prefix(state, 20)
    assert len(rows) == 7 and 20 - len(rows) == 13
    assert retired == [{"iteration": 8, "status": "budget_stop"}]
    rows[0]["status"] = "changed"
    assert state == original


@pytest.mark.parametrize("change", ["gap", "extra_candidate", "extra_stop", "non_budget_stop", "no_work"])
def test_bad_checkpoint_is_not_silently_trimmed(change):
    state = checkpoint()
    if change == "gap":
        state["history"][3]["iteration"] = 42
    elif change == "extra_candidate":
        state["candidates"]["astra_008"] = {}
    elif change == "extra_stop":
        state["history"].append({"iteration": 9, "status": "budget_stop"})
    elif change == "non_budget_stop":
        state["history"][-1]["status"] = "operational_pause"
    with pytest.raises((RuntimeError, ValueError)):
        H.completed_prefix(state, 7 if change == "no_work" else 20)


def test_metric_reproduction_ignores_runtime_but_not_tail_or_coverage():
    original = {"aggregate": {"cycle_macro_mae_pct": 2.1, "wall_s": 8, "speedup_vs_oracle": 2},
                "per_workload": {"training": {"wall_s": {"model": 2, "oracle": 3},
                    "requests": {"coverage_model": 1, "paired_tail_p99_over_L": 2.3}}}}
    replay = copy.deepcopy(original)
    replay["aggregate"]["wall_s"] = 500
    replay["per_workload"]["training"]["wall_s"]["model"] = 999
    assert H.accuracy(original) == H.accuracy(replay)
    replay["per_workload"]["training"]["requests"]["coverage_model"] = .99
    assert H.accuracy(original) != H.accuracy(replay)


def test_continuation_dispatches_iterations_8_through_20_not_20_new_ones(tmp_path, monkeypatch):
    root, seed = fake_root(tmp_path)
    config = R.read_json(root / "codex_config.json")
    config["maximum_iterations"] = 20
    atomic_write_json(root / "codex_config.json", config)
    state = checkpoint()
    state["history"] = state["history"][:-1]
    def metrics(n):
        return {"aggregate": {"cycle_macro_mae_pct": 50-n, "request_macro_mae_over_L": .8-n/100}}
    state["candidates"] = {key: {**seed, "metrics": metrics(0)} for key in state["candidates"]}
    state.update(run_id=root.name, status="running")
    atomic_write_json(root / "state.json", state)
    monkeypatch.setattr(B, "verify", lambda root: {})
    monkeypatch.setattr(B, "event", lambda *args, **kwargs: None)
    monkeypatch.setattr(U, "write_report", lambda *args: None)
    monkeypatch.setattr(B.V, "write_report", lambda *args: None)
    seen = []
    def evolve(root, state, parent_id, iteration):
        seen.append(iteration)
        return {"status": "evaluated", "candidate": {**seed, "metrics": metrics(iteration)},
                "drafts": [], "proposal": {}}
    monkeypatch.setattr(B, "evolve_one", evolve)
    final = B.search(root)
    assert seen == list(range(8, 21))
    assert len(final["history"]) == 20 and final["incumbent"] == "astra_020"
    assert final["termination"] == "iteration_limit" and final["status"] == "frozen"
    with pytest.raises(RuntimeError, match="frozen"):
        B.search(root)


def test_predecessor_path_cannot_escape_to_other_run(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    external = tmp_path / "private"
    external.write_text("not this run")
    (source / "symlink").symlink_to(external)
    with pytest.raises(RuntimeError, match="escapes"):
        H.own_file(source, source / "symlink")


def test_inherited_checkpoint_is_hash_pinned_and_immutable(tmp_path):
    config = authorize(tmp_path)
    rows, _ = H.completed_prefix(checkpoint(), 20)
    candidates = checkpoint()["candidates"]
    receipt = {"run_id": tmp_path.name, "model": T.MODEL, "effort": "xhigh", "maximum_iterations": 20,
               "completed_designs": 7, "history": rows, "candidates": candidates,
               "held_out_results_imported": False, "all_inherited_training_accuracy_reproduced": True}
    atomic_write_json(tmp_path / "continuation.json", receipt)
    config["continuation_sha256"] = P.sha((tmp_path / "continuation.json").read_bytes())
    state = {"history": rows, "candidates": candidates}
    atomic_write_json(tmp_path / "state.json", state)
    H.verify_import(tmp_path, config)
    state["history"][0]["status"] = "altered"
    atomic_write_json(tmp_path / "state.json", state)
    with pytest.raises(RuntimeError, match="provenance"):
        H.verify_import(tmp_path, config)
    atomic_write_json(tmp_path / "continuation.json", {**receipt, "held_out_results_imported": True})
    with pytest.raises(RuntimeError, match="changed"):
        H.verify_import(tmp_path, config)
