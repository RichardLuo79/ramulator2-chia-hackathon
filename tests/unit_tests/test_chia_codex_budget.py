"""Financial continuity without importing predecessor scientific context."""
import fcntl
import json

import pytest

from tools.chia_loop import recovery as R, real_core as P
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.codex_cli import transport as T, budget as F, usage as U
from test_chia_codex_usage import response


def previous(tmp_path, effort="xhigh", cap=100):
    root = tmp_path / "previous"
    root.mkdir()
    atomic_write_json(root / "codex_config.json", {"run_id": root.name, "model": T.MODEL,
        "effort": effort, "usd_cap": cap, "private_context": "DO_NOT_IMPORT"})
    atomic_write_json(root / "supervisor_state.json", {"run_id": root.name, "status": "completed"})
    for name in (".runner.lock", ".supervisor.lock"):
        (root / name).touch()
    ledger = T.Ledger(root, cap)
    request = {"model": T.MODEL, "input": "PRIVATE_PROMPT", "reasoning": {"effort": effort}}
    call = ledger.reserve(request, "proposal", "proposal_001_001")
    ledger.settle(call, response())
    ledger.reserve(request, "proposal", "proposal_001_002")  # Unknown outcome stays charged.
    (root / "state.json").write_text("DO_NOT_IMPORT_SCORES_OR_TESTS")
    return root


def successor(tmp_path, old, *, effort="xhigh", cap=100, name="fresh"):
    root = tmp_path / name
    root.mkdir()
    digest = F.prepare(root, old, model=T.MODEL, effort=effort, cap=cap)
    atomic_write_json(root / "codex_config.json", {"run_id": root.name, "model": T.MODEL,
        "effort": effort, "usd_cap": cap, "budget_carryover_sha256": digest})
    T.Ledger(root, cap).transaction(lambda data: None)
    return root


def test_restart_carries_all_usage_not_designs_and_does_not_renumber_calls(tmp_path):
    old = previous(tmp_path)
    old_ledger = (old / "ledger.json").read_bytes()
    before = T.Ledger(old, 100).totals()
    root = successor(tmp_path, old)
    ledger = T.Ledger(root, 100)
    assert ledger.totals()["attempts"] == 0
    assert ledger.totals()["carryover_attempts"] == 2
    assert ledger.totals()["cap_charge_usd"] == before["cap_charge_usd"]
    assert ledger.totals()["known_standard_usd"] == before["known_standard_usd"]
    assert ledger.totals()["unknown_usage_calls"] == 1
    assert ledger.reserve({"model": T.MODEL, "input": "NEW"}, "proposal", "proposal_001_001") == 0
    report = U.report(root)
    assert report["totals"]["generation_attempts"] == 1
    assert report["authorization_totals"]["generation_attempts"] == 3
    assert report["authorization_totals"]["conservative_guard_usd"] == ledger.totals()["cap_charge_usd"]
    assert (old / "ledger.json").read_bytes() == old_ledger
    assert "PRIVATE" not in (root / "budget_carryover.json").read_text()
    assert "DO_NOT_IMPORT" not in json.dumps(report)
    assert not (root / "state.json").exists()


def test_existing_ceiling_not_reset_by_fresh_directory(tmp_path):
    old = previous(tmp_path, cap=20)
    root = successor(tmp_path, old, cap=20)
    ledger = T.Ledger(root, 20)
    ledger.reserve({"model": T.MODEL}, "proposal", "one")
    with pytest.raises(P.BudgetExhausted):
        ledger.reserve({"model": T.MODEL}, "proposal", "two")


def test_two_successors_cannot_reuse_one_remaining_allowance(tmp_path):
    old = previous(tmp_path)
    successor(tmp_path, old)
    with pytest.raises(RuntimeError, match="different successor"):
        successor(tmp_path, old, name="another")


def test_repeated_clean_restart_keeps_ancestral_charges_exactly_once(tmp_path):
    first = previous(tmp_path)
    second = successor(tmp_path, first)
    ledger = T.Ledger(second, 100)
    ledger.reserve({"model": T.MODEL}, "proposal", "new_operation")
    expected = ledger.totals()
    atomic_write_json(second / "supervisor_state.json", {"run_id": second.name, "status": "completed"})
    for name in (".supervisor.lock", ".runner.lock"):
        (second / name).touch()
    third = successor(tmp_path, second, name="third")
    totals = T.Ledger(third, 100).totals()
    assert totals["attempts"] == 0 and totals["carryover_attempts"] == 3
    assert totals["cap_charge_usd"] == expected["cap_charge_usd"]
    assert totals["unknown_usage_calls"] == 2
    assert U.report(third)["authorization_totals"]["generation_attempts"] == 3


def test_unfinished_financial_predecessor_requires_persistent_stop(tmp_path):
    old = previous(tmp_path)
    atomic_write_json(old / "supervisor_state.json", {"run_id": old.name, "status": "needs_attention"})
    with pytest.raises(RuntimeError, match="persistent STOP"):
        successor(tmp_path, old)
    (old / "STOP").write_text("operator stop")
    successor(tmp_path, old, name="authorized")


@pytest.mark.parametrize("change", ["effort", "cap", "model", "active", "locked"])
def test_wrong_or_running_financial_predecessor_is_rejected(tmp_path, change):
    old = previous(tmp_path)
    kwargs = {}
    if change == "effort":
        kwargs["effort"] = "max"
    elif change == "cap":
        kwargs["cap"] = 200
    elif change == "model":
        config = R.read_json(old / "codex_config.json")
        config["model"] = "another-model"
        atomic_write_json(old / "codex_config.json", config)
    elif change == "active":
        atomic_write_json(old / "supervisor_state.json", {"run_id": old.name, "status": "running"})
    with (old / ".runner.lock").open("r") as lock:
        if change == "locked":
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError):
            successor(tmp_path, old, **kwargs)


@pytest.mark.parametrize("change", ["delete", "amount"])
def test_pinned_carryover_cannot_be_removed_or_reduced(tmp_path, change):
    root = successor(tmp_path, previous(tmp_path))
    path = root / "budget_carryover.json"
    if change == "delete":
        path.unlink()
    else:
        receipt = R.read_json(path)
        receipt["totals"]["conservative_guard_usd"] = 0
        atomic_write_json(path, receipt)
    with pytest.raises(RuntimeError, match="carryover"):
        T.Ledger(root, 100).reserve({"model": T.MODEL}, "proposal", "one")
