"""Finite-job mix membership, input identity, and per-core result collection."""

from pathlib import Path

import pytest

from ramulator_chia.framework.archive import Member, Payload
from ramulator_chia.framework.external_frontends import ExternalHost, TransferCase
from ramulator_chia.eval.champsim_mixes import MultiProgramCase, select_mixes
from ramulator_chia.eval.metrics import cycles_metrics


def trace(name, digest):
    member = Member(name + ".gz", digest * 64, 64, digest * 64, 64, "none")
    inventory = dict(stored_sha256=digest * 64, decoded_sha256=digest * 64,
                     decoded_bytes=23_000_000 * 64, record_bytes=64,
                     instructions=23_000_000, host_sha256="f" * 64,
                     complete_compressed_integrity=True, compression="gzip")
    return TransferCase("champsim", name, Payload(Path("/fixture") / member.name, member),
                        instruction_inventory=inventory, stage="test")


def mix(first, second):
    return MultiProgramCase(frontend="champsim", workload="mix", payload=first.payload,
                            instruction_inventory=first.instruction_inventory,
                            stage="test", companions=(second,))


def test_ordered_trace_identities_and_distinct_staging_names():
    a, b = trace("a", "a"), trace("b", "b")
    ab, ba = mix(a, b), mix(b, a)
    assert ab.identity() != ba.identity()
    assert [r["input_sha256"] for r in ab.identity()["programs"]] == ["a" * 64, "b" * 64]
    assert list(ab.input_files()) == ["core-0-trace.champsimtrace.gz", "core-1-trace.champsimtrace.gz"]
    assert ab.identity()["stopping"] == "finite-jobs-warmup-barrier-v1"
    assert ab.identity()["warmup_instructions"] == 2_000_000
    assert ab.identity()["roi_instructions"] == 20_000_000


def test_mix_selection_is_reproducible_covers_test_cohort_and_ignores_errors():
    rows = [dict(name=f"app{i}", family=f"app{i}", category="spec") for i in range(9)]
    rows += [dict(name=f"{family}{i}", family=family, category="google")
             for family in ("sierra", "tahoe") for i in range(8)]
    result = select_mixes(rows)
    assert result == select_mixes(list(reversed(rows)))
    assert len(result) == 16
    assert {n for r in result for n in r["programs"]} == {r["name"] for r in rows}
    for cores in (2, 4):
        members = [r for r in result if r["cores"] == cores]
        assert len(members) == 8
        assert [r["group"] for r in members].count("mixed") == 4
        assert all(len(r["programs"]) == len(set(r["programs"])) == cores for r in members)


def statistics(tmp_path, monkeypatch, rows, cores=2):
    monkeypatch.setattr(ExternalHost, "record", lambda _: {
        "frontend": "champsim", "settings": {"num_cores": cores}})
    (tmp_path / "simulation.log").write_text("\n".join(rows))
    observations = tmp_path / "observations"
    observations.mkdir()
    (observations / "controller.csv.ch0").write_text(
        "arrive,depart,type,source,addr\n0,10,0,0,64\n")
    return ExternalHost(tmp_path, "a" * 64).statistics(tmp_path)


def test_stats_are_ordered_by_cpu_not_finish_time(tmp_path, monkeypatch):
    stats = statistics(tmp_path, monkeypatch, [
        "Simulation finished CPU 1 instructions: 20000000 cycles: 100",
        "Simulation finished CPU 0 instructions: 20000002 cycles: 200",
    ])
    assert stats["frontend"]["per_core_cycles"] == [200, 100]
    assert stats["frontend"]["per_core_instructions"] == [20000002, 20000000]
    assert stats["frontend"]["cycles_or_ticks"] == 200
    assert mix(trace("a", "a"), trace("b", "b")).result_fields(stats)["per_core_cycles"] == [200, 100]


@pytest.mark.parametrize("ids", [[0], [0, 0], [0, 2]])
def test_missing_duplicate_or_wrong_core_is_not_success(tmp_path, monkeypatch, ids):
    with pytest.raises(ValueError, match="exactly one ROI per core"):
        statistics(tmp_path, monkeypatch, [
            f"Simulation finished CPU {i} instructions: 20000000 cycles: 100" for i in ids])


def test_positive_and_negative_core_errors_do_not_cancel():
    result = cycles_metrics({"per_core_cycles": [100, 200]}, {"per_core_cycles": [110, 180]})
    assert result["per_core_dev_pct"] == [10, -10]
    assert result["mean_abs_per_core_pct"] == 10
    assert result["makespan_dev_pct"] == -10
