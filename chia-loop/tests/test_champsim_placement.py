"""Placement-study selection and evidence are independent of model scores."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import pytest

from ramulator_chia.framework.archive import Member, Payload
from ramulator_chia.framework.evaluation import SimulationLimits
from ramulator_chia.eval.champsim_placement import balanced_mixes, select_matrix
from ramulator_chia.eval.champsim_mixes import MultiProgramCase, select_mixes
from test_champsim_mixes import trace


def test_balanced_mixes_ignore_input_order_and_cover_each_pool():
    for size in (14, 25):
        names = [f"program{i}" for i in range(size)]
        for cores in (4, 8):
            result = balanced_mixes(names, "validation", cores)
            assert result == balanced_mixes(names[::-1], "validation", cores)
            assert len(result) == 8
            assert len({tuple(sorted(m["programs"])) for m in result}) == 8
            counts = Counter(n for m in result for n in m["programs"])
            assert set(counts) == set(names)
            assert max(counts.values()) - min(counts.values()) <= 1
            assert all(len(set(m["programs"])) == cores for m in result)


def test_matrix_preserves_old_four_core_order_and_balances_hosts():
    rows = [dict(name=f"app{i}", family=f"app{i}", category="spec") for i in range(9)]
    rows += [dict(name=f"{f}{i}", family=f, category="google") for f in ("sierra", "tahoe") for i in range(8)]
    previous = dict(cases=rows, mixes=select_mixes(rows))
    matrix = select_matrix(previous, [f"val{i}" for i in range(14)])
    assert len(matrix) == 32
    assert Counter(m["site"] for m in matrix) == dict(local=8, cloud=24)
    assert [m["programs"] for m in matrix if m["stage"] == "test" and m["cores"] == 4] == [
        m["programs"] for m in previous["mixes"] if m["cores"] == 4]
    for stage in ("validation", "test"):
        for cores in (4, 8):
            group = [m for m in matrix if (m["stage"], m["cores"]) == (stage, cores)]
            assert Counter(m["site"] for m in group) == dict(local=2, cloud=6)


def test_placement_is_not_a_positional_trace_and_changes_identity():
    traces = [replace(trace(str(i), str(i)), stage="validation") for i in range(8)]
    mix = MultiProgramCase(frontend="champsim", workload="mix", stage="validation",
        payload=traces[0].payload, instruction_inventory=traces[0].instruction_inventory,
        companions=tuple(traces[1:]))
    placement = Payload(Path("/fixture/map.gz"), Member("placement.txt", "a"*64, 100, "b"*64, 300, "gzip"))
    fixed = replace(mix, placement=placement)
    assert len(fixed.trace_files()) == 8 and len(fixed.input_files()) == 9
    assert "placement.txt" not in fixed.trace_files()
    assert mix.identity() != fixed.identity()
    assert fixed.identity()["placement"]["sha256"] == "b"*64


def test_uncapped_simulation_retains_memory_and_file_guards():
    assert SimulationLimits(None, 4 << 30, 16 << 30, 1 << 20, 3).timeout_seconds is None
    with pytest.raises(ValueError):
        SimulationLimits(0, 4 << 30, 16 << 30, 1 << 20, 3)


def test_matching_concurrency_reserves_ram_and_keeps_margin():
    from ramulator_chia.eval.champsim_placement import matching_batch
    row = {"observation": {"traces": {"controller.csv.ch0": {"logical_bytes": 256 << 20}}}}
    results = {("mix", arm): row for arm in ("oracle", "a", "b")}
    pending = [("mix", "a"), ("mix", "b")]
    assert matching_batch(pending, results, 24 << 30) == pending
    assert matching_batch(pending, results, 23 << 30) == pending[:1]
    with pytest.raises(RuntimeError, match="insufficient RAM"):
        matching_batch(pending, results, 16 << 30)


@pytest.mark.parametrize("different", [False, True])
def test_cross_host_qualification_ignores_host_logs_but_not_timing(tmp_path, different):
    import json
    from ramulator_chia.eval.champsim_placement_report import qualify

    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    save("protocol.json", dict(pilots=["mix4", "mix8"], arms=["oracle", "fixedlat"]))
    save("placement-unit.json", dict(passed=True))
    for cores in (4, 8):
        save(f"fixtures{cores}/passed.json", {name: dict(matched=1, physical_mismatches=0)
            for name in ("streaming", "random", "mixed_rw", "page_crossing")})
        name = f"mix{cores}"
        for site in ("local", "cloud"):
            for arm in ("oracle", "fixedlat"):
                cycles = [100] * cores
                if different and site == "cloud" and arm == "oracle":
                    cycles[0] += 1
                save(f"{site}/completed/{name}/{arm}.json", dict(identity=dict(case=dict(num_cores=cores)), observation=dict(
                    complete=True, process=dict(host=site), frontend_stats=dict(per_core_instructions=[20_000_000]*cores,
                    per_core_cycles=cycles), controller_stats=dict(reads=10), traces={"controller.csv.ch0": dict(
                    logical_sha256="a"*64, logical_bytes=500)})))
        save(f"local/scores/fixedlat/{name}.json", dict(pairing_error=None, request=dict(matched=10),
             request_pairing=dict(address_mismatch_pairs=0)))
    save("fixtures8/negative-checks.json", {name: dict(returncode=1) for name in ("alias", "out_of_range", "missing_root", "capacity")})
    if different:
        with pytest.raises(ValueError, match="cross-host simulated results differ"):
            qualify(tmp_path)
        assert not (tmp_path / "qualification-passed.json").exists()
    else:
        assert qualify(tmp_path)["passed"] is True
        assert (tmp_path / "qualification-passed.json").exists()
