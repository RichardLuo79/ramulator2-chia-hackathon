import copy
import json

import pytest

from tools.chia_loop import evaluation_config as W, real_eval as E
from tools.chia_loop.core import atomic_write_json


def profile():
    return json.loads(W.DEFAULT.read_text())


def test_expanded_profile_and_legacy_are_separate(tmp_path):
    assert E.workloads(tmp_path, "training") == E.TRAIN
    config = W.install(tmp_path)
    assert len(E.workloads(tmp_path, "training")) == 8
    assert len(E.workloads(tmp_path, "test")) == 8
    assert len(config["transfer"]["champsim"]["workloads"]) == 6
    assert len(config["transfer"]["gem5"]["workloads"]) == 12
    with pytest.raises(RuntimeError, match="replace"):
        W.install(tmp_path)
    with pytest.raises(ValueError, match="split"):
        E.workloads(tmp_path, "transfer")


def test_family_split_catches_other_versions_and_intervals():
    config = profile()
    config["simpleo3"]["test"].append("605.mcf_s-1554B")
    with pytest.raises(ValueError, match="families overlap"):
        W.validate(config)
    assert W.family("605.mcf_s-1554B") == W.family("429.mcf")
    assert W.transfer_group(profile(), "605.mcf_s-1554B") == "training_family_new_frontend"
    assert W.transfer_group(profile(), "649.fotonik3d_s-1176B") == "simpleo3_test_family_new_frontend"
    assert W.transfer_group(profile(), "654.roms_s-1021B") == "unseen_family_new_frontend"


@pytest.mark.parametrize("change", [
    lambda c: c.update(standard="HBM4"),
    lambda c: c["simpleo3"].update(instructions_per_core=50_000),
    lambda c: c["simpleo3"].update(training=["../../other_run"]),
    lambda c: c["simpleo3"].update(training=["429.mcf", "429.mcf"]),
    lambda c: c["transfer"]["champsim"].update(roi_instructions=100),
    lambda c: c["transfer"]["gem5"].update(run_to_exit=False),
    lambda c: c["transfer"].update(mystery={}),
])
def test_invalid_profile_fails_closed(change):
    config = profile()
    change(config)
    with pytest.raises(ValueError):
        W.validate(config)


def test_window_and_diagnostics_use_run_profile(tmp_path):
    config = profile()
    config["simpleo3"]["training"] = ["462.libquantum"]
    atomic_write_json(tmp_path / "evaluation_config.json", config)
    atomic_write_json(tmp_path / "window_policy.json", {"instructions_per_core": 20_000_000})
    assert E.evaluation_insts(tmp_path) == 20_000_000
    for label in ("oracle", "seed"):
        atomic_write_json(tmp_path / "training/simpleo3/DDR5/462.libquantum" / label / "manifest.json", {"per_core_cycles": [1]})
    assert E.training_diagnostics(tmp_path, "seed", "462.libquantum", "stats")["oracle"]["per_core_cycles"] == [1]
    for workload in ("429.mcf", "433.milc", "605.mcf_s-1554B", "../../run"):
        with pytest.raises(ValueError):
            E.training_diagnostics(tmp_path, "seed", workload, "stats")
    atomic_write_json(tmp_path / "window_policy.json", {"instructions_per_core": 40_000_000})
    with pytest.raises(ValueError, match="differs"):
        E.evaluation_insts(tmp_path)


def test_inventory_refuses_aliases_and_wrapped_roi(tmp_path):
    config = copy.deepcopy(W.LEGACY)
    config["simpleo3"]["training"] = ["429.mcf"]
    config["simpleo3"]["test"] = ["433.milc"]
    atomic_write_json(tmp_path / "evaluation_config.json", config)
    trace = tmp_path / "input.trace"
    trace.write_text("30000000 64\n")
    with pytest.raises(RuntimeError, match="alias"):
        W.inventory(tmp_path, lambda _: str(trace), E.C.file_provenance)
    trace.write_text("10 64\n")
    with pytest.raises(RuntimeError, match="wrap"):
        W.inventory(tmp_path, lambda _: str(trace), E.C.file_provenance)
