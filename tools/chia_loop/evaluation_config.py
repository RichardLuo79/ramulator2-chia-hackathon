"""Operator-owned DDR5 cohorts; never chosen by an optimization agent.

Old roots without evaluation_config.json keep the original 2/4 workload split.
New preparations explicitly install a versioned profile. Transfer cases never
participate in promotion or become diagnostic-tool inputs.
"""
from __future__ import annotations

import copy
import json
import pathlib
import re
import subprocess

from tools.chia_loop.core import atomic_write_json

DEFAULT = pathlib.Path(__file__).with_name("configs") / "ddr5_frontend_transfer_v1.json"
LEGACY = {
    "schema_version": 1, "name": "legacy_simpleo3_2train_4test", "standard": "DDR5",
    "simpleo3": {
        "training": ["429.mcf", "519.lbm"],
        "test": ["433.milc", "450.soplex", "459.GemsFDTD", "549.fotonik3d"],
        "instructions_per_core": 20_000_000, "minimum_oracle_owner_reads": 10_000,
    },
    "transfer": {},
}


def family(workload):
    """Group SPEC versions and trace intervals of the same application."""
    name = re.sub(r"^\d+\.", "", workload).lower()
    name = re.sub(r"-[0-9]+[a-z]$", "", name)
    return re.sub(r"_[rs]$", "", name)


def _names(value, field):
    if (not isinstance(value, list) or not value or
            any(not isinstance(w, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", w)
                for w in value) or len(set(value)) != len(value)):
        raise ValueError(field + " requires unique, nonempty safe workload names")
    return value


def _integer(value, minimum, field):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")


def validate(config):
    if not isinstance(config, dict) or set(config) != {"schema_version", "name", "standard", "simpleo3", "transfer"}:
        raise ValueError("unexpected evaluation configuration fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["standard"] != "DDR5":
        raise ValueError("this pipeline supports schema 1 and fixed DDR5 only")
    if not isinstance(config["name"], str) or not config["name"]:
        raise ValueError("evaluation profile needs a name")
    simple = config["simpleo3"]
    if not isinstance(simple, dict) or set(simple) != {"training", "test", "instructions_per_core", "minimum_oracle_owner_reads"}:
        raise ValueError("unexpected SimpleO3 configuration fields")
    train, test = (_names(simple[k], k) for k in ("training", "test"))
    overlap = {family(w) for w in train} & {family(w) for w in test}
    if overlap:
        raise ValueError("training/test application families overlap: " + ", ".join(sorted(overlap)))
    if any(w.startswith("Mix") for w in train + test):
        raise ValueError("real CHIA profiles currently require single-core SimpleO3 workloads")
    _integer(simple["instructions_per_core"], 20_000_000, "SimpleO3 instructions")
    _integer(simple["minimum_oracle_owner_reads"], 1, "minimum oracle reads")
    transfer = config["transfer"]
    if not isinstance(transfer, dict) or set(transfer) - {"champsim", "gem5"}:
        raise ValueError("unknown transfer frontend")
    for frontend, settings in transfer.items():
        if not isinstance(settings, dict):
            raise ValueError("transfer frontend settings must be an object")
        _names(settings.get("workloads"), frontend)
        if frontend == "champsim":
            if set(settings) != {"workloads", "warmup_instructions", "roi_instructions"}:
                raise ValueError("unexpected ChampSim configuration fields")
            _integer(settings["warmup_instructions"], 2_000_000, "ChampSim warmup")
            _integer(settings["roi_instructions"], 10_000_000, "ChampSim ROI")
        elif set(settings) != {"workloads", "run_to_exit"} or settings["run_to_exit"] is not True:
            raise ValueError("gem5 transfer must run complete benchmark programs to exit")
    return copy.deepcopy(config)


def load(root):
    path = pathlib.Path(root) / "evaluation_config.json"
    return validate(json.loads(path.read_text()) if path.exists() else LEGACY)


def install(root, source=None):
    path = pathlib.Path(root) / "evaluation_config.json"
    if path.exists():
        raise RuntimeError("cannot replace a prepared evaluation profile")
    config = validate(json.loads(pathlib.Path(source or DEFAULT).read_text()))
    atomic_write_json(path, config)
    return config


def workloads(root, split):
    if split not in ("training", "test"):
        raise ValueError("unknown scoring split: " + split)
    return load(root)["simpleo3"][split]


def inventory(root, trace_path, provenance):
    """Validate trace length/content identity before any generation call."""
    config = load(root)
    inputs = {}
    for workload in config["simpleo3"]["training"] + config["simpleo3"]["test"]:
        path = trace_path(workload)
        before = provenance(path)
        result = subprocess.run(["awk", '{n++; i += $1 + ($2 != -1)} END {printf "%d %.0f\\n", n, i}', path],
                                capture_output=True, text=True, check=True)
        records, instructions = map(int, result.stdout.split())
        if instructions < config["simpleo3"]["instructions_per_core"]:
            raise RuntimeError("evaluation ROI would wrap input trace: " + workload)
        if provenance(path) != before:
            raise RuntimeError("input changed during inventory: " + workload)
        inputs[workload] = {**before, "family": family(workload), "records": records,
                            "available_instructions": instructions, "wraps": False}
    hashes = [p["sha256"] for p in inputs.values()]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("duplicate trace content/alias in evaluation cohorts")
    atomic_write_json(pathlib.Path(root) / "input_inventory.json", inputs)
    return inputs


def transfer_group(config, workload):
    """Frontend transfer is not necessarily unseen-application transfer."""
    group = family(workload)
    if group in {family(w) for w in config["simpleo3"]["training"]}:
        return "training_family_new_frontend"
    if group in {family(w) for w in config["simpleo3"]["test"]}:
        return "simpleo3_test_family_new_frontend"
    return "unseen_family_new_frontend"
