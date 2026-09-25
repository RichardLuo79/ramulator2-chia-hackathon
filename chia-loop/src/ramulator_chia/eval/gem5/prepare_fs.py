"""Freeze an FS evaluation's source inputs; build only the three model modules.

Run 'freeze' locally, copy its output to the worker, then run 'build' there.
Existing campaign databases, prompts and sessions are not copied or modified.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from fs_board import save_json, sha256


def freeze(args):
    repo = __import__("ramulator_chia.layout", fromlist=["RAMULATOR"]).RAMULATOR
    sys.path.insert(0, str(repo))
    from ramulator_chia.eval.simpleo3 import _controller_config
    from ramulator_chia.eval.config import MODEL_IMPL

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.selection.read_text())["selections"]
    api = Path("src/ramulator/controller/atomic_model/api.h")
    api_source = args.runtime / "export" / api
    manifest = json.loads((args.runtime / "runtime_manifest.json").read_text())
    if sha256(api_source) != manifest["model_api_sha256"]:
        raise RuntimeError("frozen model API differs from its recorded runtime")
    configs = root / "configs"
    configs.mkdir()
    oracle = json.loads(args.oracle.read_text())
    ctrl = oracle["memory_system"]["controllers"][0]
    ctrl.pop("controller_plugins", None)
    save_json(configs / "oracle.json", oracle)
    frozen = {}
    for label, record in selection.items():
        candidate = record["candidate"]
        origin = Path(record["original_campaign"]) / "candidates" / candidate["candidate_id"]
        destination = root / "models" / label
        destination.mkdir(parents=True)
        for name, info in candidate["files"].items():
            if name not in ("model.cpp", "parameters.json"):
                raise RuntimeError("unexpected frozen model file")
            if sha256(origin / name) != info["sha256"]:
                raise RuntimeError("frozen candidate checksum changed: " + label)
            shutil.copy2(origin / name, destination / name)
        if json.loads((destination / "parameters.json").read_text()) != candidate["parameters"]:
            raise RuntimeError("parameter values differ from selection")
        exported_api = Path(str(api).removeprefix("src/"))
        (destination / "api" / exported_api.parent).mkdir(parents=True)
        shutil.copy2(api_source, destination / "api" / exported_api)
        shutil.copy2(repo.parent / "chia-loop/native/model/CMakeLists.txt", destination / "CMakeLists.txt")
        config = copy.deepcopy(oracle)
        controller = _controller_config("candidate", {
            "model_parameters": [f"{k}={v}" for k, v in sorted(candidate["parameters"].items())]})
        controller.update(impl="Atomic", dram=ctrl["dram"], addr_mapper=ctrl["addr_mapper"])
        config["memory_system"]["controllers"] = [controller]
        save_json(configs / (label + ".json"), config)
        frozen[label] = {"candidate": candidate, "selected_iteration": record["promoted_iteration"]}
    shutil.copy2(Path(__file__).resolve().parents[1] / "calibration/mess_DDR5.txt", configs / "mess_DDR5.txt")
    for label in ("fixedlat", "md1", "wmg1", "mess"):
        config = copy.deepcopy(oracle)
        config["memory_system"]["controllers"] = [dict(_controller_config(label, {}),
                                                         impl=MODEL_IMPL[label], dram=ctrl["dram"])]
        save_json(configs / (label + ".json"), config)
    save_json(root / "selections.json", {"models": frozen, "model_api_sha256": sha256(api_source),
        "selection_source_sha256": sha256(args.selection), "optimization": "-O3",
        "no_llm_calls": True, "no_model_changes": True})
    for name in ("fs_board.py", "run_fs.py", "prepare_fs.py", "fs_workloads.json"):
        shutil.copy2(Path(__file__).with_name(name), root / name)
    shutil.copy2(args.resources, root / "resource-pins.json")
    save_json(root / "source-files.json", {str(p.relative_to(root)): sha256(p)
                                           for p in sorted(root.rglob("*")) if p.is_file()})


def build(args):
    root = args.root.resolve()
    baseline = args.baseline.resolve()
    inputs = json.loads((root / "source-files.json").read_text())
    for name, expected in inputs.items():
        if sha256(root / name) != expected:
            raise RuntimeError("frozen input changed: " + name)
    selection = json.loads((root / "selections.json").read_text())
    host = json.loads((root / "host.json").read_text())
    if sha256(host["gem5"]) != host["gem5_sha256"] or sha256(
            Path(host["ramulator_library_dir"]) / "libramulator.so") != host["ramulator_library_sha256"]:
        raise RuntimeError("retained simulator build changed")
    symbols = subprocess.check_output(["nm", "-D", "-C", str(
        Path(host["ramulator_library_dir"]) / "libramulator.so")], text=True)
    if "Ramulator::AtomicModel::load_library(" not in symbols:
        raise RuntimeError("Ramulator must be built with RAMULATOR_CHIA_MODEL_API=ON")
    pins = json.loads((root / "resource-pins.json").read_text())["resources"]
    verified = []
    for pin in pins:
        name = pin["id"] + "-" + pin["version"]
        candidates = [Path(path) / name for path in set(host["resources"].values())]
        matching = [path for path in candidates if path.is_file()]
        if not matching:
            raise RuntimeError("missing pinned resource: " + name)
        for path in matching:
            if sha256(path) != pin["raw_sha256"]:
                raise RuntimeError("resource checksum changed: " + str(path))
            verified.append(dict(path=str(path), sha256=pin["raw_sha256"]))
    save_json(root / "verified-resources.json", verified)
    api = baseline / "ramulator/src/ramulator/controller/atomic_model/api.h"
    if sha256(api) != selection["model_api_sha256"]:
        raise RuntimeError("cloud Ramulator API differs from frozen candidates")
    configs = root / "effective-configs"
    configs.mkdir(exist_ok=True)
    builds = {}
    for label in json.loads((root / "fs_workloads.json").read_text())["models"]:
        config = json.loads((root / "configs" / (label + ".json")).read_text())
        controller = config["memory_system"]["controllers"][0]
        if label in selection["models"]:
            source = root / "models" / label
            output = root / "builds" / label
            output.mkdir(parents=True, exist_ok=True)
            with (output / "build.log").open("w") as log:
                subprocess.run(["cmake", "-S", str(source), "-B", str(output), "-G", "Ninja",
                    "-DMODEL_SOURCES=model.cpp", "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG"], stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run(["cmake", "--build", str(output), "--parallel", "2"], stdout=log, stderr=subprocess.STDOUT, check=True)
            commands = json.loads((output / "compile_commands.json").read_text())
            if not commands or any("-O3" not in c["command"].split() for c in commands):
                raise RuntimeError("model build did not use -O3")
            module = output / "output/candidate.so"
            controller["model_library"] = str(module)
            builds[label] = {"module": str(module), "sha256": sha256(module), "optimization": "-O3"}
        elif label == "mess":
            controller["curve_path"] = str(root / "configs/mess_DDR5.txt")
        save_json(configs / (label + ".json"), config)
    save_json(root / "builds.json", builds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("freeze")
    for name in ("selection", "runtime", "oracle", "resources", "output"):
        local.add_argument("--" + name, type=Path, required=True)
    remote = sub.add_parser("build")
    remote.add_argument("--root", type=Path, required=True)
    remote.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else build)(args)
