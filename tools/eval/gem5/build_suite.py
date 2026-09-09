"""Build the pinned public guest suite; no downloads or simulator/model calls.

Inputs are two checksummed upstream source archives. Build output contains local
executables, source, commands and logs. Campaign export takes the source recipes,
not those executables. Compilation is sequential and uses one CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from tools.chia_loop.framework.archive import describe_payload, safe_name
from tools.chia_loop.framework.build import run_build_command
from tools.chia_loop.framework.identity import canonical_json, file_sha256
from tools.chia_loop.framework.snapshots import publish_bytes, regular_path

RECIPE = Path(__file__).with_name("public_suite.json")


def build(archives: Path, output: Path):
    recipe_bytes = RECIPE.read_bytes()
    recipe = json.loads(recipe_bytes)
    for spec in recipe["sources"].values():
        if file_sha256(archives / spec["archive"]) != spec["sha256"]:
            raise ValueError("source archive checksum differs: " + spec["archive"])
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "bin").mkdir()
    (output / "logs").mkdir()
    source_root = output / "sources"
    source_root.mkdir()
    publish_bytes(output / "recipe.json", recipe_bytes)
    for spec in recipe["sources"].values():
        with tarfile.open(archives / spec["archive"], "r:gz") as archive:
            for member in archive.getmembers():
                name = safe_name(member.name.rstrip("/"))
                if name.split("/")[0] != spec["directory"] or not (
                    member.isfile() or member.isdir()
                ):
                    raise ValueError("unexpected upstream archive member: " + member.name)
            archive.extractall(source_root, filter="data")
    inventory = {
        path.relative_to(output).as_posix(): file_sha256(path)
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
        and (
            path.suffix in {".h", ".c", ".cc"}
            or path.name in {"LICENSE", "LICENSE.txt", "README", "README.md"}
        )
    }
    programs = {}
    for name, case in recipe["programs"].items():
        spec = recipe["sources"][case["suite"]]
        source = Path("sources") / spec["directory"]
        kernel = source / case["source"]
        compiler = shutil.which(spec["compiler"])
        if compiler is None:
            raise ValueError("missing compiler: " + spec["compiler"])
        command = [compiler, *spec["flags"], "-I" + str(kernel.parent)]
        command += ["-I" + str(source / name) for name in spec["includes"]]
        command += [str(source / name) for name in spec["extra_sources"]]
        command += [str(kernel), "-lm", "-o", "bin/" + name]
        result = run_build_command(
            command,
            output / "logs" / (name + ".log"),
            cwd=output,
            cpus=1,
            timeout_seconds=300,
        )
        programs[name] = {
            **case,
            "executable": "bin/" + name,
            "sha256": file_sha256(output / "bin" / name),
            "compiler_sha256": file_sha256(Path(compiler)),
            "build": result,
        }
    record = {
        "schema_version": 1,
        "name": recipe["name"],
        "recipe": recipe,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "builder_sha256": file_sha256(Path(__file__)),
        "source_files": inventory,
        "programs": programs,
    }
    publish_bytes(output / "suite.json", canonical_json(record).encode())
    return record


def load_programs(manifest: Path):
    """Resolve a built suite and check its source and executable identities."""
    root = manifest.parent.resolve(strict=True)
    record = json.loads(manifest.read_text())
    if record["schema_version"] != 1:
        raise ValueError("unsupported guest suite manifest")
    if (
        file_sha256(root / "recipe.json") != record["recipe_sha256"]
        or json.loads((root / "recipe.json").read_text()) != record["recipe"]
        or set(record["programs"]) != set(record["recipe"]["programs"])
    ):
        raise ValueError("guest source recipe changed")
    for name, expected in record["source_files"].items():
        if file_sha256(regular_path(root, name)) != expected:
            raise ValueError("guest source changed: " + name)
    programs = {}
    for name, case in record["programs"].items():
        if any(case[key] != value for key, value in record["recipe"]["programs"][name].items()):
            raise ValueError("guest program differs from the built recipe: " + name)
        binary = regular_path(root, case["executable"])
        if file_sha256(binary) != case["sha256"]:
            raise ValueError("guest executable changed: " + name)
        programs[name] = (binary, tuple(case["arguments"]))
    return programs


def source_recipes(manifest: Path):
    """Give the existing exporter source recipes, without a binary payload."""
    load_programs(manifest)
    record = json.loads(manifest.read_text())
    recipes = {}
    for name, case in record["programs"].items():
        spec = record["recipe"]["sources"][case["suite"]]
        prefix = "sources/" + spec["directory"] + "/"
        recipes["benchmark:" + name] = {
            "metadata": {
                "upstream": spec,
                "working_directory": "external/" + name,
                "build_commands": [["mkdir", "-p", "bin"], case["build"]["argv"]],
                "original_build": case,
            },
            "payloads": [
                describe_payload(manifest.parent / path, "external/" + name + "/" + path)
                for path in record["source_files"]
                if path.startswith(prefix)
            ]
            + [
                describe_payload(
                    manifest.parent / "recipe.json", "external/" + name + "/recipe.json"
                ),
                describe_payload(
                    manifest.parent / "logs" / (name + ".log"), "external/" + name + "/build.log"
                ),
            ],
        }
    return recipes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.archives, args.output)
    print(
        json.dumps({"built": len(result["programs"]), "manifest": str(args.output / "suite.json")})
    )
