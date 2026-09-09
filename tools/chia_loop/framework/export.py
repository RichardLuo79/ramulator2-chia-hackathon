"""Source-first campaign export over the existing checked ZIP64 writer.

Input selection is explicit. This module never copies an installation, follows
binary dependencies, reconstructs an environment, or runs a missing evaluation.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from tools.chia_loop.recovery import exclusive_lock

from .archive import backup_sqlite, describe_payload, seal
from .identity import canonical_json, digest_json, file_sha256
from .snapshots import publish_bytes, regular_path

HARNESS_TREES = (
    "tools/chia_loop/framework",
    "tools/eval",
    "python/ramulator",
    "resources/champsim_bridge",
    "resources/gem5_wrappers",
)
HARNESS_FILES = (
    "pyproject.toml",
    "requirements-dev.txt",
    "tools/chia_loop/requirements.txt",
    "tools/chia_loop/core.py",
    "tools/chia_loop/recovery.py",
    "tools/chia_loop/pareto.py",
    "tools/chia_loop/evaluation_config.py",
    "tools/chia_loop/sandbox.py",
    "tools/chia_loop/codex_cli/__init__.py",
    "tools/chia_loop/codex_cli/auth.py",
    "tools/chia_loop/codex_cli/boundary.py",
    "tools/chia_loop/claude_cli/__init__.py",
    "tools/chia_loop/claude_cli/auth.py",
    "tools/chia_loop/transfer_gem5_board.py",
    "tools/chia_loop/champsim_trace_format.cpp",
    "tests/utils/check_chia_clean_campaign.py",
)
SOURCE_SUFFIXES = {
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hh",
    ".hpp",
    ".json",
    ".txt",
    ".md",
    ".patch",
}
BUILD_EVIDENCE = {
    "result.json",
    "build.json",
    "configure.log",
    "build.log",
    "symbols.log",
    "compile.policy.json",
    "compile_commands.json",
    "CMakeConfigureLog.yaml",
}
MEASUREMENT_EVIDENCE = {
    "result.json",
    "measurement.json",
    "config.json",
    "simulation.policy.json",
    "simulation.log",
    "trace_inventory.json",
    "archive_manifest.json",
    "staging_cleanup.json",
    "stats.yaml",
    "stats.txt",
    "exit.json",
    "ramulator_stats.yaml",
    "config.ini",
    "logical.csv.ch0",
    "controller.csv.ch0",
    "logical.csv.ch0.gz",
    "controller.csv.ch0.gz",
}


def tree_files(root: Path):
    """Enumerate one owned evidence/source tree without following links."""
    if root.is_symlink():
        raise ValueError("source/evidence root is a symlink")
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(n for n in names if n not in {"__pycache__", ".git", ".pytest_cache"})
        for name in names:
            if (Path(directory) / name).is_symlink():
                raise ValueError("source/evidence directory is a symlink")
        for name in sorted(files):
            path = Path(directory) / name
            yield regular_path(root, path.relative_to(root).as_posix())


def capture_sources(repo: Path, runtime: Path, destination: Path) -> dict:
    """Freeze the code actually used: compiled C++ inputs plus current harness.

    Repeat construction checks this inventory. A changed harness starts a new
    experiment instead of silently changing a continued campaign's treatment.
    This trusted snapshot is never an agent-visible directory.
    """
    compiled = json.loads((runtime / "runtime_manifest.json").read_text())["source_inventory"]
    files = {name: runtime / "runtime-source" / name for name in compiled}
    for name, path in files.items():
        if file_sha256(path) != compiled[name]:
            raise ValueError("compiled source changed: " + name)
    for name in HARNESS_FILES:
        files[name] = repo / name
    for tree in HARNESS_TREES:
        for path in tree_files(repo / tree):
            if path.suffix in SOURCE_SUFFIXES or path.name == "SConscript":
                files[path.relative_to(repo).as_posix()] = path
    inventory = {name: file_sha256(path) for name, path in sorted(files.items())}
    record = {"schema_version": 1, "files": inventory, "sha256": digest_json(inventory)}
    # Check the complete inventory before writing any new file into a resumed snapshot.
    publish_bytes(destination / "inventory.json", canonical_json(record).encode())
    for name, path in files.items():
        data = path.read_bytes()
        from hashlib import sha256

        if sha256(data).hexdigest() != inventory[name]:
            raise ValueError("source changed while being captured: " + name)
        publish_bytes(destination / name, data)
    return record


def validate_recipes(recipes: dict, frontends) -> None:
    """Check source recipes before evaluation, and recheck them when sealing."""
    frontends = set(frontends)
    if frontends - set(recipes):
        raise ValueError("each external frontend needs its pinned source/build recipe")
    for name, recipe in recipes.items():
        metadata = recipe["metadata"]
        commands = metadata.get("build_commands")
        if (
            not isinstance(commands, list)
            or not commands
            or any(
                not isinstance(command, list)
                or not command
                or any(not isinstance(arg, str) or "\0" in arg for arg in command)
                for command in commands
            )
        ):
            raise ValueError("external recipes need explicit build argument arrays")
        if name in frontends:
            upstream = metadata.get("upstream", {})
            if not re.fullmatch(r"[0-9a-f]{40}", upstream.get("revision", "")) or not (
                isinstance(upstream.get("url"), str) and upstream["url"].startswith("https://")
            ):
                raise ValueError("external frontends need an immutable upstream source pin")
        elif not recipe["payloads"]:
            raise ValueError("benchmark build recipes must include their source inputs")
        for payload in recipe["payloads"]:
            if payload.member.executable:
                raise ValueError("external recipe payloads must be source/data, not executables")
            payload.source.read_text()  # Source/patch/build log, not a renamed compiled program.


def load_frontend_recipes(path: Path) -> dict:
    """Read pinned frontend metadata and explicit source/log file references.

    Relative source paths are relative to the recipe file. Build commands are
    preserved as data; this function never executes them or fetches a repository.
    Guest recipes come from the separate verified source-build manifest.
    """

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate frontend recipe key: " + key)
            result[key] = value
        return result

    rows = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(rows, dict) or set(rows) - {"gem5", "champsim"}:
        raise ValueError("recipes must name the supported external frontends")
    recipes = {}
    for name, row in rows.items():
        if (
            not isinstance(row, dict)
            or set(row) != {"metadata", "files"}
            or not isinstance(row["metadata"], dict)
            or not isinstance(row["files"], list)
            or any(
                not isinstance(item, dict)
                or set(item) != {"source", "member"}
                or any(not isinstance(value, str) for value in item.values())
                for item in row["files"]
            )
        ):
            raise ValueError("frontend recipes need metadata and an explicit files list")
        recipes[name] = {
            "metadata": row["metadata"],
            "payloads": [
                describe_payload(path.parent / item["source"], item["member"])
                for item in row["files"]
            ],
        }
    validate_recipes(recipes, recipes.keys())
    return recipes


def export_campaign(campaign, research, output: Path, *, external_recipes: dict | None = None):
    """Archive a frozen, evaluated campaign. Externally built programs use recipes.

    Each external recipe names an immutable upstream revision, patch/source
    payloads and build commands. Guest benchmark executables also need a recipe;
    they are not smuggled into the bundle as opaque input data. Original build
    hashes remain in measurement receipts for provenance, not binary restoration.
    """
    external_recipes = external_recipes or {}
    root = campaign.root
    with exclusive_lock(root / "campaign.lock"):
        selection = campaign.state.get("selection")
        postrun = campaign.state.completed_step("postrun")
        if selection is None or postrun is None:
            raise ValueError("freeze and evaluate the campaign before exporting final results")
        sources = json.loads((root / "source/inventory.json").read_text())
        if digest_json(sources["files"]) != sources["sha256"]:
            raise ValueError("source inventory checksum differs from its contents")
        native_inputs = json.loads((root / "native-inputs.json").read_text())
        if native_inputs["source_sha256"] != sources["sha256"]:
            raise ValueError("source snapshot differs from the campaign's original binding")
        if native_inputs["cases"] != {
            group: [case.identity() for case in cases] for group, cases in research.cases.items()
        } or native_inputs["curve"] != asdict(research.curve.member):
            raise ValueError("export inputs differ from the evaluated campaign")
        if native_inputs["runtime"] != file_sha256(research.runtime / "runtime_manifest.json"):
            raise ValueError("export runtime provenance differs from the evaluated campaign")
        hosts = {name: host.identity() for name, host in research.hosts.items()}
        if native_inputs.get("hosts", {}) != hosts:
            raise ValueError("export frontends differ from the evaluated campaign")
        payloads = []

        def add(path, name):
            payloads.append(
                describe_payload(path, name, codec="gzip" if name.endswith(".gz") else "none")
            )

        for name, expected in sources["files"].items():
            path = regular_path(root / "source", name)
            if file_sha256(path) != expected:
                raise ValueError("campaign source snapshot changed: " + name)
            add(path, "source/" + name)
        add(root / "source/inventory.json", "source/inventory.json")
        add(research.runtime / "runtime_manifest.json", "campaign/runtime-build.json")
        for name in ("native-inputs.json",):
            add(root / name, "campaign/" + name)
        for name, host in research.hosts.items():
            add(host.root / "host.json", f"campaign/frontends/{name}/host.json")
        for directory in (
            "candidates",
            "summaries",
            "events",
            "native-evidence",
            "workspace-bindings",
        ):
            for path in tree_files(root / directory):
                add(path, "campaign/" + path.relative_to(root).as_posix())
        # Preserve model-authored notes, without live CLI profiles or credentials.
        for notes in sorted((root / "agents").glob("*/*/notes")):
            for path in tree_files(notes):
                add(path, "campaign/" + path.relative_to(root).as_posix())
        for directory, names in (
            ("builds", BUILD_EVIDENCE),
            ("measurements", MEASUREMENT_EVIDENCE),
        ):
            for path in tree_files(root / directory):
                if path.name in names:
                    add(path, "campaign/" + path.relative_to(root).as_posix())
        inputs, input_names = {}, set()
        for group, cases in research.cases.items():
            inputs[group] = []
            for case in cases:
                bound = {}
                for name, payload in case.input_files().items():
                    member = payload.member
                    if member.executable:
                        recipe = "benchmark:" + case.workload
                        if recipe not in external_recipes:
                            raise ValueError("missing source/build recipe for " + recipe)
                        bound[name] = {"recipe": recipe, "original_build": asdict(member)}
                        continue
                    archive_name = "inputs/" + member.stored_sha256
                    if member.codec != "none":
                        archive_name += "." + {"gzip": "gz", "xz": "xz"}[member.codec]
                    if archive_name not in input_names:
                        payloads.append(replace(payload, member=replace(member, name=archive_name)))
                        input_names.add(archive_name)
                    bound[name] = {"member": archive_name, "input": asdict(member)}
                inputs[group].append({"case": case.identity(), "files": bound})
        curve = research.curve
        payloads.append(replace(curve, member=replace(curve.member, name="inputs/mess.txt")))
        validate_recipes(external_recipes, research.hosts)
        references = {}
        for name, recipe in external_recipes.items():
            metadata = recipe["metadata"]
            references[name] = (
                {**metadata, "evaluated_host": hosts[name]} if name in hosts else metadata
            )
            payloads.extend(recipe["payloads"])
        with tempfile.TemporaryDirectory(prefix="chia-export-state.") as directory:
            database = Path(directory) / "campaign.sqlite"
            backup_sqlite(root / "campaign.sqlite", database)
            add(database, "campaign/campaign.sqlite")
            return seal(
                payloads,
                output,
                metadata={
                    "campaign": campaign.config.model_dump(mode="json"),
                    "selection": selection,
                    "postrun": postrun,
                    "source_sha256": sources["sha256"],
                    "inputs": inputs,
                    "external_sources": references,
                    "reproduction": "rebuild from source",
                    "runtime_binaries_included": False,
                    "exporter_sha256": file_sha256(Path(__file__)),
                },
            )
