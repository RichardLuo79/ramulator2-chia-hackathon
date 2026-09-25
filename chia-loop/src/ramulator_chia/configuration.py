"""Resolve public templates and bind newly prepared storage identities."""

import json
import os
from pathlib import Path
import re

from .input_data import safe_path, sha
from .layout import ROOT


def expand(value, variables):
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if not isinstance(value, str):
        return value

    def substitute(match):
        name = match.group(1)
        if not variables.get(name):
            raise ValueError(f"set {name} explicitly before using this template")
        return str(variables[name])

    return re.sub(r"\$\{([A-Z_]+)\}", substitute, value)


def template(path, data_root, work_root):
    return expand(
        json.loads(Path(path).read_text()),
        {
            "ARTIFACT_ROOT": ROOT,
            "DATA_ROOT": Path(data_root).absolute(),
            "WORK_ROOT": Path(work_root).absolute(),
            "VERTEX_PROJECT": os.environ.get("CHIA_VERTEX_PROJECT"),
        },
    )


def bind_prepared(configuration, data_root):
    """Change storage hashes only; require the frozen decoded identities.

    A receipt is permission to use these bytes in a *new* experiment, never to
    reuse a historical measurement. The ordinary campaign identity checks still
    reject changes to inputs on resume.
    """
    root = Path(data_root).absolute()
    path = safe_path(root, "prepared-inputs.json")
    if not path.exists():
        return configuration
    receipt = json.loads(path.read_text())
    if receipt.get("schema_version") != 1 or receipt.get(
        "source_manifest_sha256"
    ) != sha(ROOT / "results/manifests/input-sources.json"):
        raise ValueError("unrecognized input preparation receipt")
    cohort = configuration["experiment"]["evaluation"]["champsim"]
    inputs = list(cohort["traces"].values()) + [
        case["placement"] for case in cohort["cases"].values()
    ]
    for item in inputs:
        relative = Path(item["path"]).relative_to(root).as_posix()
        row = receipt["files"].get(relative)
        if row is None:
            raise ValueError("preparation receipt lacks input: " + relative)
        if row["decoded_sha256"] != item["decoded_sha256"] or item["sha256"] not in (
            row["reference_sha256"],
            row["sha256"],
        ):
            raise ValueError(
                "prepared input identity differs from configuration: " + relative
            )
        if sha(safe_path(root, relative)) != row["sha256"]:
            raise ValueError("prepared input changed: " + relative)
        item["sha256"] = row["sha256"]
    return configuration


def load_config(path, data_root, work_root, *, prepared=True):
    from .framework.config import CampaignConfig

    value = template(path, data_root, work_root)
    if prepared:
        value = bind_prepared(value, data_root)
    return CampaignConfig.model_validate_json(json.dumps(value))


def compaction_policy(configuration, supplied=None):
    from .framework.config import ContextCompaction

    names = {"deepseek_api": "deepseek", "vertex_gemini": "gemini"}
    kind = configuration.backend.kind
    if supplied is None and kind not in names:
        return None
    path = supplied or ROOT / "chia-loop/configs/compaction" / (names[kind] + ".json")
    value = expand(json.loads(Path(path).read_text()), {"ARTIFACT_ROOT": ROOT})
    return ContextCompaction.model_validate_json(json.dumps(value))
