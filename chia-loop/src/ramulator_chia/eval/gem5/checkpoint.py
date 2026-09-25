"""Qualified checkpoint metadata reader; no census execution."""
import json,re
from fs_board import IMAGES

def checkpoint_workload(checkpoint):
    metadata = json.loads((checkpoint / "stage.json").read_text())
    # The original NPB-only batch predates the multi-suite metadata.
    if "suite" not in metadata and re.fullmatch(r"[a-z]+\.[A-Z]\.x", metadata.get("benchmark", "")):
        metadata["suite"] = "npb"
    if metadata["stage"] != "roi" or metadata["cores"] not in (1, 4):
        raise ValueError("expected a one- or four-core workload ROI checkpoint")
    if metadata["suite"] not in IMAGES or not metadata.get("benchmark"):
        raise ValueError("checkpoint must identify its suite and workload")
    return metadata

