"""Shared test utilities — DRAM creation and layout extraction."""

import ramulator
from tools.eval.layout import extract_dram_layout as extract_dram_layout


def create_dram(cfg):
    """Instantiate a DRAM object from a testcase CONFIG dict."""
    dram_cls = getattr(ramulator.dram, cfg["dram_class"])
    return dram_cls(
        org_preset=cfg["org_preset"], timing_preset=cfg["timing_preset"], **cfg["dram_kwargs"]
    )
