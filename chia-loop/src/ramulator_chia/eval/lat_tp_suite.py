"""Spec resolution: compute theoretical targets from JEDEC timing parameters.

Resolves timing parameters via DRAMStandard.resolve() and derives peak throughput,
unloaded latency, and refresh overhead.
"""

from dataclasses import dataclass

from ramulator.dram.spec import DRAMStandard


@dataclass(frozen=True)
class Spec:
    """Theoretical targets and timing constants derived from a testcase dict."""

    # Time references
    tCK_ns: float                   # ns per CK cycle (for theoretical formulas)
    time_unit_ns: float             # ns per simulation tick (for converting sim cycle counts)

    # Throughput
    bytes_per_req: int
    channel_width: int
    rate: int
    has_pseudo_channel: bool
    max_theoretical_throughput: float  # GB/s — peak channel throughput
    max_achievable_throughput: float   # GB/s — peak after refresh overhead

    # Latency
    unloaded_latency_cycles: int    # nRP + nRCD + nCL
    unloaded_latency_ns: float

    # Refresh
    refresh_penalty_cycles: int     # nRTP + nRP + nRFC + nRCD
    refresh_overhead: float         # fraction of time unavailable to refresh

    # Resolved JEDEC timing params (CK units)
    nRCD: int
    nRTP: int
    nCL: int
    nRP: int
    nRFC: int
    nREFI: int


def resolve_spec(cfg: dict) -> Spec:
    """Resolve JEDEC timings and derive theoretical targets for one standard."""
    import ramulator

    std_cls = DRAMStandard._registry[cfg["dram_class"]]
    dram_cls = getattr(ramulator.dram, cfg["dram_class"])
    dram_obj = dram_cls(org_preset=cfg["org_preset"], timing_preset=cfg["timing_preset"])
    org_dict, timing_dict = dram_obj.resolve()

    # resolve() returns timings in CK units (pre tick_multiplier conversion).
    # Two time references:
    #   tCK_ns       — ns per CK cycle (use for theoretical formulas using timing_dict)
    #   time_unit_ns — ns per simulation tick (use for converting simulation cycle counts)
    # When tick_multiplier > 1 (HBM3/HBM4), to_config() divides tCK_ps and multiplies all
    # timing params by tick_multiplier.  resolve() returns the original (pre-mult) values.
    tick_mult = getattr(std_cls, "tick_multiplier", 1)
    tCK_ns = timing_dict["tCK_ps"] / 1000.0
    time_unit_ns = tCK_ns / tick_mult

    channel_width = org_dict["channel_width"]
    prefetch = std_cls.internal_prefetch_size
    bytes_per_req = std_cls.data_payload_bytes or (channel_width * prefetch // 8)

    rate = timing_dict["rate"]
    burst_gap = timing_dict.get("nBL_min", timing_dict.get("nBL"))
    if std_cls.data_payload_bytes is not None and burst_gap is not None:
        # Prefer modeled payload cadence when available. LPDDR6 transfers
        # metadata on the wire, so its payload bytes differ from physical width.
        max_theoretical_throughput = bytes_per_req / (burst_gap * tCK_ns)
    else:
        max_theoretical_throughput = channel_width * rate / 8 / 1000

    # Pseudo-channel standards interleave both PCs for an effective doubling
    has_pseudo_channel = "PseudoChannel" in std_cls.levels
    if has_pseudo_channel:
        max_theoretical_throughput *= org_dict.get("pseudochannel", 2)

    # Timing parameter name resolution (standards differ)
    nRCD = timing_dict.get("nRCDRD", timing_dict.get("nRCD", timing_dict.get("nRCDr")))
    nRTP = timing_dict.get("nRTPL", timing_dict.get("nRTP", timing_dict.get("nRTPSB")))
    if nRCD is None or nRTP is None:
        raise ValueError(
            f"{cfg['name']}: timing dict missing nRCD/nRCDRD/nRCDr or nRTP/nRTPL. "
            f"Available keys: {sorted(timing_dict.keys())}"
        )
    nCL = timing_dict.get("nCL", timing_dict.get("nRL"))
    nRP = timing_dict["nRP"]
    nRFC = timing_dict.get("nRFC", timing_dict.get("nRFCab"))
    nREFI = timing_dict["nREFI"]
    if nCL is None or nRFC is None:
        raise ValueError(
            f"{cfg['name']}: timing dict missing nCL/nRL or nRFC/nRFCab. "
            f"Available keys: {sorted(timing_dict.keys())}"
        )

    # Refresh penalty: full sequence between two READs when refresh intervenes:
    # close row (nRTP + nRP) + refresh (nRFC) + reopen row (nRCD)
    refresh_penalty_cycles = nRTP + nRP + nRFC + nRCD
    refresh_overhead = refresh_penalty_cycles / nREFI
    max_achievable_throughput = max_theoretical_throughput * (1 - refresh_overhead)

    # Unloaded latency: random probe hits a closed row
    # PREpb (nRP) + ACT-to-CAS (nRCD) + CAS latency (nCL)
    unloaded_latency_cycles = nRP + nRCD + nCL
    unloaded_latency_ns = unloaded_latency_cycles * tCK_ns

    return Spec(
        tCK_ns=tCK_ns,
        time_unit_ns=time_unit_ns,
        bytes_per_req=bytes_per_req,
        channel_width=channel_width,
        rate=rate,
        has_pseudo_channel=has_pseudo_channel,
        max_theoretical_throughput=max_theoretical_throughput,
        max_achievable_throughput=max_achievable_throughput,
        unloaded_latency_cycles=unloaded_latency_cycles,
        unloaded_latency_ns=unloaded_latency_ns,
        refresh_penalty_cycles=refresh_penalty_cycles,
        refresh_overhead=refresh_overhead,
        nRCD=nRCD,
        nRTP=nRTP,
        nCL=nCL,
        nRP=nRP,
        nRFC=nRFC,
        nREFI=nREFI,
    )

SUITE_CONFIG = {
    "name": "DDR5",
    "dram_class": "DDR5",
    "org_preset": "DDR5_16Gb_x8",
    "timing_preset": "DDR5_4800AN",
    "controller_class": "GenericDDR",
    "scheduler_class": "FRFCFSRowHit",
    "stagger_stream_rows": True,
    "frontend_clock_ratio": 8,
    "stream_cls": 64,
    "nop_counters": (
        1, 33, 36, 38, 41, 44, 47, 50, 52, 55, 58, 65, 70, 80,
        90, 100, 125, 150, 200, 300, 500, 750, 1000, 2000,
        5000, 10000,
    ),
}

def make_controller(ctrl_cls, cfg: dict, dram, refresh_enabled: bool):
    """Build a controller passing only the children/params its class declares.

    Cycle-level controllers take scheduler/row_policy/refresh_manager children;
    fast controllers take a 'refresh' string param instead. controller_kwargs
    from the testcase config always win.
    """
    import ramulator
    from ramulator.components import _get_descriptors

    descriptors = _get_descriptors(ctrl_cls)
    kwargs = dict(dram=dram, **cfg.get("controller_kwargs", {}))
    if "addr_mapper" in descriptors:
        kwargs.setdefault("addr_mapper", ramulator.addr_mapper.PassThroughAddrMapper())
    if "scheduler" in descriptors:
        scheduler_cls = getattr(ramulator.scheduler, cfg.get("scheduler_class", "FRFCFS"))
        kwargs.setdefault("scheduler", scheduler_cls())
    if "row_policy" in descriptors:
        kwargs.setdefault("row_policy", ramulator.row_policy.Open())
    if "refresh_manager" in descriptors:
        kwargs.setdefault(
            "refresh_manager",
            ramulator.refresh_manager.AllBank()
            if refresh_enabled
            else ramulator.refresh_manager.NoRefresh(),
        )
    elif "refresh" in descriptors:
        kwargs.setdefault("refresh", "all_bank" if refresh_enabled else "none")
    return ctrl_cls(**kwargs)

