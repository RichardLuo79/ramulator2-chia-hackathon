"""Latency-throughput simulation runners.

`run_single_config_point` covers one concrete point for one standard config,
including NOP-injected latency points and max-rate streaming-only points. It is
a top-level function (not a closure) so it pickles for ProcessPoolExecutor.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tests.utils import extract_dram_layout


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


def run_single_config_point(
    cfg: dict,
    *,
    nop_counter: int,
    read_ratio: int,
    num_probe_requests: int,
    refresh_enabled: bool,
    frontend_clock_ratio: int,
    warmup_cycles: int,
    num_streaming_requests: int = 0,
    streaming_only: bool = False,
    stream_cls: int | None = None,
    latency_measure_mode: str | None = None,
    latency_sample_count: int | None = None,
) -> dict:
    """Run one config plus one latency-throughput experiment point."""
    import ramulator

    dram_cls = getattr(ramulator.dram, cfg["dram_class"])
    dram = dram_cls(org_preset=cfg["org_preset"], timing_preset=cfg["timing_preset"])
    layout = extract_dram_layout(dram)
    if stream_cls is None:
        stream_cls = cfg.get("stream_cls", layout["num_cls"])
    if latency_measure_mode is None:
        latency_measure_mode = cfg.get("latency_measure_mode", "random-probe")
    if latency_sample_count is None:
        latency_sample_count = cfg.get("latency_sample_count", num_probe_requests)

    frontend = ramulator.frontend.LatencyThroughputTrace(
        clock_ratio=frontend_clock_ratio,
        nop_counter=nop_counter,
        num_probe_requests=num_probe_requests,
        latency_measure_mode=latency_measure_mode,
        latency_sample_count=latency_sample_count,
        num_streaming_requests=num_streaming_requests,
        streaming_only=streaming_only,
        warmup_cycles=warmup_cycles,
        seed=12345,
        read_ratio=read_ratio,
        stream_cls=stream_cls,
        stagger_stream_rows=cfg.get("stagger_stream_rows", False),
        **layout,
    )

    ctrl_cls = getattr(ramulator.controller, cfg["controller_class"])
    ctrl = make_controller(ctrl_cls, cfg, dram, refresh_enabled)

    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )

    sim = ramulator.Simulation(frontend, mem)
    sim.run()
    return sim.stats


def run_sweep(
    cfg: dict,
    *,
    read_ratios: tuple[int, ...],
    num_probe_requests: int,
    refresh_enabled: bool,
    frontend_clock_ratio: int,
    warmup_cycles: int,
    max_workers: int = 4,
    latency_measure_mode: str | None = None,
    latency_sample_count: int | None = None,
) -> dict:
    """Run a parallel NOP/read-ratio sweep.

    Returns: {(nop, read_ratio): stats_dict}
    """
    if latency_measure_mode is None:
        latency_measure_mode = cfg.get("latency_measure_mode", "random-probe")
    if latency_sample_count is None:
        latency_sample_count = cfg.get("latency_sample_count", num_probe_requests)
    jobs = [(nop, rr) for rr in read_ratios for nop in cfg["nop_counters"]]
    total = len(jobs)

    t_start = time.time()
    raw_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_single_config_point,
                cfg,
                nop_counter=nop,
                read_ratio=rr,
                num_probe_requests=num_probe_requests,
                refresh_enabled=refresh_enabled,
                frontend_clock_ratio=frontend_clock_ratio,
                warmup_cycles=warmup_cycles,
                latency_measure_mode=latency_measure_mode,
                latency_sample_count=latency_sample_count,
            ): (nop, rr)
            for nop, rr in jobs
        }
        done = 0
        for fut in as_completed(futures):
            nop, rr = futures[fut]
            raw_results[(nop, rr)] = fut.result()
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{cfg['name']}] {done}/{total} completed")

    elapsed = time.time() - t_start
    print(f"  [{cfg['name']}] All {total} runs done in {elapsed:.1f}s")
    return raw_results
