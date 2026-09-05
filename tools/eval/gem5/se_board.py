"""gem5 SE board: O3 CPU + private L1/L2 + Ramulator2 memory.

Env: R2_MODEL=oracle|candidate, R2_BINARY=/path, R2_ARGS='a b c' (optional),
     R2_REPO=/path/to/ramulator2 (default /home/dev/ramulator2),
     R2_TRACE=/path/prefix (optional per-request trace on either side).

Both sides use 64-entry read/write buffers. The oracle additionally uses the
reference write-drain watermarks. This file executes inside gem5's embedded
Python, so the small shared constants are mirrored here.
"""
import os
import sys

REPO = os.environ.get("R2_REPO", "/home/dev/ramulator2")
sys.path.insert(0, REPO + "/python")

from gem5.components.boards.simple_board import SimpleBoard  # noqa: E402
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (  # noqa: E402
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.processors.cpu_types import CPUTypes  # noqa: E402
from gem5.components.processors.simple_processor import SimpleProcessor  # noqa: E402
from gem5.isas import ISA  # noqa: E402
from gem5.resources.resource import BinaryResource  # noqa: E402
from gem5.simulate.simulator import Simulator  # noqa: E402

import ramulator  # noqa: E402

# Mirror of tools/eval/config.py REFERENCE (see module docstring).
REFERENCE = dict(read_buffer_size=64, write_buffer_size=64,
                 wr_low_watermark=0.5, wr_high_watermark=0.8)

model = os.environ["R2_MODEL"]
binary = os.environ["R2_BINARY"]
args = os.environ.get("R2_ARGS", "").split()
trace = os.environ.get("R2_TRACE", "")

ddr5 = ramulator.dram.DDR5(org_preset="DDR5_16Gb_x8", timing_preset="DDR5_4800AN")
if model == "oracle":
    plugins = []
    if trace:
        plugins = [ramulator.controller_plugin.ReqTraceRecorder(path=trace)]
    ctrl = ramulator.controller.GenericDDR(
        dram=ddr5,
        scheduler=ramulator.scheduler.FRFCFSRowHit(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        controller_plugins=plugins,
        **REFERENCE,
    )
elif model == "candidate":
    kw = dict(read_buffer_size=64, write_buffer_size=64)
    if trace:
        kw["trace_path"] = trace
    # Optional candidate parameters, encoded as JSON in the environment.
    extra = os.environ.get("R2_CANDIDATE_KW", "")
    if extra:
        import json as _json
        kw.update(_json.loads(extra))
    ctrl = ramulator.controller.Atomic(
        dram=ddr5, addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        refresh="none", **kw)
else:
    raise SystemExit(f"R2_MODEL must be oracle|candidate, got {model!r}")

mem_sys = ramulator.memory_system.GenericDRAM(
    clock_ratio=3, controllers=[ctrl],
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave())

board = SimpleBoard(
    clk_freq="3.2GHz",
    processor=SimpleProcessor(cpu_type=CPUTypes.O3, isa=ISA.X86, num_cores=1),
    memory=ramulator.gem5.Memory(mem_sys, size="3GiB"),
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="1MiB"),
)
board.set_se_binary_workload(binary=BinaryResource(local_path=binary), arguments=args)

Simulator(board=board).run()
