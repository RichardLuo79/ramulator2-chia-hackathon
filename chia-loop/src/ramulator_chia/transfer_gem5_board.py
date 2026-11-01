"""Frozen DDR5 configuration + whole-program O3 gem5 transfer (inside gem5)."""
import json
import os
import sys

sys.path.insert(0, os.environ["CHIA_PYTHON"])
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import PrivateL1PrivateL2CacheHierarchy
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
import ramulator

with open(os.environ["CHIA_MEMORY_CONFIG"]) as stream:
    config = json.load(stream)
board = SimpleBoard(
    clk_freq="3.2GHz",
    processor=SimpleProcessor(cpu_type=CPUTypes.O3, isa=ISA.X86, num_cores=1),
    memory=ramulator.gem5.Memory(config["memory_system"], size="3GiB"),
    cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
        l1d_size="32KiB", l1i_size="32KiB", l2_size="1MiB"),
)
board.set_se_binary_workload(
    binary=BinaryResource(local_path=os.environ["CHIA_BENCHMARK"]),
    arguments=json.loads(os.environ["CHIA_BENCHMARK_ARGS"]),
)
simulator = Simulator(board=board)
simulator.run()
cause = simulator.get_last_exit_event_cause()
code = simulator.get_last_exit_event_code()
with open(os.environ["CHIA_EXIT_RECEIPT"], "w") as stream:
    json.dump({"cause": cause, "code": code}, stream)
if cause != "exiting with last active thread context" or code != 0:
    raise RuntimeError(f"benchmark did not exit successfully: {cause}, code={code}")
