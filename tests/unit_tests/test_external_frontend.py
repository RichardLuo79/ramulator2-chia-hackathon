import pytest

import ramulator


def _memory():
    dram = ramulator.dram.DDR5(
        org_preset="DDR5_16Gb_x8",
        timing_preset="DDR5_4800AN",
    )
    controller = ramulator.controller.Atomic(
        dram=dram,
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        refresh="none",
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[controller],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )


def test_external_frontend_accepts_a_positive_core_count():
    frontend = ramulator.frontend.External(clock_ratio=1, num_cores=4)
    assert frontend.to_config()["num_cores"] == 4

    simulation = ramulator.Simulation(frontend, _memory())
    simulation.finalize()


def test_external_frontend_rejects_a_nonpositive_core_count():
    frontend = ramulator.frontend.External(clock_ratio=1, num_cores=0)
    with pytest.raises(ValueError, match="num_cores must be positive"):
        ramulator.Simulation(frontend, _memory())
