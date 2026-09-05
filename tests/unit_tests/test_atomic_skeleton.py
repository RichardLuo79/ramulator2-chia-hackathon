import csv

import pytest

import ramulator


def _ddr5_atomic(**kwargs):
    dram = ramulator.dram.DDR5(
        org_preset="DDR5_16Gb_x8",
        timing_preset="DDR5_4800AN",
    )
    return ramulator.controller.Atomic(
        dram=dram,
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        refresh="none",
        **kwargs,
    )


def _memory(controller, *, passthrough=False):
    mapper = (
        ramulator.channel_mapper.PassThroughChannelMapper()
        if passthrough
        else ramulator.channel_mapper.CacheLineInterleave()
    )
    return ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=[controller],
        channel_mapper=mapper,
    )


def test_atomic_seed_is_fixed_delay_with_bounded_backpressure(tmp_path):
    trace = tmp_path / "atomic_requests.csv"
    sim = ramulator.BatchSim(
        _memory(
            _ddr5_atomic(
                latency=7,
                read_buffer_size=2,
                write_buffer_size=2,
                trace_path=str(trace),
            )
        )
    )

    departs = sim.run([0, 64, 128], [0, 0, 0], [0, 0, 0])
    stats = sim.stats["memory_system"]["controller"]
    sim.finalize()

    # Two reads enter at tick 0. The third waits for the bounded pool and is
    # admitted at tick 7; every admitted read receives exactly seven ticks.
    assert departs == [7, 7, 14]
    assert stats["avg_read_latency"] == 7
    assert stats["peak_inflight_reads"] == 2
    assert stats["send_rejects"] == 7

    with (tmp_path / "atomic_requests.csv.ch0").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["frontend_id"]) for row in rows] == [0, 1, 2]
    assert [int(row["depart"]) - int(row["arrive"]) for row in rows] == [7, 7, 7]
    # Rejected attempts consume candidate admission ordinals by contract.
    assert int(rows[2]["admission_ordinal"]) > int(rows[1]["admission_ordinal"]) + 1


def test_passthrough_channel_mapper_defaults_an_empty_address_vector_to_channel_zero():
    sim = ramulator.BatchSim(
        _memory(_ddr5_atomic(latency=3), passthrough=True)
    )
    assert sim.run([0], [0], [0]) == [3]
    sim.finalize()


@pytest.mark.parametrize(
    ("standard", "org", "timing", "extra"),
    [
        ("DDR5", "DDR5_16Gb_x8", "DDR5_4800AN", {}),
        ("LPDDR5", "LPDDR5_16Gb_x16", "LPDDR5_6400", {"channel_width": 32}),
        ("LPDDR6", "LPDDR6_16Gb_x12", "LPDDR6_10667_BL24", {}),
        ("HBM4", "HBM4_32Gb_8Hi", "HBM4_8000Mbps", {"channel_width": 64}),
    ],
)
def test_one_atomic_seed_runs_unchanged_across_target_standards(
    standard, org, timing, extra
):
    dram = getattr(ramulator.dram, standard)(
        org_preset=org,
        timing_preset=timing,
        **extra,
    )
    controller = ramulator.controller.Atomic(
        dram=dram,
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        refresh="none",
    )
    sim = ramulator.BatchSim(_memory(controller))
    depart = sim.run([0], [0], [0])
    sim.finalize()
    assert depart[0] > 0


@pytest.mark.parametrize("latency", [0, -2])
def test_atomic_seed_rejects_nonpositive_explicit_latency(latency):
    with pytest.raises(RuntimeError, match="latency must be positive"):
        ramulator.Simulation(
            ramulator.frontend.External(clock_ratio=1),
            _memory(_ddr5_atomic(latency=latency)),
        )


def test_atomic_seed_rejects_refresh_until_a_policy_is_defined():
    controller = _ddr5_atomic()
    controller.refresh = "all_bank"
    with pytest.raises(RuntimeError, match="refresh='none' only"):
        ramulator.Simulation(
            ramulator.frontend.External(clock_ratio=1),
            _memory(controller),
        )


@pytest.mark.parametrize("values", [
    {"wr_low_watermark": 0.9, "wr_high_watermark": 0.5},
    {"wr_high_watermark": 1.1},
    {"model_parameters": ["unknown=1"]},
    {"model_parameters": ["gain=nan"]},
    {"model_parameters": ["gain=1", "gain=2"]},
    {"model_parameters": ["gain=1junk"]},
])
def test_atomic_model_interface_rejects_invalid_settings(values):
    with pytest.raises(RuntimeError):
        ramulator.Simulation(ramulator.frontend.External(clock_ratio=1), _memory(_ddr5_atomic(**values)))
