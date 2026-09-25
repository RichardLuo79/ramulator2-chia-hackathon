"""Generic diagnostic cases for the shared native measurement task.

Only generator axes and training workload labels are agent inputs. Controller
configuration, geometry, trace paths, builds and observation identities belong
to the trusted task. Neither case type supplies a promotion score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field, field_validator

from ramulator_chia.eval import config as C
from ramulator_chia.eval import simpleo3, synthetic
from ramulator_chia.eval.layout import extract_dram_layout

from .archive import Payload
from .config import StrictRecord, SyntheticLimits
from .evaluation import check_candidate_callbacks
from .identity import digest_json

# These are the native generator's integer representation, not search limits.
INT_MAX = 2**31 - 1
NativeCount = Annotated[int, Field(ge=1, le=INT_MAX)]
NativeDelay = Annotated[int, Field(ge=0, le=INT_MAX)]


class SyntheticAxes(StrictRecord):
    mlp: NativeCount = Field(16, description="Maximum outstanding reads per stream.")
    dep_frac: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = Field(
        0.0, description="Probability a read must wait for outstanding reads to finish."
    )
    think_time: NativeDelay = Field(
        0, description="Minimum delay between read issues, in DRAM cycles."
    )
    # The current generator evaluates the inclusive upper bound as int + 1.
    jitter: Annotated[int, Field(ge=0, lt=INT_MAX)] = Field(
        0, description="Maximum additional index-seeded random issue delay."
    )
    row_run: NativeCount = Field(
        32, description="Consecutive cache lines per bank row before advancing."
    )
    bank_spread: NativeCount = Field(
        1, description="Banks used per stream, bounded by its resolved region."
    )
    bank_random: bool = Field(
        False, description="Index-seeded random rather than round-robin bank choice."
    )
    row_random: bool = Field(
        False, description="Index-seeded random rather than sequential row choice."
    )
    wfrac_pct: Annotated[int, Field(ge=0, le=100)] = Field(
        0, description="Percentage probability of one writeback per read; ignored by mode 0."
    )
    wb_mode: Annotated[int, Field(ge=0, le=4)] = Field(
        0,
        description=(
            "0 read-only; 1 same-line callback writeback; 2 disjoint-row writes; "
            "3 split read/write banks; 4 lead writes plus optional writebacks."
        ),
    )
    wb_batch: NativeCount = Field(
        1, description="Queued-write release threshold; the tail is always drained."
    )
    phase_on: NativeDelay = Field(0, description="Reads per active phase; zero disables phases.")
    phase_off: NativeDelay = Field(0, description="Idle cycles after an active phase.")
    phase_off_random: Annotated[int, Field(ge=0, lt=INT_MAX)] = Field(
        0, description="Maximum extra index-seeded random idle cycles per phase."
    )
    pp_lead: NativeDelay = Field(40, description="Mode-4 write-to-read lead delay, in DRAM cycles.")


class SyntheticRequest(SyntheticAxes):
    num_requests: NativeCount | None = Field(
        None, description="Reads per stream; null uses the configured default."
    )
    streams: NativeCount = Field(1, description="Number of concurrent stream engines.")
    seed: Annotated[int, Field(ge=0, le=2**64 - 1)] = 1
    share_region: bool = Field(
        False, description="Share bank/row regions instead of disjoint stream regions."
    )
    mode_switch: NativeDelay = Field(
        0, description="Reads per alternating A/B block; zero uses A only."
    )
    mode_b: SyntheticAxes = Field(
        default_factory=SyntheticAxes,
        description="Partial regime-B overrides, applied after each stream's overrides.",
    )
    stream_params: tuple[SyntheticAxes, ...] = Field(
        (), description="Partial axis overrides, one per stream; empty uses the common axes."
    )

    @field_validator("stream_params", mode="before")
    @classmethod
    def ordered_overrides(cls, value):
        if type(value) not in {tuple, list}:
            raise ValueError("stream_params must be an ordered array of axis objects")
        return tuple(value)

    def parameters(self, limits: SyntheticLimits, layout: dict) -> dict:
        result = self.model_dump(exclude={"mode_b", "stream_params"})
        result["num_requests"] = self.num_requests or limits.requests_per_stream
        if result["num_requests"] > INT_MAX:
            raise ValueError("configured read count exceeds the native generator representation")
        if self.streams * result["num_requests"] > limits.maximum_total_reads:
            raise ValueError("requested population exceeds the declared total-read resource guard")
        if self.stream_params and len(self.stream_params) != self.streams:
            raise ValueError("stream_params must contain exactly one axis object per stream")
        result["mode_b"] = self.mode_b.model_dump(exclude_unset=True)
        result["stream_params"] = [
            item.model_dump(exclude_unset=True)
            for item in (self.stream_params or (SyntheticAxes(),) * self.streams)
        ]
        partitions = 1 if self.share_region else self.streams
        banks = layout["total_bank_units"] // partitions
        if not banks or not (layout["num_rows"] // 2) // partitions:
            raise ValueError("stream regions exceed the resolved DRAM organization")
        for stream in result["stream_params"]:
            for mode in ({}, result["mode_b"]):
                axes = {**result, **stream, **mode}
                if axes["row_run"] > layout["num_cls"] or axes["bank_spread"] > banks:
                    raise ValueError("row_run/bank_spread exceeds the resolved per-stream layout")
                if axes["wb_mode"] == 3 and axes["bank_spread"] > banks // 2:
                    raise ValueError("split write banks require two disjoint bank regions")
                if result["num_requests"] + axes["bank_spread"] - 2 > INT_MAX:
                    raise ValueError(
                        "random bank index would exceed the native integer representation"
                    )
        return result


def generator_parameters(parameters):
    """Serialize validated axis objects to the existing native parser's format."""

    def encoded(values):
        return ";".join(
            f"{key}={int(value) if type(value) is bool else value}"
            for key, value in sorted(values.items())
        )

    return {
        **parameters,
        "mode_b": encoded(parameters["mode_b"]),
        "stream_params": "|".join(encoded(values) for values in parameters["stream_params"]),
    }


def ddr5():
    import ramulator

    return ramulator.dram.DDR5(
        org_preset=C.STD["DDR5"]["org"], timing_preset=C.STD["DDR5"]["timing"]
    )


@dataclass(frozen=True)
class SyntheticCase:
    parameters: dict
    layout: dict
    observation_names = ("controller.csv.ch0",)

    @classmethod
    def create(cls, request: SyntheticRequest, limits: SyntheticLimits):
        layout = extract_dram_layout(ddr5())
        return cls(request.parameters(limits, layout), layout)

    def identity(self):
        return {
            "frontend": "SyntheticPattern",
            "std": "DDR5",
            "stage": "diagnostic",
            "parameters": self.parameters,
            "layout": self.layout,
            "timebase": "DRAM cycles",
            "eligible_for_promotion": False,
        }

    def input_files(self):
        return {}

    def configuration(self, model, inputs, observations, parameters, curve):
        import ramulator

        controller = simpleo3._build_controller(
            ramulator,
            model,
            ddr5(),
            "DDR5",
            observations / "controller.csv",
            parameters,
            mess_curve=curve,
        )
        controller.addr_mapper = ramulator.addr_mapper.PassThroughAddrMapper()
        frontend = ramulator.frontend.SyntheticPattern(
            clock_ratio=1, **self.layout, **generator_parameters(self.parameters)
        )
        memory = ramulator.memory_system.GenericDRAM(
            clock_ratio=1,
            controllers=[controller],
            channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
        )
        return {"frontend": frontend.to_config(), "memory_system": memory.to_config()}

    def check_statistics(self, stats, observations, *, candidate):
        frontend = stats["frontend"]
        if frontend["reads_sent"] != self.parameters["num_requests"] * self.parameters["streams"]:
            raise RuntimeError("synthetic frontend did not issue the requested population")
        for kind in ("read", "write"):
            if frontend[f"{kind}s_sent"] != frontend[f"{kind}s_completed"]:
                raise RuntimeError("synthetic callbacks were not fully drained")
        synthetic.read_population(observations / "controller.csv.ch0", self.parameters, frontend)
        if candidate:
            check_candidate_callbacks(stats, observations / "controller.csv.ch0")

    def result_fields(self, stats):
        return {"diagnostic_only": True, "eligible_for_promotion": False}


@dataclass(frozen=True)
class ReplayCase:
    workload: str
    arrivals: Payload
    source_receipt: dict
    cores: int
    reads: int
    writes: int
    observation_names = ("controller.csv.ch0", "replay.csv")

    def identity(self):
        return {
            "frontend": "ControllerReplay",
            "std": "DDR5",
            "stage": "diagnostic",
            "workload": self.workload,
            "source_receipt": self.source_receipt,
            "arrivals_sha256": self.arrivals.member.logical_sha256,
            "arrivals_bytes": self.arrivals.member.logical_bytes,
            "cores": self.cores,
            "reads": self.reads,
            "writes": self.writes,
            "ordering": "arrive,admission_ordinal",
            "timebase": "DRAM cycles",
            "population": "complete training oracle controller stream",
            "eligible_for_promotion": False,
        }

    def input_files(self):
        return {"replay.csv": self.arrivals}

    def configuration(self, model, inputs, observations, parameters, curve):
        import ramulator

        controller = simpleo3._build_controller(
            ramulator,
            model,
            ddr5(),
            "DDR5",
            observations / "controller.csv",
            parameters,
            mess_curve=curve,
        )
        frontend = ramulator.frontend.External(clock_ratio=1, num_cores=self.cores)
        memory = ramulator.memory_system.GenericDRAM(
            clock_ratio=1,
            controllers=[controller],
            channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
        )
        return {
            "frontend": frontend.to_config(),
            "memory_system": memory.to_config(),
            "batch_trace": str(inputs["replay.csv"]),
        }

    def check_statistics(self, stats, observations, *, candidate):
        batch = stats["batch"]
        if batch["reads_completed"] != self.reads or batch["writes_completed"] != self.writes:
            raise RuntimeError("replay callbacks do not cover the full offered population")
        if candidate:
            check_candidate_callbacks(stats, observations / "controller.csv.ch0")

    def result_fields(self, stats):
        return {
            "diagnostic_only": True,
            "eligible_for_promotion": False,
            "batch_stats": stats["batch"],
        }


def case_id(case):
    return digest_json(case.identity())
