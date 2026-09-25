"""Qualified Lat–Tp case: physical-byte mode is explicit, legacy default retained."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from ramulator_chia.framework.identity import file_sha256
from ramulator_chia.eval.layout import extract_dram_layout
from ramulator_chia.eval.config import REFERENCE
from ramulator_chia.eval.lat_tp_suite import make_controller, SUITE_CONFIG
READ_RATIOS=(100,90,80,70,60,50)
WARMUP_CYCLES=10000
PROBE_REQUESTS=LATENCY_SAMPLE_COUNT=10000
STREAMING_REQUESTS=50000
LATENCY_MEASURE_MODE='random-probe'
CONFIG={**SUITE_CONFIG,'controller_kwargs':dict(REFERENCE)}
@dataclass(frozen=True)
class LatTpCase:
    nop_counter: int
    read_ratio: int
    streaming_only: bool = False
    address_encoding: str = "bank-major-line"
    observation_names = ("controller.csv.ch0",)

    def identity(self):
        return dict(frontend="LatencyThroughputTrace", stage="diagnostic", std="DDR5",
                    suite=CONFIG, point=asdict(self), warmup_cycles=WARMUP_CYCLES,
                    probe_requests=PROBE_REQUESTS, latency_sample_count=LATENCY_SAMPLE_COUNT,
                    streaming_requests=STREAMING_REQUESTS, seed=12345,
                    refresh=False, address_representation=self.address_encoding,
                    driver_sha256=file_sha256(Path(__file__)))

    def input_files(self):
        return {}

    def configuration(self, model, inputs, observations, parameters, curve):
        import ramulator

        if model not in {"oracle", "candidate", "fixedlat", "md1", "wmg1", "mess"}:
            raise ValueError("unknown frozen model")
        dram = ramulator.dram.DDR5(org_preset=CONFIG["org_preset"], timing_preset=CONFIG["timing_preset"])
        frontend = ramulator.frontend.LatencyThroughputTrace(
            clock_ratio=CONFIG["frontend_clock_ratio"], nop_counter=self.nop_counter,
            num_probe_requests=0 if self.streaming_only else PROBE_REQUESTS,
            latency_measure_mode=LATENCY_MEASURE_MODE,
            latency_sample_count=LATENCY_SAMPLE_COUNT,
            num_streaming_requests=STREAMING_REQUESTS if self.streaming_only else 0,
            streaming_only=self.streaming_only, warmup_cycles=WARMUP_CYCLES,
            address_encoding=self.address_encoding,
            seed=12345, read_ratio=self.read_ratio, stream_cls=CONFIG["stream_cls"],
            stagger_stream_rows=CONFIG["stagger_stream_rows"], **extract_dram_layout(dram))
        cfg = {**CONFIG, "controller_kwargs": dict(CONFIG["controller_kwargs"])}
        if model == "oracle":
            cls = ramulator.controller.GenericDDR
            cfg["controller_kwargs"]["controller_plugins"] = [
                ramulator.controller_plugin.ReqTraceRecorder(path=str(observations / "controller.csv"))]
        elif model == "candidate":
            cls = ramulator.controller.Atomic
            cfg["controller_kwargs"].update(parameters)
            cfg["controller_kwargs"]["trace_path"] = str(observations / "controller.csv")
        if model in {"oracle", "candidate"}:
            controller = make_controller(cls, cfg, dram, refresh_enabled=False)
        else:
            # Keep the audited baseline's own admission/posted-write semantics.
            # Unlike GenericDDR/Atomic, these controllers have no queue-size
            # parameter; adding a limiter here would create a different model.
            from ramulator_chia.eval.simpleo3 import _build_controller
            controller = _build_controller(ramulator, model, dram, "DDR5",
                observations / "controller.csv", parameters, mess_curve=curve)
        memory = ramulator.memory_system.GenericDRAM(clock_ratio=1, controllers=[controller],
            channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper())
        return {"frontend": frontend.to_config(), "memory_system": memory.to_config()}

    def check_statistics(self, stats, observations, *, candidate):
        frontend, controller = stats["frontend"], stats["memory_system"]["controller"]
        if self.streaming_only:
            if frontend["streaming_requests_sent"] != STREAMING_REQUESTS:
                raise ValueError("streaming-only point did not complete the suite's request budget")
        elif (frontend["probe_requests_completed"] != LATENCY_SAMPLE_COUNT
              or not math.isfinite(frontend["avg_probe_latency"])
              or frontend["avg_probe_latency"] <= 0):
            raise ValueError("random-probe sample population or latency is invalid")
        if controller["cycles"] <= 0:
            raise ValueError("nonpositive measurement interval")
        if candidate and any(controller[f"peak_inflight_{kind}s"] > REFERENCE[f"{kind}_buffer_size"]
                             for kind in ("read", "write")):
            raise ValueError("atomic controller exceeded admission capacity")
        if self.observation_names and not (observations / "controller.csv.ch0").is_file():
            raise ValueError("missing request observations")

    def result_fields(self, stats):
        return {"diagnostic_only": True, "eligible_for_promotion": False}

