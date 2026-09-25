"""Shared SimpleO3 configuration and observation checks, without a runner.

Legacy command-line evaluation and CHIA tasks use the same controller settings,
logical request population and full-drain validation. This module does not
schedule work, select candidates or read a campaign directory.
"""

import csv
import pathlib
import re

from . import artifacts as A
from . import config as C

REQUEST_TRACE_SCOPE = "simpleo3_logical_llc"
CONTROLLER_TRACE_SCOPE = "dram_controller"
REQUEST_TRACE_TIMEBASE = "simpleo3_frontend_cycles"
FIXED_ISSUE_ROI = True
ALL_MODELS = ("oracle",) + C.MODEL_ORDER
SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
LOGICAL_TRACE_COLUMNS = (
    "arrive",
    "depart",
    "type",
    "source",
    "addr",
    "frontend_id",
    "frontend_sub_id",
    "admission_ordinal",
    "llc_path",
)
LOGICAL_PATH_CODES = {"0": "hit", "1": "mshr_merge", "2": "miss_owner"}


def _controller_config(model, candidate_overrides):
    if model == "oracle":
        return {
            **C.REFERENCE,
            "scheduler": "FRFCFSRowHit",
            "refresh_manager": "NoRefresh",
            "row_policy": "Open",
        }
    if model == "candidate":
        return {
            **C.CANDIDATE_RESOURCES,
            "refresh": "none",
            **candidate_overrides,
        }
    if model == "fixedlat":
        return {"latency": -1, "pipe": 1}
    if model == "md1":
        return {"phase_ticks": 10_000, "smoothing": 0.5}
    if model == "wmg1":
        return {"window_ns": 10_000}
    if model == "mess":
        return {"window_accesses": 1_000, "converge": 0.05}
    raise ValueError(f"unknown SimpleO3 model {model!r}")


def _build_controller(
    ramulator, model, dram, std, trace_path, candidate_overrides, *, mess_curve=None
):
    config = _controller_config(model, candidate_overrides)
    if model == "oracle":
        return ramulator.controller.GenericDDR(
            dram=dram,
            scheduler=ramulator.scheduler.FRFCFSRowHit(),
            refresh_manager=ramulator.refresh_manager.NoRefresh(),
            row_policy=ramulator.row_policy.Open(),
            addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
            controller_plugins=[ramulator.controller_plugin.ReqTraceRecorder(path=str(trace_path))],
            **C.REFERENCE,
        )
    if model == "candidate":
        reserved = {"dram", "addr_mapper", "trace_path", "refresh", "model_library"}
        overlap = reserved & set(candidate_overrides)
        if overlap:
            raise ValueError(
                "candidate overrides contain evaluator-owned keys: " + ", ".join(sorted(overlap))
            )
        return ramulator.controller.Atomic(
            dram=dram,
            addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
            trace_path=str(trace_path),
            **config,
        )

    cls = getattr(ramulator.controller, C.MODEL_IMPL[model])
    kwargs = {**config, "dram": dram, "trace_path": str(trace_path)}
    if model == "mess":
        try:
            curve = mess_curve if mess_curve is not None else C.MESS_CURVES[std]
        except KeyError as exc:
            raise ValueError(
                f"MESS has no calibrated curve for {std}; calibrate it before evaluation"
            ) from exc
        kwargs["curve_path"] = str(curve)
    return cls(**kwargs)


def build_config(
    model,
    *,
    standard,
    traces,
    instructions_per_core,
    request_trace,
    controller_trace,
    candidate_overrides=None,
    mess_curve=None,
):
    """Build the established single-channel configuration from explicit inputs.

    Paths come from the trusted caller, never from implicit workload discovery.
    The caller establishes window eligibility; small windows are useful only in
    labeled infrastructure tests. No simulation or scheduling happens here.
    """
    import ramulator

    if type(instructions_per_core) is not int or instructions_per_core <= 0 or not traces:
        raise ValueError("SimpleO3 needs input traces and a positive instruction window")
    dram = getattr(ramulator.dram, standard)(
        org_preset=C.STD[standard]["org"], timing_preset=C.STD[standard]["timing"]
    )
    controller = _build_controller(
        ramulator,
        model,
        dram,
        standard,
        controller_trace,
        candidate_overrides or {},
        mess_curve=mess_curve,
    )
    frontend = ramulator.frontend.SimpleO3(
        clock_ratio=8,
        traces=[str(trace) for trace in traces],
        num_expected_insts=instructions_per_core,
        llc_num_mshr_per_core=16,
        request_trace_path=str(request_trace),
        translation=ramulator.translation.NoTranslation(max_addr=2**33),
    )
    memory = ramulator.memory_system.GenericDRAM(
        clock_ratio=3,
        controllers=[controller],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
    )
    frontend_config = frontend.to_config()
    # Keep the shared builder usable before the generated Python wrapper is
    # refreshed. The native frontend is the authority on trace grammar/counts.
    frontend_config["allow_trace_wrap"] = False
    return {"frontend": frontend_config, "memory_system": memory.to_config()}


def audit_candidate_trace(path, stats):
    """Check committed departures against callback counters, streaming the CSV."""
    counts = {"0": 0, "1": 0}
    latency_sum = 0
    with A.open_text(path, newline="") as stream:
        for row in csv.DictReader(stream):
            counts[row["type"]] += 1
            latency = int(row["depart"]) - int(row["arrive"])
            if latency <= 0:
                raise RuntimeError("candidate committed a non-positive latency")
            if row["type"] == "0":
                latency_sum += latency
    if counts["0"] != stats["num_read_reqs"] or counts["1"] != stats["num_write_reqs"]:
        raise RuntimeError("candidate admission trace/count mismatch")
    if latency_sum != stats["read_latency"]:
        raise RuntimeError("candidate committed departures disagree with callback latency sum")


def _validate_logical_request_trace(path):
    """Require the fixed, matcher-facing SimpleO3 logical trace contract."""
    path = pathlib.Path(path)
    if not A.exists(path):
        raise RuntimeError(f"simulation produced no non-empty logical request trace: {path}")
    with A.open_text(path, newline="") as stream:
        rows = csv.reader(stream)
        try:
            header = tuple(next(rows))
        except StopIteration as exc:
            raise RuntimeError(f"logical request trace is empty: {path}") from exc
        if header != LOGICAL_TRACE_COLUMNS:
            raise RuntimeError(
                f"logical request trace {path} has columns {header!r}; "
                f"expected exactly {LOGICAL_TRACE_COLUMNS!r}"
            )
        row_count = 0
        for line_number, row in enumerate(rows, start=2):
            row_count += 1
            if len(row) != len(LOGICAL_TRACE_COLUMNS):
                raise RuntimeError(
                    f"logical request trace {path}:{line_number} has {len(row)} "
                    f"columns; expected {len(LOGICAL_TRACE_COLUMNS)}"
                )
            try:
                llc_path = int(row[-1])
            except ValueError as exc:
                raise RuntimeError(
                    f"logical request trace {path}:{line_number} has a "
                    f"non-integer llc_path {row[-1]!r}"
                ) from exc
            if llc_path not in (0, 1, 2):
                raise RuntimeError(
                    f"logical request trace {path}:{line_number} has invalid "
                    f"llc_path {llc_path}; expected 0 (hit), 1 (merge), or 2 (owner)"
                )
        if row_count == 0:
            raise RuntimeError(f"logical request trace contains no request rows: {path}")
    return row_count


def _validate_fixed_roi_frontend_stats(
    stats,
    *,
    core_count,
    insts_per_core,
    logical_rows,
):
    """Fail closed unless the frontend reports a fully drained fixed ROI."""
    if not isinstance(stats, dict):
        raise RuntimeError("frontend_stats must be an object")

    def exact_nonnegative_int(key):
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"frontend_stats.{key} must be a non-negative integer")
        return value

    for core_id in range(core_count):
        issued = exact_nonnegative_int(f"insts_issued_core_{core_id}")
        if issued != insts_per_core:
            raise RuntimeError(
                f"frontend_stats.insts_issued_core_{core_id}={issued}; "
                f"expected fixed ROI {insts_per_core}"
            )
    live = exact_nonnegative_int("logical_requests_live")
    if live != 0:
        raise RuntimeError(f"frontend_stats.logical_requests_live={live}; expected a drained LLC")
    completed = exact_nonnegative_int("logical_requests_completed")
    if completed != logical_rows:
        raise RuntimeError(
            f"frontend_stats.logical_requests_completed={completed}; "
            f"trace contains {logical_rows} rows"
        )
    path_total = sum(
        exact_nonnegative_int(key)
        for key in (
            "logical_requests_hit",
            "logical_requests_mshr_merge",
            "logical_requests_miss_owner",
        )
    )
    if path_total != completed:
        raise RuntimeError(
            f"frontend logical path counters sum to {path_total}; "
            f"expected {completed} completed requests"
        )
    internal_generated = exact_nonnegative_int("internal_writebacks_generated")
    internal_completed = exact_nonnegative_int("internal_writebacks_completed")
    internal_live = exact_nonnegative_int("internal_writebacks_live")
    if internal_live != 0 or internal_completed != internal_generated:
        raise RuntimeError(
            "frontend internal writebacks were not fully drained: "
            f"generated={internal_generated}, completed={internal_completed}, "
            f"live={internal_live}"
        )
