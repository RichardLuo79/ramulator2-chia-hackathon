"""Cross-frontend open-loop replay diagnostic for candidate controllers.

Replays the RECORDED oracle arrival streams of every frontend (SimpleO3
eval_out/simpleo3/<std>/<wl>/oracle, ChampSim eval_out/champsim/<tag>/
*_oracle_req, gem5 eval_out/gem5/<std>/<bench>/oracle) through the atomic
model with candidate overrides (BatchSim, no feedback) and reports the
per-request signed drift/L and MAE/L per stream plus per-frontend means.
Fast (minutes) and feedback-free: a candidate that moves ChampSim/gem5
streams in a different direction here merit investigation. This is not a
promotion metric and cannot substitute for closed-loop evaluation. CAVEAT:
SimpleO3 MSHR-clump traces
replayed open-loop can run away (mcf_s: +32xL — the closed loop's own
backpressure is what bounds them); read the SimpleO3 column per stream,
weight the ChampSim/gem5 columns.

Usage: python tools/eval/replay_screen.py '<overrides json>' [--std DDR5]
       [--n 150000] [--frontends simpleo3,champsim,gem5] [--champsim-tag DDR5_champreq]
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from eval import artifacts as A
from eval import config as C


def _load_stream(path, n):
    if n <= 0:
        raise ValueError("replay request limit must be positive")
    frame = pd.read_csv(A.resolve(path), on_bad_lines="error")
    required = ("arrive", "depart", "type", "source", "addr")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path}: missing required trace columns {missing}")
    for column in required:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"{path}: {column} must contain only integers")
        frame[column] = numeric.astype(np.int64)
    if not frame["type"].isin((0, 1)).all():
        raise ValueError(f"{path}: request type must be 0 (read) or 1 (write)")
    if ((frame[["arrive", "addr"]] < 0).any().any()
            or (frame["source"] < -1).any()):
        raise ValueError(
            f"{path}: arrive/addr must be non-negative and source must be at least -1"
        )
    reads = frame["type"] == 0
    if (frame.loc[reads, "depart"] < frame.loc[reads, "arrive"]).any():
        raise ValueError(f"{path}: read departure precedes arrival")

    # Newer recorders may provide an explicit admission sequence. Older raw
    # traces fall back to their file ordinal, preserving equal-arrival order.
    ordinal = next(
        (column for column in ("admission_ordinal", "admit_seq", "sequence")
         if column in frame),
        None,
    )
    if ordinal is None:
        ordinal = "_admission_ordinal"
        frame[ordinal] = np.arange(len(frame), dtype=np.int64)
    else:
        values = pd.to_numeric(frame[ordinal], errors="raise")
        if (values.isna().any() or (values % 1 != 0).any()
                or (values < 0).any() or values.duplicated().any()):
            raise ValueError(f"{path}: {ordinal} must contain unique integers")
        frame[ordinal] = values.astype(np.int64)
    return (frame.sort_values(["arrive", ordinal], kind="stable")
            .reset_index(drop=True)
            .iloc[:n])


def replay(path, std, kw, n):
    import ramulator
    o = _load_stream(path, n)
    s = C.STD[std]
    dram = getattr(ramulator.dram, std)(
        org_preset=s["org"], timing_preset=s["timing"], **s["extra"],
    )
    controller_config = {**C.CANDIDATE_RESOURCES, **kw}
    ctrl = ramulator.controller.Atomic(
        dram=dram, addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(), refresh="none",
        **controller_config)
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    dep = np.array(ramulator.BatchSim(mem).run(
        o.addr.values.tolist(), o.type.values.tolist(), o.arrive.values.tolist()))
    rd = o.type.values == 0
    ml = dep[rd] - o.arrive.values[rd]
    ol = (o.depart - o.arrive).values[rd]
    L = float(ol.mean())
    return float((ml - ol).mean() / L), float(np.abs(ml - ol).mean() / L)


def _simpleo3_controller_streams(std):
    """Yield DRAM-controller streams, never the logical LLC score trace."""
    pattern = C.OUT / "simpleo3" / std / "*" / "oracle"
    for oracle_dir_text in sorted(glob.glob(str(pattern))):
        oracle_dir = pathlib.Path(oracle_dir_text)
        manifest_path = oracle_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot identify SimpleO3 replay trace from {manifest_path}: {exc}"
            ) from exc
        if manifest.get("request_trace_scope") == "simpleo3_logical_llc":
            trace = oracle_dir / "controller_trace.csv.ch0"
            provenance = manifest.get("controller_trace")
        else:
            # Schema <=3 stored the controller lifecycle stream under the
            # historical scored-trace filename.
            trace = oracle_dir / "trace.csv.ch0"
            provenance = manifest.get("raw_trace")
        if not A.exists(trace) or not A.raw_provenance_matches(trace, provenance):
            raise RuntimeError(
                f"missing or stale SimpleO3 controller replay trace: {trace}"
            )
        yield oracle_dir.parent.name, str(trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides")
    ap.add_argument("--std", default="DDR5")
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--frontends", default="simpleo3,champsim,gem5")
    ap.add_argument("--champsim-tag", default="DDR5_champreq")
    a = ap.parse_args()
    try:
        kw = json.loads(a.overrides)
    except json.JSONDecodeError as exc:
        ap.error(f"overrides are not valid JSON: {exc}")
    if not isinstance(kw, dict):
        ap.error("overrides must decode to a JSON object")
    fes = a.frontends.split(",")
    streams = []
    if "simpleo3" in fes:
        for wl, p in _simpleo3_controller_streams(a.std):
            if wl.startswith("Mix"):
                # Matching-by-source is unnecessary here; skip mixes for speed.
                continue
            streams.append(("simpleo3", wl, p))
    if "champsim" in fes:
        pattern = C.OUT / "champsim" / a.champsim_tag / "*_oracle_req.csv.ch0"
        for path in A.glob_logical(pattern):
            p = str(path)
            streams.append(("champsim", path.name.replace("_oracle_req.csv.ch0", ""), p))
    if "gem5" in fes:
        pattern = C.OUT / "gem5" / a.std / "*" / "oracle" / "trace.csv.ch0"
        for path in A.glob_logical(pattern):
            streams.append(("gem5", path.parents[1].name, str(path)))
    rows = []
    for fe, name, p in streams:
        try:
            sgn, mae = replay(p, a.std, kw, a.n)
        except Exception as e:  # noqa: BLE001 — report and continue
            print(f"  {fe:9s} {name:24s} FAILED {e}", file=sys.stderr)
            continue
        rows.append((fe, name, sgn, mae))
        print(f"  {fe:9s} {name:24s} sgn {sgn:+.3f}  mae {mae:.3f}", flush=True)
    print(f"overrides {json.dumps(kw)}")
    for fe in fes:
        r = [x for x in rows if x[0] == fe]
        if r:
            worst = max(r, key=lambda x: abs(x[2]))
            print(f"  {fe:9s} n={len(r):2d}  mean|sgn|/L {np.mean([abs(x[2]) for x in r]):.3f}  "
                  f"worst sgn {worst[2]:+.3f} ({worst[1]})  "
                  f"mean mae/L {np.mean([x[3] for x in r]):.3f}")
    print("SCREEN_DONE")


if __name__ == "__main__":
    main()
