"""Per-request join of oracle and candidate traces.

Stable-ID keyed like matchlib.match() (with the same legacy fallback), but
returns a DataFrame with both sides' latencies and preserves any optional
candidate attribution columns so deviations can be bucketed by cause.

    from ramulator_chia.eval.reqjoin import join
    df = join(oracle_trace, candidate_trace)
    # columns: src, addr, olat, mlat, dv, cls, wbank, wact, wbus, clamped, oarr
"""
import math

import pandas as pd

from . import artifacts as A

_STABLE = ("frontend_id", "frontend_sub_id", "admission_ordinal")


def _load(path):
    df = pd.read_csv(A.resolve(path))
    df = df[df["type"] == 0].copy()
    df["lat"] = df["depart"] - df["arrive"]
    return df


def join(oracle_path, candidate_path):
    o, m = _load(oracle_path), _load(candidate_path)
    if o.empty:
        raise ValueError("oracle trace contains no reads")
    full_oracle_read_mean_latency = float(o["lat"].mean())
    if (not math.isfinite(full_oracle_read_mean_latency) or
            full_oracle_read_mean_latency <= 0):
        raise ValueError("oracle trace has an invalid mean read latency")
    full_oracle_read_count = len(o)
    stable_schema = all(column in o.columns and column in m.columns for column in _STABLE)
    stable_rows = stable_schema and (
        (o["frontend_id"] >= 0).any() or (m["frontend_id"] >= 0).any()
    )
    if stable_rows:
        o = o[o["frontend_id"] >= 0].copy()
        m = m[m["frontend_id"] >= 0].copy()
        keys = ["source", "frontend_id", "frontend_sub_id"]
        if o.duplicated(keys).any() or m.duplicated(keys).any():
            raise ValueError("duplicate stable request identity")
        mode = "stable_id"
    else:
        o["occurrence"] = o.groupby(["source", "addr"], sort=False).cumcount()
        m["occurrence"] = m.groupby(["source", "addr"], sort=False).cumcount()
        keys = ["source", "addr", "occurrence"]
        mode = "legacy_occurrence"

    pairs = o.merge(m, on=keys, how="inner", sort=False, suffixes=("_o", "_m"))
    if stable_rows and len(pairs) and (pairs["addr_o"] != pairs["addr_m"]).any():
        raise ValueError("stable request identity maps to different addresses")
    oracle_addr = "addr_o" if stable_rows else "addr"
    df = pd.DataFrame({
        "src": pairs["source"],
        "addr": pairs[oracle_addr],
        "oarr": pairs["arrive_o"],
        "olat": pairs["lat_o"],
        "mlat": pairs["lat_m"],
        "match_mode": mode,
    })
    for column in ("cls", "wbank", "wact", "wbus", "clamped"):
        model_column = f"{column}_m" if f"{column}_m" in pairs else column
        if model_column in pairs:
            df[column] = pairs[model_column].to_numpy()
    if len(df):
        df["dv"] = df["mlat"] - df["olat"]
    df.attrs.update({
        "oracle_read_mean_latency": full_oracle_read_mean_latency,
        "oracle_read_count": full_oracle_read_count,
        "normalization": "full_oracle_read_mean_latency",
    })
    return df


def bucket_report(df, edges=(0, 60, 120, 250, 500, 1000, 3000, 1e12)):
    L = df.attrs.get("oracle_read_mean_latency")
    if not isinstance(L, (int, float)) or isinstance(L, bool) or not math.isfinite(L) or L <= 0:
        raise ValueError(
            "joined frame lacks a valid full-oracle read mean normalization"
        )
    print(f"n={len(df)} L={L:.1f} mae/L={df.dv.abs().mean()/L:.3f} sgn/L={df.dv.mean()/L:+.3f}")
    print(
        f"{'oracle lat':>16s} {'n':>8s} {'share':>6s} {'mean dv':>8s} "
        f"{'dv share':>8s} {'wact':>7s} {'wbank':>7s} {'cls2':>5s}"
    )
    tot = df.dv.abs().sum()
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = df[(df.olat >= lo) & (df.olat < hi)]
        if len(b) == 0:
            continue
        print(f"[{lo:>6.0f},{hi:>7.0f}) {len(b):8d} {len(b)/len(df):6.3f} {b.dv.mean():+8.1f} "
              f"{b.dv.abs().sum()/tot:8.3f} {b.wact.mean():7.1f} {b.wbank.mean():7.1f} "
              f"{(b.cls==2).mean():5.2f}")
