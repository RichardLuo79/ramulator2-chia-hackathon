"""Trace loader for the pairing-free multicore metrics.
_load_frame is copied verbatim from ramulator_chia/eval/matchlib.py (matcher schema 4);
artifacts.resolve is inlined as _resolve. recorded() is the same as score.recorded()."""
import json, os, pathlib, hashlib
import numpy as np
import pandas as pd

GZIP_SUFFIX = ".gz"

def _resolve(path):
    path = pathlib.Path(path)
    if path.is_file():
        return path
    if not path.name.endswith(GZIP_SUFFIX):
        archived = path.with_name(path.name + GZIP_SUFFIX)
        if archived.is_file():
            return archived
    return path

class _A:
    resolve = staticmethod(_resolve)
A = _A

_BASE_COLUMNS = ("arrive", "depart", "type", "source", "addr")
_STABLE_COLUMNS = ("frontend_id", "frontend_sub_id", "admission_ordinal")
_IDENTITY_COLUMNS = ("src", "frontend_id", "frontend_sub_id")


class RequestIdentityError(ValueError):
    """Well-formed observations do not establish valid cross-run request IDs."""


def _load_frame(path, *, windows=None):
    stored_path = A.resolve(path)
    header = pd.read_csv(stored_path, nrows=0, on_bad_lines="error")
    missing = set(_BASE_COLUMNS) - set(header.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    present_stable = set(_STABLE_COLUMNS) & set(header.columns)
    if present_stable and present_stable != set(_STABLE_COLUMNS):
        missing_stable = set(_STABLE_COLUMNS) - present_stable
        raise ValueError(
            f"{path}: incomplete stable-ID schema; missing {sorted(missing_stable)}"
        )

    columns = list(_BASE_COLUMNS)
    if present_stable:
        columns.extend(_STABLE_COLUMNS)
    reader = pd.read_csv(
        stored_path,
        usecols=columns,
        dtype={column: np.int64 for column in columns},
        on_bad_lines="error",
        **({"chunksize": 200_000} if windows is not None else {}),
    )
    if windows is not None:
        if not present_stable:
            raise ValueError("admission-window filtering requires stable request metadata")
        bounds = {w["core"]: (w["begin"], w["end"]) for w in windows}
        if (len(bounds) != len(windows) or sorted(bounds) != list(range(len(bounds))) or
                any(not 0 <= a <= b for a, b in bounds.values())):
            raise ValueError("invalid per-core admission windows")
        selected = []
        for chunk in reader:
            if (not chunk["source"].isin(bounds).all() or
                    not chunk["type"].isin((0, 1)).all() or
                    (chunk[["arrive", "addr", "admission_ordinal"]] < 0).any().any() or
                    (chunk.loc[chunk["type"] == 0, "depart"] < chunk.loc[chunk["type"] == 0, "arrive"]).any()):
                raise ValueError("invalid raw controller observation, including background traffic")
            begin = chunk["source"].map({c: a for c, (a, _) in bounds.items()})
            end = chunk["source"].map({c: b for c, (_, b) in bounds.items()})
            selected.append(chunk.loc[(chunk["admission_ordinal"] >= begin) &
                                      (chunk["admission_ordinal"] < end)])
        df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)
    else:
        df = reader
    df = df.rename(columns={"source": "src"})

    if not df["type"].isin((0, 1)).all():
        raise ValueError(f"{path}: request type must be 0 (read) or 1 (write)")
    if (df[["arrive", "addr"]] < 0).any().any() or (df["src"] < -1).any():
        raise ValueError(
            f"{path}: arrive/addr must be non-negative and source must be at least -1"
        )
    reads = df["type"] == 0
    if (df.loc[reads, "depart"] < df.loc[reads, "arrive"]).any():
        raise ValueError(f"{path}: read departure precedes arrival")

    if present_stable:
        if (df["frontend_id"] < -1).any():
            raise ValueError(f"{path}: frontend_id must be at least -1")
        if (df["frontend_sub_id"] < 0).any():
            raise ValueError(f"{path}: frontend_sub_id must be non-negative")
        if (df["admission_ordinal"] < -1).any():
            raise ValueError(f"{path}: admission_ordinal must be at least -1")
        stable_eligible = df["frontend_id"] >= 0
        if (df.loc[stable_eligible, "admission_ordinal"] < 0).any():
            raise ValueError(
                f"{path}: stable-ID-eligible rows require a non-negative "
                "admission_ordinal"
            )
        admitted = df["admission_ordinal"] >= 0
        if df.loc[admitted, "admission_ordinal"].duplicated().any():
            raise ValueError(f"{path}: duplicate admission_ordinal in channel trace")

    df["lat"] = df["depart"] - df["arrive"]
    return df.loc[reads].copy(), bool(present_stable)




def recorded(result_json):
    """Trace path is <dir of result.json>/<basename(directory)>/<trace name>, as in score.recorded()."""
    d = json.load(open(result_json))
    o = d['observation']
    base = os.path.join(os.path.dirname(result_json), os.path.basename(d['directory']))
    return dict(trace=os.path.join(base, o['traces']['controller.csv.ch0']['name']),
                windows=o['frontend_stats']['admission_windows'], cycles=o['per_core_cycles'])

def load(trace, windows):
    df, _ = _load_frame(trace, windows=windows)
    return df

def sha256(path):
    h = hashlib.sha256()
    with open(_resolve(path), 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()
