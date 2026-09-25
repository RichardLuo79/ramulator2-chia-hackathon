"""Fail-closed outcome analysis for paired SimpleO3 logical request traces.

The scored SimpleO3 trace identifies a logical request by
``(source, frontend_id, frontend_sub_id)``.  This module verifies that two
closed-loop runs contain exactly the same logical population before reporting
how requests moved among LLC hit, MSHR-merge, and miss-owner paths.  It is a
descriptive audit helper: path outcomes never change which requests are paired
or scored by the latency matcher.

The trailing ``llc_path`` column is optional so old logical traces remain
auditable for population identity.  A pair must either contain the column on
both sides or omit it on both sides; path diagnostics are ``None`` when it is
absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter
from collections.abc import Iterator, Sequence

from ramulator_chia.eval import artifacts as A

LOGICAL_OUTCOME_SCHEMA_VERSION = 1

TRACE_COLUMNS = (
    "arrive",
    "depart",
    "type",
    "source",
    "addr",
    "frontend_id",
    "frontend_sub_id",
    "admission_ordinal",
)
LLC_PATH_COLUMN = "llc_path"
LLC_PATH_LABELS = ("hit", "mshr_merge", "miss_owner")

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_MISSING = object()


class LogicalTraceError(ValueError):
    """Raised when a logical trace or paired population is not trustworthy."""


def _read_header(path: pathlib.Path) -> bool:
    try:
        with A.open_text(path, newline="") as stream:
            rows = csv.reader(stream, strict=True)
            header = tuple(next(rows))
    except StopIteration as exc:
        raise LogicalTraceError(f"{path}: empty logical request trace") from exc
    except (OSError, csv.Error) as exc:
        raise LogicalTraceError(f"{path}: cannot read logical request trace: {exc}") from exc

    if header == TRACE_COLUMNS:
        return False
    if header == (*TRACE_COLUMNS, LLC_PATH_COLUMN):
        return True
    raise LogicalTraceError(
        f"{path}: logical trace columns must be exactly {TRACE_COLUMNS!r} or "
        f"{(*TRACE_COLUMNS, LLC_PATH_COLUMN)!r}; got {header!r}"
    )


def _parse_int(path: pathlib.Path, line_number: int, column: str, text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise LogicalTraceError(
            f"{path}:{line_number}: {column} must be an integer, got {text!r}"
        ) from exc
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise LogicalTraceError(f"{path}:{line_number}: {column} is outside signed 64-bit range")
    return value


def _rows(path: pathlib.Path, has_llc_path: bool) -> Iterator[tuple[int, ...]]:
    expected_columns = (*TRACE_COLUMNS, LLC_PATH_COLUMN) if has_llc_path else TRACE_COLUMNS
    try:
        with A.open_text(path, newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                observed_header = tuple(next(reader))
            except StopIteration as exc:
                raise LogicalTraceError(f"{path}: empty logical request trace") from exc
            if observed_header != expected_columns:
                raise LogicalTraceError(
                    f"{path}: logical trace header changed while it was being analyzed; "
                    f"expected {expected_columns!r}, got {observed_header!r}"
                )
            for line_number, raw in enumerate(reader, start=2):
                if len(raw) != len(expected_columns):
                    raise LogicalTraceError(
                        f"{path}:{line_number}: expected {len(expected_columns)} columns, "
                        f"got {len(raw)}"
                    )
                values = tuple(
                    _parse_int(path, line_number, column, text)
                    for column, text in zip(expected_columns, raw, strict=True)
                )
                arrive, depart, request_type, source, addr, frontend_id, sub_id, ordinal = values[
                    :8
                ]
                if arrive < 0 or depart < 0 or addr < 0:
                    raise LogicalTraceError(
                        f"{path}:{line_number}: arrive, depart, and addr must be non-negative"
                    )
                if depart < arrive:
                    raise LogicalTraceError(
                        f"{path}:{line_number}: departure precedes logical admission"
                    )
                if request_type not in (0, 1):
                    raise LogicalTraceError(
                        f"{path}:{line_number}: request type must be 0 (read) or 1 (write)"
                    )
                if source < 0:
                    raise LogicalTraceError(
                        f"{path}:{line_number}: SimpleO3 source must be non-negative"
                    )
                if frontend_id < 0 or sub_id < 0 or ordinal < 0:
                    raise LogicalTraceError(
                        f"{path}:{line_number}: stable frontend IDs and admission ordinal "
                        "must be non-negative"
                    )
                if has_llc_path and values[8] not in range(len(LLC_PATH_LABELS)):
                    raise LogicalTraceError(
                        f"{path}:{line_number}: llc_path must be 0 (hit), 1 (MSHR merge), "
                        "or 2 (miss owner)"
                    )
                yield (line_number, *values)
    except LogicalTraceError:
        raise
    except (OSError, csv.Error) as exc:
        raise LogicalTraceError(f"{path}: cannot read logical request trace: {exc}") from exc


def _increment_path(counts: dict[int, list[int]], source: int, path: int) -> None:
    counts.setdefault(source, [0, 0, 0])[path] += 1


def _format_source_counts(counts: Counter[int]) -> dict[str, int]:
    return {str(source): counts[source] for source in sorted(counts)}


def _format_path_counts(counts: dict[int, list[int]]) -> dict[str, object]:
    total = [0, 0, 0]
    for values in counts.values():
        for path, count in enumerate(values):
            total[path] += count
    return {
        "all": total,
        "by_source": {str(source): counts[source] for source in sorted(counts)},
    }


def _identity_text(identity: tuple[int, int, int]) -> str:
    source, frontend_id, sub_id = identity
    return f"(source={source}, frontend_id={frontend_id}, frontend_sub_id={sub_id})"


def analyze_pair(oracle_path, model_path) -> dict[str, object]:
    """Validate and describe a paired SimpleO3 logical request population.

    The returned object is JSON-serializable.  Transition-matrix rows are
    oracle paths and columns are model paths, both in numeric path order
    ``0, 1, 2``.  ``affected_fraction`` is the fraction of paired logical
    requests for which either side took a non-hit path.

    Any duplicate stable identity or admission ordinal, missing identity, or
    address/type disagreement raises :class:`LogicalTraceError` rather than
    returning partial diagnostics.
    """
    oracle_path = pathlib.Path(oracle_path)
    model_path = pathlib.Path(model_path)
    oracle_has_path = _read_header(oracle_path)
    model_has_path = _read_header(model_path)
    if oracle_has_path != model_has_path:
        raise LogicalTraceError(
            "paired logical traces must either both contain llc_path or both omit it: "
            f"oracle={oracle_has_path}, model={model_has_path}"
        )
    has_path = oracle_has_path

    # A value becomes None after its oracle identity is consumed.  Keeping the
    # key lets us distinguish a duplicate model identity from a model-only key
    # without allocating a second full-population key set.
    oracle_records: dict[tuple[int, int, int], tuple[int, int, int | None] | None] = {}
    oracle_ordinals: set[int] = set()
    oracle_sources: Counter[int] = Counter()
    oracle_path_counts: dict[int, list[int]] = {}
    oracle_rows = 0
    for row in _rows(oracle_path, has_path):
        line_number, _, _, request_type, source, addr, frontend_id, sub_id, ordinal, *tail = row
        identity = (source, frontend_id, sub_id)
        if identity in oracle_records:
            raise LogicalTraceError(
                f"{oracle_path}:{line_number}: duplicate stable logical identity "
                f"{_identity_text(identity)}"
            )
        if ordinal in oracle_ordinals:
            raise LogicalTraceError(
                f"{oracle_path}:{line_number}: duplicate admission_ordinal {ordinal}"
            )
        path = tail[0] if has_path else None
        oracle_records[identity] = (addr, request_type, path)
        oracle_ordinals.add(ordinal)
        oracle_sources[source] += 1
        if has_path:
            _increment_path(oracle_path_counts, source, path)
        oracle_rows += 1
    if oracle_rows == 0:
        raise LogicalTraceError(f"{oracle_path}: logical request trace contains no rows")

    # Ordinals are only needed for within-side uniqueness and can be released
    # before validating the peer, keeping peak memory bounded by one trace map
    # plus one ordinal set.
    del oracle_ordinals

    model_ordinals: set[int] = set()
    model_only_seen: set[tuple[int, int, int]] = set()
    model_sources: Counter[int] = Counter()
    model_path_counts: dict[int, list[int]] = {}
    transition_matrix = [[0, 0, 0] for _ in LLC_PATH_LABELS]
    affected_count = 0
    changed_path_count = 0
    model_rows = 0
    model_only_count = 0
    model_only_examples: list[tuple[int, int, int]] = []
    address_mismatch_count = 0
    address_mismatch_examples: list[str] = []
    type_mismatch_count = 0
    type_mismatch_examples: list[str] = []

    for row in _rows(model_path, has_path):
        line_number, _, _, request_type, source, addr, frontend_id, sub_id, ordinal, *tail = row
        identity = (source, frontend_id, sub_id)
        if ordinal in model_ordinals:
            raise LogicalTraceError(
                f"{model_path}:{line_number}: duplicate admission_ordinal {ordinal}"
            )
        model_ordinals.add(ordinal)
        model_sources[source] += 1
        model_llc_path = tail[0] if has_path else None
        if has_path:
            _increment_path(model_path_counts, source, model_llc_path)
        model_rows += 1

        oracle_record = oracle_records.get(identity, _MISSING)
        if oracle_record is None or identity in model_only_seen:
            raise LogicalTraceError(
                f"{model_path}:{line_number}: duplicate stable logical identity "
                f"{_identity_text(identity)}"
            )
        if oracle_record is _MISSING:
            model_only_seen.add(identity)
            model_only_count += 1
            if len(model_only_examples) < 3:
                model_only_examples.append(identity)
            continue

        oracle_addr, oracle_type, oracle_llc_path = oracle_record
        if addr != oracle_addr:
            address_mismatch_count += 1
            if len(address_mismatch_examples) < 3:
                address_mismatch_examples.append(
                    f"{_identity_text(identity)}: oracle={oracle_addr}, model={addr}"
                )
        if request_type != oracle_type:
            type_mismatch_count += 1
            if len(type_mismatch_examples) < 3:
                type_mismatch_examples.append(
                    f"{_identity_text(identity)}: oracle={oracle_type}, model={request_type}"
                )
        if has_path:
            transition_matrix[oracle_llc_path][model_llc_path] += 1
            if oracle_llc_path != 0 or model_llc_path != 0:
                affected_count += 1
            if oracle_llc_path != model_llc_path:
                changed_path_count += 1
        oracle_records[identity] = None

    if model_rows == 0:
        raise LogicalTraceError(f"{model_path}: logical request trace contains no rows")

    oracle_only_count = 0
    oracle_only_examples: list[tuple[int, int, int]] = []
    for identity, record in oracle_records.items():
        if record is not None:
            oracle_only_count += 1
            if len(oracle_only_examples) < 3:
                oracle_only_examples.append(identity)
    paired_keys = oracle_rows - oracle_only_count

    problems = []
    if oracle_only_count or model_only_count:
        problems.append(
            "missing logical identities: "
            f"missing_from_model={oracle_only_count} "
            f"examples={[_identity_text(key) for key in oracle_only_examples]}; "
            f"missing_from_oracle={model_only_count} "
            f"examples={[_identity_text(key) for key in model_only_examples]}"
        )
    if address_mismatch_count:
        problems.append(
            "stable identities map to different addresses: "
            f"count={address_mismatch_count}, examples={address_mismatch_examples}"
        )
    if type_mismatch_count:
        problems.append(
            "stable identities map to different request types: "
            f"count={type_mismatch_count}, examples={type_mismatch_examples}"
        )
    if problems:
        raise LogicalTraceError("; ".join(problems))

    population = {
        "oracle_rows": oracle_rows,
        "model_rows": model_rows,
        "paired_keys": paired_keys,
        "oracle_only_keys": oracle_only_count,
        "model_only_keys": model_only_count,
        "duplicate_keys_oracle": 0,
        "duplicate_keys_model": 0,
        "duplicate_admission_ordinals_oracle": 0,
        "duplicate_admission_ordinals_model": 0,
        "address_mismatch_pairs": address_mismatch_count,
        "type_mismatch_pairs": type_mismatch_count,
        "exact_one_to_one": True,
        "rows_by_source": {
            "oracle": _format_source_counts(oracle_sources),
            "model": _format_source_counts(model_sources),
        },
    }
    if has_path:
        path_counts: dict[str, object] | None = {
            "oracle": _format_path_counts(oracle_path_counts),
            "model": _format_path_counts(model_path_counts),
        }
        affected_fraction: float | None = affected_count / paired_keys
        changed_path_fraction: float | None = changed_path_count / paired_keys
        matrix: list[list[int]] | None = transition_matrix
    else:
        path_counts = None
        affected_count = None
        affected_fraction = None
        changed_path_count = None
        changed_path_fraction = None
        matrix = None

    return {
        "logical_outcome_schema_version": LOGICAL_OUTCOME_SCHEMA_VERSION,
        "population": population,
        "llc_path_available": has_path,
        "llc_path_labels": {str(i): label for i, label in enumerate(LLC_PATH_LABELS)},
        "path_counts": path_counts,
        "transition_matrix": matrix,
        "transition_orientation": {"rows": "oracle", "columns": "model"},
        "affected_count": affected_count,
        "affected_fraction": affected_fraction,
        "changed_path_count": changed_path_count,
        "changed_path_fraction": changed_path_fraction,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    json.dump(analyze_pair(args.oracle, args.model), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
