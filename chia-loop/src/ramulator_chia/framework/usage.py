"""Pure token normalization and tariff estimates from preserved native evidence.

Counting differences are reviewed protocol data, not branches on model names.
The original counters remain evidence. Missing counters are not silently zero;
derived values name their derivation. Estimates are token-tariff equivalents,
not invoices, subscription charges or account quota percentages.
"""

from __future__ import annotations

import json
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

from pydantic import Field, model_validator

from .config import Name, StrictRecord

PROTOCOL_PATH = Path(__file__).with_name("usage_protocols.json")
TOKEN_FIELDS = (
    "input_tokens",
    "uncached_input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "tool_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "non_reasoning_output_tokens",
    "total_tokens",
)


class InvalidUsage(ValueError):
    """Reported counters or their declared interpretation are inconsistent."""


def protocols() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def at_path(value, path):
    for name in path:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise InvalidUsage("usage field traverses a non-object")
        value = value.get(name)
    return value


def _sum(*values):
    return None if any(value is None for value in values) else sum(values)


def normalize(raw: dict | None, protocol: str) -> dict:
    """Normalize one reported object, without deciding its accounting scope."""
    rule = protocols()["protocols"][protocol]
    if raw is None:
        return {"tokens": None, "derived": [], "protocol": protocol}
    if not isinstance(raw, dict):
        raise InvalidUsage("provider usage must be an object")
    iterations = raw.get(rule.get("iterations_field"))
    if iterations is not None:
        if not isinstance(iterations, list) or not iterations or any(
            not isinstance(row, dict) or row.get("type") not in {"message", "compaction"}
            for row in iterations
        ):
            raise InvalidUsage("unsupported provider sampling-iteration usage")
        parts = [
            {"type": row["type"], "usage": normalize(row, rule["iteration_protocol"])}
            for row in iterations
        ]
        return {
            "tokens": {name: _sum(*(p["usage"]["tokens"][name] for p in parts))
                       for name in TOKEN_FIELDS},
            "derived": ["sampling_iterations_replace_top_level_usage"],
            "protocol": protocol,
            "iterations": parts,
        }
    values = {name: None for name in TOKEN_FIELDS}
    derived = []
    for name, path in rule["fields"].items():
        value = at_path(raw, path)
        if value is not None and (type(value) is not int or value < 0):
            raise InvalidUsage("token counts must be non-negative integers")
        values[name] = value
    for name in rule["zero_fields"]:
        values[name] = 0
        derived.append(name + ":not_a_separate_counter_in_this_protocol")

    # The provider total can identify one missing disjoint component. This is
    # arithmetic evidence, not a default-zero assumption for absent counters.
    parts = ["input_tokens", "output_tokens", "tool_input_tokens"]
    if not rule["output_includes_reasoning"]:
        parts.append("reasoning_output_tokens")
    if not rule["input_includes_cache"]:
        parts += ["cached_input_tokens", "cache_write_input_tokens"]
    missing = [name for name in parts if values[name] is None]
    total = values["total_tokens"]
    if total is not None:
        known = sum(values[name] for name in parts if values[name] is not None)
        if known > total or (not missing and known != total):
            raise InvalidUsage("disjoint provider counters disagree with total")
        if len(missing) == 1:
            values[missing[0]] = total - known
            derived.append(missing[0] + ":provider_total_minus_other_components")

    inp, cached, written = (
        values[name] for name in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens")
    )
    tool = values["tool_input_tokens"]
    if rule["input_includes_cache"]:
        if inp is not None and sum(v for v in (cached, written) if v is not None) > inp:
            raise InvalidUsage("cache counters exceed provider input")
        uncached = None if None in (inp, cached, written) else inp - cached - written
        values["input_tokens"] = _sum(inp, tool)
        values["uncached_input_tokens"] = _sum(uncached, tool)
    else:
        values["uncached_input_tokens"] = _sum(inp, tool)
        values["input_tokens"] = _sum(inp, cached, written, tool)
    short, long = values["cache_write_5m_tokens"], values["cache_write_1h_tokens"]
    if written is not None:
        known_writes = sum(value for value in (short, long) if value is not None)
        if known_writes > written or (
            short is not None and long is not None and known_writes != written
        ):
            raise InvalidUsage("cache-creation TTL counters disagree")
    out, reasoning = values["output_tokens"], values["reasoning_output_tokens"]
    if rule["output_includes_reasoning"]:
        if out is not None and reasoning is not None and reasoning > out:
            raise InvalidUsage("reasoning tokens exceed output")
        values["non_reasoning_output_tokens"] = (
            None if None in (out, reasoning) else out - reasoning
        )
    else:
        values["non_reasoning_output_tokens"] = out
        values["output_tokens"] = _sum(out, reasoning)
    derived_total = _sum(values["input_tokens"], values["output_tokens"])
    if total is not None and derived_total is not None and total != derived_total:
        raise InvalidUsage("normalized counts disagree with provider total")
    if total is None and derived_total is not None:
        values["total_tokens"] = derived_total
        derived.append("total_tokens:input_plus_output")
    return {"tokens": values, "derived": derived, "protocol": protocol}


class Rates(StrictRecord):
    # Decimal strings preserve the reviewed tariff exactly through JSON and SQL.
    input: str = Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")
    cached_input: str = Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")
    cache_write: str = Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")
    output: str = Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")
    cache_write_5m: str | None = Field(default=None, pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")
    cache_write_1h: str | None = Field(default=None, pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")

    @model_validator(mode="after")
    def ttl_rates_together(self):
        if (self.cache_write_5m is None) != (self.cache_write_1h is None):
            raise ValueError("declare both cache-TTL rates or neither")
        return self


class Tier(StrictRecord):
    maximum_input_tokens: int | None = Field(ge=1)
    rates: Rates


class RequestBounds(StrictRecord):
    maximum_input_tokens: int = Field(ge=1)
    maximum_output_tokens: int = Field(ge=1)
    source: str = Field(min_length=1)
    checked_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class Tariff(StrictRecord):
    schema_version: int = Field(ge=1, le=1)
    model: Name
    accepted_response_models: tuple[Name, ...] = ()
    source: str = Field(min_length=1)
    checked_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    units: str = Field(pattern=r"^USD per million tokens$")
    tiers: tuple[Tier, ...]
    request_bounds: RequestBounds | None = None

    @model_validator(mode="after")
    def complete_tiers(self):
        if not self.tiers or self.tiers[-1].maximum_input_tokens is not None:
            raise ValueError("tariff must end with an unbounded input tier")
        limits = [tier.maximum_input_tokens for tier in self.tiers[:-1]]
        if any(limit is None for limit in limits) or limits != sorted(set(limits)):
            raise ValueError("tariff input tiers must increase strictly")
        return self


def _input_range(tokens, rates):
    write_rates = [Decimal(rates.cache_write)]
    written = tokens["cache_write_input_tokens"]
    short, long = tokens["cache_write_5m_tokens"], tokens["cache_write_1h_tokens"]
    if rates.cache_write_5m is not None:
        write_rates = [Decimal(rates.cache_write_5m), Decimal(rates.cache_write_1h)]
    if len(write_rates) == 2 and written and short is not None and long is not None:
        exact = (short * write_rates[0] + long * write_rates[1]) / written
        write_rates = [exact]
    components = [
        (tokens["uncached_input_tokens"], [Decimal(rates.input)]),
        (tokens["cached_input_tokens"], [Decimal(rates.cached_input)]),
        (written, write_rates),
    ]
    lower = sum(count * min(rate) for count, rate in components if count is not None)
    upper = sum(count * max(rate) for count, rate in components if count is not None)
    missing_rates = [rate for count, rates in components if count is None for rate in rates]
    if missing_rates:
        if tokens["input_tokens"] is None:
            return lower, None
        remaining = tokens["input_tokens"] - sum(
            count for count, _ in components if count is not None
        )
        if remaining < 0:
            raise InvalidUsage("input partition exceeds total")
        lower += remaining * min(missing_rates)
        upper += remaining * max(missing_rates)
    return lower, upper


def quote(normalized: dict, tariff: Tariff | None, *, scope: str) -> dict:
    """A conservative interval, never a guessed tariff for aggregate context size."""
    if "iterations" in normalized:
        parts = [
            {"type": row["type"], "cost": quote(row["usage"], tariff, scope=scope)}
            for row in normalized["iterations"]
        ]
        return {
            **{key: _sum(*(row["cost"][key] for row in parts))
               for key in ("lower_micro_usd", "upper_micro_usd")},
            "assumptions": ["sum_sampling_iterations_once; not_invoice_or_quota"],
            "iterations": parts,
        }
    tokens = normalized["tokens"]
    if tariff is None or tokens is None:
        return {
            "lower_micro_usd": None,
            "upper_micro_usd": None,
            "assumptions": ["tariff_unavailable" if tariff is None else "usage_unavailable"],
        }
    tiers = list(tariff.tiers)
    assumptions = ["token_tariff_equivalent; not invoice or quota"]
    if scope == "provider_request" and tokens["input_tokens"] is not None:
        tiers = [
            next(
                t
                for t in tiers
                if t.maximum_input_tokens is None
                or tokens["input_tokens"] <= t.maximum_input_tokens
            )
        ]
    elif len(tiers) > 1:
        assumptions.append("per_request_context_lengths_unknown; interval_across_tariff_tiers")
    intervals = []
    for tier in tiers:
        lower, upper = _input_range(tokens, tier.rates)
        out = tokens["output_tokens"]
        if out is None:
            upper = None
        else:
            lower += out * Decimal(tier.rates.output)
            if upper is not None:
                upper += out * Decimal(tier.rates.output)
        intervals.append((lower, upper))
    lower = min(pair[0] for pair in intervals)
    upper = (
        None if any(pair[1] is None for pair in intervals) else max(pair[1] for pair in intervals)
    )
    if upper is None or lower != upper:
        assumptions.append("incomplete_counter_partition_or_tariff_tier; do_not_report_exact_cost")
    # USD/M tokens multiplied by tokens is already micro-USD. Round outward.
    return {
        "lower_micro_usd": int(lower.to_integral_value(rounding=ROUND_FLOOR)),
        "upper_micro_usd": None
        if upper is None
        else int(upper.to_integral_value(rounding=ROUND_CEILING)),
        "assumptions": assumptions,
    }


def reservation(tariff: Tariff) -> int:
    """Worst declared per-request token cost; no string-length token heuristic."""
    bounds = tariff.request_bounds
    if bounds is None:
        raise ValueError("spending guard requires declared provider request bounds")
    input_rate = max(
        Decimal(value)
        for tier in tariff.tiers
        for key, value in tier.rates.model_dump().items()
        if key != "output" and value is not None
    )
    output_rate = max(Decimal(tier.rates.output) for tier in tariff.tiers)
    return int(
        (
            bounds.maximum_input_tokens * input_rate + bounds.maximum_output_tokens * output_rate
        ).to_integral_value(rounding=ROUND_CEILING)
    )
