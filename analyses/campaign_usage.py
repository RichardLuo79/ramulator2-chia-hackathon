"""Offline accounting for an additive, receipt-bound single-core usage snapshot.

No provider SDK, campaign runtime, native state, or network access is required.
Counters are normalized with the campaign's recorded protocol rules. Prices are
the frozen tariffs, not today's prices. Unknown usage never becomes zero usage.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import re

CAMPAIGNS = ('astra', 'deepseek', 'gemini', 'opus')
PHASES = {'explore': 'development', 'review': 'review',
          'review_format': 'review', 'reflect': 'reflection'}
FIELDS = ('input_tokens', 'uncached_input_tokens', 'cached_input_tokens',
          'cache_write_input_tokens', 'cache_write_5m_tokens', 'cache_write_1h_tokens',
          'tool_input_tokens', 'output_tokens', 'reasoning_output_tokens',
          'non_reasoning_output_tokens', 'total_tokens')
RECEIPT = re.compile(r'^native-evidence/([1-9]|10)/(proposer|reviewer)/'
                     r'(explore|review|review_format|reflect)-([^/]+)$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def counter(value):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError('invalid token counter')
    return value


def at(raw, path):
    for key in path:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError('invalid counter path')
        raw = raw.get(key)
    return counter(raw)


def sum_known(*values):
    return None if any(x is None for x in values) else sum(values)


def normalize(raw, rule):
    """Mirror the frozen protocol arithmetic; output includes reasoning once."""
    if raw is None:
        return None
    t = {k: None for k in FIELDS}
    for k, path in rule['fields'].items():
        t[k] = at(raw, path)
    for k in rule['zero_fields']:
        t[k] = 0
    parts = ['input_tokens', 'output_tokens', 'tool_input_tokens']
    if not rule['output_includes_reasoning']:
        parts += ['reasoning_output_tokens']
    if not rule['input_includes_cache']:
        parts += ['cached_input_tokens', 'cache_write_input_tokens']
    missing = [k for k in parts if t[k] is None]
    if t['total_tokens'] is not None:
        known = sum(t[k] for k in parts if t[k] is not None)
        if known > t['total_tokens'] or (not missing and known != t['total_tokens']):
            raise ValueError('provider total disagrees')
        if len(missing) == 1:
            t[missing[0]] = t['total_tokens'] - known
    inp, cached, written, tool = [t[k] for k in
        ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'tool_input_tokens')]
    if rule['input_includes_cache']:
        if inp is not None and sum(x for x in (cached, written) if x is not None) > inp:
            raise ValueError('cache exceeds input')
        uncached = None if None in (inp, cached, written) else inp - cached - written
        t['uncached_input_tokens'] = sum_known(uncached, tool)
        t['input_tokens'] = sum_known(inp, tool)
    else:
        t['uncached_input_tokens'] = sum_known(inp, tool)
        t['input_tokens'] = sum_known(inp, cached, written, tool)
    short, long = t['cache_write_5m_tokens'], t['cache_write_1h_tokens']
    if written is not None and (sum(x for x in (short, long) if x is not None) > written
            or None not in (short, long) and short + long != written):
        raise ValueError('cache-write durations disagree')
    out, reasoning = t['output_tokens'], t['reasoning_output_tokens']
    if rule['output_includes_reasoning']:
        if out is not None and reasoning is not None and reasoning > out:
            raise ValueError('reasoning exceeds output')
        t['non_reasoning_output_tokens'] = None if None in (out, reasoning) else out - reasoning
    else:
        t['non_reasoning_output_tokens'] = out
        t['output_tokens'] = sum_known(out, reasoning)
    total = sum_known(t['input_tokens'], t['output_tokens'])
    if t['total_tokens'] is not None and total is not None and t['total_tokens'] != total:
        raise ValueError('normalized total disagrees')
    if t['total_tokens'] is None:
        t['total_tokens'] = total
    return t


def price(tokens, tariff, scope):
    """Conservative micro-USD interval over disjoint categories and allowed tiers.

    A CLI aggregate is NEVER assigned the long-context tier from its total size.
    A missing complete call has no finite upper bound. No stochastic uncertainty
    or repeat variation is implied by these accounting intervals.
    """
    if tokens is None:
        return {'lower_micro_usd': None, 'upper_micro_usd': None}
    tiers = tariff['tiers']
    if scope == 'provider_request' and tokens['input_tokens'] is not None:
        tiers = [next(t for t in tiers if t['maximum_input_tokens'] is None
                      or tokens['input_tokens'] <= t['maximum_input_tokens'])]
    bounds = []
    for tier in tiers:
        r = tier['rates']
        writes = [Decimal(r['cache_write'])]
        written = tokens['cache_write_input_tokens']
        short, long = tokens['cache_write_5m_tokens'], tokens['cache_write_1h_tokens']
        if r.get('cache_write_5m') is not None:
            writes = [Decimal(r['cache_write_5m']), Decimal(r['cache_write_1h'])]
            if written and short is not None and long is not None:
                writes = [(short * writes[0] + long * writes[1]) / written]
        components = [(tokens['uncached_input_tokens'], [Decimal(r['input'])]),
                      (tokens['cached_input_tokens'], [Decimal(r['cached_input'])]),
                      (written, writes)]
        low = sum(n * min(rates) for n, rates in components if n is not None)
        high = sum(n * max(rates) for n, rates in components if n is not None)
        unknown_rates = [rate for n, rates in components if n is None for rate in rates]
        if unknown_rates:
            if tokens['input_tokens'] is None:
                high = None
            else:
                remaining = tokens['input_tokens'] - sum(n for n, _ in components if n is not None)
                if remaining < 0:
                    raise ValueError('invalid input partition')
                low += remaining * min(unknown_rates)
                high += remaining * max(unknown_rates)
        if tokens['output_tokens'] is None:
            high = None
        else:
            output = tokens['output_tokens'] * Decimal(r['output'])
            low += output
            if high is not None:
                high += output
        bounds.append((Decimal(low), None if high is None else Decimal(high)))
    low = min(x[0] for x in bounds)
    high = None if any(x[1] is None for x in bounds) else max(x[1] for x in bounds)
    return {'lower_micro_usd': int(low.to_integral_value(rounding=ROUND_FLOOR)),
            'upper_micro_usd': None if high is None else int(high.to_integral_value(rounding=ROUND_CEILING))}


def normalize_entry(entry, rules, tariff):
    """Validate recorded accounting, then recover only a reported reasoning split."""
    protocol = entry['usage']['protocol']
    tokens = normalize(entry.get('raw'), rules[protocol])
    if tokens != entry['usage']['tokens']:
        raise ValueError('receipt normalization mismatch')
    cost = price(tokens, tariff, entry['scope'])
    if any(cost[k] != entry['cost'][k] for k in cost):
        raise ValueError('frozen tariff does not reproduce receipt cost')
    recovered = None
    # The old Anthropic normalization omitted this NEW per-invocation field.
    # Never use cumulative modelUsage.thinkingTokens from a resumed session.
    raw = entry.get('raw')
    if tokens is not None and protocol == 'anthropic_api' and raw is not None:
        reported = at(raw, ['output_tokens_details', 'thinking_tokens'])
        if reported is not None:
            if tokens['output_tokens'] is None or reported > tokens['output_tokens']:
                raise ValueError('reported thinking exceeds invocation output')
            if tokens['reasoning_output_tokens'] not in (None, reported):
                raise ValueError('conflicting reasoning counters')
            tokens['reasoning_output_tokens'] = reported
            tokens['non_reasoning_output_tokens'] = tokens['output_tokens'] - reported
            recovered = 'raw.output_tokens_details.thinking_tokens (per invocation)'
    return tokens, cost, recovered


def unique_receipts(items):
    """Deduplicate copies by invocation identity, not by equal usage values."""
    unique = {}
    for item in items:
        name = item['receipt']['attempt']
        if not RECEIPT.fullmatch(name):
            continue  # Excludes rounds 11+, smoke, tool HTTP receipts, final tests.
        if name in unique:
            if digest(unique[name]['receipt']) != digest(item['receipt']):
                raise ValueError('conflicting copies of an invocation')
        else:
            unique[name] = item
    return list(unique.values())


def endpoint_binding(sources):
    return {c: sources[c + '_single_core']['candidate']['candidate_id'] for c in CAMPAIGNS}


def summarize(rows):
    result = dict(records=len(rows), unknown_usage_records=0,
                  input_partition_unknown_records=0, reasoning_unknown_records=0,
                  successful_attempt_records=0, failed_attempt_records=0,
                  provider_requests=0, provider_requests_with_usage=0, cli_invocations=0,
                  observed_cost_lower_usd=0., observed_cost_upper_usd=0.,
                  campaign_cost_upper_usd=None)
    for key in FIELDS:
        result[key] = None
        result[key + '_unknown_records'] = 0
    low, high = 0, 0
    unbounded = False
    for row in rows:
        result['successful_attempt_records' if row['attempt_success'] else 'failed_attempt_records'] += 1
        if row.get('status') != 'interrupted_tail_usage_unknown':
            result['provider_requests' if row['scope'] == 'provider_request' else 'cli_invocations'] += 1
        t = row['tokens']
        if row['scope'] == 'provider_request' and t is not None:
            result['provider_requests_with_usage'] += 1
        if t is None or t.get('input_tokens') is None or t.get('output_tokens') is None:
            result['unknown_usage_records'] += 1
        if t is not None and (t['uncached_input_tokens'] is None or t['cached_input_tokens'] is None):
            result['input_partition_unknown_records'] += 1
        if t is None or t['reasoning_output_tokens'] is None:
            result['reasoning_unknown_records'] += 1
        for key in FIELDS:
            value = None if t is None else t[key]
            if value is None:
                result[key + '_unknown_records'] += 1
            else:
                result[key] = (result[key] or 0) + value
        a, b = row['cost']['lower_micro_usd'], row['cost']['upper_micro_usd']
        if a is not None:
            low += a
        if b is not None:
            high += b
        elif a is not None:
            unbounded = True
    result['observed_cost_lower_usd'] = low / 1e6
    result['observed_cost_upper_usd'] = None if unbounded else high / 1e6
    result['cost_scope'] = ('observed subtotal; additional usage unknown' if result['unknown_usage_records']
                            else 'recorded invocation coverage')
    if not result['unknown_usage_records'] and not unbounded:
        result['campaign_cost_upper_usd'] = high / 1e6
    return result


def aggregate(snapshot):
    output = {'headline': [], 'rounds': [], 'phases': [], 'attempt_status': []}
    for campaign in CAMPAIGNS:
        rows = [r for r in snapshot['rows'] if r['campaign'] == campaign]
        common = {'campaign': campaign}
        output['headline'].append({**common, **summarize(rows)})
        cumulative = []
        for rnd in range(1, 11):
            part = [r for r in rows if r['round'] == rnd]
            cumulative.extend(part)
            total = summarize(cumulative)
            output['rounds'].append({**common, 'round': rnd, **summarize(part),
                'cumulative_observed_cost_lower_usd': total['observed_cost_lower_usd'],
                'cumulative_observed_cost_upper_usd': total['observed_cost_upper_usd'],
                'cumulative_unknown_usage_records': total['unknown_usage_records'],
                'cumulative_campaign_cost_upper_usd': total['campaign_cost_upper_usd']})
        for phase in ('development', 'review', 'reflection', 'compaction'):
            part = [r for r in rows if r['phase'] == phase]
            if part:
                output['phases'].append({**common, 'phase': phase, **summarize(part)})
        for success in (True, False):
            part = [r for r in rows if r['attempt_success'] == success]
            if part:
                output['attempt_status'].append({**common, 'attempt_status': 'successful' if success else 'failed',
                                                **summarize(part)})
    return output


def validate(snapshot, sources=None):
    if snapshot['schema'] != 'single-core-usage-v1' or snapshot['rounds'] != [1, 10]:
        raise ValueError('unsupported usage scope')
    if sources is not None and snapshot['endpoint_binding'] != endpoint_binding(sources):
        raise ValueError('usage belongs to another selected campaign endpoint')
    if set(snapshot['campaigns']) != set(CAMPAIGNS):
        raise ValueError('missing campaign')
    ids = set()
    response_ids = set()
    invocation_groups = defaultdict(list)
    for row in snapshot['rows']:
        if not 1 <= row['round'] <= 10 or row['phase'] not in (*PHASES.values(), 'compaction'):
            raise ValueError('out-of-scope usage')
        key = row['campaign'], row['invocation_sha256'], row['accounting_ordinal']
        if key in ids:
            raise ValueError('duplicate accounting record')
        ids.add(key)
        if row.get('response_identity_sha256'):
            response = row['campaign'], row['response_identity_sha256']
            if response in response_ids:
                raise ValueError('provider response counted across resumed sessions')
            response_ids.add(response)
        invocation_groups[key[:2]].append(row)
        expected = price(row['tokens'], snapshot['campaigns'][row['campaign']]['tariff'], row['scope'])
        if expected != row['cost']:
            raise ValueError('usage cost mismatch')
    for group in invocation_groups.values():
        scopes = set(r['scope'] for r in group)
        if len(scopes) != 1 or (next(iter(scopes)) != 'provider_request' and len(group) != 1):
            raise ValueError('mixed accounting layers for one invocation')
    for c in CAMPAIGNS:
        meta = snapshot['campaigns'][c]
        if meta['completed_rounds'] != list(range(1, 11)):
            raise ValueError('incomplete ten-round record coverage')
        if meta['tariff_sha256'] != digest(meta['tariff']):
            raise ValueError('tariff identity mismatch')
        invocations = [i for i in snapshot['invocations'] if i['campaign'] == c]
        if len(invocations) != meta['invocations']:
            raise ValueError('invocation reconciliation failed')
        for i in invocations:
            group = invocation_groups[(c, i['invocation_sha256'])]
            if not group or len(group) != i['accounting_records']:
                raise ValueError('unaccounted invocation')
            if any(r['round'] != i['round'] or r['receipt_sha256'] != i['receipt_sha256']
                   or r['attempt_success'] != i['success'] for r in group):
                raise ValueError('invocation binding mismatch')
    if snapshot['tables'] != aggregate(snapshot):
        raise ValueError('usage table reconciliation failed')
    # Export uses an allowlist. This check is a final accidental-content guard.
    text = json.dumps(snapshot, ensure_ascii=False)
    for pattern in (r'/home/', r'/tmp/', r'gs://', r'"raw_stdout"', r'"session_id"',
                    r'"access_token"', r'"refresh_token"', r'"api_key"', r'"conversation"'):
        if re.search(pattern, text):
            raise ValueError('private content in usage supplement')


def merge_usage(data, snapshot):
    validate(snapshot, data['sources'])
    if 'single_core_usage' in data:
        raise ValueError('usage snapshot already present; do not overwrite')
    result = deepcopy(data)
    result['single_core_usage'] = snapshot
    result['provenance']['single_core_usage_sha256'] = digest(snapshot)
    return result
