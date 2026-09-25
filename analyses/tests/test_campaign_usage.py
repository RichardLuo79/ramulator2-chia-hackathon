"""Receipt accounting, not live model calls or simulations."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve()
MODULES = HERE.parents[2] / 'tools/eval' if (HERE.parents[2] / 'tools/eval').is_dir() else HERE.parents[1]
sys.path.insert(0, str(MODULES))
u = importlib.import_module('campaign_usage')
imp = importlib.import_module('campaign_usage_import')


def tariff(long_context=False, ttl=False):
    rates = dict(input='10', cached_input='1', cache_write='12.5', output='50')
    if ttl:
        rates.update(cache_write_5m='12.5', cache_write_1h='20')
    tiers = [dict(maximum_input_tokens=None, rates=rates)]
    if long_context:
        tiers = [dict(maximum_input_tokens=272000, rates=rates),
                 dict(maximum_input_tokens=None,
                      rates=dict(input='20', cached_input='2', cache_write='25', output='75'))]
    return dict(tiers=tiers)


def tokens(**changes):
    t = dict.fromkeys(u.FIELDS, None)
    t.update(input_tokens=100, uncached_input_tokens=60, cached_input_tokens=30,
             cache_write_input_tokens=10, output_tokens=20, reasoning_output_tokens=12,
             non_reasoning_output_tokens=8, total_tokens=120, tool_input_tokens=0)
    t.update(changes)
    return t


def rule(protocol):
    # Small fixture declarations use the same documented counter relationships.
    base = dict(input_includes_cache=True, output_includes_reasoning=True,
                fields={'input_tokens':['input'], 'cached_input_tokens':['cache'],
                        'output_tokens':['output'], 'reasoning_output_tokens':['reasoning'],
                        'total_tokens':['total']}, zero_fields=['tool_input_tokens', 'cache_write_input_tokens'])
    if protocol == 'gemini':
        base['output_includes_reasoning'] = False
    if protocol == 'anthropic_api':
        base['input_includes_cache'] = False
        base['fields']['cache_write_input_tokens'] = ['writes']
        base['fields'].pop('reasoning_output_tokens')
        base['zero_fields'] = ['tool_input_tokens']
    return base


def test_normalized_output_includes_reasoning_once():
    raw = dict(input=100, cache=40, output=20, reasoning=12, total=120)
    a = u.normalize(raw, rule('openai'))
    assert a['output_tokens'] == 20 and a['non_reasoning_output_tokens'] == 8
    raw.update(output=8)
    assert u.normalize(raw, rule('gemini')) == a


def test_missing_cache_is_not_zero():
    raw = dict(input=100, output=20, reasoning=12, total=120)
    t = u.normalize(raw, rule('openai'))
    assert t['cached_input_tokens'] is None and t['uncached_input_tokens'] is None
    assert u.price(t, tariff(), 'provider_request') == dict(lower_micro_usd=1100, upper_micro_usd=2000)


def test_exact_context_tier_vs_cli_aggregate_bounds():
    t = tokens(input_tokens=1_000_000, uncached_input_tokens=600_000,
               cached_input_tokens=400_000, cache_write_input_tokens=0, output_tokens=1000)
    assert u.price(t, tariff(True), 'provider_request') == dict(lower_micro_usd=12875000, upper_micro_usd=12875000)
    assert u.price(t, tariff(True), 'native_cli_turn') == dict(lower_micro_usd=6450000, upper_micro_usd=12875000)
    t.update(input_tokens=272000, uncached_input_tokens=272000, cached_input_tokens=0)
    assert u.price(t, tariff(True), 'provider_request')['lower_micro_usd'] == 2770000
    t.update(input_tokens=272001, uncached_input_tokens=272001)
    assert u.price(t, tariff(True), 'provider_request')['lower_micro_usd'] == 5515020


def test_cache_write_durations_and_unknown_duration_bounds():
    t = tokens()
    unknown = u.price(t, tariff(ttl=True), 'main_agent_cli_turn')
    assert unknown == dict(lower_micro_usd=1755, upper_micro_usd=1830)
    t.update(cache_write_5m_tokens=4, cache_write_1h_tokens=6)
    assert u.price(t, tariff(ttl=True), 'main_agent_cli_turn') == dict(lower_micro_usd=1800, upper_micro_usd=1800)


def test_claude_resume_uses_invocation_not_cumulative_totals_and_recovers_reasoning():
    raw = dict(input=4, cache=100, writes=10, output=20, total=134,
               output_tokens_details={'thinking_tokens':12})
    t = u.normalize(raw, rule('anthropic_api'))
    assert t['input_tokens'] == 114 and t['reasoning_output_tokens'] is None
    entry = dict(raw=raw, scope='main_agent_cli_turn', usage=dict(tokens=t, protocol='anthropic_api'),
                 cost=u.price(t, tariff(), 'main_agent_cli_turn'),
                 cumulative_model_usage_not_added={'model': {'inputTokens':99999999, 'thinkingTokens':9999999}})
    normalized, cost, recovery = u.normalize_entry(entry, {'anthropic_api':rule('anthropic_api')}, tariff())
    assert normalized['input_tokens'] == 114 and normalized['output_tokens'] == 20
    assert normalized['reasoning_output_tokens'] == 12 and recovery
    assert cost == entry['cost']


def test_codex_native_requests_replace_cumulative_cli_and_preserve_failed_partial_usage():
    def event(turn, response, raw, turn_sum, thread_sum):
        return {'type':'token_usage_record', 'payload':dict(turn_id=turn, response_id=response,
            usage=raw, turn_token_usage=turn_sum, thread_token_usage=thread_sum)}
    a = dict(input=100, cache=20, output=10, reasoning=3, total=110)
    b = dict(input=120, cache=40, output=20, reasoning=5, total=140)
    total = {k:a[k]+b[k] for k in a}
    events = [dict(type='event_msg', payload={'type':'task_started', 'turn_id':'one'}),
              event('one', 'response-one', a, a, a),
              dict(type='event_msg', payload={'type':'task_started', 'turn_id':'two'}),
              event('two', 'response-two', b, b, total)]
    receipt = dict(success=True, usage=[{'raw':total}])
    rows = imp.native_request_records(events, receipt, rule('openai'), tariff())
    assert len(rows)==1 and rows[0]['tokens']['input_tokens']==120
    assert rows[0]['cost']['lower_micro_usd']==1840  # Not the cumulative 3,160.
    receipt = dict(success=False, usage=[{'raw':None}])
    rows = imp.native_request_records(events, receipt, rule('openai'), tariff())
    assert len(rows)==2 and rows[-1]['tokens'] is None
    assert rows[-1]['status']=='interrupted_tail_usage_unknown'
    broken = deepcopy(events)
    broken[-1]['payload']['turn_token_usage'] = dict(b, input=121)
    with pytest.raises(ValueError, match='turn usage'):
        imp.native_request_records(broken, receipt, rule('openai'), tariff())


def item(rnd=1, phase='explore', ident='one', success=True):
    role = 'reviewer' if phase.startswith('review') else 'proposer'
    return dict(receipt=dict(attempt=f'native-evidence/{rnd}/{role}/{phase}-{ident}',
                            result_sha256='native-result', success=success, usage=['same-counters']),
                receipt_sha256='receipt', prompt_sha256='prompt')


def test_round_exclusion_and_duplicate_receipts_not_equal_value_dedup():
    a = item(); b = item(ident='two')
    assert len(u.unique_receipts([a, deepcopy(a), b, item(11), item(15), item(0)])) == 2
    changed = deepcopy(a); changed['receipt']['usage'] = ['changed']
    with pytest.raises(ValueError, match='conflicting'):
        u.unique_receipts([a, changed])


def reconciled_fixture():
    records, attempts, items = {}, [], []
    for r in range(1, 11):
        records[f'iteration:{r}'] = dict(stage='single_core', selection_reason='unchanged_candidate' if r == 4 else None)
        for phase in ('explore', 'review', 'reflect'):
            if r == 4 and phase == 'review':
                continue
            it = item(r, phase); items.append(it)
            attempts.append(dict(step=f'iteration:{r}:{phase}', number=1, status='complete',
                                 result=it['receipt'], error=None))
    return records, attempts, items


def test_skipped_review_and_recovered_attempt_are_not_new_calls():
    records, attempts, items = reconciled_fixture()
    records['recovery:iteration:1:explore:1'] = {'receipt_sha256':'receipt'}
    items += [deepcopy(items[0])]
    _, matched, skipped = imp.reconcile(records, attempts, items)
    assert len(matched) == 29 and skipped[0]['round'] == 4
    assert skipped[0]['reason'] == 'unchanged_candidate'
    records['iteration:4']['selection_reason'] = None
    with pytest.raises(ValueError, match='mechanical skip'):
        imp.reconcile(records, attempts, items)


def test_failed_attempt_and_format_repair_are_counted_and_bound():
    records, attempts, items = reconciled_fixture()
    fail = item(1, ident='failed', success=False); items.append(fail)
    attempts.append(dict(step='iteration:1:explore', number=2, status='failed', result=None,
                         error={'message':'evidence retained at '+fail['receipt']['attempt']}))
    items.append(item(2, phase='review_format'))
    _, matched, _ = imp.reconcile(records, attempts, items)
    assert len(matched) == 31
    assert matched[fail['receipt']['attempt']]['database_status'] == 'failed'
    assert matched[items[-1]['receipt']['attempt']]['kind'] == 'format_only_review_repair'


def usage_row(rnd=1, missing=False, phase='development'):
    t = None if missing else tokens()
    return dict(campaign='astra', round=rnd, phase=phase, attempt_success=not missing,
                scope='native_cli_turn', tokens=t, cost=u.price(t, tariff(), 'native_cli_turn'))


def test_unknown_call_does_not_make_observed_interval_a_campaign_bound():
    rows = [usage_row(), usage_row(2, True)]
    s = u.summarize(rows)
    assert s['input_tokens'] == 100 and s['input_tokens_unknown_records'] == 1
    assert s['campaign_cost_upper_usd'] is None and s['observed_cost_upper_usd'] is not None
    a = u.aggregate({'rows':rows})
    rnd10 = next(x for x in a['rounds'] if x['campaign']=='astra' and x['round']==10)
    assert rnd10['cumulative_unknown_usage_records'] == 1
    assert rnd10['cumulative_campaign_cost_upper_usd'] is None
    assert u.summarize([rows[1]])['input_tokens'] is None


def test_compaction_is_a_disjoint_phase_and_totals_reconcile():
    rows = [usage_row(), usage_row(1, phase='compaction'), usage_row(2, True)]
    a = u.aggregate({'rows':rows})
    h = a['headline'][0]
    assert h['input_tokens'] == sum(x['input_tokens'] or 0 for x in a['phases'])
    assert h['output_tokens'] == sum(x['output_tokens'] or 0 for x in a['attempt_status'])
    assert abs(h['observed_cost_lower_usd'] - sum(x['observed_cost_lower_usd'] for x in a['rounds'])) < 1e-12


def test_real_snapshot_identity_privacy_and_immutable_scientific_evidence():
    root = HERE.parents[2]
    path = root / 'reports/chia_campaign_results.single-core-usage.json'
    if not path.exists():
        path = root / 'results/foundation/single-core-usage.json'
    if not path.exists():
        pytest.skip('usage evidence not collected')
    snap = json.loads(path.read_text()); u.validate(snap)
    data = dict(sources={c+'_single_core':{'candidate':{'candidate_id':snap['endpoint_binding'][c]}}
                         for c in u.CAMPAIGNS}, provenance={}, rows=[{'accuracy':'unchanged'}])
    before = deepcopy(data); merged = u.merge_usage(data, snap)
    assert data == before and merged['rows'] == before['rows']
    altered = deepcopy(data); altered['sources']['astra_single_core']['candidate']['candidate_id'] = 'draft'
    with pytest.raises(ValueError, match='endpoint'):
        u.merge_usage(altered, snap)
    for modification, match in [
        (lambda x:x['rows'].append(deepcopy(x['rows'][0])), 'duplicate'),
        (lambda x:x['rows'][0].update(round=11), 'out-of-scope'),
        (lambda x:x.update(private_path='/home/private'), 'private content'),
        (lambda x:x['tables']['headline'][0].update(input_tokens=0), 'reconciliation')]:
        bad = deepcopy(snap); modification(bad)
        with pytest.raises(ValueError, match=match): u.validate(bad)
