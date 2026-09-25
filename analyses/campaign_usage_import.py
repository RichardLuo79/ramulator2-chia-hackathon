"""Read-only operator import of preserved single-core LLM usage.

Invoked by ``campaign_results_import.py usage --sources ... --output ...``.
The private sources manifest contains local paths; it is never copied into the
public supplement. No inference, simulation, or campaign write is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import re
import sqlite3
import tarfile
import tempfile
import zipfile

from campaign_usage import (CAMPAIGNS, PHASES, RECEIPT, aggregate, digest,
                            normalize, normalize_entry, price, unique_receipts, validate)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_database(path):
    """SQLite online backup creates a consistent private temporary snapshot."""
    with tempfile.TemporaryDirectory(prefix='usage-sqlite-') as tmp:
        dest = Path(tmp) / 'snapshot.sqlite'
        with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(dest) as target:
                source.backup(target)
        with sqlite3.connect(dest.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('campaign SQLite integrity check failed')
            records = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM records')}
            attempts = [dict(step=s, number=n, status=st, result=json.loads(result) if result else None,
                             error=json.loads(err) if err else None, input_sha256=inp)
                        for s, n, st, result, err, inp in db.execute(
                            'SELECT step,number,status,result,error,input_sha256 FROM attempts')
                        if re.fullmatch(r'iteration:([1-9]|10):(explore|review|reflect)', s)]
        return records, attempts, file_sha(dest)


def local_source(spec):
    root = Path(spec['root'])
    records, attempts, dbsha = read_database(root / 'campaign.sqlite')
    receipts = []
    for path in root.glob('native-evidence/*/*/*/receipt.json'):
        if not RECEIPT.fullmatch(str(path.parent.relative_to(root))):
            continue
        raw = path.read_bytes()
        prompt = path.parent / 'prompt.txt'
        receipts.append(dict(receipt=json.loads(raw), receipt_sha256=sha(raw),
                             prompt_sha256=file_sha(prompt)))
    return records, attempts, receipts, {'consistent_database_sha256': dbsha}


def native_request_records(events, receipt, rule, tariff):
    """Use the last resumed turn, not cumulative thread/terminal counters.

    Public output contains counter values and digests only. Each response ID
    establishes a real request identity; equal counters do not imply duplication.
    """
    starts = [e['payload']['turn_id'] for e in events
              if e.get('payload', {}).get('type') == 'task_started']
    if not starts:
        raise ValueError('native invocation has no turn boundary')
    current = starts[-1]
    records = [e for e in events if e['type'] == 'token_usage_record'
               and e['payload']['turn_id'] == current]
    if not records:
        raise ValueError('no native request counters for current invocation')
    ids = [e['payload']['response_id'] for e in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate native provider response')
    last = records[-1]['payload']
    summed = {k: sum(e['payload']['usage'][k] for e in records) for k in last['usage']}
    if summed != last['turn_token_usage']:
        raise ValueError('native request sum differs from turn usage')
    terminal = receipt['usage'][0].get('raw')
    if terminal is not None and any(v != last['thread_token_usage'][k] for k, v in terminal.items()):
        raise ValueError('native history does not reproduce terminal thread totals')
    rows = []
    for event in records:
        raw = event['payload']['usage']
        t = normalize(raw, rule)
        rows.append(dict(scope='provider_request', purpose='agent', protocol='openai_cli',
            tokens=t, cost=price(t, tariff, 'provider_request'), status='response_recorded',
            recorded_accounting_sha256=digest(event), raw_counter_sha256=digest(raw),
            response_identity_sha256=digest(event['payload']['response_id']),
            reasoning_recovery=None, cumulative_cli_usage_excluded=True,
            accounting_source='verified_native_request'))
    if not receipt['success']:
        # The responses above are known. A terminally interrupted request/retry
        # tail has no complete usage receipt; neither zero nor one paid request
        # is asserted by this marker.
        rows.append(dict(scope='provider_request', purpose='agent', protocol='openai_cli',
            tokens=None, cost=price(None, tariff, 'provider_request'),
            status='interrupted_tail_usage_unknown', response_identity_sha256=None,
            recorded_accounting_sha256=digest(receipt['usage'][0]), raw_counter_sha256=digest(None),
            reasoning_recovery=None, cumulative_cli_usage_excluded=True,
            accounting_source='unaccounted_interruption_marker'))
    return rows


def astra_requests(root, receipt, rule, tariff):
    folder = Path(root) / receipt['attempt']
    checkpoint = read(folder / 'checkpoint.json')
    names = [n for n in checkpoint['files'] if n.startswith('sessions/') and n.endswith('.jsonl')]
    if len(names) != 1:
        raise ValueError('ambiguous native history file')
    name = names[0]
    if '..' in Path(name).parts or Path(name).is_absolute():
        raise ValueError('unsafe native history path')
    raw = gzip.decompress((folder / 'state' / (name + '.gz')).read_bytes())
    if sha(raw) != checkpoint['files'][name]:
        raise ValueError('native request history checksum mismatch')
    events = [json.loads(line) for line in raw.splitlines()]
    rows = native_request_records(events, receipt, rule, tariff)
    history_sha, checkpoint_sha = sha(raw), file_sha(folder / 'checkpoint.json')
    for row in rows:
        row.update(native_counter_history_sha256=history_sha, checkpoint_sha256=checkpoint_sha)
    return rows


def opus_source(spec):
    """Use verified reviewed receipts plus the final quiesced original database."""
    index = read(spec['preservation_index'])
    archive = Path(spec['database_archive'])
    expected = next(a for a in index['archives'] if a['sha256'] == archive.name.split('.')[0])
    if file_sha(archive) != expected['sha256']:
        raise ValueError('Opus database archive checksum differs')
    with tarfile.open(archive) as tar:
        rawdb = tar.extractfile('campaign/campaign.sqlite').read()
    if sha(rawdb) != index['files']['campaign/campaign.sqlite']['sha256']:
        raise ValueError('Opus original database checksum differs')
    with tempfile.TemporaryDirectory(prefix='opus-usage-') as tmp:
        path = Path(tmp) / 'db.sqlite'; path.write_bytes(rawdb)
        records, attempts, _, = read_database(path)
    redactions = {r['path']: r for r in read(spec['redactions'])['files']}
    public = Path(spec['receipt_archive'])
    # Hash the complete archive, not just its name or ZIP directory.
    if file_sha(public) != spec['receipt_archive_sha256']:
        raise ValueError('Opus reviewed archive checksum differs')
    receipts = []
    with zipfile.ZipFile(public) as bundle:
        for member in bundle.namelist():
            if not member.startswith('logs/opus/') or not member.endswith('/receipt.json'):
                continue
            relative = member.removeprefix('logs/opus/').removesuffix('/receipt.json')
            if not RECEIPT.fullmatch(relative):
                continue
            raw = bundle.read(member); record = redactions[member]
            if sha(raw) != record['public_sha256']:
                raise ValueError('Opus reviewed member checksum differs')
            original = index['files']['campaign/' + relative + '/receipt.json']['sha256']
            if original != record['original_sha256']:
                raise ValueError('Opus original-to-reviewed receipt link differs')
            receipts.append(dict(receipt=json.loads(raw), receipt_sha256=original,
                reviewed_receipt_sha256=sha(raw),
                prompt_sha256=index['files']['campaign/' + relative + '/prompt.txt']['sha256']))
    return records, attempts, receipts, {
        'original_database_sha256': sha(rawdb), 'database_archive_sha256': expected['sha256'],
        'receipt_archive_sha256': spec['receipt_archive_sha256'],
        'preservation_index_sha256': file_sha(spec['preservation_index']),
        'redaction_inventory_sha256': file_sha(spec['redactions'])}


def reconcile(records, attempts, items):
    """Bind every invocation to a committed attempt or an explicit review repair."""
    selected = {i['receipt']['attempt']: i for i in unique_receipts(items)}
    attempts = [a for a in attempts if re.fullmatch(
        r'iteration:([1-9]|10):(explore|review|reflect)', a['step'])]
    matched = {}
    for a in attempts:
        _, rnd, phase = a['step'].split(':')
        if a['result']:
            key = a['result']['attempt']
            item = selected.get(key)
            if item is None:
                raise ValueError('committed invocation receipt is missing')
            # Public Opus text can be redacted. Usage and native result hashes
            # must still match the original database exactly.
            for field in ('usage', 'success', 'result_sha256'):
                if item['receipt'][field] != a['result'][field]:
                    raise ValueError('database and receipt disagree')
        else:
            message = (a['error'] or {}).get('message', '')
            keys = re.findall(r'native-evidence/[^\s\"\']+', message)
            if keys:
                key = keys[0]
            else:
                # The Gemini recovery record replaced the error text. One
                # otherwise-unmatched failed receipt binds it unambiguously.
                choices = [k for k, item in selected.items() if not item['receipt']['success']
                           and RECEIPT.fullmatch(k).group(1) == rnd
                           and RECEIPT.fullmatch(k).group(3) == phase and k not in matched]
                if len(choices) != 1:
                    raise ValueError('ambiguous failed-attempt receipt')
                key = choices[0]
            if key not in selected or selected[key]['receipt']['success']:
                raise ValueError('failed-attempt receipt missing or inconsistent')
        if key in matched:
            raise ValueError('two attempts claim the same invocation')
        m = RECEIPT.fullmatch(key)
        if m.group(1) != rnd or m.group(3) != phase:
            raise ValueError('attempt phase or round differs')
        matched[key] = dict(attempt_number=a['number'], database_status=a['status'],
                            database_attempt_sha256=digest(a), kind='database_attempt')
    for key in selected.keys() - matched.keys():
        m = RECEIPT.fullmatch(key)
        if m.group(3) != 'review_format':
            raise ValueError('unreconciled invocation')
        parents = [a for a in attempts if a['step'] == f'iteration:{m.group(1)}:review']
        if len(parents) != 1:
            raise ValueError('ambiguous review-format parent')
        matched[key] = dict(attempt_number=parents[0]['number'], database_status='complete',
                            database_attempt_sha256=digest(parents[0]), kind='format_only_review_repair')
    skipped = []
    for rnd in range(1, 11):
        row = records[f'iteration:{rnd}']
        if row['stage'] != 'single_core':
            raise ValueError('wrong campaign stage')
        phases = {a['step'].split(':')[-1] for a in attempts if a['step'].split(':')[1] == str(rnd)}
        if not {'explore', 'reflect'} <= phases:
            raise ValueError('missing development/reflection attempt')
        if 'review' not in phases:
            if row.get('selection_reason') != 'unchanged_candidate':
                raise ValueError('missing review without a mechanical skip')
            skipped.append(dict(round=rnd, phase='review', reason='unchanged_candidate',
                                record_sha256=digest(row)))
    return selected, matched, skipped


def collect(specification, rules):
    snapshot = dict(schema='single-core-usage-v1', rounds=[1, 10],
        created_utc=datetime.now(timezone.utc).isoformat(),
        campaigns={}, endpoint_binding={}, invocations=[], rows=[],
        normalization_rules_sha256=digest(rules),
        scope='Single-core development, independent review, reflection, recorded retries and compaction only',
        units='provider-reported tokens and estimated API-equivalent USD; not actual spending')
    for name in CAMPAIGNS:
        spec = specification[name]
        records, attempts, items, provenance = opus_source(spec) if name == 'opus' else local_source(spec)
        selected, matched, skipped = reconcile(records, attempts, items)
        config = read(spec['config']); tariff = read(spec['tariff'])
        candidate = records['stage-selection:single_core']['candidate']
        retained = records['iteration:10']['selected']
        retained = retained.get('candidate', retained)
        if candidate['candidate_id'] != retained['candidate_id']:
            raise ValueError('stage endpoint is not round-10 retained selection')
        snapshot['endpoint_binding'][name] = candidate['candidate_id']
        metadata = dict(tariff=tariff, tariff_sha256=digest(tariff),
            original_tariff_file_sha256=file_sha(spec['tariff']),
            campaign_identity_sha256=digest(config['campaign_id']),
            configuration_sha256=file_sha(spec['config']),
            selected_source_sha256=candidate['files']['model.cpp']['sha256'],
            completed_rounds=list(range(1, 11)), invocations=len(selected),
            database_attempts=len([a for a in matched.values() if a['kind'] == 'database_attempt']),
            format_only_repairs=len([a for a in matched.values() if a['kind'] != 'database_attempt']),
            skipped_reviews=skipped, provenance=provenance,
            billing='subscription; API-equivalent only' if name in ('astra', 'opus') else
                    'Vertex credits; API-equivalent, not credit-adjusted' if name == 'gemini' else
                    'direct API; frozen peak-rate equivalent, not time-adjusted invoice',
            compaction='included in native invocation totals; not separately attributable' if name in ('astra', 'opus') else
                       'separate only when the recorded request purpose identifies compaction',
            scope_note='main-agent CLI invocations, not cumulative modelUsage or a claim of subagent coverage'
                       if name == 'opus' else 'individual native request counters; cumulative resumed-thread CLI totals excluded'
                       if name == 'astra' else 'recorded provider requests, including failed requests with unknown usage',
            recovered_attempt_records=[])
        for key, value in records.items():
            if re.fullmatch(r'recovery:iteration:([1-9]|10):(explore|review|reflect):\d+', key):
                metadata['recovered_attempt_records'].append(dict(
                    round=int(key.split(':')[2]), phase=PHASES[key.split(':')[3]], record_sha256=digest(value)))
        snapshot['campaigns'][name] = metadata
        for key, item in sorted(selected.items()):
            m = RECEIPT.fullmatch(key); rnd = int(m.group(1)); phase = PHASES[m.group(3)]
            receipt = item['receipt']; invocation = digest([config['campaign_id'], key])
            evidence = {k: v for k, v in item.items() if k != 'receipt'}
            identity = dict(campaign=name, round=rnd, parent_phase=phase, invocation_sha256=invocation,
                            result_sha256=receipt['result_sha256'], **evidence, **matched[key])
            rows = receipt['usage']
            if not rows:
                raise ValueError('empty invocation accounting; unknown usage must be explicit')
            normalized_rows = []
            for entry in rows:
                if entry['scope'] not in ('provider_request', 'native_cli_turn', 'main_agent_cli_turn'):
                    raise ValueError('unqualified accounting layer')
                tokens, cost, recovery = normalize_entry(entry, rules, tariff)
                purpose = entry.get('purpose', 'agent')
                if purpose not in ('agent', 'context_compaction'):
                    raise ValueError('unrecognized accounting purpose')
                normalized_rows.append({'purpose': purpose,
                    'scope': entry['scope'], 'status': entry.get('status', 'terminal_cli_usage' if tokens else 'usage_unknown'),
                    'protocol': entry['usage']['protocol'], 'tokens': tokens, 'cost': cost,
                    'recorded_accounting_sha256': digest(entry), 'raw_counter_sha256': digest(entry.get('raw')),
                    'reasoning_recovery': recovery,
                    'cumulative_cli_usage_excluded': 'cumulative_model_usage_not_added' in entry,
                    'accounting_source': 'receipt'})
            if name == 'astra':
                normalized_rows = astra_requests(spec['root'], receipt, rules['openai_cli'], tariff)
            snapshot['invocations'].append({**identity, 'success': receipt['success'],
                                            'accounting_records': len(normalized_rows),
                                            'terminal_receipt_records_not_added': len(rows) if name == 'astra' else 0})
            for ordinal, row in enumerate(normalized_rows):
                purpose = row.pop('purpose')
                snapshot['rows'].append({**identity, 'phase': 'compaction' if purpose == 'context_compaction' else phase,
                    'accounting_ordinal': ordinal, 'attempt_success': receipt['success'], **row})
        print(name, 'reconciled', len(selected), 'invocations;', len(skipped), 'skipped reviews', flush=True)
    snapshot['tables'] = aggregate(snapshot)
    validate(snapshot)
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True, help='Private operator paths; never exported')
    parser.add_argument('--protocols', type=Path, required=True, help='Frozen campaign usage_protocols.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mirror-output', type=Path, help='Write the identical generated public supplement here too')
    args = parser.parse_args(argv)
    snapshot = collect(read(args.sources), read(args.protocols)['protocols'])
    snapshot['normalization_protocol_file_sha256'] = file_sha(args.protocols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    args.output.write_text(payload)
    if args.mirror_output:
        args.mirror_output.parent.mkdir(parents=True, exist_ok=True)
        args.mirror_output.write_text(payload)
    print(json.dumps({'output_sha256': file_sha(args.output), 'accounting_records': len(snapshot['rows'])}))


if __name__ == '__main__':
    main()
