"""Import committed endpoint evidence without accessing a running campaign.

This operator-only tool reads a consistent database backup or reviewed record
export, completed measurement receipts, and preserved observations. It neither
runs evaluations nor calls providers. The small additive snapshot keeps the
original campaign evidence unchanged. Notebook execution needs only that snapshot.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import zipfile

from campaign_results import (behavior, bind_score, choose_receipt, digest,
                              measurements, read, save, scan_extreme,
                              score_row, select_extremes, sha)


def merge_snapshot(base, snapshot):
    """Add only new, verified arms/cases. Never rewrite an earlier observation."""
    if digest(base) != snapshot['base_payload_sha256']:
        raise ValueError('base evidence identity differs')
    out = deepcopy(base)
    existing = {(r['arm'], r['case']) for r in out['rows']}
    for row in snapshot['rows']:
        key = row['arm'], row['case']
        if key in existing:
            raise ValueError('snapshot would overwrite a completed score')
        existing.add(key)
    if set(out['sources']) & set(snapshot['sources']):
        raise ValueError('snapshot would overwrite a frozen model')
    for key in ('rows', 'convergence', 'decisions', 'oracle_equivalence',
                'extremes', 'extreme_selection'):
        out[key].extend(snapshot[key])
    out['sources'].update(snapshot['sources'])
    out['availability'] = snapshot['availability']
    out['provenance']['single_core_additions'] = snapshot['provenance']
    out['provenance']['single_core_additions_sha256'] = digest(snapshot)
    return out


def consistent_records(backup):
    """Verify the backup manifest before opening a temporary immutable SQLite copy."""
    manifest = read(backup / 'manifest.json')
    packed = (backup / 'campaign.sqlite.gz').read_bytes()
    # The supervisor's database_sha256 is the compressed backup checksum.
    expected = manifest['database_sha256']
    if hashlib.sha256(packed).hexdigest() != expected:
        raise ValueError('database backup checksum differs')
    raw = gzip.decompress(packed)
    with tempfile.TemporaryDirectory(prefix='analysis-sqlite-') as tmp:
        path = Path(tmp) / 'snapshot.sqlite'
        path.write_bytes(raw)
        with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('consistent backup failed SQLite integrity check')
            records = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM records')
                       if k.startswith(('iteration:', 'stage-entry:', 'stage-selection:'))
                       or k == 'postrun'}
    return records, dict(database_sha256=hashlib.sha256(raw).hexdigest(),
                         backup_sha256=expected, completed_rounds=manifest['completed_rounds'],
                         backup_utc=datetime.fromtimestamp(manifest['time'], timezone.utc).isoformat())


def rounds(records, stage):
    return sorted((v for k, v in records.items() if k.startswith('iteration:')
                   and len(k.split(':')) == 2 and v.get('stage') == stage),
                  key=lambda r: r['iteration'])


def candidate_of(value):
    return value['candidate'] if 'candidate' in value else value


def selected_round(records, stage):
    endpoint = records['stage-selection:' + stage]
    cid = endpoint['candidate']['candidate_id']
    # Never infer selection from the newest submitted candidate or active draft.
    retained = records['iteration:' + str(endpoint['iteration'])]['selected']
    if candidate_of(retained)['candidate_id'] != cid:
        raise ValueError('stage endpoint is not the committed retained selection')
    eligible = [r for r in rounds(records, stage) if r['iteration'] <= endpoint['iteration']
                and r['promoted'] and r['candidate']['candidate_id'] == cid]
    if not eligible:
        raise ValueError('selected source has no committed promotion in this stage')
    origin = eligible[0]
    if candidate_of(endpoint['training'])['candidate_id'] != cid:
        raise ValueError('endpoint training result belongs to another source')
    return endpoint, origin


def oracle_proof(campaign, case, previous, reference):
    if previous['identity']['case'] != reference['identity']['case']:
        raise ValueError('oracle scientific case identity differs')
    if not previous['observation']['complete'] or not reference['observation']['complete']:
        raise ValueError('oracle evidence is incomplete')
    if behavior(previous) != behavior(reference):
        raise ValueError('oracle counters or decoded observation hashes differ')
    return dict(campaign=campaign, case=case, equal=True,
                old_receipt=previous['receipt_sha256'], new_receipt=reference['receipt_sha256'],
                old_runtime=previous['identity']['runtime'], new_runtime=reference['identity']['runtime'],
                behavior_sha256=digest(behavior(reference)))


def verify_candidate(receipt, candidate):
    if receipt['identity']['model'] != 'candidate' or receipt['identity']['candidate'] != candidate:
        raise ValueError('measurement is not from the exact selected source and parameters')
    if not receipt['observation']['complete']:
        raise ValueError('selected measurement is incomplete')


def archived_receipts(archive, inventory):
    index = defaultdict(list)
    with zipfile.ZipFile(archive) as bundle:
        for member in inventory['receipts']:
            raw = bundle.read(member['path'])
            if hashlib.sha256(raw).hexdigest() != member['public_sha256']:
                raise ValueError('archived measurement receipt checksum differs')
            result = json.loads(raw)
            identity = result['identity']; case = identity['case']
            if case.get('frontend') != 'champsim' or not result['observation']['complete']:
                continue
            candidate = identity.get('candidate') or {}
            index[identity['model'], candidate.get('candidate_id'), case['workload']].append(result)
    return index


def group_values(value):
    return value.get('measurement', value)['groups']


def trajectories(records, campaign, stage):
    entry = records['stage-entry:' + stage]
    committed = rounds(records, stage)
    known = {entry['candidate']['candidate_id']: entry}
    points, decisions = [], []

    def add(value, number, kind, promoted):
        for split in ('training', 'validation'):
            for cores, group in group_values(value[split]).items():
                mean = group.get('review_means', group.get('means'))
                points.append(dict(campaign=campaign, stage=stage, round=number, split=split,
                                   cores=int(cores), kind=kind, promoted=promoted,
                                   core_abs_pct=mean['core_error_pct'], mae_L=mean['request']['mae']))

    add(entry, 0 if stage == 'single_core' else 10, 'retained', None)
    for r in committed:
        known[r['candidate']['candidate_id']] = r
        selected_id = candidate_of(r['selected'])['candidate_id']
        if selected_id not in known:
            raise ValueError('retained trajectory has no matching evaluated candidate')
        add(r, r['iteration'], 'submitted', r['promoted'])
        add(known[selected_id], r['iteration'], 'retained', r['promoted'])
        decisions.append(dict(campaign=campaign, stage=stage, round=r['iteration'],
                              promoted=r['promoted'], candidate=r['candidate']['candidate_id'],
                              selected=selected_id, decision=r['promotion_review']))
    return points, decisions


def import_stage(records, config, campaign, stage, source_dir, index, baseline_index, *, independent_tests=None):
    endpoint, origin = selected_round(records, stage)
    candidate = endpoint['candidate']; cid = candidate['candidate_id']; arm = campaign + '_' + stage
    source_files = {}
    for name, identity in candidate['files'].items():
        raw = (source_dir / name).read_bytes()
        if len(raw) != identity['bytes'] or hashlib.sha256(raw).hexdigest() != identity['sha256']:
            raise ValueError('frozen source or parameter file differs')
        source_files[name] = raw.decode()
    allowed = (1,) if stage == 'single_core' else (1, 4, 8)
    cohort = config['experiment']['evaluation']['champsim']
    validation = [c for c in cohort['validation'] if len(cohort['cases'][c]['programs']) in allowed]
    aliases = {f'val{i}': c for i, c in enumerate(validation, 1)}
    rows = []
    for group in group_values(endpoint['training']).values():
        rows.extend(score_row(c, arm, s, cohort='training', stage=stage)
                    for c, s in group['workloads'].items())
    for group in origin['validation']['groups'].values():
        rows.extend(score_row(aliases[c], arm, s, cohort='validation', stage=stage, projected=True)
                    for c, s in group['cases'].items())
    post = records.get('postrun', {}).get('stage_endpoints', {}).get(stage)
    if independent_tests is not None:
        if post is not None:
            raise ValueError('do not replace committed final tests with an independent run')
        post = independent_tests
    if post:
        if post['candidate'] != candidate:
            raise ValueError('final tests belong to a different selected model')
        for count, group in post['reports']['test']['candidate']['groups'].items():
            if int(count) not in allowed:
                continue
            rows.extend(score_row(c, arm, s, cohort='test', stage=stage)
                        for c, s in group['workloads'].items())
    proofs, receipts = [], {}
    for row in rows:
        case = row['case']
        model = choose_receipt(index['candidate', cid, case])
        old_oracle = choose_receipt(index['oracle', None, case])
        reference = choose_receipt(baseline_index['oracle', None, case])
        verify_candidate(model, candidate)
        proofs.append(oracle_proof(campaign, case, old_oracle, reference))
        bind_score(row, model, reference)
        row['selection_record_sha256'] = digest(endpoint)
        row['score_record_sha256'] = digest(origin['validation'] if row['cohort'] == 'validation'
                                             else post if row['cohort'] == 'test' else endpoint['training'])
        row['runtime_sha256'] = model['identity']['runtime']
        receipts[arm, case] = model, reference
    for split in ('training', 'validation', 'test'):
        wanted = {c for c in cohort[split] if len(cohort['cases'][c]['programs']) in allowed}
        actual = {r['case'] for r in rows if r['cohort'] == split}
        if actual != wanted and not (split == 'test' and post is None and not actual):
            raise ValueError(f'{campaign}/{stage}/{split}: incomplete or unexpected case population')
    convergence, decisions = trajectories(records, campaign, stage)
    return dict(rows=rows, oracle_equivalence=proofs, convergence=convergence, decisions=decisions,
                sources={arm: dict(candidate=candidate, source=source_files['model.cpp'],
                                   parameters_source=source_files['parameters.json'],
                                   selected_after_round=endpoint['iteration'], introduced_round=origin['iteration'],
                                   backend=campaign, report_sha256=digest(endpoint),
                                   configuration_sha256=digest(config))}), receipts


def independent_test_evidence(root, candidate, index):
    """Verify an authorized test study without inventing a campaign postrun."""
    inventory = read(root / 'evidence-inventory.json')
    for name, expected in inventory['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(root / relative) != expected:
            raise ValueError('independent test evidence member failed verification')
    report = read(root / 'test-results.json')
    if (not report['complete'] or report['candidate'] != candidate or report['campaign_feedback']
            or report['policy'] != 'operator-authorized-early-frozen-SC-tests-v1'):
        raise ValueError('independent tests are incomplete or use another selection/policy')
    cases = {}
    for row in report['rows']:
        if not row['complete'] or row['case'] in cases:
            raise ValueError('incomplete or duplicate independent test case')
        cases[row['case']] = row['score']
        for owner in ('oracle', 'candidate'):
            receipt = row[owner]
            if receipt['identity']['case']['workload'] != row['case']:
                raise ValueError('independent score/receipt case mismatch')
            recorded = root / receipt['directory'] / 'measurement.json'
            if sha(recorded) != receipt['receipt_sha256'] or read(recorded) != receipt['observation']:
                raise ValueError('independent observation contradicts its completed receipt')
            identity = receipt['identity']
            cid = (identity['candidate'] or {}).get('candidate_id')
            index[identity['model'], cid, row['case']].append(receipt)
    if len(cases) != 24:
        raise ValueError('independent SC test population must contain all 24 cases')
    return dict(candidate=candidate, reports={'test': {'candidate': {'groups': {'1': {'workloads': cases}}}}},
                evidence_kind='independent frozen endpoint; not a committed campaign postrun',
                source_sha256=sha(root / 'test-results.json'),
                inventory_sha256=sha(root / 'evidence-inventory.json'))


def main():
    import sys
    if sys.argv[1:2] == ['usage']:
        from campaign_usage_import import main as import_usage
        return import_usage(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'opus-records', 'opus-config', 'opus-model', 'opus-archive',
                 'opus-inventory', 'gemini-backup', 'gemini-root', 'baselines', 'output', 'work'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--include-gemini-final', action='store_true',
                        help='Import final tests/MC only after both committed endpoints and postrun exist.')
    parser.add_argument('--gemini-independent-tests', type=Path,
                        help='Verified isolated 24-case frozen-SC test evidence; never changes campaign records.')
    args = parser.parse_args()
    # Only scientific/analysis records are exported; no native state or credentials.
    from ramulator_chia.public_export import redact, POLICY
    args.work.mkdir(parents=True, exist_ok=True)
    base = read(args.base); baseline_index = measurements(args.baselines)
    opus = read(args.opus_records)
    gemini, backup_identity = consistent_records(args.gemini_backup)
    if not args.include_gemini_final:
        # Freeze this partial update even if the campaign finishes during collection.
        gemini.pop('postrun', None)
    elif not (gemini.get('stage-selection:multicore') and gemini.get('postrun')):
        raise ValueError('Gemini final endpoints/tests are not committed')
    snapshot = dict(schema=1, base_payload_sha256=digest(base), rows=[], sources={},
                    convergence=[], decisions=[], oracle_equivalence=[], extremes=[], extreme_selection=[])
    all_receipts = {}
    oi = archived_receipts(args.opus_archive, read(args.opus_inventory))
    gi = measurements(args.gemini_root)
    independent = (independent_test_evidence(args.gemini_independent_tests,
                    gemini['stage-selection:single_core']['candidate'], gi)
                   if args.gemini_independent_tests else None)
    for name, records, config, source, index in (
        ('opus', opus['records'], read(args.opus_config), args.opus_model, oi),
        ('gemini', gemini, read(args.gemini_root / 'config.json'),
         args.gemini_root / 'candidates' / gemini['stage-selection:single_core']['candidate']['candidate_id'], gi)):
        addition, receipts = import_stage(records, config, name, 'single_core', source, index, baseline_index,
                                          independent_tests=independent if name == 'gemini' else None)
        for key in ('rows', 'convergence', 'decisions', 'oracle_equivalence'):
            snapshot[key].extend(addition[key])
        snapshot['sources'].update(addition['sources']); all_receipts.update(receipts)
        print(name, 'verified', len(addition['rows']), 'selected-source scores and oracle equivalences', flush=True)
    if args.include_gemini_final:
        cid = gemini['stage-selection:multicore']['candidate']['candidate_id']
        addition, receipts = import_stage(gemini, read(args.gemini_root / 'config.json'), 'gemini',
             'multicore', args.gemini_root / 'candidates' / cid, gi, baseline_index)
        for key in ('rows', 'convergence', 'decisions', 'oracle_equivalence'):
            snapshot[key].extend(addition[key])
        snapshot['sources'].update(addition['sources']); all_receipts.update(receipts)
    snapshot['extreme_selection'] = select_extremes(snapshot['rows'], ('opus_single_core', 'gemini_single_core'))
    for selected in snapshot['extreme_selection']:
        key = selected['score']['arm'], selected['score']['case']
        model, reference = all_receipts[key]
        cache = args.work / (digest({'selection': selected, 'scanner': sha(scan_extreme.__code__.co_filename)}) + '.json')
        if cache.exists():
            snapshot['extremes'].append(read(cache)); continue
        root = args.gemini_independent_tests or args.gemini_root
        if key[0].startswith('opus'):
            root = args.work / 'opus'
            member = model['observation']['traces']['controller.csv.ch0']
            relative = Path(model['directory']) / member['name']
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe receipt member path')
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with zipfile.ZipFile(args.opus_archive) as bundle, bundle.open('observations/' + member['stored_sha256'] + '.gz') as src, path.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            if sha(path) != member['stored_sha256']:
                raise ValueError('archived observation checksum differs')
        print('Scanning full eligible observations:', *key, flush=True)
        result = scan_extreme(selected, ((args.baselines, reference), (root, model)))
        save(cache, result); snapshot['extremes'].append(result)
    snapshot['availability'] = {
        'astra': dict(schedule='10+5', completed_rounds=15, single_core_test='complete', multicore='complete'),
        'deepseek': dict(schedule='10+5', completed_rounds=15, single_core_test='complete', multicore='complete'),
        'opus': dict(schedule='10 only', completed_rounds=10, single_core_test='complete', multicore='not part of campaign'),
        'gemini': dict(schedule='10+5', completed_rounds=backup_identity['completed_rounds'],
                       single_core_test='complete' if (args.include_gemini_final or independent) else 'test evaluation pending',
                       multicore='complete' if args.include_gemini_final else 'selection pending')}
    snapshot['provenance'] = dict(policy='committed-endpoint-import-v1', redaction_policy=POLICY,
        importer_sha256=sha(__file__), created_utc=datetime.now(timezone.utc).isoformat(),
        gemini_backup=backup_identity, opus_records_sha256=sha(args.opus_records),
        opus_receipt_inventory_sha256=sha(args.opus_inventory),
        opus_archive_sha256=sha(args.opus_archive),
        gemini_independent_test=None if independent is None else {
            k: independent[k] for k in ('source_sha256','inventory_sha256','evidence_kind')},
        scope='Selected endpoints only. Test evidence never comes from active drafts. No campaign writes.',
        unavailable='Opus/Gemini multicore, hardware transfer, gem5, Lat–Tp and throughput not added in this snapshot.')
    snapshot = redact(snapshot)
    merge_snapshot(base, snapshot)  # Guard against accidental overlap before publication.
    save(args.output, snapshot)
    print(json.dumps(dict(output=str(args.output), scores=len(snapshot['rows']),
                          oracle_checks=len(snapshot['oracle_equivalence']), extremes=len(snapshot['extremes']))))


if __name__ == '__main__':
    main()
