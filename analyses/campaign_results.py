"""Collect the finished 10+5 campaigns without changing their evidence.

This collector preserves the historical comprehensive evidence. The paper-only
notebook uses the versioned paper bundle derived from it. Raw observations are
needed only to reconstruct request-level case studies; ordinary notebook
execution is offline. No simulator or model provider is called here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import os

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / 'chia-loop').is_dir() else HERE.parents[1]
CAMPAIGNS = Path(os.environ.get('RAMULATOR_CHIA_CAMPAIGNS', REPO/'.work/campaigns'))
BASELINES = Path(os.environ.get('RAMULATOR_CHIA_BASELINES', REPO/'.work/baselines'))
STUDY = Path(os.environ.get('RAMULATOR_CHIA_TRANSFER', REPO/'.work/transfer'))
CAMPAIGN_NAMES = {'astra': 'astra_xhigh', 'deepseek': 'deepseek_max'}
STAGES = ('single_core', 'multicore')
BASELINE_NAMES = ('fixedlat', 'md1', 'wmg1', 'mess')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def measurements(root):
    """Index immutable result pointers, not incomplete attempt directories."""
    index = defaultdict(list)
    for path in sorted((root / 'measurements').glob('*/result.json')):
        result = read(path)
        identity = result['identity']
        case = identity.get('case', {}).get('workload')
        candidate = identity.get('candidate') or {}
        candidate_id = candidate.get('candidate_id') if isinstance(candidate, dict) else candidate
        if (case and identity['case'].get('frontend') == 'champsim'
                and result['observation']['complete']):
            index[identity['model'], candidate_id, case].append(result)
    return index


def behavior(result):
    """Exact simulated evidence; exclude gzip bytes and host wall clocks."""
    observation = result['observation']
    return {
        **{k: observation[k] for k in ('case', 'frontend_stats', 'controller_stats', 'per_core_cycles')},
        'traces': {k: {f: v[f] for f in ('logical_sha256', 'logical_bytes')}
                   for k, v in observation['traces'].items()},
    }


def choose_receipt(options):
    if not options:
        raise ValueError('missing completed measurement')
    first = behavior(options[0])
    if any(behavior(other) != first for other in options[1:]):
        raise ValueError('duplicate measurement identities disagree in simulated behavior')
    return options[0]


def bind_score(row, model, oracle):
    """Reconcile a compact score with its completed measurement receipts."""
    a, b = model['observation'], oracle['observation']
    if not a['complete'] or not b['complete']:
        raise ValueError('incomplete measurement cannot supply an accuracy score')
    if model['identity']['case'] != oracle['identity']['case']:
        raise ValueError('case identity differs across comparison versions')
    if len(a['per_core_cycles']) != len(b['per_core_cycles']) or len(a['per_core_cycles']) != row['cores']:
        raise ValueError('per-core count differs')
    if any(c <= 0 for c in b['per_core_cycles']):
        raise ValueError('nonpositive oracle cycle count')
    signed = [100 * (m - o) / o for m, o in zip(a['per_core_cycles'], b['per_core_cycles'])]
    if not np.allclose(signed, row['per_core_signed_pct'], rtol=1e-12, atol=1e-12):
        raise ValueError(f'{row["arm"]}/{row["case"]}: score is not the selected source measurement')
    row.update(instructions=a['frontend_stats']['per_core_instructions'],
               background=a['frontend_stats'].get('background'),
               model_receipt=model['receipt_sha256'], oracle_receipt=oracle['receipt_sha256'])


def score_row(case, arm, score, *, cohort=None, stage=None, projected=False):
    """Normalize the existing full and reviewer-projected score schemas."""
    if projected:
        per_core = score['per_core_signed_error_pct']
        core = score['core_error_pct']
        makespan = score['makespan_signed_error_pct']
        populations = score['populations']
    else:
        per_core = score['cycles']['per_core_dev_pct']
        core = score['cycles']['mean_abs_per_core_pct']
        makespan = score['cycles']['makespan_dev_pct']
        populations = {k: score['request_pairing'][k] for k in ('oracle', 'model')}
        if score.get('pairing_error'):
            raise ValueError(f'{case}: {score["pairing_error"]}')
    request = score['request']
    if request is None or request['address_mismatch_pairs'] != 0:
        raise ValueError(f'{case}: invalid or missing physical request matching')
    if not np.isclose(core, np.mean(np.abs(per_core)), rtol=1e-12, atol=1e-12):
        raise ValueError(f'{case}: per-core aggregation differs')
    if not np.isfinite([*per_core, makespan, request['mae'], request['sgn'], request['tail'],
                        request['p999_over_L'], request['oracle_read_mean_latency']]).all():
        raise ValueError(f'{case}: nonfinite accuracy metric')
    row = dict(case=case, arm=arm, cohort=cohort or case.split('-')[0], stage=stage,
               cores=len(per_core), core_abs_pct=core, core_signed_pct=float(np.mean(per_core)),
               per_core_signed_pct=per_core, worst_core_pct=max(map(abs, per_core)),
               makespan_signed_pct=makespan, makespan_abs_pct=abs(makespan),
               mae_L=request['mae'], drift_L=request['sgn'], abs_drift_L=abs(request['sgn']),
               mae_cycles=request['mae_cycles'], p99_L=request['tail'], p999_L=request['p999_over_L'],
               p99_cycles=request['paired_p99_cycles'], p999_cycles=request['paired_p999_cycles'],
               min_cycles=request['extreme_min_cycles'], max_cycles=request['extreme_max_cycles'],
               min_L=request['extreme_min_over_L'], max_L=request['extreme_max_over_L'],
               L=request['oracle_read_mean_latency'], pairs=request['matched'],
               oracle_coverage=request['cov_o'], model_coverage=request['cov_m'],
               populations=populations, address_mismatches=request['address_mismatch_pairs'],
               low_traffic=request['n_oracle'] < 10000)
    for multiple in (1, 5):
        tail = request['tail_thresholds'][f'{multiple}L']
        row[f'tail_{multiple}L_fraction'] = tail['fraction']
        row[f'tail_{multiple}L_count'] = tail['count']
        row[f'tail_{multiple}L_share'] = tail['absolute_error_share']
    for owner in ('oracle', 'model'):
        pop = populations[owner]
        for field in ('recorded_reads', 'without_stable_id', 'unmatched_reads',
                      'admissions_without_recorded_latency', 'admitted_foreground_reads',
                      'total_admitted_reads'):
            row[f'{owner}_{field}'] = pop[field]
        if pop['recorded_reads'] <= 0 or row['pairs'] > pop['recorded_reads']:
            raise ValueError(f'{case}: impossible recorded-read population')
        if not np.isclose(row[f'{owner}_coverage'], row['pairs'] / pop['recorded_reads']):
            raise ValueError(f'{case}: coverage denominator changed')
    return row


def select_extremes(rows, arms=('astra_single_core', 'deepseek_single_core')):
    """Predeclared complementary examples, with stable name tie-breaking."""
    chosen = []
    for arm in arms:
        cases = [r for r in rows if r['arm'] == arm and r['cohort'] == 'test' and r['cores'] == 1]
        if not cases:
            continue  # A pending test cohort is not an empty accuracy population.
        core_mid = np.median([r['core_abs_pct'] for r in cases])
        request_mid = np.median([r['mae_L'] for r in cases])
        selectors = [
            ('largest core error', cases, lambda r: r['core_abs_pct']),
            ('largest request MAE/L', cases, lambda r: r['mae_L']),
            ('largest P99.9/L below median core error',
             [r for r in cases if r['core_abs_pct'] < core_mid], lambda r: r['p999_L']),
            ('largest absolute request error below both medians',
             [r for r in cases if r['core_abs_pct'] < core_mid and r['mae_L'] < request_mid],
             lambda r: max(abs(r['min_cycles']), abs(r['max_cycles']))),
        ]
        by_case = {}
        for reason, pool, key in selectors:
            if not pool:
                continue
            row = sorted(pool, key=lambda r: (-key(r), r['case']))[0]
            by_case.setdefault(row['case'], dict(score=row, reasons=[]))['reasons'].append(reason)
        chosen.extend(by_case.values())
    return chosen


def collect(campaigns=CAMPAIGNS, baselines=BASELINES, study=STUDY):
    final_path = REPO / 'reports/final_model_transfer_0923.json'
    final = read(final_path)
    if not all(r['accepted'] for r in final['rows']):
        raise ValueError('final-study comparison failures require explicit handling')
    baseline_path = baselines / 'baseline-accuracy.json'
    baseline = read(baseline_path)
    inventory = read(baselines / 'coverage.json')
    if inventory['complete'] != 500 or inventory['required'] != 500:
        raise ValueError('expected the verified 500-entry baseline inventory')
    bindex = measurements(baselines)
    rows = [score_row(r['case'], r['model'], r['score']) for r in baseline['per_case']]
    for row in rows:
        bind_score(row, choose_receipt(bindex[row['arm'], None, row['case']]),
                   choose_receipt(bindex['oracle', None, row['case']]))
    sources, convergence, decisions, proofs, receipts, memberships, inputs = {}, [], [], [], {}, {}, {}
    for short, name in CAMPAIGN_NAMES.items():
        root = campaigns / name
        report_path = REPO / 'reports' / f'chia_staged_{short}_0923.json'
        report = read(report_path)
        if report['completed_rounds'] != 15:
            raise ValueError(f'{short}: unfinished campaign')
        config = read(root / 'config.json')
        cohort = config['experiment']['evaluation']['champsim']
        memberships.update(cohort['cases'])
        for case, item in cohort['cases'].items():
            inputs[case] = dict(programs=item['programs'], placement_sha256=item['placement']['sha256'])
        index = measurements(root)
        with sqlite3.connect((root / 'campaign.sqlite').as_uri() + '?mode=ro', uri=True) as db:
            entries = {k: json.loads(v) for k, v in db.execute(
                "SELECT key,value FROM records WHERE key LIKE 'stage-entry:%'")}
        # Every oracle case, not just a favorable representative, must match.
        for (model, _, case), values in bindex.items():
            if model != 'oracle':
                continue
            current = choose_receipt(values)
            candidates = index.get(('oracle', None, case), [])
            compatible = [r for r in candidates if behavior(r) == behavior(current)]
            if not compatible:
                raise ValueError(f'{short}/{case}: historical and corrected oracle evidence differs')
            old = choose_receipt(compatible)
            proofs.append(dict(campaign=short, case=case, equal=True,
                               old_receipt=old['receipt_sha256'], new_receipt=current['receipt_sha256'],
                               old_runtime=old['identity']['runtime'], new_runtime=current['identity']['runtime'],
                               behavior_sha256=digest(behavior(current))))
        for stage in STAGES:
            selected = report['stage_endpoints'][stage]
            candidate = selected['candidate']; cid = candidate['candidate_id']; arm = short + '_' + stage
            originating = next(r for r in report['rounds'] if r['promoted'] and
                               r['stage'] == stage and r['candidate']['candidate_id'] == cid)
            source_path = study / 'candidates' / cid / 'model.cpp'
            if sha(source_path) != candidate['files']['model.cpp']['sha256']:
                raise ValueError('frozen endpoint source differs')
            sources[arm] = dict(candidate=candidate, source=source_path.read_text(),
                                selected_after_round=selected['iteration'], introduced_round=originating['round'],
                                backend=report['backend'], report_sha256=sha(report_path),
                                configuration_sha256=sha(root / 'config.json'))
            # The selected record contains its exact retained training measurement.
            for group in selected['training']['measurement']['groups'].values():
                rows.extend(score_row(case, arm, score, stage=stage)
                            for case, score in group['workloads'].items())
            allowed = (1,) if stage == 'single_core' else (1, 4, 8)
            validation = [n for n in cohort['validation'] if len(cohort['cases'][n]['programs']) in allowed]
            alias = {f'val{i}': n for i, n in enumerate(validation, 1)}
            for group in originating['validation']['groups'].values():
                rows.extend(score_row(alias[case], arm, score, cohort='validation', stage=stage, projected=True)
                            for case, score in group['cases'].items())
            # The SC endpoint was evaluated on expanded cohorts before round 11.
            # These are transfer measurements of that same frozen source, not MC training results.
            if stage == 'single_core':
                entry = entries['stage-entry:multicore']
                if entry['candidate']['candidate_id'] != cid:
                    raise ValueError('multicore stage entry does not preserve the SC endpoint')
                for split in ('training', 'validation'):
                    for count, group in entry[split]['measurement']['groups'].items():
                        if int(count) > 1:
                            rows.extend(score_row(case, arm, score, cohort=split, stage=stage)
                                        for case, score in group['workloads'].items())
            test = report['postrun']['stage_endpoints'][stage]
            if test['candidate']['candidate_id'] != cid:
                raise ValueError('postrun evaluated a different endpoint')
            for group in test['reports']['test']['candidate']['groups'].values():
                rows.extend(score_row(case, arm, score, stage=stage)
                            for case, score in group['workloads'].items())
            # Bind every compact score to actual measurement counts and inputs.
            for row in [r for r in rows if r['arm'] == arm]:
                case = row['case']
                model = choose_receipt(index.get(('candidate', cid, case), []))
                oracle = choose_receipt(bindex['oracle', None, case])
                bind_score(row, model, oracle)
                receipts[arm, case] = ((baselines, oracle), (root, model))
        # Preserve candidate and incumbent trajectories separately, including the seed.
        known = {}
        for r in report['rounds']:
            known.setdefault((r['stage'], r['candidate']['candidate_id']), r)
        for stage in STAGES:
            entry = entries['stage-entry:' + stage]
            stage_start = 0 if stage == 'single_core' else 10
            for split in ('training', 'validation'):
                measurement = entry[split].get('measurement')
                view = entry[split]
                groups = measurement['groups'] if measurement else view['groups']
                for cores, g in groups.items():
                    means = g.get('review_means', g.get('means'))
                    convergence.append(dict(campaign=short, stage=stage, round=stage_start, split=split,
                        cores=int(cores), kind='retained', core_abs_pct=means['core_error_pct'],
                        mae_L=means['request']['mae'], promoted=None))
            for r in [v for v in report['rounds'] if v['stage'] == stage]:
                selected_id = r['selected']['candidate_id']
                retained = known.get((stage, selected_id))
                for kind, value in [('submitted', r), ('retained', retained)]:
                    for split in ('training', 'validation'):
                        if value is None:
                            groups = entries['stage-entry:' + stage][split]
                            groups = groups.get('measurement', groups)['groups']
                        else:
                            groups = value[split]['groups']
                        for cores, g in groups.items():
                            means = g.get('means', g.get('review_means'))
                            convergence.append(dict(campaign=short, stage=stage, round=r['round'], split=split,
                                cores=int(cores), kind=kind, core_abs_pct=means['core_error_pct'],
                                mae_L=means['request']['mae'], promoted=r['promoted']))
                decisions.append(dict(campaign=short, round=r['round'], stage=stage,
                                      promoted=r['promoted'], candidate=r['candidate']['candidate_id'],
                                      selected=selected_id, decision=r['decision']))
    # Independently rerun default-point transfer results must agree with campaign scores.
    default_proofs = []
    for r in final['rows']:
        if r['study'] != 'hardware' or r['point'] != 'DDR5_16Gb_x8_q64':
            continue
        old = next(v for v in rows if v['case'] == r['case'] and v['arm'] == r['arm'])
        for current_key, original_key in [('core_abs_pct', 'core_abs_pct'),
                                          ('request_mae_over_L', 'mae_L'), ('L', 'L'),
                                          ('pairs', 'pairs'), ('negative_cycles', 'min_cycles'),
                                          ('positive_cycles', 'max_cycles')]:
            if not np.isclose(r[current_key], old[original_key], rtol=1e-12, atol=1e-12):
                raise ValueError(f'default transfer point differs: {r["arm"]}/{r["case"]}/{current_key}')
        default_proofs.append(dict(case=r['case'], arm=r['arm'], equal=True))
    keys = [(r['arm'], r['case']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate score rows')
    data = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
                rows=rows, sources=sources, convergence=convergence, decisions=decisions,
                membership={k: v['programs'] for k, v in memberships.items()}, inputs=inputs,
                transfer=final, oracle_equivalence=proofs, default_transfer_equivalence=default_proofs,
                provenance={'baseline_accuracy_sha256': sha(baseline_path),
                            'baseline_inventory_sha256': sha(baselines / 'coverage.json'),
                            'baseline_manifest': read(baselines / 'manifest.json')['baseline_revision'],
                            'baseline_source_audit': (REPO / 'doc/baseline_source_audit.md').read_text(),
                            'transfer_data_sha256': sha(final_path), 'collector_sha256': sha(__file__)},
                extremes=[], extreme_selection=select_extremes(rows), lat_tp=None)
    return data, receipts


def scan_extreme(selected, measurements_pair):
    """Full-population statistics, compact distribution curves and plot samples."""
    from ramulator_chia.eval import matchlib
    from ramulator_chia.eval.metrics import paired_error_statistics

    score = selected['score']; frames, paths, identities = [], [], []
    for root, result in measurements_pair:
        observation = result['observation']; member = observation['traces']['controller.csv.ch0']
        path = root / result['directory'] / member['name']
        if not path.is_file():
            raise FileNotFoundError(f'preserved request evidence needs retrieval: {path}')
        if sha(path) != member['stored_sha256']:
            raise ValueError('compressed request evidence checksum differs')
        frame, stable = matchlib._load_frame(path, windows=observation['frontend_stats']['admission_windows'])
        if not stable:
            raise ValueError('stable request IDs required')
        frames.append(frame); paths.append(path)
        identities.append(dict(receipt=result['receipt_sha256'], trace=member,
                               windows=observation['frontend_stats']['admission_windows']))
    pairs, _, _ = matchlib._stable_pairs(*frames, *paths)
    if (pairs.addr_o != pairs.addr_m).any():
        raise ValueError('physical-address mismatch')
    error = (pairs.lat_m - pairs.lat_o).to_numpy()
    scale = float(frames[0].lat.mean())
    metric = paired_error_statistics(error, scale, include_tails=True)
    for a, b in [(len(pairs), score['pairs']), (scale, score['L']), (metric['mae'], score['mae_L']),
                 (metric['sgn'], score['drift_L']), (metric['tail'], score['p99_L']),
                 (metric['p999_over_L'], score['p999_L']),
                 (metric['extreme_min_cycles'], score['min_cycles']),
                 (metric['extreme_max_cycles'], score['max_cycles'])]:
        if not np.isclose(a, b, rtol=1e-12, atol=1e-12):
            raise ValueError(f'raw observations do not reproduce the saved score: {a} vs {b}')
    # Curve probabilities are evaluated on the entire population, not a sample.
    absolute = np.abs(error) / scale
    thresholds = np.unique(np.r_[0, np.geomspace(max(1 / scale, 1e-5), max(absolute.max(), 1 / scale), 240),
                                 score['p99_L'], score['p999_L'], 1, 5, absolute.max()])
    ordered = np.sort(absolute)
    exceedance = (len(ordered) - np.searchsorted(ordered, thresholds, side='right')) / len(ordered)
    latency_max = max(float(frames[0].lat.max()), float(frames[1].lat.max()))
    edges = np.r_[0, np.geomspace(1, max(2, latency_max + 1), 160)]
    histogram = {key: np.histogram(frame.lat.to_numpy(), bins=edges)[0].tolist()
                 for key, frame in zip(('oracle', 'model'), frames)}
    pairs = pairs.assign(error_cycles=error).sort_values(['frontend_id', 'frontend_sub_id'])
    bins = []
    for number, block in pairs.groupby(pairs.frontend_id // 100000):
        values = block.error_cycles.to_numpy() / scale
        bins.append(dict(instruction_million=(int(number) + .5) / 10, count=len(values),
                         mean=float(values.mean()), q01=float(np.quantile(values, .01)),
                         q99=float(np.quantile(values, .99)), minimum=float(values.min()),
                         maximum=float(values.max()), over_5L=int((np.abs(values) > 5).sum())))
    columns = ['src', 'frontend_id', 'frontend_sub_id', 'addr_o', 'arrive_o', 'depart_o',
               'arrive_m', 'depart_m', 'lat_o', 'lat_m', 'error_cycles']
    extreme_ids = pairs.error_cycles.nsmallest(12).index.union(pairs.error_cycles.nlargest(12).index)
    sample = pairs.iloc[np.linspace(0, len(pairs) - 1, min(1600, len(pairs)), dtype=int)]
    return dict(case=score['case'], arm=score['arm'], reasons=selected['reasons'],
                score=score, identities=identities, verified=True,
                exact_statistics=metric, ccdf=dict(thresholds=thresholds.tolist(), probability=exceedance.tolist()),
                absolute_error_quantiles_cycles={str(q):float(np.quantile(np.abs(error),q))
                                                for q in (.25,.5,.75,.9,.99,.999)},
                histogram=dict(edges=edges.tolist(), **histogram), bins=bins,
                sample=sample[columns].to_dict('records'),
                extreme_requests=pairs.loc[extreme_ids, columns].to_dict('records'),
                sample_policy='evenly spaced logical request IDs; plot only, not metric estimation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaigns', type=Path, default=CAMPAIGNS)
    parser.add_argument('--baselines', type=Path, default=BASELINES)
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--output', type=Path, default=REPO / 'reports/chia_campaign_results.data.json')
    parser.add_argument('--scan-extremes', action='store_true')
    parser.add_argument('--lat-tp', type=Path)
    args = parser.parse_args()
    data, receipts = collect(args.campaigns, args.baselines, args.study)
    cache = args.output.parent / 'chia_campaign_results.extremes'
    for selected in data['extreme_selection']:
        row = selected['score']; key = (row['arm'], row['case'])
        identity = digest({'score': row, 'scanner': sha(__file__)})
        path = cache / (identity + '.json')
        if path.exists():
            data['extremes'].append(read(path))
        elif args.scan_extremes:
            print('Scanning', *key, flush=True)
            result = scan_extreme(selected, receipts[key])
            save(path, result); data['extremes'].append(result)
    if args.lat_tp:
        data['lat_tp'] = read(args.lat_tp)
        for arm, candidate in data['lat_tp']['protocol']['model_sources'].items():
            if data['sources'][arm]['candidate'] != candidate:
                raise ValueError('Lat–Tp uses a different source or parameter selection')
        data['provenance']['lat_tp_sha256'] = sha(args.lat_tp)
    save(args.output, data)
    print(json.dumps(dict(output=str(args.output), rows=len(data['rows']),
                         oracle_checks=len(data['oracle_equivalence']), extremes=len(data['extremes'])), indent=2))


if __name__ == '__main__':
    main()
