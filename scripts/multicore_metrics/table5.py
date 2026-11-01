"""Table 5 (multicore request metrics) from the per-mix data in this repository; no simulation.

Inputs, relative to the repository root:
  results/extensions/multicore/metrics/pairfree.json     per-core W1/L, mean/L and windowed/L per test
                                                         mix and arm (pairfree.py output)
  results/extensions/multicore/metrics/alt_metrics.json  the same metrics (same w1()) for the
                                                         +1-cycle oracle, Astra MC and DeepSeek MC
  results/extensions/multicore/metrics/scores_p_rl1.json paired MAE_Req of the oracle with read latency
                                                         +1 cycle (score.py)
  results/extensions/multicore/metrics/pair_stats.json   paired MAE_Req of a per-core constant (each
                                                         core's oracle median) on the Astra MC pairs
  results/foundation/campaigns.json                      paired MAE_Req of Astra MC, DeepSeek MC, Mess
  results/extensions/multicore/mc_endpoints_test_by_group.csv  paired MAE_Req of Gemini MC, Opus MC
Each value is averaged over cores, then over the 8 test mixes of a core count. The +1-cycle oracle and
Astra MC rows of pairfree.json must equal alt_metrics.json exactly.

Writes per_mix.csv, summary.csv and table5.csv to the output directory.
Usage: python3 -I table5.py [repository root] [output directory]
"""
import csv, json, os, re, sys, statistics

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results/extensions/multicore/metrics')
D = os.path.join(ROOT, 'results/extensions/multicore/metrics')
M = ('w1', 'mean', 'win')


def read(name):
    with open(os.path.join(D, name)) as f:
        return json.load(f)


def cores(case):
    return int(re.match(r'test-c(\d)', case).group(1))


pairfree, alt = read('pairfree.json'), read('alt_metrics.json')
cases = sorted(pairfree)

per_mix = []
for case in cases:
    for arm, x in sorted(pairfree[case].items()):
        if arm == '__oracle__':
            continue
        per_mix.append(dict(arm=arm, case=case, cores=cores(case), **{k: x[k] for k in M}, L=x['L'],
                            cores_scored=len(x['cores_scored']), trace_sha256=x['trace_sha256']))
    x = alt[f'{case}|ds_mc']
    per_mix.append(dict(arm='deepseek_multicore', case=case, cores=cores(case), **{k: x[k] for k in M}))

for arm, key in (('oracle_plus1', 'p_rl1'), ('astra_multicore', 'astra_mc')):
    diff = max(abs(pairfree[c][arm][k] - alt[f'{c}|{key}'][k]) for c in cases for k in M)
    assert diff == 0.0, (arm, diff)
    print(f'{arm}: equal to alt_metrics.json (max |diff| {diff})')


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


summary = []
for arm in sorted({r['arm'] for r in per_mix}):
    for c in (4, 8):
        g = [r for r in per_mix if r['arm'] == arm and r['cores'] == c]
        summary.append(dict(arm=arm, cores=c, mixes=len(g), **{k: mean(r[k] for r in g) for k in M}))
w1 = {(r['arm'], r['cores']): r['w1'] for r in summary}

campaign = json.load(open(os.path.join(ROOT, 'results/foundation/campaigns.json')))['rows']
endpoints = list(csv.DictReader(open(os.path.join(ROOT, 'results/extensions/multicore/mc_endpoints_test_by_group.csv'))))
scores, pairs = read('scores_p_rl1.json'), read('pair_stats.json')


def mae(arm, c):
    if arm == 'oracle_plus1':
        return mean(v['mae'] for k, v in scores.items() if cores(k) == c)
    if arm == 'constant':
        return mean(v['mae_const_median_core'] for k, v in pairs.items()
                    if k.endswith('|astra_mc') and cores(k) == c)
    if arm in ('opus_multicore', 'gemini_multicore'):
        row, = [r for r in endpoints if r['arm'] == arm and int(r['cores']) == c]
        return float(row['mean_request_MAE_over_L'])
    return mean(r['mae_L'] for r in campaign if r['arm'] == arm and r['cohort'] == 'test' and r['cores'] == c)


ROWS = (('Oracle, read latency +1 cycle', 'oracle_plus1', 'oracle_plus1'),
        ('Per-core constant (median)', 'constant', None),
        ('Mess', 'mess', 'mess'),
        ('GPT-6 Astra', 'astra_multicore', 'astra_multicore'),
        ('DeepSeek V4.1 Flash', 'deepseek_multicore', 'deepseek_multicore'),
        ('Gemini 3.8 Flash', 'gemini_multicore', 'gemini_multicore'),
        ('Opus 5.5', 'opus_multicore', 'opus_multicore'))
table = [dict(row=label, mae_req_4=mae(m, 4), mae_req_8=mae(m, 8),
              w1_4=w1.get((a, 4)) if a else None, w1_8=w1.get((a, 8)) if a else None)
         for label, m, a in ROWS]

os.makedirs(OUT, exist_ok=True)
for name, rows in (('per_mix.csv', per_mix), ('summary.csv', summary), ('table5.csv', table)):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
for r in table:
    print(f"{r['row']:32s}" + ''.join(f"{r[k]:8.3f}" if r[k] is not None else '       -'
                                     for k in ('mae_req_4', 'mae_req_8', 'w1_4', 'w1_8')))
