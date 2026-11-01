"""Pairing-free multicore request metrics over a list of test mixes.
Per test mix, against one oracle trace: per-core Wasserstein-1 of read-latency distributions (w1),
per-core |mean latency error| (mean), windowed (K=50 equal-count admission-order windows) mean-latency
error (win); each / L (L = mean oracle read latency inside the oracle's admission windows), averaged
over cores with >= 10*K reads in both traces. alt_metrics.json was computed with the same w1() and metrics().

Usage: python3 -I pairfree.py jobs.json out.json [nproc]
jobs.json: {case: {"oracle": {"trace":..,"windows":..}, "arms": {arm: {"trace":..,"windows":..}}}}
One worker per case (oracle loaded once per case)."""
import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from loadlib import load, sha256


def w1(a, b):
    q = np.linspace(0.0005, 0.9995, 2000)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def metrics(O, M, K=50):
    L = O.lat.mean(); res = {'w1': [], 'mean': [], 'win': []}
    for c, go in O.groupby('src'):
        gm = M[M.src == c]
        if len(gm) < K * 10 or len(go) < K * 10: continue
        res['w1'].append(w1(go.lat.values, gm.lat.values) / L)
        res['mean'].append(abs(gm.lat.mean() - go.lat.mean()) / L)
        so = go.sort_values('admission_ordinal').lat.values; sm = gm.sort_values('admission_ordinal').lat.values
        wo = np.array([x.mean() for x in np.array_split(so, K)]); wm = np.array([x.mean() for x in np.array_split(sm, K)])
        res['win'].append(float(np.abs(wo - wm).mean()) / L)
    return {k: float(np.mean(v)) for k, v in res.items()}


def extra(O, M, K=50):
    used = [int(c) for c, go in O.groupby('src') if len(go) >= K * 10 and (M.src == c).sum() >= K * 10]
    return dict(L=float(O.lat.mean()), n_oracle_reads=int(len(O)), n_model_reads=int(len(M)),
                cores_scored=used, n_cores_oracle=int(O.src.nunique()))


def run_case(item):
    case, spec = item
    out = {}
    t0 = time.time()
    O = load(spec['oracle']['trace'], spec['oracle']['windows'])
    out['__oracle__'] = dict(sha256=sha256(spec['oracle']['trace']), n=int(len(O)))
    for arm, m in spec['arms'].items():
        try:
            M = load(m['trace'], m['windows'])
            r = metrics(O, M); r.update(extra(O, M)); r['trace_sha256'] = sha256(m['trace'])
        except Exception as e:  # report, do not hide
            r = dict(error=f'{type(e).__name__}: {e}')
        out[arm] = r
        print(time.strftime('%T'), case, arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ('w1', 'mean', 'win', 'error')}, flush=True)
    print(time.strftime('%T'), case, 'done in', round(time.time() - t0), 's', flush=True)
    return case, out


if __name__ == '__main__':
    jobs = json.load(open(sys.argv[1]))
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    res = {}
    with Pool(min(nproc, 16, len(jobs))) as p:
        for case, out in p.imap_unordered(run_case, sorted(jobs.items())):
            res[case] = out
            json.dump(res, open(sys.argv[2], 'w'), indent=1, sort_keys=True)
