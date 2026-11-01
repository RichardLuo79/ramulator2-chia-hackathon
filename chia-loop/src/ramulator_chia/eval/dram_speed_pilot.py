"""Small frozen-runtime feasibility pilot, not publication throughput.

No controller or batch-kernel changes. The native runner's simulation timer
includes replay CSV serialization and its existing polling admission retries.
"""
import argparse
import copy
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

ARMS = ('oracle', 'astra_single_core', 'deepseek_single_core', 'fixedlat', 'md1',
        'wmg1', 'mess', 'astra_multicore', 'deepseek_multicore')
INTERVALS = (1, 4, 16, 64, 256)
STOP = float(os.environ.get('RAMULATOR_CHIA_STOP_AT', 'inf'))


def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(row, bank, group, slot):
    assert 0 <= row < 65536 and 0 <= bank < 4 and 0 <= group < 8 and 0 <= slot < 64
    return ((((row * 4 + bank) * 8 + group) * 64 + slot) * 64)


def fixtures(root):
    """Same Lat-Tp coordinate recipes; separate deterministic address/type RNGs.

    This pilot's Python RNG is explicitly recorded, not claimed to reproduce
    the native mt19937_64 draw sequence used by the production generator.
    """
    records = []
    for pattern in ('streaming', 'random'):
        address_rng, type_rng = random.Random(12345), random.Random(12345 ^ 0x54595045)
        addresses, choices = [], []
        for i in range(500000):
            if pattern == 'streaming':
                flat = i % 32
                per_bank = i // 32 + flat * 2
                address = encode((per_bank // 64) % 65536, flat // 8, flat % 8, per_bank % 64)
            else:
                address = encode(address_rng.randrange(65536), address_rng.randrange(4),
                                 address_rng.randrange(8), address_rng.randrange(64))
            addresses.append(address)
            choices.append(type_rng.randrange(100))
        for ratio in (100, 50):
            for interval in INTERVALS:
                for n in (100000, 500000):
                    name = f'{pattern}-r{ratio}-i{interval}-n{n}'
                    path = root / 'inputs' / (name + '.csv')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    reads = 0
                    with path.open('w') as out:
                        out.write('arrive,addr,type,source\n')
                        for i in range(n):
                            kind = int(choices[i] >= ratio)
                            reads += kind == 0
                            out.write(f'{i*interval},{addresses[i]},{kind},0\n')
                    records.append(dict(name=name, pattern=pattern, read_percent=ratio,
                                        interval=interval, n=n, reads=reads, writes=n-reads,
                                        path=str(path.relative_to(root)), sha256=sha(path)))
    save(root / 'fixtures.json', dict(seed=12345, type_seed=12345 ^ 0x54595045,
         rng='Python random.Random; generation outside timing', points=records))


def estimate(root):
    rows = [read(p) for p in (root / 'pilot').glob('*/receipt.json')]
    complete = [r for r in rows if r.get('complete')]
    estimates = []
    for arm in ARMS:
        selected = [r for r in complete if r['arm'] == arm]
        small = {r['point']['name'].replace('n100000', 'n500000'):r for r in selected if r['point']['n'] == 100000}
        large = [r for r in selected if r['point']['n'] == 500000]
        ratios = [r['simulation_wall_seconds'] / small[r['point']['name']]['simulation_wall_seconds']
                  for r in large if r['point']['name'] in small]
        # Twenty large cases cover both endpoint read ratios; interpolate the
        # four middle ratios. Scale 0.5M to 5.1M total work including warmup.
        one = sum(r['simulation_wall_seconds'] for r in large) * 3 * 10.2 if len(large) == 20 else None
        estimates.append(dict(arm=arm, completed=len(selected), expected=40,
             projected_one_pass_seconds=one,
             projected_five_pass_seconds=None if one is None else one*5,
             size_runtime_ratios=ratios,
             strongest_superlinear_ratio=max(ratios, default=None)))
    full = all(r['projected_one_pass_seconds'] is not None for r in estimates)
    save(root / 'pilot-estimate.json', dict(time=time.time(), successful=len(complete),
         attempts=len(rows), expected=360, models=estimates, full_coverage=full,
         projected_one_pass_seconds=sum(r['projected_one_pass_seconds'] for r in estimates) if full else None,
         projected_five_pass_seconds=sum(r['projected_five_pass_seconds'] for r in estimates) if full else None,
         caveats=['actual Granite Rapids measurements; not final publication timings',
                  'no simulated warmup in pilot; production has 100k warmup + 5M requests',
                  'pilot includes native CSV serialization and per-cycle backpressure retries',
                  'middle read ratios interpolated from 100% and 50%; queue growth can be nonlinear']))


def run(root):
    import yaml
    manifest = read(root / 'pilot-manifest.json')
    for relative, digest in manifest['files'].items():
        if sha(root / relative) != digest:
            raise RuntimeError('staged file checksum mismatch: ' + relative)
    if os.cpu_count() != 8 or 2 not in os.sched_getaffinity(0):
        raise RuntimeError('unexpected pilot CPU topology')
    points = read(root / 'fixtures.json')['points']
    process = None
    def stop(signum, frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise KeyboardInterrupt('pilot interrupted by preservation/cutoff')
    signal.signal(signal.SIGTERM, stop)
    for arm in ARMS:
        for point in points:
            if time.time() >= STOP:
                save(root / 'pilot-status.json', dict(time=time.time(), state='cutoff'))
                return
            target = root / 'pilot' / (arm + '--' + point['name'])
            receipt_path = target / 'receipt.json'
            if receipt_path.exists():
                old = read(receipt_path)
                if old.get('complete') and old['manifest_sha256'] == sha(root / 'pilot-manifest.json'):
                    continue
                # Never overwrite failed attempts or repeatedly try a bad point.
                continue
            available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                                 if s.startswith('MemAvailable:'))) * 1024
            if available < 16 * 1024**3:
                save(root / 'pilot-status.json', dict(time=time.time(), state='memory_hold', available=available))
                return
            target.mkdir(parents=True)
            observations = target / 'observations'
            observations.mkdir()
            cfg = read(root / 'configs' / (arm + '.json'))
            cfg['batch_trace'] = str(root / point['path'])
            save(target / 'config.json', cfg)
            library = manifest['models'][arm].get('library')
            library = str(root / library) if library else '-'
            cmd = ['/usr/bin/time', '-f', '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"peak_rss_kib":%M}',
                   '-o', str(target / 'resources.json'), 'taskset', '-c', '2',
                   str(root / 'runtime/isolated_sim'), str(target / 'config.json'), library,
                   str(observations), str(observations / 'stats.yaml')]
            save(root / 'pilot-status.json', dict(time=time.time(), state='running', arm=arm, point=point))
            began = time.time()
            error = None
            with (target / 'output.log').open('w') as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'LD_LIBRARY_PATH':str(root / 'runtime'), 'OMP_NUM_THREADS':'1'})
                try:
                    code = process.wait(timeout=1800)
                    if code: error = f'exit status {code}'
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    error = type(exc).__name__
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
            row = dict(arm=arm, point=point, complete=False, started=began, ended=time.time(),
                       error=error, manifest_sha256=sha(root / 'pilot-manifest.json'))
            if error is None:
                stats = yaml.safe_load((observations / 'stats.yaml').read_text())
                batch = stats['batch']
                if batch['reads_completed'] != point['reads'] or batch['writes_completed'] != point['writes']:
                    raise RuntimeError('incomplete pilot callback population')
                row.update(complete=True, simulation_wall_seconds=stats['simulation_wall_s'],
                           batch=batch, resources=read(target / 'resources.json'),
                           stats_sha256=sha(observations / 'stats.yaml'),
                           replay_sha256=sha(observations / 'replay.csv'))
            save(receipt_path, row)
            print(json.dumps({k:row.get(k) for k in ('arm','complete','error','simulation_wall_seconds')}
                             | {'point':point['name']}), flush=True)
            estimate(root)
            if error == 'KeyboardInterrupt': return
    save(root / 'pilot-status.json', dict(time=time.time(), state='complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('fixtures', 'run', 'estimate'))
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    globals()[args.command](args.root)
