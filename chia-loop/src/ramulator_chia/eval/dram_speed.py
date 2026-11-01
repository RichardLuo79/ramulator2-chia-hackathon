"""Standalone DRAM throughput: opt-in measurements around existing batch replay.

No model code or admission policy is modified. Source/configuration manifests
and per-attempt receipts are immutable; short qualification never counts as a
production result. This module uses the pilot's small evidence helpers.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import statistics
import subprocess
import time

import yaml

try:
    from .dram_speed_pilot import read, save, sha, STOP
except ImportError:
    from dram_speed_pilot import read, save, sha, STOP

ARMS = ('oracle', 'astra_single_core', 'deepseek_single_core',
        'astra_multicore', 'deepseek_multicore', 'fixedlat', 'md1', 'wmg1', 'mess')
CRITICAL = ARMS[:6]
RATIOS = (100, 75, 50)
INTERVALS = (1, 4, 16, 64, 256)
WARMUP, MEASURED, REPEATS = 100000, 5000000, 5


def settings(root):
    """Optional new-study manifest; absent manifests retain the original study."""
    path = root/'study.json'
    value = read(path) if path.exists() else {}
    arms = tuple(value.get('models', ARMS))
    allowed = set(ARMS) | {'opus_single_core', 'gemini_single_core'}
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= allowed or 'oracle' not in arms:
        raise ValueError('invalid standalone speed model population')
    critical = tuple(value.get('critical_models', [a for a in CRITICAL if a in arms]))
    if len(set(critical)) != len(critical) or not set(critical) <= set(arms):
        raise ValueError('invalid standalone speed priority population')
    return {**value, 'models': arms, 'critical_models': critical}


def cutoff(root):
    value = settings(root)
    if 'stop_utc' not in value: return STOP
    return (datetime.fromisoformat(value['stop_utc'].replace('Z', '+00:00')).timestamp()
            if value['stop_utc'] else None)


def points():
    rows = [dict(pattern=p, read_percent=r, interval=i)
            for p in ('streaming','random') for r in RATIOS for i in INTERVALS]
    random.Random(12345).shuffle(rows)
    return rows


def point_name(p):
    return f'{p["pattern"]}-r{p["read_percent"]}-i{p["interval"]}'


def schedule(arms=ARMS, critical=CRITICAL):
    traffic = points()
    corners = [p for p in traffic if p['read_percent'] in (100,50) and p['interval'] in (1,256)]
    other = [p for p in traffic if p not in corners]
    jobs = []
    for repeat in range(1, REPEATS+1):
        # User priority: complete the key six-model population before the other
        # baselines. Complete the whole first pass before collecting repeats.
        for group in (critical, tuple(a for a in arms if a not in critical)):
            if not group: continue
            for j, p in enumerate((corners+other) if repeat == 1 else traffic):
                order = list(group)
                offset = (j+repeat-1) % len(order)
                order = order[offset:] + order[:offset]
                for arm in order:
                    jobs.append(dict(arm=arm, point=p, repeat=repeat))
    return jobs


def invoke(root, target, config, arm, *, legacy=False):
    target.mkdir(parents=True)
    observations = target/'observations'
    observations.mkdir()
    save(target/'config.json', config)
    manifest = read(root/'pilot-manifest.json')
    library = manifest['models'][arm].get('library')
    runtime = root/('runtime' if legacy else 'production/runtime')
    cmd = ['/usr/bin/time','-f',
           '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"peak_rss_kib":%M}',
           '-o',str(target/'resources.json'),'taskset','-c','2',str(runtime/'isolated_sim'),
           str(target/'config.json'),str(root/library) if library else '-',
           str(observations),str(observations/'stats.yaml')]
    began = time.time()
    proc = None
    def interrupt(signum, frame):
        raise InterruptedError('preservation/cutoff requested')
    old = signal.signal(signal.SIGTERM, interrupt)
    error = None
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:')))*1024
    # Host reserve without changing any controller queue/admission semantics.
    budget = available - 16*1024**3
    if budget < 1024**3:
        raise RuntimeError('insufficient memory before simulator launch')
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (budget,budget))
    with (target/'output.log').open('w') as log:
        try:
            proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits,
                env={**os.environ,'LD_LIBRARY_PATH':str(runtime),'OMP_NUM_THREADS':'1'})
            code = proc.wait(timeout=1800)
            if code: error = f'exit status {code}'
        except (subprocess.TimeoutExpired, InterruptedError, KeyboardInterrupt) as exc:
            error = 'diagnostic exceeded its runtime limit' if isinstance(exc, subprocess.TimeoutExpired) else str(exc)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        finally:
            signal.signal(signal.SIGTERM,old)
    result = dict(started=began,ended=time.time(),error=error,complete=False,command=cmd,
                  config_sha256=sha(target/'config.json'),deadline_seconds=1800,memory_limit_bytes=budget)
    if error is None:
        try:
            result['stats'] = yaml.safe_load((observations/'stats.yaml').read_text())
            result['resources'] = read(target/'resources.json')
            result['complete'] = True
        except Exception as exc:
            result['error'] = 'invalid completed output: '+repr(exc)
    result['files'] = {str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
    save(target/'process.json',result)
    return result


def simulation_identity(stats, *, legacy_precision=False, ignore_retries=False):
    """Exact native integer counters; floats compared at legacy print precision."""
    def clean(value):
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items() if not k.endswith(('_wall_s','_cpu_s'))
                    and not (ignore_retries and k == 'send_rejects')}
        if isinstance(value,float):
            if not math.isfinite(value): raise ValueError('non-finite simulation statistic')
            return float(format(value,'.6g')) if legacy_precision else value
        if isinstance(value,list): return [clean(v) for v in value]
        return value
    return clean(stats)


def qualify(root):
    destination=root/'production/qualification'
    destination.mkdir(parents=True,exist_ok=True)
    rng=random.Random(12345)
    fixture=destination/'traffic.csv'
    if not fixture.exists():
        rows=['arrive,addr,type,source']
        t=0
        for i in range(4096):
            if i%251==0: t+=1000
            else: t+=int(i%8==0)
            # Repeated writes/read forwarding plus row conflicts, independent
            # random banks/rows, sparse traffic and simultaneous arrivals.
            address=(i%16)*64 if i%7<3 else rng.randrange(8*1024**3//64)*64
            kind=(i%7 != 2) if i%7<3 else rng.randrange(2)
            rows.append(f'{t},{address},{int(kind)},0')
        fixture.write_text('\n'.join(rows)+'\n')
    evidence=[]
    for arm in settings(root)['models']:
        records={}
        for mode in ('legacy','tick','skip','compact'):
            target=destination/(arm+'-'+mode)
            if (target/'process.json').exists():
                result=read(target/'process.json')
            else:
                config=read(root/'configs'/f'{arm}.json')
                config['batch_trace']=str(fixture)
                if mode!='legacy':
                    config['batch_options']=dict(warmup_requests=257,tick_only=mode=='tick',
                        skip_idle_retries=mode!='tick',record_requests=mode!='compact')
                result=invoke(root,target,config,arm,legacy=mode=='legacy')
            if not result['complete']: raise RuntimeError(f'qualification {arm}/{mode}: {result["error"]}')
            records[mode]=result
        # Actual event records, not just totals, are identical across the
        # original replay, tick reference and optimized replay.
        hashes=[records[m]['files']['observations/replay.csv'] for m in ('legacy','tick','skip')]
        if len(set(hashes))!=1: raise RuntimeError('admission/completion mismatch: '+arm)
        native=[simulation_identity(records[m]['stats'],ignore_retries=True) for m in ('tick','skip','compact')]
        if not all(v==native[0] for v in native): raise RuntimeError('recording/skip changes simulation: '+arm)
        old=simulation_identity(records['legacy']['stats'],legacy_precision=True)
        compatible=simulation_identity(records['tick']['stats'],legacy_precision=True)
        if any(old[k] != compatible[k] for k in ('frontend','memory_system')):
            raise RuntimeError('legacy runtime simulation counters changed: '+arm)
        batch=records['compact']['stats']['batch']
        if batch['measured_reads_completed']+batch['measured_writes_completed']!=4096-257:
            raise RuntimeError('warmup population incorrect')
        evidence.append(dict(arm=arm,passed=True,event_sha256=hashes[0],
                             admission_digest=batch['admission_digest'],completion_digest=batch['completion_digest']))
    # The extracted helper must also leave the real Lat-Tp frontend unchanged,
    # including legacy address encoding and random-probe draw ordering.
    if (root/'lat-tp-qualification-config.json').exists():
        cfg=read(root/'lat-tp-qualification-config.json')
    else:
        original=Path(os.environ.get('RAMULATOR_CHIA_LAT_TP_REFERENCE', '.work/required-input'))
        point=read(original/'study/points/oracle/rr100_nop10000.json')
        cfg=read(original/'study'/point['receipt']['directory']/'config.json')
    ctrl=cfg['memory_system']['controllers'][0]
    ctrl.pop('trace_path',None); ctrl.pop('controller_plugins',None)
    for encoding in ('bank-major-line','ro-ba-ra-co-ch-byte'):
        cfg['frontend'].update(address_encoding=encoding,nop_counter=1,warmup_cycles=100,
                              num_probe_requests=300,latency_sample_count=300,read_ratio=50)
        results=[]
        for old in (True,False):
            target=destination/('lat-tp-'+encoding+'-'+str(old))
            result=read(target/'process.json') if (target/'process.json').exists() else invoke(root,target,cfg,'oracle',legacy=old)
            if not result['complete']: raise RuntimeError('Lat-Tp legacy qualification failed')
            results.append(simulation_identity(result['stats'],legacy_precision=True))
        if results[0]!=results[1]: raise RuntimeError('Lat-Tp behavior changed after helper extraction')
    save(root/'production/qualification.json',dict(passed=True,time=time.time(),
         fixture_sha256=sha(fixture),models=evidence,legacy_lat_tp_parity=True,
         build_sha256=sha(root/'production/build.json')))


def prepare(root):
    if not read(root/'production/qualification.json')['passed']: raise RuntimeError('qualification required')
    study=settings(root)
    inputs=root/'production/inputs'
    if 'frozen_inputs' in study:
        files=study['frozen_inputs']
        expected={(p,r) for p in ('streaming','random') for r in RATIOS}
        if {(r['pattern'],r['read_percent']) for r in files}!=expected or len(files)!=6:
            raise ValueError('frozen input matrix differs from protocol')
        for row in files:
            if row['requests']!=WARMUP+MEASURED or sha(inputs/row['path'])!=row['sha256']:
                raise ValueError('frozen traffic input mismatch')
    else:
        result=subprocess.run([str(root/'production/runtime/ramulator-speed-inputs'),str(inputs),
                               str(WARMUP+MEASURED)],check=True,capture_output=True,text=True)
        files=[json.loads(line) for line in result.stdout.splitlines()]
        for row in files:
            row['sha256']=sha(inputs/row['path'])
    configurations=len(study['models'])*len(points())
    protocol=dict(schema_version=1,models=study['models'],critical_models=study['critical_models'],read_percentages=RATIOS,
         intervals=INTERVALS,warmup_requests=WARMUP,measured_requests=MEASURED,repetitions=REPEATS,
         configurations=configurations,timed_runs=configurations*REPEATS,input_format='RMSPD001',seed=12345,type_seed=12345^0x54595045,
         rng='std::mt19937_64; separate address and type streams',
         geometry=dict(bytes=8*1024**3,banks=32,bank_groups=8,banks_per_group=4,rows=65536,
                       cache_lines_per_row=64,row_stagger=True,transaction_bytes=64),
         mapping='RoBaRaCoCh physical bytes',refresh=False,
         time_boundary='before first measured request processing, through all required callbacks; no warmup drain/reset',
         writes='native posted acknowledgement/terminal command issue, not physical write-burst completion',
         timing='one process, CPU 2, measured wall includes drain; input loading/initialization excluded',
         skip='guaranteed-idle controller hooks; oracle remains cycle-level',stop_utc=study.get('stop_utc'),
         timeout_seconds=1800,unit='requests per host second, not simulated DRAM bandwidth',
         inputs=files,source_sha256=sha(root/'production/source/source_inventory.json'),
         build_sha256=sha(root/'production/build.json'),
         qualification_sha256=sha(root/'production/qualification.json'),
         original_pilot_sha256=sha(root/'pilot-manifest.json'))
    save(root/'production/protocol.json',protocol)


def verify_cloud(root):
    results=[]
    for arm in settings(root)['models']:
        prior=root/'production/qualification'/(arm+'-compact')
        cfg=read(prior/'config.json')
        target=root/'production/cloud-qualification'/arm
        result=invoke(root,target,cfg,arm)
        expected=read(prior/'process.json')
        if not result['complete'] or simulation_identity(result['stats'])!=simulation_identity(expected['stats']):
            raise RuntimeError('cloud simulated-counter parity failed: '+arm)
        results.append(dict(arm=arm,passed=True,process_sha256=sha(target/'process.json')))
    save(root/'production/cloud-qualification.json',dict(time=time.time(),passed=True,models=results))


def backup_boundary(root, terminal=False):
    token=str(time.time_ns())
    save(root/'production/backup-ready.json',dict(token=token,time=time.time(),terminal=terminal))
    required=settings(root).get('require_backup_ack',False)
    deadline=time.monotonic()+(180 if required else 60)
    released=False
    while time.monotonic()<deadline:
        path=root/'production/backup-release.json'
        if path.exists() and read(path).get('token')==token:
            released=True
            break
        time.sleep(1)
    save(root/'production/backup-ready.json',dict(token=token,time=time.time(),terminal=terminal,closed=True))
    if required and not released:
        raise RuntimeError('verified backup acknowledgement missing; timed work remains stopped')


def run(root):
    study=settings(root)
    stop=cutoff(root)
    manifest=read(root/'production/transfer-manifest.json')
    for relative,digest in manifest['files'].items():
        if sha(root/relative)!=digest: raise RuntimeError('production staged hash mismatch: '+relative)
    if os.cpu_count()!=8 or 2 not in os.sched_getaffinity(0): raise RuntimeError('unexpected CPU topology')
    if not read(root/'production/qualification.json')['passed']: raise RuntimeError('qualification missing')
    if not read(root/'production/cloud-qualification.json')['passed']: raise RuntimeError('cloud qualification missing')
    key=sha(root/'production/protocol.json')
    last_backup=time.monotonic()
    for job in schedule(study['models'],study['critical_models']):
        name=f'{point_name(job["point"])}--{job["arm"]}--r{job["repeat"]}'
        target=root/'production/runs'/name
        if (target/'receipt.json').exists(): continue  # Never reroll a failed attempt.
        if stop is not None and time.time()>=stop:
            save(root/'production/status.json',dict(time=time.time(),state='cutoff'))
            break
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        if mem<16*1024**3:
            save(root/'production/status.json',dict(time=time.time(),state='memory_hold',available=mem))
            break
        p=job['point']; arm=job['arm']
        cfg=read(root/'configs'/f'{arm}.json')
        cfg['batch_speed_input']=str(root/'production/inputs'/f'{p["pattern"]}-r{p["read_percent"]}.bin')
        cfg['batch_interval']=p['interval']
        cfg['batch_options']=dict(warmup_requests=WARMUP,record_requests=False,skip_idle_retries=True)
        save(root/'production/status.json',dict(time=time.time(),state='running',job=job))
        result=invoke(root,target,cfg,arm)
        receipt=dict(**job,protocol_sha256=key,process_sha256=sha(target/'process.json'),
                     complete=result['complete'],error=result['error'],started=result['started'],ended=result['ended'])
        if result['complete']:
            b=result['stats']['batch']
            population=b['measured_reads']+b['measured_writes']
            if (population!=MEASURED or b['reads_completed']+b['writes_completed']!=WARMUP+MEASURED or
                b['measured_reads_completed']+b['measured_writes_completed']!=MEASURED or b['measured_wall_s']<=0):
                receipt.update(complete=False,error='incomplete or invalid measured callback population')
            else:
                receipt.update(batch=b,resources=result['resources'],
                    input_loading_wall_s=result['stats']['input_loading_wall_s'],
                    requests_per_second=MEASURED/b['measured_wall_s'],
                    simulated_identity=simulation_identity(result['stats']))
                earlier=list((root/'production/runs').glob(f'{point_name(p)}--{arm}--r*/receipt.json'))
                for old in earlier:
                    previous=read(old)
                    if previous.get('complete') and previous['simulated_identity']!=receipt['simulated_identity']:
                        receipt.update(complete=False,error='simulated results differ across repetitions')
        save(target/'receipt.json',receipt)
        export(root)
        print(json.dumps(dict(job=name,complete=receipt['complete'],error=receipt['error'],
                              req_s=receipt.get('requests_per_second'))),flush=True)
        if time.monotonic()-last_backup>=540:
            backup_boundary(root); last_backup=time.monotonic()
        if result['error'] and 'cutoff' in result['error']: break
    else:
        save(root/'production/status.json',dict(time=time.time(),state='terminal'))
    export(root)
    backup_boundary(root,terminal=True)


def export(root):
    study=settings(root)
    expected=len(study['models'])*len(points())*REPEATS
    rows=[read(p) for p in (root/'production/runs').glob('*/receipt.json')]
    good=[r for r in rows if r['complete']]
    summary=[]
    for p in points():
        oracle=[r['requests_per_second'] for r in good if r['arm']=='oracle' and r['point']==p]
        for arm in study['models']:
            chosen=[r for r in good if r['arm']==arm and r['point']==p]
            values=[r['requests_per_second'] for r in chosen]
            summary.append(dict(**p,arm=arm,n=len(values),median=statistics.median(values) if values else None,
              mean=statistics.mean(values) if values else None,
              standard_deviation=statistics.stdev(values) if len(values)>1 else None,
              oracle_relative_speedup=statistics.median(values)/statistics.median(oracle) if values and oracle else None,
              repetitions=[dict(repeat=r['repeat'],requests_per_second=r['requests_per_second'],
                  wall_seconds=r['batch']['measured_wall_s'],cpu_seconds=r['batch']['measured_cpu_s'],
                  drain_wall_seconds=r['batch']['drain_wall_s'],peak_rss_kib=r['resources']['peak_rss_kib'],
                  warmup_wall_seconds=r['batch']['warmup_wall_s'],warmup_cpu_seconds=r['batch']['warmup_cpu_s'],
                  drain_cpu_seconds=r['batch']['drain_cpu_s'],input_loading_wall_seconds=r['input_loading_wall_s'],
                  total_process_wall_seconds=r['resources']['wall_seconds'],
                  total_process_cpu_seconds=r['resources']['user_seconds']+r['resources']['system_seconds'],
                  simulated_cycles=r['batch']['cycles'],
                  measured_simulated_cycles=r['batch']['cycles']-r['batch']['measurement_start_cycle'],
                  drain_simulated_cycles=r['batch']['cycles']-r['batch']['final_admission_cycle'],
                  admission_wait_sum=r['batch']['measured_admission_wait_sum'],
                  admission_wait_max=r['batch']['measured_admission_wait_max'],
                  warmup_reads_outstanding=r['batch']['warmup_reads_outstanding'],
                  warmup_writes_outstanding=r['batch']['warmup_writes_outstanding'],
                  measured_reads=r['batch']['measured_reads'],measured_writes=r['batch']['measured_writes'],
                  measured_reads_completed=r['batch']['measured_reads_completed'],
                  measured_writes_completed=r['batch']['measured_writes_completed'])
                  for r in sorted(chosen,key=lambda r:r['repeat'])]))
    save(root/'production/results.json',dict(time=time.time(),expected=expected,successful=len(good),
         failed=len(rows)-len(good),pending=expected-len(rows),summary=summary,
         failures=[r for r in rows if not r['complete']],protocol=read(root/'production/protocol.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('qualify','prepare','verify_cloud','run','export'))
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    globals()[args.command](args.root)
