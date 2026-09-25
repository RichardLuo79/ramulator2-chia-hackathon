"""Frozen endpoint evaluations, using the existing native executor and ROI driver.

This is an operator study, not an agent loop. A manifest binds every job to its
sources, inputs, placement and protocol. Cloud lifecycle is deliberately outside
this module. Old campaign records are read, never modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from ramulator_chia.eval import chia_baselines as baseline
from ramulator_chia.eval.champsim_mixes import MultiProgramCase
from ramulator_chia.eval.hardware_sweep import HardwarePoint, GRID
from ramulator_chia.framework.archive import describe_payload
from ramulator_chia.framework.candidate import BuildLimits
from ramulator_chia.framework.dram import MODEL_FILES, build_model, measurement
from ramulator_chia.framework.identity import digest_json, file_sha256
from ramulator_chia.framework.snapshots import snapshot, publish_bytes

REPO = __import__("ramulator_chia.layout", fromlist=["RAMULATOR"]).RAMULATOR
BASE = Path(__import__("os").environ.get("RAMULATOR_CHIA_BASELINES", ".work/required-input"))
RUNTIME = Path(__import__("os").environ.get("RAMULATOR_CHIA_RUNTIME", ".work/required-input"))
CAMPAIGNS = Path(__import__("os").environ.get("RAMULATOR_CHIA_CAMPAIGNS", ".work/required-input"))
PRIOR_GEM5 = Path(__import__("os").environ.get("RAMULATOR_CHIA_GEM5_EVIDENCE", ".work/required-input"))
GEM5_PYTHON = Path(__import__("os").environ.get("RAMULATOR_CHIA_GEM5_PYTHON", ".work/required-input"))
# Optional operator limit; there is no hard-coded historical cloud expiry.
STOP_AT = float(os.environ.get('RAMULATOR_CHIA_STOP_AT', 'inf'))
read, save, status = baseline.read, baseline.publish, baseline.status


@dataclass(frozen=True, kw_only=True)
class HardwareMix(MultiProgramCase):
    hardware: HardwarePoint

    def identity(self):
        return {**super().identity(), 'hardware': self.hardware.identity()}

    def configuration(self, model, inputs, observations, parameters, curve):
        config = super().configuration(model, inputs, observations, parameters, curve)
        config['memory_system']['controllers'][0].update(self.hardware.settings())
        return config

    def check_statistics(self, stats, observations, *, candidate):
        super().check_statistics(stats, observations, candidate=candidate)
        controller = read(observations.parent/'config.json')['memory_system']['controllers'][0]
        if any(controller.get(k) != v for k,v in self.hardware.settings().items()):
            raise ValueError('measured hardware differs from its frozen identity')


def remap(case, host, root, capacity):
    """Reuse the complete virtual-page set; allocate frames with native ChampSim."""
    original = gzip.decompress(case.placement.source.read_bytes())
    header = original.splitlines()[0].split()
    if int(header[5]) == capacity:
        return case.placement
    destination = root/'maps'/str(capacity)/(case.workload+'.txt.gz')
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        pages = [line.split()[1:3] for line in original.splitlines()[1:] if line.startswith(b'D ')]
        if len(pages) != int(header[7]):
            raise ValueError('source placement data-page population mismatch')
        inventory = f'CHAMPSIM_PAGES_V1 {case.cores} 4096 {capacity}\n'.encode()
        inventory += b''.join(b' '.join(row)+b'\n' for row in pages)
        source = destination.with_suffix('.pages'); output = destination.with_suffix('')
        publish_bytes(source, inventory)
        binary = host.root/host.record()['files']['binary']['member']['name']
        env = {k:v for k,v in os.environ.items() if k not in ('CHAMPSIM_PLACEMENT_FILE','RAMULATOR_CONFIG')}
        env['LD_LIBRARY_PATH'] = str(RUNTIME/'runtime')
        with destination.with_suffix('.log').open('w') as log:
            subprocess.run([str(binary),'--prepare-placement',str(source),str(output)],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        raw = output.read_bytes()
        new = raw.splitlines()[0].split()
        if int(new[5]) != capacity or new[7:] != header[7:]:
            raise ValueError('capacity remapping changed logical pages or page-table geometry')
        publish_bytes(destination,gzip.compress(raw,compresslevel=3,mtime=0))
        # These two files are temporary inputs/outputs created above, not evidence.
        source.unlink(); output.unlink()
    return describe_payload(destination,'placement.txt',codec='gzip')


def load_hardware(root):
    from ramulator_chia.framework.inputs import _staged_champsim
    from ramulator_chia.framework.evaluation import SimulationLimits
    settings=read(root/'baseline-settings.json')
    cohort=settings['evaluation']['champsim'].copy()
    for key in ('builds','traces'):
        cohort[key]={n:SimpleNamespace(**v) for n,v in cohort[key].items()}
    cohort['cases']={n:SimpleNamespace(programs=v['programs'],placement=SimpleNamespace(**v['placement']))
                     for n,v in cohort['cases'].items()}
    # This is a held-out study, not a search configuration requiring training.
    groups,hosts=_staged_champsim(SimpleNamespace(**cohort),root)
    limits=SimulationLimits(**settings['limits'])
    chosen={c.workload:c for c in groups['test']}
    result = {}
    for point in GRID:
        capacity = (8 if point.organization=='DDR5_16Gb_x8' else 4)*1024**3
        for name,case in chosen.items():
            payload = remap(case,hosts[f'champsim:{case.cores}'],root,capacity)
            values = {field.name:getattr(case,field.name) for field in fields(case)}
            values.update(placement=payload,hardware=point)
            result[point.name,name] = HardwareMix(**values)
    return result,hosts,limits


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    settings = read(BASE/'baseline-settings.json')
    cohort = settings['evaluation']['champsim']
    cohort['training'] = []; cohort['validation'] = []
    cohort['cases'] = {n:cohort['cases'][n] for n in cohort['test']}
    programs = {p for case in cohort['cases'].values() for p in case['programs']}
    cohort['traces'] = {n:v for n,v in cohort['traces'].items() if n in programs}
    settings['mess_curve'] = str(Path(__file__).resolve().parent/'calibration/mess_DDR5.txt')
    save(root/'baseline-settings.json',settings)
    if not (root/'frontend-builds').exists():
        shutil.copytree(BASE/'frontend-builds',root/'frontend-builds',copy_function=os.link)
    cases,hosts,limits = load_hardware(root)
    if len(cases)!=360: raise ValueError('expected 40 cases and nine hardware configurations')
    endpoints = {}
    build_limits = BuildLimits(cpus=1,timeout_seconds=600,source_bytes=1<<20,
                               memory_bytes=8<<30,file_bytes=1<<30)
    for short, campaign in (('astra','astra_xhigh'),('deepseek','deepseek_max')):
        report = read(REPO/'reports'/f'chia_staged_{short}_0923.json')
        for stage,row in report['stage_endpoints'].items():
            selected = row['candidate']
            copied = snapshot(CAMPAIGNS/campaign/'candidates'/selected['candidate_id'],
                              root/'candidates',MODEL_FILES,maximum_bytes=1<<20)
            if copied != selected: raise ValueError('selected model source changed')
            build = build_model._chia_original(RUNTIME,root,selected,build_limits)
            if not build['passed']: raise RuntimeError('frozen endpoint build failed')
            name = short+'_'+stage
            endpoints[name] = dict(candidate=selected,build=build,campaign=campaign,stage=stage,
                                   report_sha256=file_sha256(REPO/'reports'/f'chia_staged_{short}_0923.json'))
    save(root/'endpoints.json',endpoints)
    jobs = []
    for (point,name),case in cases.items():
        for arm in ('oracle',*endpoints):
            identity = baseline.identity_for(RUNTIME,case,'oracle' if arm=='oracle' else 'candidate',
                                            hosts[f'champsim:{case.cores}'],limits,None)
            if arm!='oracle': identity['candidate']=endpoints[arm]['candidate']
            jobs.append(dict(id=digest_json(identity),study='hardware',case=name,point=point,
                             arm=arm,cores=case.cores,identity=identity))
    g5 = read(PRIOR_GEM5/'manifest.json')
    driver = root/'gem5-driver'; driver.mkdir(exist_ok=True)
    for name in ('roi_study.py','roi_report.py','fs_board.py','count_fs.py'):
        path=driver/name
        if path.exists() and file_sha256(path)!=file_sha256(Path(__file__).parent/'gem5'/name):
            raise ValueError('frozen driver changed')
        if not path.exists():shutil.copy2(Path(__file__).parent/'gem5'/name,path)
    import ramulator
    from ramulator_chia.eval.simpleo3 import _build_controller
    for arm in (*baseline.MODELS,*endpoints):
        model = arm if arm in baseline.MODELS else 'candidate'
        parameters = {} if model!='candidate' else {'model_parameters':[
            f'{name}={value}' for name,value in sorted(endpoints[arm]['candidate']['parameters'].items())]}
        dram = ramulator.dram.DDR5(org_preset='DDR5_16Gb_x8',timing_preset='DDR5_4800AN')
        ctrl = _build_controller(ramulator,model,dram,'DDR5','',parameters,
                                 mess_curve=Path(settings['mess_curve'])).to_config()
        # Request observations are not scored in gem5. Preserve the earlier
        # qualification's observation-free polling configuration.
        ctrl.pop('trace_path',None);ctrl.pop('controller_plugins',None)
        if model=='candidate':
            ctrl['model_library']=str(root/endpoints[arm]['build']['directory']/'candidate.so')
        config={'memory_system':{'impl':'GenericDRAM','clock_ratio':1,'controllers':[ctrl],
                                'channel_mapper':{'impl':'CacheLineInterleave'}}}
        save(root/'gem5-configs'/(arm+'.json'),config)
        for case in g5['jobs']:
            identity = dict(study='gem5',case=case,arm=arm,
                source=endpoints[arm]['candidate'] if model=='candidate' else None,
                runtime_sha256=file_sha256(RUNTIME/'runtime_manifest.json'),
                gem5_sha256=g5['gem5_sha256'],configuration=config,
                driver={p.name:file_sha256(p) for p in driver.glob('*.py')},
                warmup=2_000_000,instructions=2_000_000_000,timeout_seconds=28800,
                resources=g5['resources'],ramulator_python=str(GEM5_PYTHON))
            jobs.append(dict(id=digest_json(identity),study='gem5',case=case['name'],arm=arm,
                             cores=1,identity=identity))
    save(root/'gem5-inputs.json',g5)
    manifest=dict(schema=1,jobs=jobs,endpoints=endpoints,
        hardware_points={p.name:p.identity() for p in GRID},
        runtime=str(RUNTIME),runtime_sha256=file_sha256(RUNTIME/'runtime_manifest.json'),
        no_model_tuning=True,no_agent_feedback=True,test_cohort='previously examined historical cohort',
        instruction_protocol='2M+20M/core continuous contention; gem5 2M+up to2B or ROI end',
        stop_utc=None)
    save(root/'manifest.json',manifest)
    print(json.dumps(dict(prepared=len(jobs),hardware=1800,gem5=135)),flush=True)


def execute_gem5(root,job):
    """The existing eight-hour screening subprocess protocol, for frozen models."""
    identity=job['identity'];case=identity['case'];output=root/'gem5-runs'/job['id']
    pointer=output/'receipt.json'
    if pointer.exists():return read(pointer)
    output.mkdir(parents=True,exist_ok=True)
    g5=read(root/'gem5-inputs.json')
    if file_sha256(Path(g5['gem5']))!=identity['gem5_sha256']:raise ValueError('gem5 changed')
    config=root/'gem5-configs'/(job['arm']+'.json')
    if read(config)!=identity['configuration']:raise ValueError('memory configuration changed')
    for name,sha in identity['driver'].items():
        if file_sha256(root/'gem5-driver'/name)!=sha:raise ValueError('ROI driver changed')
    args=[g5['gem5'],'--outdir='+str(output/'simulation'),str(root/'gem5-driver/roi_study.py'),
          '--memory-config='+str(config),'--ramulator-python='+str(GEM5_PYTHON)]
    args += (['--checkpoint='+case['checkpoint'],'--resources='+g5['resources']] if case['mode']=='fs'
             else ['--binary='+case['binary']])
    started=time.time();save(output/'command.json',dict(argv=args,identity=identity,start=started))
    env={**os.environ,'LD_LIBRARY_PATH':str(RUNTIME/'runtime'),'OMP_NUM_THREADS':'1'}
    failure=None
    with (output/'gem5.log').open('w') as log:
        child=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status(output/'process.json',dict(pid=child.pid,pgid=child.pid,start=started))
        try:
            code=child.wait(timeout=28800)
            if code:failure='process_exit_'+str(code)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            failure='eight_hour_watchdog: incomplete and unscored'
    result_path=output/'simulation/result.json'
    if not failure and not result_path.exists():failure='missing completed ROI result'
    result=read(result_path) if not failure else None
    if result and result['loaded_libraries'].get(str(RUNTIME/'runtime/libramulator.so'))!=file_sha256(RUNTIME/'runtime/libramulator.so'):
        failure='loaded runtime identity mismatch';result=None
    record=dict(job=job['id'],case=job['case'],arm=job['arm'],complete=not failure,error=failure,
                wall_seconds=time.time()-started,result=result,identity=identity)
    if record['complete']:
        sys.path.insert(0,str(root/'gem5-driver'))
        from roi_report import evidence
        record['metrics']=evidence(output,record)
    save(pointer,record)
    return record


def verify_gem5_inputs(root):
    """Audit staged bytes and live model mappings without touching simulations."""
    data=read(root/'gem5-inputs.json');checked={}
    def check(path,expected):
        path=Path(path);actual=file_sha256(path)
        if actual!=expected:raise ValueError('staged input/build mismatch: '+str(path))
        checked[str(path)]=actual
    check(data['gem5'],data['gem5_sha256'])
    for case in data['jobs']:
        if case['mode']=='se':check(case['binary'],data['guest_builds'][case['name']]['binary_sha256'])
        else:
            for name,sha in case['checkpoint_hashes'].items():check(Path(case['checkpoint'])/name,sha)
    for item in read(root/'resource-identities.json')['files'].values():check(item['path'],item['sha256'])
    endpoints=read(root/'endpoints.json');libraries={}
    for arm,endpoint in endpoints.items():
        directory=root/endpoint['build']['directory']
        check(directory/'build.json',endpoint['build']['receipt_sha256'])
        record=read(directory/'build.json');path=directory/'candidate.so'
        check(path,record['binary_sha256']);libraries[arm]=str(path.resolve())
    # The trusted loader rejects missing/incompatible libraries and never falls
    # back to a seed. For still-running jobs, independently inspect actual maps.
    jobs={j['id']:j for j in read(root/'manifest.json')['jobs'] if j['study']=='gem5'}
    live=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            out=next((a.decode() for a in argv if a.startswith(('--outdir='+str(root/'gem5-runs')+'/').encode())),None)
            if not out:continue
            key=Path(out.split('=',1)[1]).relative_to(root/'gem5-runs').parts[0].removeprefix('qualification-')
            job=jobs[key]
            if job['arm'] not in libraries:continue
            paths={line.split()[-1] for line in (proc/'maps').read_text().splitlines() if line.split()}
            expected=libraries[job['arm']]
            if expected not in paths:raise ValueError('running model is not mapped: '+job['id'])
            live.append(dict(job=job['id'],pid=int(proc.name),model_library=expected,sha256=checked[expected]))
        except (FileNotFoundError,ProcessLookupError):continue
    result=dict(time=time.time(),host=os.uname().nodename,passed=True,checked_files=checked,
        live_model_mappings=live,completed_model_loading='trusted loader: exact explicit path, API check, no fallback')
    status(root/'gem5-asset-verification.json',result);return result


_CONTEXT=None


def initialize(root):
    global _CONTEXT
    root=Path(root);cases,hosts,limits=load_hardware(root)
    _CONTEXT=(root,cases,hosts,limits,read(root/'endpoints.json'))


def execute_hardware(job):
    root,cases,hosts,limits,endpoints=_CONTEXT
    case=cases[job['point'],job['case']]
    build=None if job['arm']=='oracle' else endpoints[job['arm']]['build']
    store=root/'qualification' if job.get('qualification') else root
    result=measurement._chia_original(RUNTIME,store,case,'candidate' if build else 'oracle',limits,
                       build,None,hosts[f'champsim:{case.cores}'])
    if result['identity']!=job['identity']:raise ValueError('measurement differs from frozen manifest')
    return result


def qualification_jobs(root,kind):
    refs=read(root/'qualification-references.json')[kind]
    jobs=read(root/'manifest.json')['jobs']
    result=[]
    for name in refs:
        job=next(j for j in jobs if j['study']==kind and j['case']==name and j['arm']=='oracle'
                 and (kind=='gem5' or j['point']=='DDR5_16Gb_x8_q64'))
        # The scientific identity is unchanged. Only this repeat's execution
        # destination is separate, so qualification never overwrites production.
        result.append({**job,'id':'qualification-'+job['id'],'qualification':True})
    return result


def qualify(root,job,result):
    reference=read(root/'qualification-references.json')[job['study']][job['case']]
    if job['study']=='gem5':
        actual=result['result']
        if (actual['measured_instructions'],actual['simulated_ticks'],actual['stop_reason']) != (
                reference['instructions'],reference['ticks'],reference['stop_reason']):
            raise ValueError('gem5 cross-host simulated-result mismatch')
    else:
        actual=result['observation']
        for key in ('per_core_cycles','frontend_stats','controller_stats'):
            if actual[key]!=reference[key]:raise ValueError('ChampSim cross-host mismatch: '+key)
        signature=lambda r:{n:(v['logical_sha256'],v['logical_bytes']) for n,v in r['traces'].items()}
        if signature(actual)!=signature(reference):raise ValueError('decoded observations differ across hosts')


def execute(root,job):
    try:
        result=execute_gem5(root,job) if job['study']=='gem5' else execute_hardware(job)
        if job.get('qualification') and (job['study']!='gem5' or result['complete']):
            qualify(root,job,result)
        record=dict(id=job['id'],state='success',complete=True,result=result)
        if job['study']=='gem5' and not result['complete']:
            record.update(state='failed',complete=False,error=result['error'])
    except Exception as exc:
        record=dict(id=job['id'],state='failed',complete=False,error=type(exc).__name__+': '+str(exc))
    status(root/'job-status'/(job['id']+'.json'),{**record,'host':os.uname().nodename,'finished':time.time()})
    return record


def run(root,queue,workers,status_name='worker-status.json'):
    import multiprocessing
    manifest=read(root/'manifest.json');assignment=read(queue)
    if assignment['manifest_sha256']!=file_sha256(root/'manifest.json'):raise ValueError('queue/manifest mismatch')
    jobs={row['id']:row for row in manifest['jobs']}
    # Prepare once in the parent. Forked workers reuse verified immutable inputs;
    # do not decompress all input traces separately in 120 worker initializers.
    if assignment['kind']=='hardware':initialize(root)
    wanted=qualification_jobs(root,assignment['kind'])+[jobs[key] for key in assignment['ids']]
    pending=[job for job in wanted if not (root/'job-status'/(job['id']+'.json')).exists()]
    active={};done=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        while pending or active:
            while pending and len(active)<workers and time.time()<STOP_AT:
                available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                   if line.startswith('MemAvailable:')))*1024
                if available<64*1024**3 or shutil.disk_usage(root).free<64*1024**3:break
                job=pending.pop(0);future=pool.submit(execute,root,job);active[future]=job['id']
            completed=[f for f in active if f.done()]
            for future in completed:
                key=active.pop(future);done.append(future.result())
                print(json.dumps({'completed':key,'state':done[-1]['state']}),flush=True)
            status(root/status_name,dict(time=time.time(),pending=len(pending),active=list(active.values()),
                successes=sum(r['complete'] for r in done),failures=sum(not r['complete'] for r in done),
                terminal=not pending and not active))
            if pending and time.time()>=STOP_AT and not active:break
            if not completed:time.sleep(5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','verify-gem5'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--queue',type=Path)
    parser.add_argument('--workers',type=int,default=120)
    parser.add_argument('--status-name',default='worker-status.json',choices=('worker-status.json','retry-status.json'))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root)
    elif args.action=='verify-gem5':print(json.dumps(verify_gem5_inputs(args.root)))
    else:run(args.root,args.queue,args.workers,args.status_name)


if __name__=='__main__':main()
