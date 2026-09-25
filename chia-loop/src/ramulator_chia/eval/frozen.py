"""Portable entry adapters to the existing frozen-model diagnostic executors.

No campaign state, cloud lifecycle or provider access. Each command runs one
point, so an operator can schedule the documented matrix without hidden work.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from ramulator_chia.framework.candidate import BuildLimits, compile_snapshot
from ramulator_chia.framework.dram import MODEL_FILES
from ramulator_chia.framework.identity import file_sha256, canonical_json
from ramulator_chia.framework.snapshots import snapshot, publish_bytes

MODELS=('oracle','fixedlat','md1','wmg1','mess','astra_single_core','astra_multicore',
        'deepseek_single_core','deepseek_multicore','opus_single_core','gemini_single_core')


def speed_memory_budget(meminfo=Path('/proc/meminfo')):
    """Match the qualified speed runner's 16-GiB available-memory reserve."""
    available = int(next(line.split()[1] for line in meminfo.read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    budget = available - (16 << 30)
    if budget < (1 << 30):
        raise RuntimeError('insufficient available memory before speed simulation')
    return budget


def model_library(root,runtime,output,model):
    if model in MODELS[:5]:return None,None
    value=snapshot(root/'results/models'/model,output/'candidates',MODEL_FILES,maximum_bytes=1<<20)
    receipt=compile_snapshot(runtime,output/'candidates',value['candidate_id'],MODEL_FILES,
        output/'model-build',limits=BuildLimits(1,600,1<<20,8<<30,4<<30))
    if not receipt['passed']:raise RuntimeError('frozen model compilation failed')
    return output/'model-build/candidate.so',value


def diagnostic_cli(suite,arguments,root):
    if suite not in ('lat-tp','speed'):raise SystemExit('supported diagnostic suites: lat-tp, speed')
    p=argparse.ArgumentParser(description='One frozen '+suite+' point; no LLM access.')
    p.add_argument('--runtime',type=Path,default=root/'.work/runtime')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--read-percent',type=int,default=100)
    if suite=='lat-tp':
        p.add_argument('--nop',type=int,default=10000)
        p.add_argument('--streaming-only',action='store_true')
        p.add_argument('--record-requests',action='store_true')
    else:
        p.add_argument('--pattern',choices=('streaming','random'),default='streaming')
        p.add_argument('--interval',type=int,choices=(1,4,16,64,256),default=16)
        p.add_argument('--warmup',type=int,default=100000)
        p.add_argument('--requests',type=int,default=5000000)
        p.add_argument('--cpu',type=int,required=True,help='one allowed logical CPU; run on a quiet host')
    a=p.parse_args(arguments);a.output=a.output.resolve();a.runtime=a.runtime.resolve()
    if not 0<=a.read_percent<=100:p.error('read percentage must be in [0,100]')
    a.output.mkdir(parents=True,exist_ok=False)
    library,candidate=model_library(root,a.runtime,a.output,a.model)
    curve=root/'chia-loop/src/ramulator_chia/eval/calibration/mess_DDR5.txt'
    if suite=='lat-tp':
        from ramulator_chia.eval.lat_tp_campaigns import LatTpCase,UnobservedLatTpCase
        from ramulator_chia.eval.lat_tp_suite import resolve_spec
        from ramulator_chia.eval.lat_tp_models import CONFIG
        from ramulator_chia.framework.archive import describe_payload
        from ramulator_chia.framework.evaluation import measure_native,SimulationLimits,CandidateBuild
        cls=LatTpCase if a.record_requests else UnobservedLatTpCase
        case=cls(a.nop,a.read_percent,a.streaming_only,'ro-ba-ra-co-ch-byte')
        build=(CandidateBuild(a.output/'candidates',candidate['candidate_id'],MODEL_FILES,
                a.output/'model-build',file_sha256(a.output/'model-build/build.json')) if candidate else None)
        receipt=measure_native(a.runtime,case,'candidate' if candidate else a.model,a.output/'measurement',
            limits=SimulationLimits(1800,16<<30,8<<30,1<<20,3),candidate=build,
            mess_curve=describe_payload(curve,'inputs/mess.txt') if a.model=='mess' else None)
        observed=receipt;c=observed['controller_stats'];f=observed['frontend_stats']
        spec=resolve_spec(CONFIG)
        result=dict(model=a.model,receipt=receipt,throughput_GBps=(c['num_read_reqs_served']+
            c['num_write_reqs_served'])*spec.bytes_per_req/(c['cycles']*spec.time_unit_ns),
            latency_ns=None if a.streaming_only else f['total_probe_latency']/f['probe_requests_completed']*spec.time_unit_ns)
    else:
        if a.read_percent not in (100,75,50):p.error('speed protocol uses 100,75,50% reads')
        if a.cpu not in os.sched_getaffinity(0):p.error('CPU is outside current affinity mask')
        if a.warmup<0 or a.requests<=0:p.error('invalid measurement population')
        budget = speed_memory_budget()
        import ramulator
        from ramulator_chia.eval.simpleo3 import _build_controller
        from ramulator_chia.eval import dram_speed
        runtime=a.runtime/'runtime'
        generator=runtime/'ramulator-speed-inputs'
        if not generator.exists():
            raise ValueError('speed input generator not built; run setup --component speed-inputs first')
        # Generation, allocation and model construction are outside native timing.
        inputs=a.output/'inputs'
        subprocess.run([str(generator),str(inputs),str(a.warmup+a.requests)],check=True,
                       stdout=subprocess.DEVNULL)
        params={} if candidate is None else {'model_parameters':[f'{k}={v}' for k,v in sorted(candidate['parameters'].items())]}
        dram=ramulator.dram.DDR5(org_preset='DDR5_16Gb_x8',timing_preset='DDR5_4800AN')
        ctrl=_build_controller(ramulator,'candidate' if candidate else a.model,dram,'DDR5','',params,mess_curve=curve).to_config()
        ctrl.pop('trace_path',None);ctrl.pop('controller_plugins',None)
        stream=inputs/f'{a.pattern}-r{a.read_percent}.bin'
        config={'frontend':{'impl':'External','clock_ratio':1,'num_cores':1},
                'memory_system':{'impl':'GenericDRAM','clock_ratio':1,'controllers':[ctrl],
                'channel_mapper':{'impl':'CacheLineInterleave'}},'batch_speed_input':str(stream),
                'batch_interval':a.interval,'batch_options':{'warmup_requests':a.warmup,
                    'record_requests':False,'skip_idle_retries':True}}
        cfg=a.output/'config.json';publish_bytes(cfg,canonical_json(config).encode())
        out=a.output/'observations';out.mkdir()
        cmd=['/usr/bin/time','-f','{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"peak_rss_kib":%M}',
             '-o',str(a.output/'resources.json'),'taskset','-c',str(a.cpu),str(runtime/'isolated_sim'),
             str(cfg),str(library) if library else '-',str(out),str(out/'stats.yaml')]
        import signal,yaml,resource
        def limits():
            resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
        started=time.time();error=None
        with (a.output/'simulation.log').open('w') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits,
                env={**os.environ,'LD_LIBRARY_PATH':str(runtime),'OMP_NUM_THREADS':'1'})
            try:
                code=process.wait(timeout=1800)
                if code:error='simulator exit '+str(code)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                error='diagnostic interrupted or exceeded its runtime limit'
        result=dict(model=a.model,complete=False,error=error,deadline_seconds=1800,
                    started=started,ended=time.time(),input_sha256=file_sha256(stream),
                    config_sha256=file_sha256(cfg),runtime_sha256=file_sha256(a.runtime/'runtime_manifest.json'),
                    candidate=candidate,cpu=a.cpu,memory_budget_bytes=budget)
        if error is None:
            stats=yaml.safe_load((out/'stats.yaml').read_text());b=stats['batch']
            if b['measured_reads_completed']+b['measured_writes_completed']!=a.requests:
                result['error']='incomplete measured request population'
            else:
                result.update(complete=True,stats=stats,resources=json.loads((a.output/'resources.json').read_text()),
                    requests_per_second=a.requests/b['measured_wall_s'],
                    publication_protocol=(a.requests==5000000 and a.warmup==100000),
                    simulated_identity=dram_speed.simulation_identity(stats))
    publish_bytes(a.output/'result.json',canonical_json(result).encode())
    print(json.dumps(result,indent=2))
    if result.get('complete') is False:raise SystemExit(1)
