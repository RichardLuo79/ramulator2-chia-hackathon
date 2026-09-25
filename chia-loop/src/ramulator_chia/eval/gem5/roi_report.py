"""Summarize preserved single-core ROI qualification, without running models."""
import argparse
import configparser
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import yaml
from fs_board import check_workload_output
from roi_study import last_statistics


def statistics(path):
    values={}
    for line in last_statistics(path.read_text()).splitlines()[1:]:
        fields=line.split()
        if len(fields)>1:
            try:values[fields[0]]=float(fields[1])
            except ValueError:pass
    return values


def active_o3_cycles(config, stats):
    """Select the instantiated, active O3 core, not an inactive switch target.

    FS checkpoints retain a switchable processor with a zero-cycle Atomic
    object. Its presence is not execution under AtomicSimpleCPU.
    """
    objects = configparser.ConfigParser(interpolation=None, strict=True)
    objects.read_string(config)
    active = [name for name in objects.sections()
              if objects[name].get('type') == 'BaseO3CPU'
              and not objects[name].getboolean('switched_out', fallback=False)]
    if len(active) != 1:
        raise ValueError('not the requested single-core O3/Ramulator board')
    core = active[0]
    cycles = stats.get(core + '.numCycles', 0)
    if cycles <= 0:
        raise ValueError('active O3 core has no measured cycles')
    return core, cycles


def evidence(directory,receipt):
    """Failed processes never become measurements through report generation."""
    if not receipt['complete']:
        return {'accepted':False,'reason':receipt['error']}
    result=receipt['result'];simulation=directory/'simulation'
    config=(simulation/'config.ini').read_text()
    if config.count('type=BaseO3CPU\n')!=1 or 'type=Ramulator2\n' not in config:
        raise ValueError('not the requested single-core O3/Ramulator board')
    if result['metadata'].get('mode')!='SE' and ('type=CowDiskImage' not in config or 'read_only=true' not in config):
        raise ValueError('missing private FS COW overlay')
    stats=statistics(simulation/'roi.stats.txt')
    if int(stats['simInsts'])!=result['measured_instructions'] or int(stats['simTicks'])!=result['simulated_ticks']:
        raise ValueError('ROI statistics disagree with instruction-tracker boundaries')
    end=next(e for e in result['events'] if e['event']=='measurement_end')
    memory=yaml.safe_load((simulation/f"ramulator_stats.{end['tick']}.yaml").read_text())['memory_system']
    controller=memory['controller']
    core,cycles=active_o3_cycles(config,stats)
    user_instructions=sum(int(value) for name,value in stats.items()
                          if name==core+'.commitStats0.numUserInsts')
    if not 0 <= user_instructions <= result['measured_instructions']:
        raise ValueError('invalid user/total committed-instruction counts')
    verification='unavailable: instruction-capped interval'
    output_hash=None
    if result['stop_reason']=='roi_end':
        if result['metadata'].get('mode')=='SE':
            text=(directory/'gem5.log').read_text(errors='replace')
            match=re.search(r'==BEGIN DUMP_ARRAYS==.*?==END\s+DUMP_ARRAYS==',text,re.S)
            if not match:raise ValueError('missing PolyBench result-array output')
            output_hash=hashlib.sha256(match[0].encode()).hexdigest()
            verification='array dump retained; compare oracle/FixedLat at printed precision'
        else:
            suite=result['metadata']['suite']
            name='npb-qualification.txt' if suite=='npb' else 'workload-qualification.txt'
            text=(simulation/name).read_text(errors='replace')
            verification=check_workload_output(suite,text,exit_marker='QUALIFICATION_EXIT_CODE')
    reads=controller['num_read_reqs'];writes=controller['num_write_reqs']
    return {'accepted':True,'instructions':result['measured_instructions'],'stop_reason':result['stop_reason'],
        'simulated_ticks':result['simulated_ticks'],'cycles':int(cycles),'active_cpu':core,
        'user_instructions':user_instructions,
        'user_instruction_fraction':user_instructions/result['measured_instructions'],
        'ipc':result['measured_instructions']/cycles,
        'dram_reads_admitted':reads,'dram_writes_admitted':writes,
        'dram_reads_served':controller['num_read_reqs_served'],
        'dram_reads_per_kinst':1000*reads/result['measured_instructions'],
        'low_traffic':reads<10000,'total_wall_hours':receipt['wall_seconds']/3600,
        'roi_wall_seconds':result['wall_seconds'],'roi_cpu_seconds':result['cpu_seconds'],'mips':result['mips'],
        'max_rss_gib':result['max_rss_kib']/1024**2,'verification':verification,
        'output_sha256':output_hash,**{key:result[key] for key in
          ('initialization_seconds','setup_seconds','warmup_seconds','verification_seconds')}}


def report(root,output):
    manifest=json.loads((root/'manifest.json').read_text());records=[]
    reference_path=root/'native-verification.json'
    references=json.loads(reference_path.read_text()) if reference_path.exists() else {}
    for job in manifest['jobs']:
        for model in manifest['models']:
            directory=root/'runs/full'/(job['name']+'--'+model)
            row={'case':job['name'],'model':model,'mode':job['mode'],'accepted':False,'reason':'pending'}
            path=directory/'receipt.json'
            if path.exists():
                try:row.update(evidence(directory,json.loads(path.read_text())))
                except Exception as error:row['reason']='qualification rejected: '+str(error)
            records.append(row)
    for job in manifest['jobs']:
        pair=[r for r in records if r['case']==job['name']]
        if all(r['accepted'] for r in pair) and pair[0].get('output_sha256'):
            identical=pair[0]['output_sha256']==pair[1]['output_sha256']
            for row in pair:
                row['verification']='matching oracle/FixedLat array dump' if identical else 'array mismatch'
                if not identical:row.update(accepted=False,reason='PolyBench array dump differs between models')
            reference=references.get(job['name'])
            if reference:
                for row in pair:
                    if row['output_sha256']==reference['output_sha256']:
                        row['verification']='matches native original-source array dump at printed precision'
                    else:row.update(accepted=False,reason='PolyBench array dump differs from native original source')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps({'updated':time.time(),'records':records,
         'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()},indent=2)+'\n')
    fields=sorted(set().union(*(r.keys() for r in records)))
    with output.with_suffix('.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
    lines=['# Single-core gem5 ROI qualification','',
        'O3CPU; 2M warmup inside the ROI, then up to 2B additional committed instructions or ROI end. '
        'No replay and no per-request matching. Natural ROI endings exclude the warmup prefix; '
        'capped intervals cannot claim final numerical verification. All privilege levels are counted in FS.','',
        'These are local concurrent screening timings, not a controlled simulator-speed comparison. '
        'DRAM reads below 10,000 are flagged as low traffic, not silently excluded. '
        'Per-case source and checkpoint hashes are in the bound input manifest.','',
        '| Case | Model | Status | Measured M instructions | Ending | Read / kinst | IPC | Total hours |',
        '|---|---|---|---:|---|---:|---:|---:|']
    for row in records:
        if row['accepted']:
            lines.append(f"| {row['case']} | {row['model']} | qualified{' / low traffic' if row['low_traffic'] else ''} | "
                f"{row['instructions']/1e6:.3f} | {row['stop_reason']} | {row['dram_reads_per_kinst']:.3f} | "
                f"{row['ipc']:.3f} | {row['total_wall_hours']:.3f} |")
        else:lines.append(f"| {row['case']} | {row['model']} | {row['reason'].replace('|','/')} | — | — | — | — | — |")
    lines+=['','Detailed JSON/CSV records include phase wall time, peak memory, traffic counts, verification and exclusions.',
            'This is qualification of the proposed protocol, not the full frozen-model transfer comparison.']
    output.write_text('\n'.join(lines)+'\n')
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        records=report(args.root,args.output)
        if not args.watch or (args.root/'qualification-terminal.json').exists():break
        pid=json.loads((args.root/'screen-process.json').read_text())['pid']
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(120)
