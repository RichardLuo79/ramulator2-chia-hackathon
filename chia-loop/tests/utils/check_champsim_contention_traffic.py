"""Small end-to-end replay checks; never used as accuracy measurements."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess

from ramulator_chia.eval import config as C, simpleo3, matchlib


def records(core):
    data, pages = bytearray(), set()
    for i in range(30_000):
        ip = 0x400000 + 4 * (i % 8)
        # One dependent, cache-missing stream; the other cores finish early
        # and revisit their short streams several times.
        addr = 0x10000000 + (i * 4096 if core == 1 or i % 32 == 0 else (i % 16) * 64)
        data += struct.pack('<QBB2B4B2Q4Q', ip, 0, 0, 1, 0,
                            1 if core == 1 else 0, 0, 0, 0, 0, 0, addr, 0, 0, 0)
        pages.update((ip >> 12, addr >> 12))
    return data, pages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    args=parser.parse_args()
    import ramulator
    root=args.work/'study/replay-fixtures'
    root.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,LD_LIBRARY_PATH=str(args.runtime/'runtime'),RAMULATOR_TICKS_PER_8='12')
    for name in ('RAMULATOR_CONFIG','CHAMPSIM_PLACEMENT_FILE','CHAMPSIM_COMPLETION_POLICY'):
        env.pop(name,None)
    summary={}
    for cores in (1,4,8):
        directory=root/f'c{cores}'
        directory.mkdir(exist_ok=True)
        traces=[]
        pages=[f'CHAMPSIM_PAGES_V1 {cores} 4096 {8<<30}\n']
        for core in range(cores):
            data, inventory=records(core)
            path=directory/f'core{core}.gz'
            path.write_bytes(gzip.compress(data,compresslevel=3,mtime=0))
            traces.append(str(path))
            pages.extend(f'{core} {page}\n' for page in sorted(inventory))
        inventory=directory/'pages.txt'
        inventory.write_text(''.join(pages))
        placement=directory/'placement.txt'
        binary=args.work/f'source{cores}/bin/champsim-{cores}core'
        subprocess.run([str(binary),'--prepare-placement',str(inventory),str(placement)],env=env,check=True,stdout=subprocess.DEVNULL)
        runs={}
        arms=[('oracle','background-replay'),('fixedlat','background-replay')]
        if cores==1:
            arms.append(('fixedlat','finite'))
        for model,policy in arms:
            out=directory/(model+'-'+policy)
            out.mkdir(exist_ok=True)
            dram=ramulator.dram.DDR5(org_preset=C.STD['DDR5']['org'],timing_preset=C.STD['DDR5']['timing'])
            controller=simpleo3._build_controller(ramulator,model,dram,'DDR5',out/'controller.csv',{},mess_curve=None)
            configuration={'frontend':{'impl':'External','clock_ratio':1},'memory_system':
                ramulator.memory_system.GenericDRAM(clock_ratio=1,controllers=[controller],
                    channel_mapper=ramulator.channel_mapper.CacheLineInterleave()).to_config()}
            cfg=out/'config.json'
            cfg.write_text(json.dumps(configuration))
            live=dict(env,RAMULATOR_CONFIG=str(cfg),CHAMPSIM_PLACEMENT_FILE=str(placement),CHAMPSIM_COMPLETION_POLICY=policy)
            with (out/'simulation.log').open('w') as log:
                subprocess.run([str(binary),'-w','1000','-i','5000',*traces],env=live,
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            text=(out/'simulation.log').read_text()
            timing=sorted(tuple(map(int,row)) for row in re.findall(
                r'Simulation finished CPU (\d+) instructions: (\d+) cycles: (\d+)',text))
            assert len(timing)==cores and all(5000<=r[1]<5005 for r in timing)
            windows=[json.loads(r) for r in re.findall(r'^CONTENTION_WINDOW (.+)$',text,re.M)]
            background=[json.loads(r) for r in re.findall(r'^CONTENTION_CORE (.+)$',text,re.M)]
            if policy=='background-replay':
                assert len(windows)==len(background)==cores
                if cores>1:
                    assert any(r['wraps']>0 and r['background_instructions']>0 for r in background)
                else:
                    assert background==[dict(core=0,background_instructions=0,wraps=0)]
                raw,_=matchlib._load_frame(out/'controller.csv.ch0')
                # Full-stream demand IDs stay unique even after many wraps.
                eligible=raw[raw.frontend_id>=0]
                assert not eligible.duplicated(['src','frontend_id','frontend_sub_id']).any()
                foreground,_=matchlib._load_frame(out/'controller.csv.ch0',windows=windows)
                assert len(foreground)<=sum(w['admitted_reads'] for w in windows)
                if cores>1:
                    assert len(raw)>len(foreground)
            runs[model+'-'+policy]=dict(timing=timing,windows=windows,background=background,
                csv_sha256=hashlib.sha256((out/'controller.csv.ch0').read_bytes()).hexdigest())
        paired=matchlib.match(directory/'oracle-background-replay/controller.csv.ch0',
            directory/'fixedlat-background-replay/controller.csv.ch0',
            oracle_windows=runs['oracle-background-replay']['windows'],
            model_windows=runs['fixedlat-background-replay']['windows'])
        assert paired['address_mismatch_pairs']==0 and len(paired['dv'])>0
        if cores==1:
            assert runs['fixedlat-finite']['timing']==runs['fixedlat-background-replay']['timing']
            assert runs['fixedlat-finite']['csv_sha256']==runs['fixedlat-background-replay']['csv_sha256']
        summary[str(cores)]=dict(runs=runs,matched=len(paired['dv']),physical_mismatches=0)
        print('PASS',cores,'forced replay / measurement boundaries',flush=True)
    (root/'passed.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
