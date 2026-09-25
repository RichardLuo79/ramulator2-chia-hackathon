"""Exercise warning-only low IPC and a deliberate no-progress diagnostic.

Artificial FixedLat delays stress the watchdog, not controller accuracy. These
tiny fixtures are kept outside the production matrix and never enter scores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess

spec=importlib.util.spec_from_file_location('replay_fixture',Path(__file__).with_name('check_champsim_contention_traffic.py'))
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    args=parser.parse_args()
    root=args.work/'study/watchdog-fixtures'
    root.mkdir(parents=True,exist_ok=False)
    raw,pages=fixture.records(1)
    trace=root/'dependent.gz'
    trace.write_bytes(gzip.compress(raw,compresslevel=3,mtime=0))
    inventory=root/'pages.txt'
    inventory.write_text(f'CHAMPSIM_PAGES_V1 1 4096 {8<<30}\n'+''.join(f'0 {p}\n' for p in sorted(pages)))
    placement=root/'placement.txt'
    binary=args.work/'source1/bin/champsim-1core'
    env=dict(os.environ,LD_LIBRARY_PATH=str(args.runtime/'runtime'),RAMULATOR_TICKS_PER_8='12')
    for name in ('RAMULATOR_CONFIG','CHAMPSIM_PLACEMENT_FILE','CHAMPSIM_COMPLETION_POLICY'):
        env.pop(name,None)
    subprocess.run([str(binary),'--prepare-placement',str(inventory),str(placement)],
                   env=env,check=True,stdout=subprocess.DEVNULL)
    base=json.loads((args.work/'study/replay-fixtures/c1/fixedlat-background-replay/config.json').read_text())

    def run(item):
        name,latency=item
        directory=root/name
        directory.mkdir()
        config=json.loads(json.dumps(base))
        controller=config['memory_system']['controllers'][0]
        controller.update(latency=latency,trace_path=str(directory/'controller.csv'))
        path=directory/'config.json'
        path.write_text(json.dumps(config))
        live=dict(env,RAMULATOR_CONFIG=str(path),CHAMPSIM_PLACEMENT_FILE=str(placement),
                  CHAMPSIM_COMPLETION_POLICY='background-replay')
        with (directory/'simulation.log').open('w') as log:
            done=subprocess.run([str(binary),'-w','10','-i','240',str(trace)],env=live,
                cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        text=(directory/'simulation.log').read_text()
        if name=='positive':
            assert done.returncode==0 and 'Simulation finished CPU 0' in text
            assert 'warning: IPC' in text and 'NO_PROGRESS' not in text
            cycles=int(re.search(r'Simulation finished CPU 0 instructions: \d+ cycles: (\d+)',text)[1])
            assert cycles>=20_000_000
        else:
            assert done.returncode==-signal.SIGABRT, (done.returncode,text[-2000:])
            assert 'NO_PROGRESS CPU 0' in text
            # A segmentation fault while printing used to obscure this diagnosis.
            assert 'CPU 0' in text and 'DEADLOCK' in text.upper()
        print('PASS retirement fixture',name,flush=True)
        return name,dict(returncode=done.returncode,passed=True,latency_cycles=latency)

    with ThreadPoolExecutor(max_workers=2) as pool:
        result=dict(pool.map(run,[('positive',1_000_000),('no_progress',50_000_000)]))
    (root/'passed.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
