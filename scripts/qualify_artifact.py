"""Six opt-in full-window checks; never inference. Keeps every attempt."""
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys,time
from pathlib import Path
from artifact import ROOT

def main():
    refs=json.loads((ROOT/'results/provenance/qualification-reference.json').read_text())
    folder=ROOT/'.work/full-window';folder.mkdir(parents=True,exist_ok=True)
    def one(item):
        key,reference=item;case,model=key.split('/')
        target=folder/(case+'-'+model)
        log=target.with_suffix('.launcher.log')
        if target.exists():raise FileExistsError(target)
        start=time.time()
        with log.open('w') as f:
            run=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate'),'--case',case,'--model',model,
                                '--output',str(target)],stdout=f,stderr=subprocess.STDOUT)
        result=dict(case=case,model=model,returncode=run.returncode,wall_seconds=time.time()-start,passed=False)
        if run.returncode==0:
            observed=json.loads((target/'measurement/measurement.json').read_text())
            checks={k:observed[k]==reference[k] for k in ('per_core_cycles','frontend_stats','controller_stats')}
            checks['observations']=({k:v['logical_sha256'] for k,v in observed['traces'].items()}==
                                     {k:v['logical_sha256'] for k,v in reference['traces'].items()})
            result.update(checks=checks,passed=all(checks.values()))
        target.with_suffix('.qualification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(one,refs.items()))
    output=ROOT/'results/provenance/full-window-qualification.json'
    output.write_text(json.dumps(dict(schema_version=1,results=results,passed=all(x['passed'] for x in results)),indent=2)+'\n')
    if not all(x['passed'] for x in results):raise SystemExit(1)

if __name__=='__main__':main()
