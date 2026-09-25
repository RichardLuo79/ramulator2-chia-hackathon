"""Small offline differential checks against pinned baseline source functions.

Reference implementations are external source inputs, never campaign inputs.
zsim and Sniper function bodies are compiled unchanged with clock/stat shims;
MeSS is compiled directly. No reference simulator or model is run by an LLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PINS = {
    "zsim": "f524c05647a152f44f85ad445de3dcf9e4610bfd",
    "sniper": "56505e42fd98bca863fac181e769bd3c98d2bb33",
    "mess": "7feba12ce4af081116592fb5f045895a3b86de34",
    "gem5": "7a2b0e413d06c5ce7097104abef3b1d9eaabca91",
}


def function(source, marker):
    """Retain an upstream definition verbatim, without its simulator includes."""
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def run(command, log):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed: {command}; see {log}")
    return result.stdout


def references(root):
    """Make test-only shims; none of these are linked into Ramulator."""
    zsim = root / "upstream/zsim"
    sniper = root / "upstream/sniper"
    for name, directory in (("zsim", zsim), ("sniper", sniper)):
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != PINS[name]:
            raise ValueError("reference checkout changed: " + name)
    md1 = (zsim / "src/mem_ctrls.cpp").read_text()
    wmg = (sniper / "common/performance_model/queue_model_windowed_mg1.cc").read_text()
    header = '#include <bits/stdc++.h>\nusing namespace std;\n'
    md1_shim = header + '''
struct Counter { void inc(unsigned = 1) {} };
struct Clock { uint64_t numPhases=0, phaseLength=10000; } clock_info;
Clock* zinfo=&clock_info;
struct MD1Memory {
  uint64_t lastPhase=0;
  double smoothedPhaseAccesses=0, maxRequestsPerCycle=1.0/8;
  uint32_t curPhaseAccesses=0, zeroLoadLatency=42, curLatency=42;
  Counter profClampedLoads, profLoad, profUpdates;
  void updateLatency();
};
'''
    md1_shim += 'void ' + function(md1, 'MD1Memory::updateLatency()')
    md1_shim += '''
int main(int argc,char** argv) {
  cout << unitbuf;
  MD1Memory model; model.zeroLoadLatency=model.curLatency=stoul(argv[1]);
  model.maxRequestsPerCycle=1.0/stod(argv[2]);
  uint64_t t; int type,reset;
  while(cin>>t>>type>>reset) {
    zinfo->numPhases=(t+1)/10000;
    if(zinfo->numPhases>model.lastPhase) model.updateLatency();
    ++model.curPhaseAccesses;
    cout << model.curLatency << '\\n';
  }
}
'''
    wmg_shim = header + '''
using UInt32=uint32_t; using core_id_t=int;
struct SubsecondTime {
  double ps=0;
  static SubsecondTime PS(double p) {return {floor(p)};}
  static SubsecondTime Zero() {return {0};}
  double getPS() const {return ps;}
  auto operator<=>(const SubsecondTime&) const = default;
  SubsecondTime operator-(SubsecondTime b) const {return {ps-b.ps};}
  SubsecondTime& operator+=(SubsecondTime b) {ps+=b.ps;return *this;}
};
SubsecondTime operator*(int a,SubsecondTime b) {return {a*b.ps};}
struct Clock {SubsecondTime now; SubsecondTime getGlobalTime(){return now;}
 Clock* getClockSkewMinimizationServer(){return this;} } clock_info;
Clock* Sim(){return &clock_info;}
struct QueueModelWindowedMG1 {
  SubsecondTime m_window_size{10000000},m_total_utilized_time{},m_total_queue_delay{};
  uint64_t m_total_requests=0,m_num_arrivals=0;
  double m_service_time_sum=0,m_service_time_sum2=0;
  multimap<SubsecondTime,SubsecondTime> m_window;
  SubsecondTime computeQueueDelay(SubsecondTime,SubsecondTime,core_id_t);
  void addItem(SubsecondTime,SubsecondTime); void removeItems(SubsecondTime);
};
'''
    for return_type, marker in (("SubsecondTime", "QueueModelWindowedMG1::computeQueueDelay("),
                                ("void", "QueueModelWindowedMG1::addItem("),
                                ("void", "QueueModelWindowedMG1::removeItems(")):
        wmg_shim += return_type + ' ' + function(wmg, marker) + '\n'
    wmg_shim += '''
int main(int argc,char** argv) {
  cout << unitbuf;
  QueueModelWindowedMG1 model; double tck=stod(argv[1]),burst=stod(argv[2]);
  // Match the adapter's explicitly quantized window, then test PS->tick rounding.
  model.m_window_size={floor(stod(argv[3])*1000/tck)*tck};
  uint64_t t; int type,reset;
  while(cin>>t>>type>>reset) {
    clock_info.now={(t+1)*tck};
    cout << (uint64_t)(model.computeQueueDelay(clock_info.now,{burst*tck},0).ps/tck) << '\\n';
  }
}
'''
    mess_shim = '''#include <bits/stdc++.h>
#include "mess_mem_ctrl.h"
int main(int argc,char** argv) {
  std::cout << std::unitbuf;
  MessMemCtrl model(argv[1],1000,1000.0/std::stod(argv[2]),1);
  uint64_t t; int type,reset;
  while(std::cin>>t>>type>>reset) {
    std::cout << model.access(t+1,type==1) << '\\n';
  }
}
'''
    out = root / "differential"
    out.mkdir(exist_ok=True)
    source_paths = {"md1": zsim / "src/mem_ctrls.cpp",
                    "wmg1": sniper / "common/performance_model/queue_model_windowed_mg1.cc",
                    "mess": root / "reference/mess/mess_mem_ctrl.cpp"}
    for name, body in (("md1", md1_shim), ("wmg1", wmg_shim), ("mess", mess_shim)):
        source = out / (name + "_reference.cpp")
        source.write_text(body)
        extra = [] if name != "mess" else [str(source_paths[name]), '-I'+str(source_paths[name].parent)]
        run(["c++", "-O3", "-DNDEBUG", "-std=c++20", str(source), *extra, "-o", str(out / name)],
            out / (name + "_build.log"))
    save(out / "reference-identities.json", {"pins": PINS, "sources": {
        name: {"path": str(p), "sha256": sha(p)} for name,p in source_paths.items()}})


def qualify(root, runtime, original_runtime):
    import random
    import re
    import ramulator
    out = root / 'differential'
    repo = Path(__file__).resolve().parents[2]
    # Compile the public-API driver against each real shared library. A changed
    # controller is not tested by reimplementing it in Python.
    for version, build in (('original', original_runtime), ('corrected', runtime)):
        source = build/'runtime-source'
        command = ['c++','-O3','-DNDEBUG','-std=c++20',str(repo/'tests/utils/baseline_trace_driver.cpp'),
                   '-I'+str(source/'src'), '-I'+str(source/'ext/fmt/include'),
                   '-L'+str(build/'runtime'), '-Wl,-rpath,'+str(build/'runtime'),
                   '-lramulator','-o',str(out/(version+'-driver'))]
        run(command,out/(version+'-build.log'))
    curves = {}
    for line in (repo/'tools/eval/calibration/mess_DDR5.txt').read_text().splitlines():
        pct,bw,lat=map(float,line.split())
        curves.setdefault(str(int(pct)),[]).append([1000*bw,lat])
    # The standalone expects decreasing bandwidth, unlike the local txt file.
    for curve in curves.values(): curve.sort(reverse=True)
    save(out/'curve.json', {'measuredChannels':1,'curves':curves})
    tests=[]
    randomizer=random.Random(2251)
    stream=[];t=0
    for i in range(120_000):
        # Saturation, load changes, idle gaps, simultaneous arrivals and all
        # read/write ratios. Data/address patterns do not enter these models.
        phase=i//10000
        t += (0,1,2,8,100,0,1,50,1,500,8,2)[phase]
        if i in (1000,41000,81000): t+=100_000
        kind = 1 if phase in (0,7) else int(randomizer.random()<(.1 if phase%2 else .7))
        stream.append((t,kind,int(i in (21001,62000))))
    tests.append(('load-transitions',stream))
    tests.append(('boundary-startup',[(0,0,0),(0,0,0),(0,1,0),(1,0,1),(9999,0,0),
                                    (10000,0,0),(10001,1,0),(99999,0,0),(100000,0,0)]))
    report=[]
    for model, cls in (('md1',ramulator.controller.MD1),('wmg1',ramulator.controller.WMG1),
                       ('mess',ramulator.controller.Mess),('fixedlat',ramulator.controller.FixedLat)):
        kwargs={'dram':ramulator.dram.DDR5(org_preset='DDR5_16Gb_x8',timing_preset='DDR5_4800AN')}
        if model=='mess':kwargs['curve_path']=str(repo/'tools/eval/calibration/mess_DDR5.txt')
        cfg=cls(**kwargs).to_config()
        save(out/(model+'.json'),{'controller':cfg})
        selected_tests=[(name,rows,{}) for name,rows in tests]
        if model=='wmg1':
            selected_tests.append(('short-window-startup',[(0,0,0),(0,0,0),(0,0,1),(10,0,0),(10,0,0)],{'window_ns':2}))
        if model=='fixedlat':
            selected_tests.append(('no-bandwidth',tests[-1][1],{'pipe':0}))
        for name,rows,options in selected_tests:
            config_path=out/(model+'-'+name+'.json')
            save(config_path,{'controller':{**cfg,**options}})
            path=out/(name+'.txt');path.write_text(''.join(f'{t} {k} {r}\n' for t,k,r in rows))
            outputs={}
            for version in ('original','corrected'):
                result=subprocess.run([str(out/(version+'-driver')),str(config_path),str(path)],
                                      capture_output=True,text=True,check=True)
                outputs[version]=[list(map(int,line.split())) for line in result.stdout.splitlines()]
                (out/(model+'-'+name+'-'+version+'.state.log')).write_text(result.stderr)
                info=re.search(r'read_latency=(\d+) tCK_ps=(\d+)',result.stderr)
                if not info:raise ValueError(result.stderr)
                rl,tck=map(int,info.groups())
            if model=='fixedlat':
                free=0;expected=[]
                for t,kind,_ in rows:
                    start=max(t+1,free) if options.get('pipe',1) else t+1
                    expected.append(start+rl-t-1);free=start+8
            else:
                args=([str(rl),'8'] if model=='md1' else [str(tck),'8',str(options.get('window_ns',10000))] if model=='wmg1'
                      else [str(out/'curve.json'),str(tck)])
                values=subprocess.run([str(out/model),*args],input=path.read_text(),capture_output=True,text=True,check=True)
                expected=list(map(int,values.stdout.splitlines()))
                (out/(model+'-'+name+'-reference.state.log')).write_text(values.stderr)
                if model=='wmg1': expected=[v+rl for v in expected]
            errors={v:[] for v in outputs}
            for version,observed in outputs.items():
                for i,((t,kind,_),(_,accepted,depart,_),prediction) in enumerate(zip(rows,observed,expected)):
                    if not accepted:raise ValueError('unexpected admission rejection')
                    if kind==0 and depart-t-1!=prediction:
                        errors[version].append({'request':i,'actual':depart-t-1,'reference':prediction})
                if len(observed)!=len(rows):raise ValueError('missing driver output')
            row={'model':model,'fixture':name,'requests':len(rows),
                 'old_mismatches':len(errors['original']),'corrected_mismatches':len(errors['corrected']),
                 'old_examples':errors['original'][:5], 'corrected_examples':errors['corrected'][:5],
                 'read_latency':rl,'tCK_ps':tck,'burst_ticks':8,'input_sha256':sha(path)}
            report.append(row);print(json.dumps(row),flush=True)
    save(out/'results.json', report)
    if any(row['corrected_mismatches'] for row in report):raise ValueError('corrected reference mismatch')


def closed_loop(root):
    """Run independent latency-feedback clients, not only a replayed arrival list.

    One outstanding read models a dependent chain; larger windows model
    streaming concurrency. At each load transition outstanding reads drain,
    then an idle gap tests recovery. Writes update predictors but acknowledge
    immediately, matching the deliberately adapted Ramulator interface.
    """
    import heapq
    import random
    out=root/'differential'
    results=[]
    for model in ('md1','wmg1','mess','fixedlat'):
        expected=None
        for version in ('reference','corrected','original'):
            args=([str(out/model),*({'md1':['42','8'],'wmg1':['416','8','10000'],
                    'mess':[str(out/'curve.json'),'416']}.get(model,[]))]
                  if version=='reference' and model!='fixedlat' else
                  [str(out/(version+'-driver')),str(out/(model+'.json')),'-'])
            process=None
            log=out/(model+'-feedback-'+version+'.log')
            with log.open('w') as errors:
                if not (model=='fixedlat' and version=='reference'):
                    process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1)
                rows=[];clock=0;pipe_free=0;randomizer=random.Random(2251)
                phases=[]
                try:
                    for pressure in (1,64,512,1):
                        pending=[];start=clock
                        for i in range(4000):
                            while pending and pending[0]<=clock:heapq.heappop(pending)
                            if len(pending)>=pressure:
                                clock=heapq.heappop(pending)
                                while pending and pending[0]<=clock:heapq.heappop(pending)
                            kind=int(pressure!=1 and randomizer.random()<.2)
                            reset=int(i==2111)
                            if process:
                                process.stdin.write(f'{clock} {kind} {reset}\n');process.stdin.flush()
                                line=process.stdout.readline()
                                if not line:raise RuntimeError('predictor exited during feedback fixture: '+str(log))
                                values=list(map(int,line.split()))
                                if version=='reference':latency=values[0]+(42 if model=='wmg1' else 0)
                                else:
                                    if values[1]!=1:raise ValueError('unexpected admission rejection')
                                    latency=values[2]-clock-1 if kind==0 else None
                            else:
                                service=max(clock+1,pipe_free);pipe_free=service+8
                                latency=service+42-clock-1
                            # The write predictor's return value is deliberately
                            # unscored: the adapter uses posted writes.
                            rows.append((clock,kind,latency if not kind else None))
                            if not kind:heapq.heappush(pending,clock+1+latency)
                            clock+=1
                        clock=max([clock,*pending])
                        phases.append({'outstanding_limit':pressure,'requests':4000,'ticks':clock-start})
                        clock+=100_000
                finally:
                    if process:
                        process.stdin.close()
                        code=process.wait(timeout=30)
                        if code:raise RuntimeError('feedback driver failed to drain: '+str(log))
                if expected is None:expected=rows
                mismatches=sum(a!=b for a,b in zip(rows,expected))
                save(out/(model+'-feedback-'+version+'.json'),{'requests':rows,'phases':phases})
                results.append({'model':model,'version':version,'requests':len(rows),'mismatches':mismatches,'phases':phases})
    save(out/'closed-loop-results.json',results)
    if any(x['mismatches'] for x in results if x['version']=='corrected'):
        raise ValueError('corrected closed-loop mismatch')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--original-runtime',type=Path)
    args = parser.parse_args()
    references(args.root)
    if args.runtime:
        qualify(args.root,args.runtime,args.original_runtime)
        closed_loop(args.root)
