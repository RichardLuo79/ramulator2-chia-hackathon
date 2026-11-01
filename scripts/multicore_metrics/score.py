"""Score a model/perturbed request trace against an oracle trace with the campaign metric:
stable-ID matching with ChampSim physical-mismatch filter, per-core admission windows,
L = mean latency of all oracle reads inside oracle windows. Core error per core.
Also returns per-core breakdowns. Usage as module: score(oracle_trace, oracle_windows, oracle_cycles, model_trace, model_windows, model_cycles)"""
import sys,json,re,os
import numpy as np
from ramulator_chia.eval import matchlib
from ramulator_chia.eval.metrics import paired_error_statistics

def windows_from_log(log):
    t=open(log).read()
    w=[json.loads(x) for x in re.findall(r'^CONTENTION_WINDOW (.+)$',t,re.M)]
    w.sort(key=lambda r:r['core'])
    cyc=[int(c) for _,_,c in sorted(re.findall(r'Simulation finished CPU (\d+) instructions: (\d+) cycles: (\d+)',t),key=lambda x:int(x[0]))]
    return w,cyc

def recorded(result_json):
    d=json.load(open(result_json))
    o=d['observation']
    base=os.path.join(os.path.dirname(result_json),os.path.basename(d['directory']))
    return dict(trace=os.path.join(base,o['traces']['controller.csv.ch0']['name']),windows=o['frontend_stats']['admission_windows'],cycles=o['per_core_cycles'])

def load(trace,windows):
    df,_=matchlib._load_frame(trace,windows=windows)
    return df

def score_frames(o,m,ocyc,mcyc,keep_pairs=False):
    eo=o[o.frontend_id>=0]; em=m[m.frontend_id>=0]
    pairs=eo.merge(em,on=['src','frontend_id','frontend_sub_id'],how='inner',suffixes=('_o','_m'))
    mism=(pairs.addr_o!=pairs.addr_m)
    pairs=pairs[~mism]
    dv=(pairs.lat_m-pairs.lat_o).to_numpy()
    L=float(o.lat.mean())
    st=paired_error_statistics(dv,L,include_tails=True)
    st.update(cov_o=len(dv)/len(o),cov_m=len(dv)/len(m),n_o=len(o),n_m=len(m),matched=len(dv),addr_mismatch=int(mism.sum()),L=L)
    per_core=[100*(mc-oc)/oc for oc,mc in zip(ocyc,mcyc)]
    st['per_core_dev_pct']=per_core; st['core_mae_pct']=float(np.mean(np.abs(per_core)))
    pc={}
    for c,g in pairs.groupby('src'):
        oc=o[o.src==c]
        d=(g.lat_m-g.lat_o).to_numpy()
        pc[int(c)]=dict(n_o=len(oc),matched=len(g),cov=len(g)/max(len(oc),1),mae_over_L=float(np.abs(d).mean())/L,
                        mae_over_Lc=float(np.abs(d).mean())/float(oc.lat.mean()),sgn_over_L=float(d.mean())/L,Lc=float(oc.lat.mean()))
    st['per_core']=pc
    if keep_pairs: return st,pairs
    return st
