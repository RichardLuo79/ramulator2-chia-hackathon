"""Distributional diagnostics of paired request errors: correlation, std ratio, best-constant and
shuffled-pair MAE, and MAE of a time-smoothed version of the model's own predictions."""
import sys,json,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,pandas as pd
from score import load,recorded,windows_from_log
def pairs_of(O,M):
    eo=O[O.frontend_id>=0]; em=M[M.frontend_id>=0]
    p=eo.merge(em,on=['src','frontend_id','frontend_sub_id'],how='inner',suffixes=('_o','_m'))
    return p[p.addr_o==p.addr_m]
def stats(O,M,seed=1):
    p=pairs_of(O,M); L=O.lat.mean()
    lo=p.lat_o.to_numpy(float); lm=p.lat_m.to_numpy(float)
    rng=np.random.default_rng(seed)
    res=dict(L=L,n=len(p),mae=np.abs(lm-lo).mean()/L,corr=float(np.corrcoef(lo,lm)[0,1]),std_o=lo.std()/L,std_m=lm.std()/L)
    # best constant per core (oracle median per core) -> lower bound for any context-free predictor
    res['mae_const_median_core']=float(sum(np.abs(g.lat_o-g.lat_o.median()).sum() for _,g in p.groupby('src'))/len(p)/L)
    # shuffled within core: model's own marginal distribution, no pairing information
    sh=0
    for _,g in p.groupby('src'):
        a=g.lat_m.to_numpy(float).copy(); rng.shuffle(a); sh+=np.abs(a-g.lat_o.to_numpy(float)).sum()
    res['mae_shuffled_core']=float(sh/len(p)/L)
    # model smoothed over time: rolling mean of model latency over a window of K admissions (per core, by oracle arrival order)
    for K in (16,256):
        s=0
        for _,g in p.sort_values('arrive_o').groupby('src'):
            sm=g.lat_m.rolling(K,center=True,min_periods=1).mean().to_numpy()
            s+=np.abs(sm-g.lat_o.to_numpy(float)).sum()
        res[f'mae_model_smoothed_{K}']=float(s/len(p)/L)
    # oracle smoothed with itself (how much per-request detail exists beyond local mean)
    for K in (16,256):
        s=0
        for _,g in p.sort_values('arrive_o').groupby('src'):
            sm=g.lat_o.rolling(K,center=True,min_periods=1).mean().to_numpy()
            s+=np.abs(sm-g.lat_o.to_numpy(float)).sum()
        res[f'mae_oracle_local_mean_{K}']=float(s/len(p)/L)
    return res
