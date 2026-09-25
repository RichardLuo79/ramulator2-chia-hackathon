"""Read-only checks for endpoint selection, populations and offline reporting."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import campaign_results as collect
import campaign_results_plots as plots


def score_fixture():
    def population(recorded, admitted):
        return dict(recorded_reads=recorded, without_stable_id=0,
                    unmatched_reads=recorded-4, admissions_without_recorded_latency=admitted-recorded,
                    admitted_foreground_reads=admitted, total_admitted_reads=admitted+3)
    request = dict(address_mismatch_pairs=0, mae=.15, sgn=.05, tail=.297,
        p999_over_L=.2997, mae_cycles=15, paired_p99_cycles=29.7, paired_p999_cycles=29.97,
        extreme_min_cycles=-20, extreme_max_cycles=30, extreme_min_over_L=-.2,
        extreme_max_over_L=.3, oracle_read_mean_latency=100, matched=4, cov_o=.8,
        cov_m=4/6, n_oracle=5,
        tail_thresholds={f'{k}L': dict(fraction=0, count=0, absolute_error_share=0) for k in (1,5)})
    return dict(cycles=dict(per_core_dev_pct=[-10,20], mean_abs_per_core_pct=15,makespan_dev_pct=20),
                request=request,request_pairing={'oracle':population(5,7),'model':population(6,9)})


def test_full_and_projected_scores_have_same_populations():
    full=score_fixture()
    projected=dict(per_core_signed_error_pct=[-10,20],core_error_pct=15,makespan_signed_error_pct=20,
                   populations=full['request_pairing'],request=full['request'])
    a=collect.score_row('validation-c4-01','astra_single_core',full)
    b=collect.score_row('validation-c4-01','astra_single_core',projected,projected=True)
    assert a==b
    assert a['core_abs_pct']==15 and a['core_signed_pct']==5
    assert a['oracle_recorded_reads']==5 and a['pairs']==4
    assert a['oracle_admissions_without_recorded_latency']==2
    assert a['oracle_coverage']==.8  # Not 4 / 7 admissions or 4 / 10 including background.


@pytest.mark.parametrize('change,match',[
    (lambda x:x['request'].update(address_mismatch_pairs=1),'physical'),
    (lambda x:x['request'].update(cov_o=4/7),'coverage'),
    (lambda x:x['cycles'].update(mean_abs_per_core_pct=5),'aggregation'),
    (lambda x:x['request'].update(mae=float('nan')),'nonfinite'),
    (lambda x:x.update(pairing_error='missing observations'),'missing observations'),
])
def test_invalid_scores_cannot_enter_report(change,match):
    full=score_fixture();change(full)
    with pytest.raises(ValueError,match=match):collect.score_row('test-c4-01','mess',full)


def receipt(cycles=(90,240), complete=True):
    return dict(receipt_sha256='receipt',identity={'case':{'workload':'test-c4-01','placement':'p'}},
                observation=dict(complete=complete,case={'workload':'test-c4-01'},
                    frontend_stats={'per_core_instructions':[20000000,20000002]},
                    controller_stats={'reads':7},per_core_cycles=list(cycles),
                    traces={'controller.csv':dict(logical_sha256='decoded',logical_bytes=123,
                              stored_sha256='gzip-bytes',name='controller.csv.gz')}))


def test_receipt_equivalence_uses_decoded_not_gzip_bytes():
    a=receipt();b=deepcopy(a)
    b['observation']['traces']['controller.csv']['stored_sha256']='different-gzip'
    b['wall_seconds']=27;b['host']='another host'
    assert collect.choose_receipt([a,b])==a
    b['observation']['controller_stats']['reads']+=1
    with pytest.raises(ValueError,match='disagree'):collect.choose_receipt([a,b])


def test_score_is_bound_to_real_core_counts_and_case_identity():
    row=collect.score_row('test-c4-01','mess',score_fixture())
    model=receipt();oracle=receipt((100,200))
    collect.bind_score(row,model,oracle)
    assert row['instructions']==[20000000,20000002]
    oracle['identity']['case']['placement']='old-first-touch'
    with pytest.raises(ValueError,match='case identity'):collect.bind_score(row,model,oracle)
    oracle=receipt((100,200));model['observation']['per_core_cycles']=[90]
    with pytest.raises(ValueError,match='count'):collect.bind_score(row,model,oracle)
    with pytest.raises(ValueError,match='incomplete'):collect.bind_score(row,receipt(complete=False),oracle)


def test_headlines_are_equal_case_weight_and_shared_cohort():
    base=collect.score_row('test-c4-01','mess',score_fixture())
    rows=[]
    for arm in ['mess','astra_single_core']:
        for case,error,n in [('test-a',1,100),('test-b',9,10000)]:
            rows.append({**base,'arm':arm,'case':case,'core_abs_pct':error,'pairs':n})
    rows.append({**base,'arm':'mess','case':'test-unshared','core_abs_pct':100})
    result=plots.headline(pd.DataFrame(rows),['mess','astra_single_core'])
    assert result.cases.tolist()==[2,2]
    assert result.core_abs_pct.tolist()==[5,5]
    assert result.paired_reads_total.tolist()==[10100,10100]


def test_outlier_selection_conditions_and_deduplication():
    rows=[]
    for arm in ['astra_single_core','deepseek_single_core']:
        for case,core,mae,p999,extreme in [
            ('largest-core',9,.8,4,90),('largest-mae',1,9,20,20),
            ('low-core-tail',.1,.5,100,1000),('below-both',.2,.1,5,4000),
            ('middle',2,.4,1,50)]:
            rows.append(dict(arm=arm,case=case,cohort='test',cores=1,core_abs_pct=core,
                             mae_L=mae,p999_L=p999,min_cycles=-extreme,max_cycles=1))
    selection=collect.select_extremes(rows)
    assert len(selection)==8
    assert [x['score']['case'] for x in selection[:4]]==[
        'largest-core','largest-mae','low-core-tail','below-both']
    # Same case can satisfy several selectors; combine reasons instead of plotting twice.
    for r in rows:
        if r['case']=='below-both':r['p999_L']=200
    selection=collect.select_extremes(rows)
    assert len(selection)==6
    assert len(selection[2]['reasons'])==2


def test_complete_observations_not_plot_samples_define_tail_statistics(tmp_path,monkeypatch):
    from ramulator_chia.eval import matchlib
    from ramulator_chia.eval.metrics import paired_error_statistics
    oracle=pd.DataFrame({'lat':[100,100,100,100,600]})
    model=pd.DataFrame({'lat':[80,100,110,130,50,50]})
    # The unpaired high-latency oracle read must be included in L.
    error=np.array([-20,0,10,30])
    scale=200
    stats=paired_error_statistics(error,scale,include_tails=True)
    pairs=pd.DataFrame(dict(src=[0]*4,frontend_id=[2000001,2000002,2100000,2100001],
        frontend_sub_id=[0]*4,addr_o=[64,128,192,256],addr_m=[64,128,192,256],
        arrive_o=[1,2,3,4],depart_o=[101,102,103,104],arrive_m=[1,2,3,4],
        depart_m=[81,102,113,134],lat_o=[100]*4,lat_m=[80,100,110,130]))
    results=[]
    for n in range(2):
        path=tmp_path/f'obs{n}';path.write_bytes(b'fixture')
        results.append((tmp_path,dict(directory='.',receipt_sha256=f'r{n}',
            observation={'traces':{'controller.csv.ch0':{'name':path.name,'stored_sha256':collect.sha(path)}},
                         'frontend_stats':{'admission_windows':['fixture-window']}})))
    frames=iter([oracle,model])
    monkeypatch.setattr(matchlib,'_load_frame',lambda *a,**k:(next(frames),True))
    monkeypatch.setattr(matchlib,'_stable_pairs',lambda *a:(pairs,None,None))
    score=dict(case='test-c1-case',arm='astra_single_core',pairs=4,L=scale,
               mae_L=stats['mae'],drift_L=stats['sgn'],p99_L=stats['tail'],
               p999_L=stats['p999_over_L'],min_cycles=-20,max_cycles=30)
    result=collect.scan_extreme({'score':score,'reasons':['fixture']},results)
    assert result['verified']
    assert result['score']['L']==200  # Mean of five oracle reads, not just four pairs.
    assert sum(result['histogram']['oracle'])==5
    assert sum(result['histogram']['model'])==6
    assert result['exact_statistics']['p999_over_L']==pytest.approx(.14985)
    assert result['extreme_requests'][-1]['error_cycles']==30
    assert result['ccdf']['probability'][0]==.75  # Strictly above zero.


def test_notebook_evidence_and_endpoints_if_collected():
    path=collect.REPO/'results/foundation/campaigns.json'
    if not path.exists():pytest.skip('optional preserved campaign evidence is not in a clean source checkout')
    data=collect.read(path)
    assert len(data['rows'])==800
    assert len(data['oracle_equivalence'])==200
    assert all(x['equal'] for x in data['oracle_equivalence'])
    assert len(data['default_transfer_equivalence'])==160
    assert len(data['extremes'])==len(data['extreme_selection'])==8
    for arm,source in data['sources'].items():
        assert hashlib.sha256(source['source'].encode()).hexdigest()==source['candidate']['files']['model.cpp']['sha256']
        assert source['selected_after_round']==(10 if arm.endswith('single_core') else 15)
    assert data['sources']['deepseek_single_core']['introduced_round']==7
    assert data['sources']['deepseek_multicore']['introduced_round']==12
    assert len([x for x in data['transfer']['rows'] if x['study']=='hardware'])==1440
    assert len([x for x in data['transfer']['rows'] if x['study']=='gem5'])==120


def test_workflow_has_two_panels_and_current_selection_policy(monkeypatch):
    import matplotlib.pyplot as plt
    captured = {}
    monkeypatch.setattr(plots, 'finish', lambda fig, name: captured.update(fig=fig, name=name))
    plots.workflow()
    fig = captured['fig']
    try:
        assert captured['name'] == 'chia_loop_workflow'
        assert len(fig.axes) == 2
        schedule = '\n'.join(t.get_text() for t in fig.axes[0].texts)
        loop = '\n'.join(t.get_text() for t in fig.axes[1].texts)
        assert 'Rounds 1–10' in schedule and 'Rounds 11–15' in schedule
        assert 'After round 15: test BOTH frozen endpoints' in schedule
        for text in ('Same model + effort', 'Compare incumbent + stage entry',
                     'Justified trade-offs allowed', 'No same-round correction',
                     'Optional synthetic tests / replay',
                     'Validation: anonymous numerical feedback only',
                     'An LLM cannot waive failed integrity checks'):
            assert text in loop
        assert 'Pareto promotion' not in loop
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for text in ax.texts:
                bounds = text.get_window_extent(renderer)
                assert bounds.x0 >= ax.bbox.x0 and bounds.x1 <= ax.bbox.x1
                assert bounds.y0 >= ax.bbox.y0 and bounds.y1 <= ax.bbox.y1
    finally:
        plt.close(fig)


def lat_tp_fixture():
    rows=[]
    for arm in ['oracle','fixedlat']:
        for nop,state,tp,lat in [(10000,'complete',.3,45), (100,'failed',None,None),
                                 (1,'complete',600,4000)]:
            rows.append(dict(arm=arm,read_ratio=100,nop_counter=nop,streaming_only=False,
                             status=state,throughput_GBps=tp,probe_latency_ns=lat,reason=None))
        rows.append(dict(arm=arm,read_ratio=100,nop_counter=0,streaming_only=True,
                         status='pending',throughput_GBps=None,probe_latency_ns=None,reason=None))
    return dict(point_statuses=rows,successful=4,expected=8,failed=2,pending=2,
                execution={'phase':'sweep'},failed_attempt_count=2,
                protocol={'case':{'suite':{'nop_counters':[1,100,10000]}}})


def test_lat_tp_missing_points_break_curves_and_are_never_zero():
    study=lat_tp_fixture()
    frame=plots.lat_tp_status(study)
    curve=plots.lat_tp_curve(frame,'fixedlat',100,[1,100,10000])
    assert curve.index.tolist()==[10000,100,1]
    assert curve.loc[100].isna().all()
    assert curve.loc[1,'throughput_GBps']==600  # No physical-rate clipping.
    study['point_statuses'][1]['throughput_GBps']=0
    with pytest.raises(ValueError,match='unfinished'):plots.lat_tp_status(study)


def test_lat_tp_plot_tolerates_failed_streaming_and_preserves_coverage(monkeypatch):
    import matplotlib.pyplot as plt
    tables={};figures={}
    monkeypatch.setattr(plots,'DATA',{'lat_tp':lat_tp_fixture()},raising=False)
    monkeypatch.setattr(plots,'table',lambda frame,name,**kw:tables.update({name:frame.copy()}))
    monkeypatch.setattr(plots,'display',lambda *a:None)
    monkeypatch.setattr(plots,'finish',lambda fig,name:figures.update({name:fig}))
    plots.lat_tp()
    try:
        summary=tables['lat_tp_foundation_summary'].set_index('arm')
        assert summary.loc['fixedlat','Low-load probe latency (ns)']==45
        assert np.isnan(summary.loc['fixedlat','Pure-read streaming (GB/s)'])
        coverage=tables['lat_tp_foundation_coverage'].set_index('arm')
        assert coverage.loc['fixedlat','complete']==2
        assert coverage.loc['fixedlat','failed']==1
        fig=figures['lat_tp_foundation']
        assert len(fig.axes)==6
        fixedlat=fig.axes[0].lines[1]
        assert fixedlat.get_xdata()[2]==600
        assert np.isnan(fixedlat.get_xdata()[1])
        assert all(ax.get_xscale()=='log' and ax.get_yscale()=='log' for ax in fig.axes)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        assert not fig.legends[0].get_window_extent(renderer).overlaps(
            fig._supxlabel.get_window_extent(renderer))
    finally:
        for fig in figures.values():plt.close(fig)


def test_generated_notebook_is_paper_only_if_present():
    import nbformat
    path=collect.REPO/'analyses/chia_campaign_results.ipynb'
    if not path.exists():pytest.skip('notebook is a generated deliverable')
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    manifest_path=collect.REPO/'results/paper/paper-v1/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==notebook.metadata.paper_manifest_sha256
    assert len(notebook.cells)==17
    captions=[c.source for c in notebook.cells if c.cell_type=='markdown']
    assert captions==[f'## Fig. {f["number"]}. {f["caption_markdown"]}' for f in manifest['figures']]
    for c in notebook.cells:
        if c.cell_type=='code':
            assert c.execution_count is not None
            assert not any(o.output_type=='error' for o in c.outputs)
            assert 'from tools.' not in c.source and 'import tools.' not in c.source
            assert 'requests.get(' not in c.source and 'subprocess.' not in c.source
            assert 'base64.b64decode' not in c.source
            assert not c.metadata.get('jupyter',{}).get('source_hidden',False)
