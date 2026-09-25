"""Partial campaigns cannot change complete cohorts or substitute a draft."""
from copy import deepcopy
import hashlib
from pathlib import Path
import gzip
import json
import sqlite3

import pandas as pd
import pytest

import campaign_results as collect
import campaign_results_import as imp
import campaign_results_plots as plots
from test_campaign_results import receipt, score_fixture


def test_pending_tests_do_not_erase_the_completed_test_cohort():
    row=collect.score_row('test-c1-a','astra_single_core',score_fixture())
    row.update(cores=1)
    rows=[]
    for arm in plots.FOUNDATION:
        for split in ('training','validation','test'):
            if arm=='gemini_single_core' and split=='test': continue
            rows.append({**row,'arm':arm,'cohort':split})
    result=plots.foundation_numbers(pd.DataFrame(rows))
    assert len(result[result.cohort=='test'])==7
    assert len(result[result.cohort=='training'])==8
    assert 'gemini_single_core' not in result[result.cohort=='test'].arm.tolist()
    assert 'opus_single_core' in result[result.cohort=='test'].arm.tolist()


def test_missing_whole_test_cohort_creates_no_outlier():
    assert collect.select_extremes([], ('gemini_single_core',))==[]
    assert 'opus_single_core' not in plots.SYNTH
    assert 'gemini_single_core' not in plots.LAT_TP_FOUNDATION


def test_selected_model_not_last_submitted_or_current_draft():
    chosen={'candidate_id':'retained'}
    r7=dict(iteration=7, stage='single_core', promoted=True, candidate=chosen)
    r10=dict(iteration=10, stage='single_core', promoted=False,
             candidate={'candidate_id':'rejected'}, selected={'candidate':chosen})
    endpoint=dict(iteration=10,candidate=chosen,training={'candidate':chosen})
    records={'stage-selection:single_core':endpoint, 'iteration:7':r7,'iteration:10':r10,
             'iteration:15:explore':{'draft':'not selected'}}
    selected, origin=imp.selected_round(records,'single_core')
    assert selected['candidate']==chosen and origin['iteration']==7
    records['iteration:10']['selected']['candidate']={'candidate_id':'different'}
    with pytest.raises(ValueError,match='committed retained'):
        imp.selected_round(records,'single_core')


@pytest.mark.parametrize('change',[
    lambda x:x['identity']['case'].update(placement='another-map'),
    lambda x:x['observation']['per_core_cycles'].__setitem__(0,123),
    lambda x:x['observation']['traces']['controller.csv'].update(logical_sha256='different'),
    lambda x:x['observation']['frontend_stats'].update(per_core_instructions=[1,2]),
    lambda x:x['observation'].update(complete=False),
])
def test_equivalence_rejects_identity_or_observation_mismatches(change):
    a=receipt(); a['identity']['runtime']='old'
    b=deepcopy(a); b['identity']['runtime']='new'
    assert imp.oracle_proof('gemini','test-c1-a',a,b)['equal']
    change(b)
    with pytest.raises(ValueError): imp.oracle_proof('gemini','test-c1-a',a,b)


def test_exact_candidate_source_and_parameters_are_required():
    candidate={'candidate_id':'selected','files':{'model.cpp':{'sha256':'source'}},'parameters':{'k':1}}
    r=receipt();r['identity'].update(model='candidate',candidate=deepcopy(candidate))
    imp.verify_candidate(r,candidate)
    r['identity']['candidate']['parameters']['k']=2
    with pytest.raises(ValueError,match='exact selected'):imp.verify_candidate(r,candidate)


def test_additive_snapshot_preserves_all_existing_numerical_evidence():
    base=collect.read(collect.REPO/'results/foundation/campaigns.json')
    supplement=collect.read(collect.REPO/'results/foundation/single-core-additions.json')
    result=imp.merge_snapshot(base,supplement)
    assert result['rows'][:800]==base['rows']
    for key in ('transfer','lat_tp','default_transfer_equivalence','membership','inputs'):
        assert result[key]==base[key]
    assert result['extremes'][:8]==base['extremes']
    assert len([r for r in result['rows'] if r['arm']=='opus_single_core'])==52
    assert len([r for r in result['rows'] if r['arm']=='gemini_single_core'])==52
    assert supplement['availability']['gemini']['single_core_test']=='complete'
    assert supplement['provenance']['gemini_independent_test']['evidence_kind'].endswith('not a committed campaign postrun')
    assert not any(r['cores']!=1 for r in supplement['rows'] if r['arm'].startswith('opus'))
    for source in supplement['sources'].values():
        assert hashlib.sha256(source['source'].encode()).hexdigest()==source['candidate']['files']['model.cpp']['sha256']
    old=deepcopy(base);old['rows'][0]['core_abs_pct']+=.1
    with pytest.raises(ValueError,match='base evidence'):imp.merge_snapshot(old,supplement)
    duplicate=deepcopy(supplement);duplicate['rows'].append(base['rows'][0])
    with pytest.raises(ValueError,match='overwrite'):imp.merge_snapshot(base,duplicate)


def test_new_full_population_extremes_match_bound_scalar_scores():
    supplement=collect.read(collect.REPO/'results/foundation/single-core-additions.json')
    cases=[r for r in supplement['extremes'] if r['arm'] in ('opus_single_core','gemini_single_core')]
    assert len(cases)==8
    for item in cases:
        assert item['verified'] and item['score']['pairs']>len(item['sample'])
        assert item['exact_statistics']['mae']==pytest.approx(item['score']['mae_L'],abs=1e-12)
        assert item['exact_statistics']['extreme_min_cycles']==item['score']['min_cycles']


def test_consistent_backup_is_verified_and_private_native_records_are_not_exported(tmp_path):
    database=tmp_path/'source.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE records (key TEXT,value TEXT)')
        db.executemany('INSERT INTO records VALUES (?,?)',[
            ('iteration:10',json.dumps({'stage':'single_core'})),
            ('native-session-private',json.dumps({'secret':'not analysis evidence'}))])
    packed=gzip.compress(database.read_bytes())
    (tmp_path/'campaign.sqlite.gz').write_bytes(packed)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        database_sha256=hashlib.sha256(packed).hexdigest(),completed_rounds=14,time=1790294006)))
    records,identity=imp.consistent_records(tmp_path)
    assert records=={'iteration:10':{'stage':'single_core'}}
    assert identity['database_sha256']==hashlib.sha256(database.read_bytes()).hexdigest()
    (tmp_path/'campaign.sqlite.gz').write_bytes(packed+b'corruption')
    with pytest.raises(ValueError,match='checksum'):imp.consistent_records(tmp_path)


def test_independent_test_report_must_bind_the_selected_source(tmp_path):
    report=dict(complete=True,candidate={'candidate_id':'wrong'},campaign_feedback=False,
                policy='operator-authorized-early-frozen-SC-tests-v1',rows=[])
    (tmp_path/'test-results.json').write_text(json.dumps(report))
    (tmp_path/'evidence-inventory.json').write_text(json.dumps({'files':{
        'test-results.json':collect.sha(tmp_path/'test-results.json')}}))
    with pytest.raises(ValueError,match='selection/policy'):
        imp.independent_test_evidence(tmp_path,{'candidate_id':'selected'}, {})


def test_vector_and_raster_exports_keep_fixed_axis_label_positions(tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    plots.setup({'rows':[]},output=tmp_path/'figures')
    fig,ax=plt.subplots(layout='constrained')
    ax.plot([1,10],[.1,.45]);ax.set_xscale('log')
    ax.set_xlabel('Recorded read latency (DRAM cycles)')
    ax.set_ylabel('Fraction per log-spaced bin')
    monkeypatch.setattr(plt,'show',lambda:None)
    plots.finish(fig,'label-check')
    assert not ax.xaxis._autolabelpos and not ax.yaxis._autolabelpos
    for suffix in ('pdf','svg','png'):
        assert (tmp_path/'figures'/('label-check.'+suffix)).stat().st_size>100
