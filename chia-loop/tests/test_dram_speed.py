"""Small standalone scheduling/receipt checks; no cloud operations."""
import json
from pathlib import Path

from ramulator_chia.eval import dram_speed as speed


def test_matrix_and_user_priority():
    jobs=speed.schedule()
    assert len(jobs)==1350
    assert len({(j['arm'],j['repeat'],speed.point_name(j['point'])) for j in jobs})==1350
    assert {j['point']['read_percent'] for j in jobs}=={100,75,50}
    assert all(j['repeat']==1 for j in jobs[:270])
    assert all(j['arm'] in speed.CRITICAL for j in jobs[:180])
    assert all(j['arm'] not in speed.CRITICAL for j in jobs[180:270])
    assert speed.schedule()==jobs


def test_simulated_identity_excludes_only_timing_and_explicit_retry_exception():
    source={'simulation_wall_s':2.,'input_loading_wall_s':1.,
            'batch':{'cycles':1024,'measured_cpu_s':1.,'admission_digest':2**63+11},
            'memory_system':{'send_rejects':20,'avg_latency':1.123456789}}
    result=speed.simulation_identity(source)
    assert result=={'batch':{'cycles':1024,'admission_digest':2**63+11},
                   'memory_system':{'send_rejects':20,'avg_latency':1.123456789}}
    assert speed.simulation_identity(source,ignore_retries=True)['memory_system']=={'avg_latency':1.123456789}
    assert speed.simulation_identity(source,legacy_precision=True)['memory_system']['avg_latency']==1.12346


def test_failures_and_pending_results_never_become_zero_performance(tmp_path):
    p=tmp_path/'production'; p.mkdir()
    (p/'protocol.json').write_text('{}')
    failed=p/'runs'/'failed'; failed.mkdir(parents=True)
    (failed/'receipt.json').write_text(json.dumps(dict(complete=False,error='fixture timeout',
        arm='oracle',point=speed.points()[0],repeat=1)))
    speed.export(tmp_path)
    result=json.loads((p/'results.json').read_text())
    assert (result['successful'],result['failed'],result['pending'])==(0,1,1349)
    assert len(result['summary'])==270
    assert all(row['n']==0 and row['median'] is None and row['oracle_relative_speedup'] is None
               for row in result['summary'])


def test_nonfinite_simulated_statistics_are_rejected():
    import pytest
    for value in (float('nan'),float('inf'),float('-inf')):
        with pytest.raises(ValueError,match='non-finite'):
            speed.simulation_identity({'memory_system':{'latency':value}})


def test_extension_matrix_and_explicit_self_funded_cutoff(tmp_path):
    arms=['oracle','opus_single_core','gemini_single_core']
    (tmp_path/'study.json').write_text(json.dumps(dict(models=arms,critical_models=arms,stop_utc=None)))
    config=speed.settings(tmp_path)
    jobs=speed.schedule(config['models'],config['critical_models'])
    assert len(jobs)==450
    assert len({(j['arm'],j['repeat'],speed.point_name(j['point'])) for j in jobs})==450
    assert all(j['repeat']==1 for j in jobs[:90])
    assert all(j['repeat']>1 for j in jobs[90:])
    assert speed.cutoff(tmp_path) is None
    (tmp_path/'study.json').unlink()
    assert speed.cutoff(tmp_path)==speed.STOP


def test_extension_failure_export_uses_its_own_population(tmp_path):
    (tmp_path/'study.json').write_text(json.dumps(dict(models=['oracle','opus_single_core','gemini_single_core'])))
    p=tmp_path/'production';p.mkdir();(p/'protocol.json').write_text('{}')
    failed=p/'runs'/'failed';failed.mkdir(parents=True)
    (failed/'receipt.json').write_text(json.dumps(dict(complete=False,error='incomplete callbacks',
        arm='gemini_single_core',point=speed.points()[0],repeat=1)))
    speed.export(tmp_path)
    result=json.loads((p/'results.json').read_text())
    assert (result['expected'],result['failed'],result['pending'])==(450,1,449)
    assert len(result['summary'])==90
    assert all(row['median'] is None for row in result['summary'])


def test_invalid_extension_identity_fails_before_work(tmp_path):
    import pytest
    for models in (['opus_single_core'],['oracle','oracle'],['oracle','gemini_draft']):
        (tmp_path/'study.json').write_text(json.dumps(dict(models=models)))
        with pytest.raises(ValueError,match='population'):
            speed.settings(tmp_path)
