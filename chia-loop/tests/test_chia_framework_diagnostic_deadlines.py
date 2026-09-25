"""Diagnostic containment, legacy continuation and saved-response recovery."""
import gzip
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_chia_framework_campaign import configuration
from test_chia_framework_evaluation import CONTRACT, LIMITS

from ramulator_chia.framework.config import (
    CampaignConfig, EvaluationResources, SyntheticLimits, preserved_configuration,
)
from ramulator_chia.framework.dram import DramResearch
from ramulator_chia.framework import evaluation as E
from ramulator_chia.framework.identity import digest_json, file_sha256
from ramulator_chia.framework.records import CampaignState, RecordConflict, UnresolvedStep


@pytest.mark.parametrize('value', [None, 0, -1, True, 1.5, '1800'])
def test_diagnostic_deadline_requires_positive_integer(value):
    with pytest.raises(ValueError):
        EvaluationResources(diagnostic_timeout_seconds=value)


@pytest.mark.parametrize('frontend', ['ControllerReplay', 'SyntheticPattern', 'champsim', 'SimpleO3'])
@pytest.mark.parametrize('ordinary,diagnostic', [(None, 1800), (60, 1800), (3600, 900)])
def test_only_diagnostic_simulations_receive_the_deadline(frontend, ordinary, diagnostic):
    raw = configuration().model_dump(mode='json')
    raw['run']['resources'].update(simulation_timeout_seconds=ordinary,
                                   diagnostic_timeout_seconds=diagnostic)
    research = DramResearch.__new__(DramResearch)
    research.configuration = CampaignConfig.model_validate(raw)
    research.hosts, research.runtime, research.root = {}, Path('/runtime'), Path('/campaign')
    research._dispatch = lambda function, *args: args
    case = SimpleNamespace(identity=lambda: {'frontend': frontend})
    args = research.measure(case, 'oracle')
    limits = args[4]
    expected = min(ordinary, diagnostic) if ordinary is not None else diagnostic
    assert limits.timeout_seconds == (expected if frontend in {'ControllerReplay', 'SyntheticPattern'} else ordinary)
    assert research.limits.timeout_seconds == ordinary


def test_legacy_configuration_is_preserved_and_amendment_is_explicit(tmp_path):
    from ramulator_chia.framework.campaign import Campaign
    from test_chia_framework_campaign import FixtureResearch

    config = configuration(iterations=1)
    original = config.model_dump(mode='json')
    del original['run']['resources']['diagnostic_timeout_seconds']
    campaign = Campaign(tmp_path, config, FixtureResearch(tmp_path, config))
    campaign.state.save('configuration', original)
    campaign._save_configuration()
    assert campaign.state.get('configuration') == original
    assert campaign.state.get('diagnostic-policy')['legacy_configuration_amendment']
    assert campaign.state.get('diagnostic-policy')['effective_timeout_seconds'] == 1800
    changed = config.model_dump(mode='json')
    changed['run']['maximum_iterations'] += 1
    with pytest.raises(ValueError, match='configuration changed'):
        preserved_configuration(CampaignConfig.model_validate(changed), original)


def test_verified_saved_result_is_not_reinferred_or_double_counted(tmp_path):
    state = CampaignState(tmp_path / 'campaign.sqlite')
    inputs = {'round': 1, 'model': 'fixture'}
    receipt = {'success': True, 'text': 'already answered', 'usage': [{'tokens': 100}]}
    calls = []

    def interrupted():
        calls.append('provider')
        raise RuntimeError('response saved, then cleanup interrupted')

    with pytest.raises(RuntimeError):
        state.step('iteration:1:explore', inputs, interrupted,
                   maximum_attempts=3, retry_delay_seconds=0)
    with pytest.raises(UnresolvedStep):
        state.step('iteration:1:explore', inputs, interrupted,
                   maximum_attempts=3, retry_delay_seconds=0)
    with pytest.raises(RecordConflict):
        state.reconcile_saved_result('iteration:1:explore', 'wrong', receipt, {'verified': True})
    state.reconcile_saved_result('iteration:1:explore', digest_json(inputs), receipt, {'verified': True})
    assert state.step('iteration:1:explore', inputs, interrupted,
                      maximum_attempts=3, retry_delay_seconds=0) == receipt
    assert calls == ['provider']
    with pytest.raises(RecordConflict):
        state.reconcile_saved_result('iteration:1:explore', digest_json(inputs),
                                     {**receipt, 'text': 'reroll'}, {'verified': True})


def test_timed_out_job_releases_slot_and_preserves_late_feedback(tmp_path):
    from test_chia_framework_tools import make_tool
    from test_chia_framework_workspace import workspace
    from ramulator_chia.framework.workspace import recorded

    view = workspace(tmp_path)
    failure = {'message': 'diagnostic exceeded its runtime limit', 'reason': 'runtime_limit',
               'diagnostic': 'open_loop', 'training_case': 'train', 'deadline_seconds': 1}

    @recorded
    def failing_replay(self):
        raise E.MeasurementFailed({'error': json.dumps(failure), 'diagnostic_failure': failure})

    tool, _ = make_tool()
    assert tool._job_start(lambda: failing_replay(view))['started']
    terminal = tool._job_status(2)
    assert not terminal['running'] and terminal['job_status'] == 'failed'
    assert view.diagnostic_failures() == [failure]
    assert tool._job_start(lambda: {'new_draft': True})['started']
    assert tool._job_status(2)['job_status'] == 'complete'


def native_runtime():
    value = os.environ.get('CHIA_NATIVE_TEST_RUNTIME')
    if not value:
        pytest.skip('native diagnostics are opt-in')
    return Path(value)


def diagnostic_case(tmp_path, kind):
    from ramulator_chia.framework.archive import describe_payload
    from ramulator_chia.framework.diagnostic_cases import ReplayCase, SyntheticCase, SyntheticRequest

    if kind == 'synthetic':
        return SyntheticCase.create(SyntheticRequest(num_requests=128), SyntheticLimits())
    path = tmp_path / 'arrivals.csv.gz'
    path.write_bytes(gzip.compress(('arrive,addr,type,source\n' +
                                   ''.join(f'0,{i*64},0,0\n' for i in range(128))).encode()))
    return ReplayCase('training-fixture', describe_payload(path, 'arrivals.csv.gz', codec='gzip'),
                      {'sha256': '0'*64}, 1, 128, 0)


def diagnostic_candidate(runtime, tmp_path, body):
    from ramulator_chia.framework.candidate import BuildLimits, compile_snapshot
    from ramulator_chia.framework.snapshots import snapshot

    draft = tmp_path / 'draft'
    draft.mkdir()
    source = (runtime / 'runtime-source/tools/chia_loop/model/seed.cpp').read_text()
    (draft / 'model.cpp').write_text(source.replace('return now + latency;', body))
    (draft / 'parameters.json').write_text('{}')
    submitted = snapshot(draft, tmp_path / 'snapshots', CONTRACT, maximum_bytes=LIMITS.source_bytes)
    build = tmp_path / 'build'
    compile_snapshot(runtime, tmp_path / 'snapshots', submitted['candidate_id'], CONTRACT, build,
                     limits=BuildLimits(1, 60, LIMITS.source_bytes, LIMITS.memory_bytes, LIMITS.file_bytes))
    return E.CandidateBuild(tmp_path / 'snapshots', submitted['candidate_id'], CONTRACT,
                            build, file_sha256(build / 'build.json'))


@pytest.mark.parametrize('kind,body', [
    ('synthetic', 'volatile unsigned long counter = 0; for (;;) { ++counter; }'),
    ('replay', 'return now + 1000000000000LL;'),
])
def test_native_pathological_diagnostics_leave_failed_compressed_evidence(tmp_path, kind, body):
    runtime = native_runtime()
    case = diagnostic_case(tmp_path, kind)
    candidate = diagnostic_candidate(runtime, tmp_path, body)
    output = tmp_path / 'measurement'
    with pytest.raises(E.MeasurementFailed) as error:
        E.measure_native(runtime, case, 'candidate', output,
                         limits=replace(LIMITS, timeout_seconds=1), candidate=candidate)
    receipt = error.value.receipt
    assert receipt['complete'] is False
    assert receipt['diagnostic_failure']['deadline_seconds'] == 1
    assert receipt['diagnostic_failure']['elapsed_seconds'] >= 1
    assert receipt['diagnostic_failure']['candidate_id'] == candidate.candidate_id
    assert receipt['diagnostic_failure']['diagnostic'] == ('open_loop' if kind == 'replay' else kind)
    assert 'diagnostic exceeded its runtime limit' in receipt['error']
    assert (output / 'measurement.json').is_file()
    assert not list((output / 'observations').glob('*.ch0'))
    for trace in receipt['traces'].values():
        assert file_sha256(output / trace['name']) == trace['stored_sha256']
        with gzip.open(output / trace['name'], 'rb') as stream:
            stream.read()
    with pytest.raises(ValueError, match='only complete'):
        E.verify_native_measurement(output, file_sha256(output / 'measurement.json'),
                                    frontend=case.identity()['frontend'], observations=case.observation_names)


@pytest.mark.parametrize('kind', ['replay', 'synthetic'])
def test_native_normal_diagnostics_are_identical_with_or_without_deadline(tmp_path, kind):
    runtime = native_runtime()
    case = diagnostic_case(tmp_path, kind)
    candidate = diagnostic_candidate(runtime, tmp_path, 'return now + latency;')
    results = []
    for label, timeout in [('uncapped', None), ('bounded', 30)]:
        result = E.measure_native(runtime, case, 'candidate', tmp_path / label,
                                  limits=replace(LIMITS, timeout_seconds=timeout), candidate=candidate)
        results.append(result)
    assert results[0]['frontend_stats'] == results[1]['frontend_stats']
    assert results[0]['controller_stats'] == results[1]['controller_stats']
    assert {k:v['logical_sha256'] for k,v in results[0]['traces'].items()} == {
        k:v['logical_sha256'] for k,v in results[1]['traces'].items()}
