"""Offline checks for separately qualified, host-matched speed evidence."""
from copy import deepcopy
import itertools
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import campaign_results_notebook as notebook
import campaign_results_plots as plots


def fixture():
    arms = ['oracle', 'opus_single_core', 'gemini_single_core']
    protocol = dict(models=arms, geometry={'bytes': 8 << 30, 'banks': 32},
        input_format='RMSPD001', intervals=[1, 4, 16, 64, 256],
        mapping='RoBaRaCoCh physical bytes', measured_requests=5_000_000,
        read_percentages=[100, 75, 50], refresh=False, repetitions=5,
        rng='fixed independent streams', seed=12345, skip='guaranteed idle',
        time_boundary='measurement plus drain', timeout_seconds=1800,
        type_seed=1415143548, unit='requests per host second',
        warmup_requests=100_000, writes='native acknowledgements', timed_runs=450,
        inputs=[dict(path='input.bin', pattern='streaming', read_percent=100,
                     sha256='frozen-input', requests=5_100_000)])
    candidates = {arm: {'candidate_id': arm, 'parameters': {'frozen': 1},
                       'files': {'model.cpp': {'sha256': arm + '-source'},
                                 'parameters.json': {'sha256': arm + '-params'}}}
                  for arm in arms[1:]}
    qualification = dict(passed=True, models=[dict(arm=arm, passed=True) for arm in arms])
    runtime = {'binaries': {'isolated_sim': 'binary', 'libramulator.so': 'library'}}
    study = dict(protocol=deepcopy(protocol), successful=90, expected=450, failed=0,
        pending=360, summary=[], provenance=dict(
            models={arm: {'candidate': deepcopy(value)} for arm, value in candidates.items()},
            runtime=deepcopy(runtime), qualification=deepcopy(qualification),
            cloud_qualification=deepcopy(qualification)))
    rates = {'oracle': 100_000, 'opus_single_core': 200_000, 'gemini_single_core': 400_000}
    for arm, pattern, ratio, interval in itertools.product(
            arms, ['streaming', 'random'], [100, 75, 50], [1, 4, 16, 64, 256]):
        trial = dict(repeat=1, requests_per_second=rates[arm],
            wall_seconds=5_000_000 / rates[arm], measured_reads=5_000_000,
            measured_writes=0, measured_reads_completed=5_000_000,
            measured_writes_completed=0, simulated_cycles=5_200_000,
            measured_simulated_cycles=5_000_000, drain_simulated_cycles=100,
            admission_wait_sum=0, admission_wait_max=0,
            warmup_reads_outstanding=1, warmup_writes_outstanding=0)
        study['summary'].append(dict(arm=arm, pattern=pattern, read_percent=ratio,
            interval=interval, n=1, repetitions=[trial], median=rates[arm],
            mean=rates[arm], standard_deviation=None,
            oracle_relative_speedup=rates[arm] / rates['oracle']))
    original = dict(protocol=deepcopy(protocol), provenance={'runtime': deepcopy(runtime)})
    data = dict(dram_speed=original, sources={arm: {'candidate': deepcopy(value)}
                                            for arm, value in candidates.items()})
    return study, data


class SpeedAdditionTests(unittest.TestCase):
    def test_valid_separate_study_leaves_original_unchanged(self):
        study, data = fixture()
        before = deepcopy(data)
        notebook.validate_speed_addition(study, data)
        self.assertEqual(data, before)

    def test_changed_model_parameter_runtime_input_and_protocol_are_rejected(self):
        changes = [
            (lambda s: s['provenance']['models']['opus_single_core']['candidate'].update(
                candidate_id='draft'), 'frozen selection'),
            (lambda s: s['provenance']['models']['gemini_single_core']['candidate']['parameters'].update(
                frozen=2), 'frozen selection'),
            (lambda s: s['provenance']['runtime']['binaries'].update(isolated_sim='changed'), 'runtime'),
            (lambda s: s['protocol']['inputs'][0].update(sha256='other-input'), 'input'),
            (lambda s: s['protocol'].update(measured_requests=1_000), 'protocol'),
            (lambda s: s['protocol']['geometry'].update(banks=16), 'protocol'),
            (lambda s: s['provenance']['cloud_qualification'].update(passed=False), 'unqualified'),
            (lambda s: s['summary'].pop(), 'inventory'),
            (lambda s: s['summary'][0].update(n=2), 'repetition'),
            (lambda s: s['summary'][0]['repetitions'][0].update(measured_reads_completed=4_999_999),
             'callback'),
            (lambda s: s['summary'][0]['repetitions'][0].update(wall_seconds=123), 'wall time'),
            (lambda s: s['summary'][30].update(oracle_relative_speedup=99), 'this study'),
            (lambda s: s['summary'][0].update(mean=1), 'mean'),
            (lambda s: s['summary'][0].update(standard_deviation=1), 'standard deviation'),
        ]
        for change, message in changes:
            with self.subTest(message=message):
                study, data = fixture()
                change(study)
                with self.assertRaisesRegex(ValueError, message):
                    notebook.validate_speed_addition(study, data)

    def test_simulated_counters_must_repeat_exactly(self):
        study, data = fixture()
        row = study['summary'][0]
        second = deepcopy(row['repetitions'][0])
        second.update(repeat=2, simulated_cycles=second['simulated_cycles'] + 1)
        row['repetitions'].append(second)
        row.update(n=2, standard_deviation=0.0)
        study.update(successful=91, pending=359)
        with self.assertRaisesRegex(ValueError, 'differ across repetitions'):
            notebook.validate_speed_addition(study, data)

    def test_missing_original_is_not_silently_replaced(self):
        study, data = fixture()
        data.pop('dram_speed')
        with self.assertRaisesRegex(ValueError, 'original protocol'):
            notebook.validate_speed_addition(study, data)

    def test_failed_point_is_missing_not_zero(self):
        study, data = fixture()
        study['summary'][30].update(n=0, repetitions=[], median=None, mean=None,
                                    oracle_relative_speedup=None)
        study.update(successful=89, failed=1)
        notebook.validate_speed_addition(study, data)
        self.assertIsNone(study['summary'][30]['median'])

    def test_input_location_is_not_a_scientific_difference(self):
        study, data = fixture()
        study['protocol']['inputs'][0]['path'] = '/new/study/input.bin'
        notebook.validate_speed_addition(study, data)

    def test_separate_three_model_plots_keep_export_names_and_oracle_isolated(self):
        study, data = fixture()
        data['dram_speed_opus_gemini'] = study
        tables, figures = {}, {}
        with patch.object(plots, 'DATA', data, create=True), \
             patch.object(plots, 'table', lambda frame, name, **kw: tables.update({name: frame.copy()})), \
             patch.object(plots, 'display', lambda *a: None), \
             patch.object(plots, 'finish', lambda fig, name: figures.update({name: fig})):
            try:
                plots.standalone_speed('dram_speed_opus_gemini', 'dram_speed_opus_gemini')
                self.assertEqual(set(figures), {'dram_speed_opus_gemini_throughput',
                                               'dram_speed_opus_gemini_speedup'})
                self.assertTrue(all(name.startswith('dram_speed_opus_gemini_') for name in tables))
                coverage = tables['dram_speed_opus_gemini_coverage'].set_index('arm')
                self.assertAlmostEqual(coverage.loc['opus_single_core', 'geomean_speedup'], 2)
                self.assertAlmostEqual(coverage.loc['gemini_single_core', 'geomean_speedup'], 4)
                self.assertEqual(coverage.shared_points.tolist(), [30, 30, 30])
                self.assertEqual(len(tables['dram_speed_opus_gemini_repetitions']), 90)
                for fig in figures.values():
                    self.assertEqual(len(fig.axes), 6)
                    self.assertTrue(all(len(ax.lines) == 3 for ax in fig.axes))
                    fig.canvas.draw()
            finally:
                for fig in figures.values():
                    plt.close(fig)


if __name__ == '__main__':
    unittest.main()
