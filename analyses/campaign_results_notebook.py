"""Compatibility entry point for the paper-only notebook.

The historical speed validator remains available to existing evidence tests.
The old exploratory notebook is preserved in Git history, not generated here.
"""
import itertools
import math
import statistics

from paper_notebook import build, main


def validate_speed_addition(study, data):
    """Bind the separately timed SC study to frozen sources and the same traffic."""
    reference = data.get('dram_speed')
    if not reference:
        raise ValueError('supplementary speed study requires the original protocol reference')
    protocol = study['protocol']
    arms = ['oracle', 'opus_single_core', 'gemini_single_core']
    if protocol['models'] != arms:
        raise ValueError('unexpected supplementary speed models or ordering')
    for key in ('geometry', 'input_format', 'intervals', 'mapping', 'measured_requests',
                'read_percentages', 'refresh', 'repetitions', 'rng', 'seed', 'skip',
                'time_boundary', 'timeout_seconds', 'type_seed', 'unit',
                'warmup_requests', 'writes'):
        if protocol[key] != reference['protocol'][key]:
            raise ValueError('speed protocol differs: ' + key)
    def inputs(value):
        return sorted(({k: v for k, v in row.items() if k != 'path'}
                       for row in value['inputs']),
                      key=lambda row: (row['pattern'], row['read_percent']))
    if inputs(protocol) != inputs(reference['protocol']):
        raise ValueError('speed input identities differ')
    provenance = study['provenance']
    if provenance['runtime']['binaries'] != reference['provenance']['runtime']['binaries']:
        raise ValueError('speed runtime binaries differ')
    for arm in arms[1:]:
        if provenance['models'][arm]['candidate'] != data['sources'][arm]['candidate']:
            raise ValueError('speed model is not the frozen selection: ' + arm)
    for key in ('qualification', 'cloud_qualification'):
        qualification = provenance[key]
        passed = {row['arm'] for row in qualification['models'] if row['passed'] is True}
        if qualification.get('passed') is not True or passed != set(arms):
            raise ValueError('unqualified speed study: ' + key)
    keys = set(itertools.product(arms, ('streaming', 'random'),
                                 protocol['read_percentages'], protocol['intervals']))
    rows = study['summary']
    row_keys = [(row['arm'], row['pattern'], row['read_percent'], row['interval'])
                for row in rows]
    if len(rows) != len(keys) or set(row_keys) != keys:
        raise ValueError('incomplete or duplicate speed point inventory')
    expected = len(keys) * protocol['repetitions']
    if study['expected'] != expected or protocol['timed_runs'] != expected:
        raise ValueError('incorrect speed matrix size')
    successful = 0
    medians = {}
    counter_fields = ('simulated_cycles', 'measured_simulated_cycles',
        'drain_simulated_cycles', 'admission_wait_sum', 'admission_wait_max',
        'warmup_reads_outstanding', 'warmup_writes_outstanding',
        'measured_reads', 'measured_writes',
        'measured_reads_completed', 'measured_writes_completed')
    for key, row in zip(row_keys, rows):
        trials = row['repetitions']
        repeats = [trial['repeat'] for trial in trials]
        if (row['n'] != len(trials) or len(repeats) != len(set(repeats))
                or not set(repeats) <= set(range(1, protocol['repetitions'] + 1))):
            raise ValueError('invalid speed repetition coverage')
        successful += row['n']
        first_counters = None
        for trial in trials:
            rate, wall = trial['requests_per_second'], trial['wall_seconds']
            if not all(math.isfinite(v) and v > 0 for v in (rate, wall)):
                raise ValueError('invalid speed timing')
            if not math.isclose(rate, protocol['measured_requests'] / wall, rel_tol=1e-10):
                raise ValueError('speed throughput disagrees with measured wall time')
            if (trial['measured_reads'] + trial['measured_writes'] != protocol['measured_requests']
                    or trial['measured_reads_completed'] != trial['measured_reads']
                    or trial['measured_writes_completed'] != trial['measured_writes']):
                raise ValueError('incomplete speed callback population')
            counters = tuple(trial[name] for name in counter_fields)
            if any(type(value) is not int or value < 0 for value in counters):
                raise ValueError('invalid simulated speed counters')
            if first_counters is not None and counters != first_counters:
                raise ValueError('simulated speed counters differ across repetitions')
            first_counters = counters
        median = statistics.median(t['requests_per_second'] for t in trials) if trials else None
        if row['median'] != median:
            raise ValueError('speed median disagrees with repetitions')
        values = [t['requests_per_second'] for t in trials]
        if row['mean'] != (statistics.mean(values) if values else None):
            raise ValueError('speed mean disagrees with repetitions')
        if row['standard_deviation'] != (statistics.stdev(values) if len(values) > 1 else None):
            raise ValueError('speed standard deviation disagrees with repetitions')
        medians[key] = median
    if (successful != study['successful'] or min(study['failed'], study['pending']) < 0
            or successful + study['failed'] + study['pending'] != expected):
        raise ValueError('speed coverage counters disagree')
    for key, row in zip(row_keys, rows):
        oracle = medians[('oracle', *key[1:])]
        expected_speedup = medians[key] / oracle if oracle and medians[key] else None
        if ((expected_speedup is None) != (row['oracle_relative_speedup'] is None)
                or expected_speedup is not None and not math.isclose(
                    expected_speedup, row['oracle_relative_speedup'], rel_tol=1e-10)):
            raise ValueError('speedup is not matched to this study\'s oracle')



if __name__ == "__main__":
    main()
