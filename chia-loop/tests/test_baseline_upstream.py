"""Upstream conformance without network access or a full simulator build.

Set BASELINE_REUSE_RUNTIME to additionally test the actual Ramulator adapter.
The reference MeSS files are reconstructed and hash-checked against upstream,
not taken from the patched production implementation.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from ramulator_chia.eval.baseline_audit import function

from ramulator_chia.layout import RAMULATOR as REPO, PACKAGE


@pytest.fixture(scope='module')
def pristine(tmp_path_factory):
    root = tmp_path_factory.mktemp('upstream-originals')
    for name in ('mess', 'sniper_wmg1'):
        directory = root / name
        shutil.copytree(REPO / 'ext' / name, directory)
        record = json.loads((directory / 'UPSTREAM.json').read_text())
        subprocess.run(['patch', '--batch', '--reverse', '-p1', '-i', 'ramulator.patch'],
                       cwd=directory, check=True, capture_output=True)
        for filename, expected in record['upstream_sha256'].items():
            assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
    return root


def test_prediction_routines_are_upstream_code(pristine):
    original = (pristine / 'mess/mess_mem_ctrl.cpp').read_text()
    current = (REPO / 'ext/mess/mess_mem_ctrl.cpp').read_text()
    # Everything after the anonymous loader namespace is unchanged, including
    # construction, channel/frequency conversion and all feedback arithmetic.
    marker = '} // namespace\n'
    assert current.split(marker, 1)[1] == original.split(marker, 1)[1] + '\n'
    original = (pristine / 'sniper_wmg1/queue_model_windowed_mg1.cc').read_text()
    current = (REPO / 'ext/sniper_wmg1/queue_model_windowed_mg1.cc').read_text()
    for name in ('addItem(', 'removeItems('):
        marker = 'QueueModelWindowedMG1::' + name
        assert function(current, marker) == function(original, marker)
    marker = 'QueueModelWindowedMG1::computeQueueDelay('
    # Only the simulator-specific clock/window-advancement lines differ.
    assert function(current, marker).split('   if (m_num_arrivals > 1)', 1)[1] == \
           function(original, marker).split('   if (m_num_arrivals > 1)', 1)[1]


@pytest.fixture(scope='module')
def mess_binaries(pristine, tmp_path_factory):
    root = tmp_path_factory.mktemp('mess-conformance')
    source = root / 'driver.cpp'
    source.write_text(r'''
#include "mess_mem_ctrl.h"
#include <iostream>
int main(int argc, char** argv) {
  try {
    MessMemCtrl model(argv[1], std::stoul(argv[2]), std::stod(argv[3]), 1);
    uint64_t time; int write;
    while (std::cin >> time >> write) std::cout << model.access(time, write) << '\n';
  } catch (const std::exception& e) { std::cerr << e.what(); return 2; }
}
''')
    binaries = {}
    for name, sources in [('upstream', pristine/'mess'), ('production', REPO/'ext/mess')]:
        binary = root/name
        subprocess.run(['c++', '-O3', '-DNDEBUG', '-std=c++20', '-I'+str(sources),
                        str(source), str(sources/'mess_mem_ctrl.cpp'), '-o', str(binary)],
                       check=True, capture_output=True)
        binaries[name] = binary
    return binaries


@pytest.fixture
def curves(tmp_path):
    legacy = PACKAGE/'eval/calibration/mess_DDR5.txt'
    entries = {}
    for line in legacy.read_text().splitlines():
        pct, bw, latency = map(float, line.split())
        entries.setdefault(str(int(pct)), []).append([bw*1000, latency])
    for points in entries.values():
        points.sort(reverse=True)
    native = tmp_path/'curve.json'
    native.write_text(json.dumps({'measuredChannels': 1, 'curves': entries}))
    return legacy, native


@pytest.mark.parametrize('window', [1, 7, 1000])
@pytest.mark.parametrize('frequency', [1000/416, 1.6])
def test_mess_legacy_json_and_unmodified_upstream_match(mess_binaries, curves, window, frequency):
    time = 0
    requests = []
    for i in range(12_000):
        time += (0, 1, 1000, 2, 500, 8)[i//2000]
        if i in (2000, 4000, 6000): time += 100_000
        requests.append(f'{time} {int(i % 5 < (i//2000))}\n')
    stream = ''.join(requests)
    expected = subprocess.run([str(mess_binaries['upstream']), str(curves[1]),
                               str(window), str(frequency)], input=stream,
                              capture_output=True, text=True, check=True).stdout
    assert len(expected.splitlines()) == 12_000
    for curve in curves:
        actual = subprocess.run([str(mess_binaries['production']), str(curve),
                                 str(window), str(frequency)], input=stream,
                                capture_output=True, text=True, check=True).stdout
        assert actual == expected


@pytest.mark.parametrize('contents', [
    '', '50 1 20 trailing\n', '50 -1 20\n', '101 1 20\n', '50 1 -1\n',
    '50 1 20\n50 1 21\n', '50 1e309 20\n',
    '{"measuredChannels":0,"curves":{"50":[[1000,20]]}}',
    '{"measuredChannels":1.5,"curves":{"50":[[1000,20]]}}',
    '{"measuredChannels":1,"curves":{"50":[]}}',
    '{"measuredChannels":1,"curves":{"50":[[1,20],[2,30]]}}',
    '{"measuredChannels":1,"curves":{"50":[[1000,20]]}} trailing',
])
def test_mess_rejects_bad_calibration(mess_binaries, tmp_path, contents):
    curve = tmp_path/'bad.txt'
    curve.write_text(contents)
    run = subprocess.run([str(mess_binaries['production']), str(curve), '1000', '2.4'],
                         capture_output=True, text=True)
    assert run.returncode == 2
    assert run.stderr


@pytest.fixture(scope='module')
def wmg1_binaries(pristine, tmp_path_factory):
    """Compile the untouched upstream .cc/.h with clock/config/stat stubs.

    Unsigned femtosecond storage and integer-PS conversion mirror Sniper time.
    A common positive time offset avoids its unsigned startup subtraction;
    production also runs at zero to exercise our signed-cutoff adaptation.
    """
    root = tmp_path_factory.mktemp('wmg1-conformance')
    stubs = root/'stubs'
    stubs.mkdir()
    (stubs/'fixed_types.h').write_text('''#pragma once
#include <cstdint>
#include <string>
using UInt64=uint64_t; using UInt32=uint32_t;
using core_id_t=int; using String=std::string;
constexpr core_id_t INVALID_CORE_ID=-1;
''')
    (stubs/'queue_model.h').write_text('''#pragma once
#include "fixed_types.h"
#include <compare>
#include <algorithm>
struct SubsecondTime {
  uint64_t fs=0;
  static SubsecondTime PS(uint64_t p) {return {p*1000};}
  static SubsecondTime NS(uint64_t n) {return PS(n*1000);}
  static SubsecondTime Zero() {return {};}
  uint64_t getPS() const {return fs/1000;}
  auto operator<=>(const SubsecondTime&) const=default;
  SubsecondTime operator-(SubsecondTime b) const {return {fs-b.fs};}
  SubsecondTime& operator+=(SubsecondTime b) {fs+=b.fs;return *this;}
};
inline SubsecondTime operator*(int a,SubsecondTime b) {return {a*b.fs};}
struct QueueModel {};
''')
    (stubs/'simulator.h').write_text('''#pragma once
#include "queue_model.h"
struct Simulation {
  uint64_t window_ns=0;
  SubsecondTime now;
  Simulation* getCfg(){return this;}
  uint64_t getInt(const char*){return window_ns;}
  Simulation* getClockSkewMinimizationServer(){return this;}
  SubsecondTime getGlobalTime(){return now;}
};
inline Simulation simulation;
inline Simulation* Sim(){return &simulation;}
''')
    (stubs/'stats.h').write_text('template<class... T> void registerStatsMetric(T...) {}\n')
    for name in ('config.hpp','log.h','contention_model.h'):
        (stubs/name).write_text('')
    source = root/'driver.cpp'
    source.write_text(r'''
#include "queue_model_windowed_mg1.h"
#include <iostream>
#ifdef REFERENCE
#include "simulator.h"
#else
using namespace Ramulator::Sniper;
#endif
int main(int argc,char** argv) {
  const uint64_t window_ns=std::stoull(argv[1]), offset=std::stoull(argv[2]);
#ifdef REFERENCE
  simulation.window_ns=window_ns;
  QueueModelWindowedMG1 model("reference",0);
#else
  QueueModelWindowedMG1 model(SubsecondTime::PS(window_ns*1000));
#endif
  uint64_t t,service;
  while(std::cin>>t>>service) {
    auto time=SubsecondTime::PS(t+offset);
#ifdef REFERENCE
    simulation.now=time;
#endif
    std::cout<<model.computeQueueDelay(time,SubsecondTime::PS(service)).getPS()<<'\n';
  }
}
''')
    binaries = {}
    for name, sources in [('upstream',pristine/'sniper_wmg1'),
                          ('production',REPO/'ext/sniper_wmg1')]:
        binary = root/name
        subprocess.run(['c++','-O3','-DNDEBUG','-std=c++20',
                        *(['-DREFERENCE'] if name=='upstream' else []),
                        '-I'+str(sources),'-I'+str(stubs),str(source),
                        str(sources/'queue_model_windowed_mg1.cc'),'-o',str(binary)],
                       check=True,capture_output=True)
        binaries[name] = binary
    return binaries


@pytest.mark.parametrize('window_ns',[1,2,10,10000])
def test_wmg1_exact_upstream_moments_expiry_and_rounding(wmg1_binaries, window_ns):
    import random
    rng = random.Random(731)
    window = window_ns*1000
    time = 0
    requests = []
    for i in range(12000):
        # Include the expiration edge itself and its immediate neighbors,
        # startup, simultaneous packets, saturation/recovery and nonintegral
        # second-moment averages. Compare PS before conversion to DRAM ticks.
        time += rng.choice([0,1,416,window-1,window,window+1])
        service = rng.choice([1,415,416,417,3328,3329])
        requests.append(f'{time} {service}\n')
    stream = ''.join(requests)
    def run(name, offset):
        return subprocess.run([str(wmg1_binaries[name]),str(window_ns),str(offset)],
                              input=stream,text=True,capture_output=True,check=True).stdout
    expected = run('upstream',20*window)
    assert len(expected.splitlines()) == 12000
    assert run('production',20*window) == expected
    assert run('production',0) == expected


@pytest.fixture(scope='module')
def native_driver(tmp_path_factory):
    setting = os.environ.get('BASELINE_REUSE_RUNTIME')
    if not setting: pytest.skip('set BASELINE_REUSE_RUNTIME for actual adapter tests')
    runtime = Path(setting)
    source = runtime/'runtime-source'
    binary = tmp_path_factory.mktemp('native-adapter')/'driver'
    subprocess.run(['c++','-O3','-DNDEBUG','-std=c++20',
                        str(Path(__file__).parent/'utils/baseline_trace_driver.cpp'),
                    '-I'+str(source/'src'),'-I'+str(source/'ext/fmt/include'),
                    '-L'+str(runtime/'runtime'),'-Wl,-rpath,'+str(runtime/'runtime'),
                    '-lramulator','-o',str(binary)],check=True,capture_output=True)
    return binary


def native_config(curve, **overrides):
    import ramulator
    config = ramulator.controller.Mess(
        dram=ramulator.dram.DDR5(org_preset='DDR5_16Gb_x8',timing_preset='DDR5_4800AN'),
        curve_path=str(curve),**overrides).to_config()
    return {'controller':config}


@pytest.mark.parametrize('overrides', [{'converge':0.1},{'window_accesses':0},
                                       {'transaction_bytes':32}])
def test_native_adapter_rejects_unsupported_settings(native_driver, curves, tmp_path, overrides):
    bad_tx = overrides.get('transaction_bytes')
    cfg = native_config(curves[0], **({} if bad_tx else overrides))
    if bad_tx: cfg['controller']['dram']['channel_width'] = 16
    path = tmp_path/'config.json'
    path.write_text(json.dumps(cfg))
    run = subprocess.run([str(native_driver), str(path), '-'], input='',
                         capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode != 0
    assert 'Mess:' in run.stderr


@pytest.mark.parametrize('model', ['mess-text','mess-json','wmg1'])
def test_native_reset_preserves_predictor_and_completion_state(native_driver, curves, tmp_path, model):
    path = tmp_path/'config.json'
    if model=='wmg1':
        import ramulator
        cfg = {'controller':ramulator.controller.WMG1(
            dram=ramulator.dram.DDR5(org_preset='DDR5_16Gb_x8',timing_preset='DDR5_4800AN'),
            window_ns=2).to_config()}
    else:
        cfg = native_config(curves[int(model=='mess-json')], window_accesses=7)
    path.write_text(json.dumps(cfg))
    outputs = []
    for reset in (False, True):
        stream = ''.join(f'{i//4} {int(i%5==0)} {int(reset and i%19==0)}\n' for i in range(4000))
        run = subprocess.run([str(native_driver), str(path), '-'], input=stream,
                             capture_output=True, text=True, check=True)
        outputs.append(run.stdout)
    assert outputs[0] == outputs[1]


def test_vendored_sources_and_notices_are_staged():
    from ramulator_chia.framework.build import _source_files
    from ramulator_chia.layout import build_key
    inventory = {build_key(REPO,p) for p in _source_files(REPO)}
    for name in ('mess','sniper_wmg1'):
        for filename in ('LICENSE','UPSTREAM.json','ramulator.patch'):
            assert f'ext/{name}/{filename}' in inventory


def test_vendored_sources_survive_source_archive(tmp_path, monkeypatch):
    from ramulator_chia.framework import archive, export
    from ramulator_chia.framework.identity import file_sha256
    # A tiny compiled-source inventory exercises the real exporter path rather
    # than merely assuming .json, .patch and LICENSE survive suffix filtering.
    runtime = tmp_path/'runtime'
    files = {}
    for name in ('mess','sniper_wmg1'):
        for source in sorted((REPO/'ext'/name).iterdir()):
            path = source.relative_to(REPO)
            target = runtime/'runtime-source'/path
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            files[str(path)] = file_sha256(target)
    (runtime/'runtime_manifest.json').write_text(json.dumps({'source_inventory':files}))
    monkeypatch.setattr(export,'HARNESS_TREES',())
    monkeypatch.setattr(export,'HARNESS_FILES',())
    captured = export.capture_sources(REPO,runtime,tmp_path/'source')
    assert captured['files']==files
    payloads = [archive.describe_payload(tmp_path/'source'/n,'source/'+n) for n in files]
    sealed = archive.seal(payloads,tmp_path/'archives',metadata={'runtime_binaries_included':False})
    checked = archive.verify(sealed.path,expected_sha256=sealed.sha256)
    assert {m['name'] for m in checked['manifest']['members']} == {'source/'+n for n in files}
