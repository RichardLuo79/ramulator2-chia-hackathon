#!/usr/bin/env python3
"""Small operator entry points; no cloud provisioning or implicit model calls."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tarfile
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'chia-loop/src'), str(ROOT/'ramulator/python'), str(ROOT/'ramulator')]

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def run(argv, **kwargs):
    subprocess.run([str(v) for v in argv], check=True, **kwargs)

def load_config(path, data_root, work_root, vertex_project=None):
    from ramulator_chia.framework.config import CampaignConfig
    text=Path(path).read_text().replace('${DATA_ROOT}',str(Path(data_root).resolve()))
    text=text.replace('${WORK_ROOT}',str(Path(work_root).resolve()))
    text=text.replace('${ARTIFACT_ROOT}',str(ROOT))
    if '${VERTEX_PROJECT}' in text:
        if not vertex_project:raise SystemExit(f'{path} names no Vertex project; pass --vertex-execution-project')
        text=text.replace('${VERTEX_PROJECT}',vertex_project)
    return CampaignConfig.model_validate_json(text)

def load_compaction(path):
    from ramulator_chia.framework.config import ContextCompaction
    return ContextCompaction.model_validate_json(Path(path).read_text().replace('${ARTIFACT_ROOT}',str(ROOT)))

def setup(args):
    parser=argparse.ArgumentParser(description='Install a Python environment (analysis, campaign, context) or build '
                                   'the native runtime, ChampSim or the speed-measurement inputs.')
    parser.add_argument('--component', choices=['analysis','campaign','context','runtime','champsim','speed-inputs'],default='analysis')
    parser.add_argument('--venv',type=Path,help='environment directory (default: .venv-<component>)')
    parser.add_argument('--output',type=Path,default=ROOT/'.work/runtime',help='where to build the runtime')
    parser.add_argument('--workers',type=int,default=8,help='parallel build jobs (1-16)')
    parser.add_argument('--cores',type=int,choices=[1,4,8],default=1,help='ChampSim core count')
    parser.add_argument('--work-root',type=Path,default=ROOT/'.work',help='where to build ChampSim')
    parser.add_argument('--runtime',type=Path,default=ROOT/'.work/runtime',help='the built runtime')
    a=parser.parse_args(args)
    if a.workers<1 or a.workers>16:parser.error('build workers must be in [1,16]')
    if a.component=='champsim':
        from build_frontend import build
        print(build(ROOT,a.work_root.resolve(),a.runtime.resolve(),a.cores,a.workers));return
    if a.component=='speed-inputs':
        run(['cmake','--build',a.runtime/'runtime-build','--target','ramulator-speed-inputs','--parallel',a.workers]);return
    if a.component=='runtime':
        from ramulator_chia.framework.build import prepare_runtime
        result=prepare_runtime(ROOT/'ramulator',a.output,cpus=a.workers,timeout_seconds=3600)
        print(json.dumps({'runtime':str(a.output),'optimization':result['optimization']})); return
    target=a.venv or ROOT/('.venv-'+a.component)
    if not (target/'bin/python').exists():venv.EnvBuilder(with_pip=True).create(target)
    python=target/'bin/python'
    requirements=ROOT/{'analysis':'analyses/requirements.lock.txt','campaign':'chia-loop/requirements.lock.txt',
                       'context':'chia-loop/requirements.context.lock.txt'}[a.component]
    run([python,'-m','pip','install','-r',requirements])
    if a.component=='campaign':
        run([python,'-m','pip','install','--no-deps','--no-build-isolation',
             '-e',ROOT/'chia-loop/vendor/chia','-e',ROOT/'chia-loop','-e',ROOT/'ramulator'])
    print('Prepared',target)

def fetch_data(args):
    parser=argparse.ArgumentParser(description='Download the DPC-4 trace prefixes and rebuild the placement maps '
                                   '(--download), or verify the inputs in the data root.')
    parser.add_argument('--manifest',type=Path,default=ROOT/'results/manifests/inputs.json',help='files to check')
    parser.add_argument('--data-root',type=Path,default=ROOT/'.data',help='input directory')
    parser.add_argument('--archive-dir',type=Path,help='directory holding archive files listed in the manifest')
    parser.add_argument('--download',action='store_true',help='download the traces and rebuild the maps')
    parser.add_argument('--config',type=Path,default=ROOT/'chia-loop/configs/astra.json',
                        help='configuration whose workloads to fetch')
    parser.add_argument('--work-root',type=Path,default=ROOT/'.work',help='where ChampSim is built')
    parser.add_argument('--workers',type=int,default=2,help='parallel downloads (1-4)')
    parser.add_argument('--disk-reserve-gib',type=float,default=32,help='free disk space to keep, in GiB')
    parser.add_argument('--extract',type=Path,help='check an archive and extract it into a new directory')
    parser.add_argument('--destination',type=Path,help='directory for --extract')
    parser.add_argument('--sha256',help='expected SHA-256 of the archive for --extract')
    a=parser.parse_args(args)
    if a.extract:
        if not a.destination or not a.sha256:parser.error('--extract requires --destination and --sha256')
        from ramulator_chia.framework.archive import extract
        print(extract(a.extract,a.destination,expected_sha256=a.sha256));return
    if a.download:
        from ramulator_chia.input_data import prepare
        receipt=prepare(load_config(a.config,a.data_root,a.work_root),a.data_root,a.work_root,
                        workers=a.workers,reserve_gib=a.disk_reserve_gib)
        print(json.dumps({'verified':len(receipt['files']),'receipt':str(a.data_root/'prepared-inputs.json')}))
        return
    manifest=json.loads(a.manifest.read_text()); missing=[];checked=0
    # Rebuilt files are recompressed locally; their receipt binds them to the published decoded identity.
    receipt=a.data_root/'prepared-inputs.json'
    prepared=json.loads(receipt.read_text())['files'] if receipt.exists() else {}
    for item in manifest['files']:
        base=a.archive_dir if item.get('kind')=='archive' and a.archive_dir else a.data_root
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe manifest path')
        path=base/relative
        if not path.exists():missing.append(item['path']);continue
        stored=item['sha256']
        if item['path'] in prepared:
            row=prepared[item['path']]
            if row['reference_sha256']!=stored or row['decoded_sha256']!=item.get('decoded_sha256'):
                raise ValueError('preparation identity mismatch: '+item['path'])
            stored=row['sha256']
        if path.is_symlink() or sha(path)!=stored:raise ValueError('checksum mismatch: '+item['path'])
        checked+=1
    print(json.dumps({'verified':checked,'missing':missing},indent=2))
    if missing:raise SystemExit(2)

def reproduce_figures(args):
    parser=argparse.ArgumentParser(description='Run analyses/paper_results.ipynb, which writes Figs. 2-9 to '
                                   'results/figures/ and Tables 4-5 to results/tables/.')
    parser.add_argument('--notebook',type=Path,default=ROOT/'analyses/paper_results.ipynb')
    a=parser.parse_args(args)
    import nbformat
    from nbclient import NotebookClient
    nb=nbformat.read(a.notebook,as_version=4)
    NotebookClient(nb,timeout=1800,kernel_name='python3',
                   resources={'metadata':{'path':str(a.notebook.parent)}}).execute()
    nbformat.write(nb,a.notebook)
    print(json.dumps({'executed':str(a.notebook),'figures':str(ROOT/'results/figures'),
                      'tables':str(ROOT/'results/tables')},indent=2))

def evaluate(args):
    if '--suite' in args:
        at=args.index('--suite')
        if at+1>=len(args):raise SystemExit('--suite needs speed')
        suite=args[at+1];rest=args[:at]+args[at+2:]
        from ramulator_chia.eval.frozen import diagnostic_cli
        diagnostic_cli(suite,rest,ROOT);return
    parser=argparse.ArgumentParser(description='Run one model or baseline on one ChampSim workload (no LLM). '
                                   'For a speed measurement, use --suite speed.')
    parser.add_argument('--config',type=Path,default=ROOT/'chia-loop/configs/astra.json',
                        help='configuration listing the workloads')
    parser.add_argument('--data-root',type=Path,default=ROOT/'.data',help='input directory')
    parser.add_argument('--work-root',type=Path,default=ROOT/'.work',help='where ChampSim is built')
    parser.add_argument('--runtime',type=Path,default=ROOT/'.work/runtime',help='the built runtime')
    parser.add_argument('--output',type=Path,required=True,help='new output directory')
    parser.add_argument('--case',required=True,help='workload, e.g. validation-c1-bwaves or test-c4_mixed_05')
    parser.add_argument('--organization',choices=['DDR5_16Gb_x8','DDR5_8Gb_x8','DDR5_16Gb_x16'],
                        help='DRAM organization (default: DDR5_16Gb_x8)')
    parser.add_argument('--queue-entries',type=int,choices=[32,64,128],default=64,
                        help='read/write queue depth with --organization')
    parser.add_argument('--model',required=True,choices=['oracle','fixedlat','md1','wmg1','mess',
                        'astra_single_core','astra_multicore','deepseek_single_core','deepseek_multicore',
                        'opus_single_core','gemini_single_core','opus_multicore','gemini_multicore'])
    a=parser.parse_args(args)
    from ramulator_chia.framework.inputs import _staged_champsim
    from ramulator_chia.framework.evaluation import measure_native, SimulationLimits, CandidateBuild
    from ramulator_chia.framework.candidate import BuildLimits,compile_snapshot
    from ramulator_chia.framework.snapshots import ModelFiles,snapshot
    from ramulator_chia.framework.archive import describe_payload
    cfg=load_config(a.config,a.data_root,a.work_root);cohort=cfg.experiment.evaluation.champsim
    if a.case not in cohort.cases:parser.error('case is not in this frozen cohort')
    spec=cohort.cases[a.case]
    group=next(g for g in ('training','validation','test') if a.case in getattr(cohort,g))
    update={g:((a.case,) if g==group else ()) for g in ('training','validation','test')}
    update.update(cases={a.case:spec},traces={p:cohort.traces[p] for p in spec.programs},
                  builds={k:v for k,v in cohort.builds.items() if k in ('1',str(len(spec.programs)))})
    a.output.mkdir(parents=True,exist_ok=False)
    cases,hosts=_staged_champsim(cohort.model_copy(update=update),a.output)
    case=cases[group][0];resources=cfg.run.resources
    if a.organization:
        from dataclasses import fields
        from ramulator_chia.eval import final_study
        from ramulator_chia.eval.hardware_sweep import HardwarePoint
        final_study.RUNTIME=a.runtime
        capacity=(8 if a.organization=='DDR5_16Gb_x8' else 4)*1024**3
        placement=final_study.remap(case,hosts['champsim:'+str(len(spec.programs))],a.output,capacity)
        values={f.name:getattr(case,f.name) for f in fields(case)}
        values.update(placement=placement,hardware=HardwarePoint(a.organization,a.queue_entries))
        case=final_study.HardwareMix(**values)
    limits=SimulationLimits(resources.simulation_timeout_seconds,resources.memory_bytes,
                            resources.file_bytes,resources.source_bytes,resources.gzip_level)
    candidate=None;model=a.model
    if a.model not in ('oracle','fixedlat','md1','wmg1','mess'):
        contract=ModelFiles(('model.cpp',),'parameters.json')
        selected=snapshot(ROOT/'results/models'/a.model,a.output/'candidates',contract,
                          maximum_bytes=resources.source_bytes)
        out=a.output/'model-build'
        receipt=compile_snapshot(a.runtime,a.output/'candidates',selected['candidate_id'],contract,out,
            limits=BuildLimits(1,resources.build_timeout_seconds,resources.source_bytes,
                               resources.memory_bytes,resources.file_bytes))
        if not receipt['passed']:raise RuntimeError('frozen model build failed')
        candidate=CandidateBuild(a.output/'candidates',selected['candidate_id'],contract,out,sha(out/'build.json'))
        model='candidate'
    curve=describe_payload(ROOT/'chia-loop/src/ramulator_chia/eval/calibration/mess_DDR5.txt','inputs/mess.txt') if model=='mess' else None
    result=measure_native(a.runtime,case,model,a.output/'measurement',limits=limits,candidate=candidate,
                         mess_curve=curve,host=hosts['champsim:'+str(len(spec.programs))])
    print(json.dumps(result,indent=2))

def run_campaign(args):
    if '--help' in args or '-h' in args:
        sys.argv=['run-campaign',*args]
        runpy.run_module('ramulator_chia.framework.launch',run_name='__main__');return
    if '--allow-paid' not in args:
        raise SystemExit('No inference started: run-campaign requires explicit --allow-paid.')
    # Resolve paths once; the launcher's normal configuration checks still apply.
    resolved=list(args)
    if '--config' not in resolved or '--output' not in resolved:
        raise SystemExit('--config and --output are required; use --help')
    source=Path(resolved[resolved.index('--config')+1])
    output=Path(resolved[resolved.index('--output')+1]).resolve()
    # ${VERTEX_PROJECT} is the Vertex execution project the launcher records.
    vertex=(resolved[resolved.index('--vertex-execution-project')+1]
            if '--vertex-execution-project' in resolved else None)
    configuration=load_config(source,os.environ.get('RAMULATOR_CHIA_DATA',ROOT/'.data'),
                              os.environ.get('RAMULATOR_CHIA_WORK',ROOT/'.work'),vertex)
    output.mkdir(parents=True,exist_ok=True)
    from ramulator_chia.framework.snapshots import publish_bytes
    from ramulator_chia.framework.identity import canonical_json
    effective=output/'resolved-launch-config.json'
    publish_bytes(effective,canonical_json(configuration.model_dump(mode='json')).encode())
    resolved[resolved.index('--config')+1]=str(effective)
    if '--context-compaction' in resolved:
        at=resolved.index('--context-compaction')+1
        policy=output/'resolved-compaction.json'
        publish_bytes(policy,canonical_json(load_compaction(resolved[at]).model_dump(mode='json')).encode())
        resolved[at]=str(policy)
    # The existing launcher retains all session, isolation and visibility checks.
    sys.argv=['run-campaign',*resolved]
    runpy.run_module('ramulator_chia.framework.launch',run_name='__main__')

def verify_artifact(args):
    parser=argparse.ArgumentParser(description='Check every file against results/manifests/release.json.')
    parser.add_argument('--manifest',type=Path,default=ROOT/'results/manifests/release.json')
    a=parser.parse_args(args)
    data=json.loads(a.manifest.read_text());bad=[]
    for rel,digest in data['files'].items():
        relative=Path(rel)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe release manifest path')
        p=ROOT/rel
        if not p.is_file() or p.is_symlink() or sha(p)!=digest:bad.append(rel)
    print(json.dumps({'verified':len(data['files'])-len(bad),'mismatches':bad},indent=2))
    if bad:raise SystemExit(1)

COMMANDS={'setup':setup,'fetch-data':fetch_data,'reproduce-figures':reproduce_figures,
          'evaluate':evaluate,'run-campaign':run_campaign,'verify-artifact':verify_artifact}
def main(command=None):
    command=command or (sys.argv.pop(1) if len(sys.argv)>1 else '')
    if command not in COMMANDS:raise SystemExit('Commands: '+', '.join(COMMANDS))
    COMMANDS[command](sys.argv[1:])
if __name__=='__main__':main()
