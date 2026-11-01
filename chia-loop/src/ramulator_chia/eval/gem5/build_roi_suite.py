"""Rebuild the six pinned PolyBench kernels with the recorded ROI markers.

Source archive and gem5 source are operator-supplied. No network or simulator
execution. The full original archive and every patched kernel are checked.
"""
import argparse,json,shutil,subprocess,tarfile
from pathlib import Path
from ramulator_chia.framework.archive import safe_name
from ramulator_chia.framework.identity import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--gem5-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();here=Path(__file__).parent
    recipe=json.loads((here/'roi_guests.json').read_text());spec=recipe['source']
    if file_sha256(a.archive)!=spec['sha256']:raise ValueError('PolyBench archive checksum mismatch')
    if file_sha256(here/'polybench-roi.patch')!=recipe['patch_sha256']:raise ValueError('ROI patch changed')
    a.output=a.output.resolve();a.gem5_source=a.gem5_source.resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.gem5_source,text=True).strip()!='7a2b0e413d06c5ce7097104abef3b1d9eaabca91':
        raise ValueError('gem5 source pin mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(a.archive) as t:
        for m in t:
            name=safe_name(m.name.rstrip('/'))
            if name.split('/')[0]!=spec['directory'] or not (m.isfile() or m.isdir()):raise ValueError('unsafe source archive')
        t.extractall(a.output,filter='data')
    source=a.output/spec['directory']
    for item in recipe['programs'].values():
        if file_sha256(source/item['source'])!=item['original_sha256']:raise ValueError('original kernel differs')
    with (a.output/'patch.log').open('w') as log:
        subprocess.run(['patch','-p1','--input',str(here/'polybench-roi.patch')],cwd=source,check=True,stdout=log,stderr=subprocess.STDOUT)
    records={};(a.output/'bin').mkdir()
    for name,item in recipe['programs'].items():
        kernel=source/item['source']
        if file_sha256(kernel)!=item['patched_sha256']:raise ValueError('patched kernel differs')
        command=['gcc',*recipe['flags'],'-I'+str(source/'utilities'),'-I'+str(kernel.parent),
            '-I'+str(a.gem5_source/'include'),str(kernel),str(source/'utilities/polybench.c'),
            str(a.gem5_source/'util/m5/src/abi/x86/m5op.S'),'-lm','-o',str(a.output/'bin'/name)]
        with (a.output/(name+'.log')).open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        records[name]=dict(source_sha256=file_sha256(kernel),binary_sha256=file_sha256(a.output/'bin'/name),command=command)
    (a.output/'build.json').write_text(json.dumps(dict(recipe=recipe,programs=records),indent=2)+'\n')

if __name__=='__main__':main()
