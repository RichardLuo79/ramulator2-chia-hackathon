"""Build the pinned ChampSim source; only its paths are generated."""
from pathlib import Path
import json,shutil,subprocess,sys

def build(root,work,runtime,cores,workers):
    upstream=root/'ramulator/integrations/champsim'
    target=work/'champsim'/f'source{cores}'
    if target.exists():
        raise FileExistsError('Keep an existing frontend build immutable; choose a new work directory')
    shutil.copytree(upstream/'source',target,ignore=shutil.ignore_patterns('.git'))
    for name in ('champsim_config.json','global.options'):
        shutil.copyfile(upstream/f'configs/c{cores}'/name,target/name)
    # No private include or RPATH survives the export.
    (target/'absolute.options').write_text(
        f'-I{target}/inc\n-I{runtime}/runtime-source/src\n-DCHAMPSIM_RAMULATOR\n')
    environment=__import__('os').environ.copy()
    environment['RAMULATOR_LIBRARY_DIR']=str(runtime/'runtime')
    with (target/'build.log').open('w') as log:
        subprocess.run([sys.executable,'config.sh','champsim_config.json'],cwd=target,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['make','-j',str(workers)],cwd=target,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    return target/'bin'/f'champsim-{cores}core'
