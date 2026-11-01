cd "$(dirname "$0")" && python3 - <<'EOF'
src=open('../../chia-loop/src/ramulator_chia/eval/matchlib.py').read()
start=src.index('_BASE_COLUMNS = ')
end=src.index('def parse(path):')
body=src[start:end]
hdr='''"""Trace loader for the pairing-free multicore metrics.
_load_frame is copied verbatim from ramulator_chia/eval/matchlib.py (matcher schema 4);
artifacts.resolve is inlined as _resolve. recorded() is the same as score.recorded()."""
import json, os, pathlib, hashlib
import numpy as np
import pandas as pd

GZIP_SUFFIX = ".gz"

def _resolve(path):
    path = pathlib.Path(path)
    if path.is_file():
        return path
    if not path.name.endswith(GZIP_SUFFIX):
        archived = path.with_name(path.name + GZIP_SUFFIX)
        if archived.is_file():
            return archived
    return path

class _A:
    resolve = staticmethod(_resolve)
A = _A

'''
tail='''

def recorded(result_json):
    """Trace path is <dir of result.json>/<basename(directory)>/<trace name>, as in score.recorded()."""
    d = json.load(open(result_json))
    o = d['observation']
    base = os.path.join(os.path.dirname(result_json), os.path.basename(d['directory']))
    return dict(trace=os.path.join(base, o['traces']['controller.csv.ch0']['name']),
                windows=o['frontend_stats']['admission_windows'], cycles=o['per_core_cycles'])

def load(trace, windows):
    df, _ = _load_frame(trace, windows=windows)
    return df

def sha256(path):
    h = hashlib.sha256()
    with open(_resolve(path), 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()
'''
open('loadlib.py','w').write(hdr+body+tail)
EOF
python3 -c "import ast;ast.parse(open('loadlib.py').read())" && grep -n "def \|^class" loadlib.py