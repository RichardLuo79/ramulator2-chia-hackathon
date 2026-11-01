"""Locations in a relocated artifact checkout; no operator-machine defaults."""
from pathlib import Path
import os

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('RAMULATOR_CHIA_ROOT', PACKAGE.parents[2])).resolve()
RAMULATOR = ROOT / 'ramulator'
NATIVE = ROOT / 'chia-loop/native'
WORK = Path(os.environ.get('RAMULATOR_CHIA_WORK', ROOT / '.work')).resolve()
DATA = Path(os.environ.get('RAMULATOR_CHIA_DATA', ROOT / '.data')).resolve()

def build_source(repo, relative):
    """Keep the recorded native staging layout, not a dependency on old paths."""
    repo = Path(repo)
    if relative.startswith('tools/chia_loop/') and not (repo / relative).exists():
        return repo.parent / 'chia-loop/native' / relative.removeprefix('tools/chia_loop/')
    return repo / relative

def build_key(repo, source):
    source, repo = Path(source), Path(repo)
    try:
        return source.relative_to(repo).as_posix()
    except ValueError:
        return 'tools/chia_loop/' + source.relative_to(repo.parent / 'chia-loop/native').as_posix()
