"""Frozen-study invariants without cloud access or a simulator matrix."""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import pytest

from ramulator_chia.eval import final_study as study


def test_original_capacity_reuses_exact_placement(tmp_path):
    path=tmp_path/'map.gz'
    path.write_bytes(gzip.compress(b'CHAMPSIM_PLACEMENT_V1 1 4096 4096 5 8589934592 1 1 1\nP 0 5 0 1048576\nD 0 1 257\n'))
    payload=SimpleNamespace(source=path)
    case=SimpleNamespace(placement=payload)
    assert study.remap(case,None,tmp_path,8<<30) is payload


def test_new_capacity_keeps_virtual_pages_and_uses_native_allocator(tmp_path,monkeypatch):
    original=tmp_path/'original.gz'
    original.write_bytes(gzip.compress(b'CHAMPSIM_PLACEMENT_V1 1 4096 4096 5 8589934592 1 1 1\nP 0 5 0 1048576\nD 0 123 257\n'))
    case=SimpleNamespace(placement=SimpleNamespace(source=original),workload='test-c1-fixture',cores=1)
    host=SimpleNamespace(root=tmp_path,record=lambda:{'files':{'binary':{'member':{'name':'champsim'}}}})
    def native(args,**kwargs):
        assert args[1]=='--prepare-placement'
        assert Path(args[2]).read_text()=='CHAMPSIM_PAGES_V1 1 4096 4294967296\n0 123\n'
        Path(args[3]).write_text('CHAMPSIM_PLACEMENT_V1 1 4096 4096 5 4294967296 1 1 1\nP 0 5 0 1048576\nD 0 123 257\n')
    monkeypatch.setattr(study.subprocess,'run',native)
    mapped=study.remap(case,host,tmp_path,4<<30)
    assert b'4294967296' in gzip.decompress(mapped.source.read_bytes())


def test_hardware_extension_retains_continuous_contention_fields():
    names={f.name for f in fields(study.HardwareMix)}
    assert {'companions','placement','completion_policy','hardware'}<=names


def test_artifact_has_no_historical_cloud_expiry():
    assert study.STOP_AT == float('inf')


def test_qualification_rejects_same_cycles_but_changed_observations(tmp_path):
    reference={'per_core_cycles':[20], 'frontend_stats':{'x':1},'controller_stats':{},
               'traces':{'controller':{'logical_sha256':'a','logical_bytes':10}}}
    (tmp_path/'qualification-references.json').write_text(json.dumps({'hardware':{'case':reference}}))
    changed={**reference,'traces':{'controller':{'logical_sha256':'b','logical_bytes':10}}}
    with pytest.raises(ValueError,match='observations'):
        study.qualify(tmp_path,{'study':'hardware','case':'case'},{'observation':changed})
