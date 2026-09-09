"""Source recipes and the new cohort, without CHIA, accounts or simulation."""

import io
import json
import tarfile

import pytest

from tools.chia_loop.evaluation_config import validate
from tools.chia_loop.framework.archive import extract, seal
from tools.chia_loop.framework.build import run_build_command
from tools.chia_loop.framework.identity import file_sha256
from tools.eval.gem5 import build_suite


def test_public_cohort_is_explicit_and_keeps_training_fixed():
    recipe = json.loads(build_suite.RECIPE.read_text())
    configs = build_suite.RECIPE.parents[2] / "chia_loop/configs"
    original = validate(json.loads((configs / "ddr5_frontend_transfer_v1.json").read_text()))
    current = validate(json.loads((configs / "ddr5_public_transfer_v2.json").read_text()))
    assert current["simpleo3"] == original["simpleo3"]
    assert current["transfer"]["champsim"] == original["transfer"]["champsim"]
    assert set(current["transfer"]["gem5"]["workloads"]) == set(recipe["programs"])
    assert len(recipe["programs"]) == 12
    assert all("-O3" in spec["flags"] for spec in recipe["sources"].values())
    assert "-DLARGE_DATASET" in recipe["sources"]["polybench"]["flags"]
    assert "-DPOLYBENCH_DUMP_ARRAYS" in recipe["sources"]["polybench"]["flags"]
    assert "-fopenmp" not in recipe["sources"]["gapbs"]["flags"]


def fixture_archive(tmp_path, monkeypatch, *, link=False):
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("fixture/main.c")
        data = b"int main(void) { return 0; }\n"
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/not-an-approved-source"
            stream.addfile(member)
        else:
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    recipe = {
        "schema_version": 1,
        "name": "unit-fixture",
        "sources": {
            "fixture": {
                "archive": archive.name,
                "sha256": file_sha256(archive),
                "directory": "fixture",
                "compiler": "gcc",
                "flags": ["-O3"],
                "includes": [],
                "extra_sources": [],
            }
        },
        "programs": {"tiny": {"suite": "fixture", "source": "main.c", "arguments": []}},
    }
    path = tmp_path / "recipe-input.json"
    path.write_text(json.dumps(recipe))
    monkeypatch.setattr(build_suite, "RECIPE", path)
    return archive


def test_source_build_and_source_only_export(tmp_path, monkeypatch):
    fixture_archive(tmp_path, monkeypatch)
    output = tmp_path / "built"
    record = build_suite.build(tmp_path, output)
    assert record["programs"]["tiny"]["build"]["returncode"] == 0
    assert build_suite.load_programs(output / "suite.json") == {"tiny": (output / "bin/tiny", ())}
    recipe = build_suite.source_recipes(output / "suite.json")["benchmark:tiny"]
    assert all(not payload.member.executable for payload in recipe["payloads"])
    assert {payload.member.name for payload in recipe["payloads"]} == {
        "external/tiny/sources/fixture/main.c",
        "external/tiny/recipe.json",
        "external/tiny/build.log",
    }
    # Rebuild from exactly the exported source, not the original build tree.
    archive = seal(recipe["payloads"], tmp_path / "archives", metadata=recipe["metadata"])
    rebuilt = tmp_path / "extracted"
    extract(archive.path, rebuilt, expected_sha256=archive.sha256)
    work = rebuilt / recipe["metadata"]["working_directory"]
    assert not (work / "bin/tiny").exists()
    for index, command in enumerate(recipe["metadata"]["build_commands"]):
        result = run_build_command(
            command, work / f"rebuild-{index}.log", cwd=work, cpus=1, timeout_seconds=30
        )
        assert result["returncode"] == 0
    assert (work / "bin/tiny").is_file()
    (output / "sources/fixture/main.c").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        build_suite.load_programs(output / "suite.json")


def test_source_archive_hash_is_checked_before_extraction(tmp_path, monkeypatch):
    archive = fixture_archive(tmp_path, monkeypatch)
    archive.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        build_suite.build(tmp_path, tmp_path / "built")
    assert not (tmp_path / "built").exists()


def test_source_archive_links_are_not_extracted(tmp_path, monkeypatch):
    fixture_archive(tmp_path, monkeypatch, link=True)
    with pytest.raises(ValueError, match="unexpected upstream archive member"):
        build_suite.build(tmp_path, tmp_path / "built")
