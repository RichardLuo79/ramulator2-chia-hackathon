"""Source/evidence selection, not environment reconstruction or simulator tests."""

import gzip
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from test_chia_framework_campaign import no_services as no_services
from test_chia_framework_campaign import setup

from ramulator_chia.framework import export
from ramulator_chia.framework.archive import describe_payload, verify
from ramulator_chia.framework.identity import canonical_json, digest_json, file_sha256
from ramulator_chia.framework.snapshots import InvalidSnapshot, publish_bytes

pytestmark = pytest.mark.usefixtures("no_services")


def bundle_fixture(tmp_path):
    campaign, research = setup(tmp_path / "campaign", iterations=1)
    root = research.root
    campaign.run_search()
    campaign.evaluate()
    research.runtime = tmp_path / "runtime"
    publish_bytes(research.runtime / "runtime_manifest.json", b'{"fixture":true}')
    publish_bytes(root / "source/model-api.h", b"// source-only fixture\n")
    inventory = {"model-api.h": file_sha256(root / "source/model-api.h")}
    publish_bytes(
        root / "source/inventory.json",
        canonical_json({"files": inventory, "sha256": digest_json(inventory)}).encode(),
    )
    publish_bytes(tmp_path / "trace.txt", b"input fixture\n")
    publish_bytes(tmp_path / "curve.txt", b"0 100\n")
    payload = describe_payload(tmp_path / "trace.txt", "original.trace")
    research.cases = {
        "training": (
            SimpleNamespace(
                input_files=lambda: {"trace": payload}, identity=lambda: {"fixture": True}
            ),
        )
    }
    research.curve = describe_payload(tmp_path / "curve.txt", "mess.txt")
    research.hosts = {}
    publish_bytes(
        root / "native-inputs.json",
        canonical_json(
            {
                "source_sha256": digest_json(inventory),
                "cases": {
                    key: [case.identity() for case in cases]
                    for key, cases in research.cases.items()
                },
                "curve": asdict(research.curve.member),
                "runtime": file_sha256(research.runtime / "runtime_manifest.json"),
            }
        ).encode(),
    )
    return campaign, research


def test_export_contains_source_inputs_raw_evidence_not_live_profiles_or_binaries(tmp_path):
    campaign, research = bundle_fixture(tmp_path)
    for name, data in {
        "native-evidence/1/proposer/turn/result.json.gz": gzip.compress(b"{}"),
        "builds/fixture/attempt/build.log": b"compiler output",
        "builds/fixture/attempt/candidate.so": b"compiled model",
        "private/auth.json": b"CREDENTIAL_CANARY",
        "agents/1/proposer/notes/summary.md": b"model notes",
    }.items():
        publish_bytes(research.root / name, data)
    before = research.calls[:]
    receipt = export.export_campaign(campaign, research, tmp_path / "artifacts")
    result = verify(receipt.path, expected_sha256=receipt.sha256)
    members = {p["name"] for p in result["manifest"]["members"]}
    assert "source/model-api.h" in members and "campaign/campaign.sqlite" in members
    assert "campaign/native-evidence/1/proposer/turn/result.json.gz" in members
    assert "campaign/builds/fixture/attempt/build.log" in members
    assert not any("candidate.so" in p or "auth.json" in p or "private/" in p for p in members)
    assert research.calls == before  # Export never starts missing science.
    assert not result["manifest"]["metadata"]["runtime_binaries_included"]


def test_export_rejects_changed_source_and_missing_benchmark_recipe(tmp_path):
    campaign, research = bundle_fixture(tmp_path)
    (research.root / "source/model-api.h").chmod(0o600)
    (research.root / "source/model-api.h").write_text("changed source")
    with pytest.raises(ValueError, match="source snapshot changed"):
        export.export_campaign(campaign, research, tmp_path / "artifacts")
    (research.root / "source/model-api.h").write_text("// source-only fixture\n")
    publish_bytes(tmp_path / "benchmark", b"compiled benchmark fixture")
    binary = describe_payload(tmp_path / "benchmark", "benchmark", executable=True)
    research.cases["gem5"] = (
        SimpleNamespace(
            workload="sample",
            input_files=lambda: {"benchmark": binary},
            identity=lambda: {"frontend": "gem5"},
        ),
    )
    with pytest.raises(ValueError, match="export inputs differ"):
        export.export_campaign(campaign, research, tmp_path / "artifacts")
    inputs = research.root / "native-inputs.json"
    binding = json.loads(inputs.read_text())
    binding["cases"]["gem5"] = [research.cases["gem5"][0].identity()]
    inputs.chmod(0o600)
    inputs.write_text(canonical_json(binding))
    with pytest.raises(ValueError, match="missing source/build recipe"):
        export.export_campaign(campaign, research, tmp_path / "artifacts")


def test_capture_checks_compiled_and_harness_source_without_collecting_builds(
    tmp_path, monkeypatch
):
    runtime, repo, destination = tmp_path / "runtime", tmp_path / "repo", tmp_path / "source"
    publish_bytes(runtime / "runtime-source/src/model.cpp", b"// original build input\n")
    publish_bytes(
        runtime / "runtime_manifest.json",
        canonical_json(
            {
                "source_inventory": {
                    "src/model.cpp": file_sha256(runtime / "runtime-source/src/model.cpp")
                }
            }
        ).encode(),
    )
    publish_bytes(repo / "driver.py", b"# driver\n")
    monkeypatch.setattr(export, "HARNESS_TREES", ())
    monkeypatch.setattr(export, "HARNESS_FILES", ("driver.py",))
    record = export.capture_sources(repo, runtime, destination)
    assert set(record["files"]) == {"src/model.cpp", "driver.py"}
    assert export.capture_sources(repo, runtime, destination) == record
    (repo / "driver.py").chmod(0o600)
    (repo / "driver.py").write_text("# changed driver\n")
    with pytest.raises(InvalidSnapshot):
        export.capture_sources(repo, runtime, destination)
    assert json.loads((destination / "inventory.json").read_text()) == record


def test_export_binds_frontend_build_evidence_without_its_executable(tmp_path):
    campaign, research = bundle_fixture(tmp_path)
    host_root = tmp_path / "gem5-build"
    publish_bytes(host_root / "host.json", b'{"fixture":"local build provenance"}')
    publish_bytes(host_root / "gem5.opt", b"not an archive payload")
    identity = {"kind": "gem5", "receipt_sha256": file_sha256(host_root / "host.json")}
    research.hosts["gem5"] = SimpleNamespace(root=host_root, identity=lambda: identity)
    inputs = research.root / "native-inputs.json"
    binding = json.loads(inputs.read_text())
    binding["hosts"] = {"gem5": identity}
    inputs.chmod(0o600)
    inputs.write_text(canonical_json(binding))
    recipe = {
        "metadata": {
            "upstream": {"url": "https://github.com/gem5/gem5.git", "revision": "a" * 40},
            "build_commands": [["scons", "build/X86/gem5.opt", "-j1"]],
        },
        "payloads": [],
    }
    receipt = export.export_campaign(
        campaign, research, tmp_path / "artifacts", external_recipes={"gem5": recipe}
    )
    result = verify(receipt.path, expected_sha256=receipt.sha256)["manifest"]
    names = {member["name"] for member in result["members"]}
    assert "campaign/frontends/gem5/host.json" in names
    assert not any(name.endswith("gem5.opt") for name in names)
    assert result["metadata"]["external_sources"]["gem5"]["evaluated_host"] == identity
    identity["receipt_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="frontends differ"):
        export.export_campaign(
            campaign, research, tmp_path / "changed", external_recipes={"gem5": recipe}
        )


def test_frontend_recipe_file_is_data_not_an_executable_recipe(tmp_path):
    patch = tmp_path / "bridge.patch"
    patch.write_text("source patch fixture\n")
    description = {
        "gem5": {
            "metadata": {
                "upstream": {"url": "https://github.com/gem5/gem5.git", "revision": "a" * 40},
                "build_commands": [["not-executed", "build/X86/gem5.opt"]],
            },
            "files": [{"source": patch.name, "member": "external/gem5/bridge.patch"}],
        }
    }
    path = tmp_path / "frontends.json"
    path.write_text(json.dumps(description))
    recipes = export.load_frontend_recipes(path)
    assert recipes["gem5"]["metadata"] == description["gem5"]["metadata"]
    assert recipes["gem5"]["payloads"][0].source == patch
    with pytest.raises(ValueError, match="each external frontend"):
        export.validate_recipes(recipes, {"gem5", "champsim"})
    description["gem5"]["metadata"]["upstream"]["revision"] = "main"
    path.write_text(json.dumps(description))
    with pytest.raises(ValueError, match="immutable upstream"):
        export.load_frontend_recipes(path)
    path.write_text('{"gem5":{},"gem5":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        export.load_frontend_recipes(path)


def test_multicore_builds_share_one_source_recipe(tmp_path):
    campaign, research = bundle_fixture(tmp_path)
    identities = {}
    for cores in (1, 4, 8):
        host_root = tmp_path / f"champsim-{cores}"
        publish_bytes(host_root / "host.json", canonical_json({"cores": cores}).encode())
        identity = {"cores": cores, "receipt_sha256": file_sha256(host_root / "host.json")}
        identities[f"champsim:{cores}"] = identity
        research.hosts[f"champsim:{cores}"] = SimpleNamespace(root=host_root, identity=lambda value=identity: value)
    inputs = research.root / "native-inputs.json"
    binding = json.loads(inputs.read_text())
    binding["hosts"] = identities
    inputs.chmod(0o600)
    inputs.write_text(canonical_json(binding))
    recipe = {"metadata": {"upstream": {"url": "https://github.com/ChampSim/ChampSim.git", "revision": "a" * 40},
                           "build_commands": [["make", "-j1"]]}, "payloads": []}
    sealed = export.export_campaign(campaign, research, tmp_path / "artifacts", external_recipes={"champsim": recipe})
    manifest = verify(sealed.path, expected_sha256=sealed.sha256)["manifest"]
    assert manifest["metadata"]["external_sources"]["champsim"]["evaluated_builds"] == identities
    assert {f"campaign/frontends/champsim/{n}/host.json" for n in (1, 4, 8)} <= {m["name"] for m in manifest["members"]}
