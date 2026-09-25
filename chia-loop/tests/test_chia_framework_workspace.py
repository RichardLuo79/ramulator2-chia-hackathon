"""Positive file/tool grants, with no native processes or provider calls."""

import gzip
import json
from types import SimpleNamespace

import pytest
from test_chia_framework_campaign import FixtureResearch, configuration
from test_chia_framework_campaign import no_services as no_services

from ramulator_chia.framework.build import MODEL_API_HEADER
from ramulator_chia.framework.config import CampaignConfig
from ramulator_chia.framework.snapshots import InvalidSnapshot, publish_bytes
from ramulator_chia.framework.workspace import Workspace

pytestmark = pytest.mark.usefixtures("no_services")


def workspace(tmp_path, *, role="proposer", features=True):
    record = configuration().model_dump(mode="json")
    record["experiment"]["features"] = {
        "synthetic_diagnostics": features,
        "open_loop_diagnostics": features,
    }
    research = FixtureResearch(tmp_path, CampaignConfig.model_validate(record))
    research.resources = research.configuration.run.resources
    research.runtime = tmp_path / "runtime"
    publish_bytes(research.runtime / "export" / MODEL_API_HEADER, b"// fixture API\n")
    seed = research.prepare()["seed"]
    return Workspace(research, 1, role, seed, [])


def test_role_phase_allowlist_and_summary_freeze(tmp_path):
    proposer = workspace(tmp_path)
    proposer.set_phase("explore")
    assert set(proposer.methods()) == {
        "files",
        "read",
        "write",
        "build",
        "evaluate_training",
        "synthetic",
        "open_loop",
        "inspect_training",
    }
    assert proposer.root / "draft" in proposer.grants()["write"]
    proposer.write("draft/model.cpp", "// new model\n")
    proposer.set_phase("reflect")
    assert set(proposer.methods()) == {"files", "read", "write"}
    assert proposer.root / "draft" not in proposer.grants()["write"]
    with pytest.raises(PermissionError):
        proposer.write("draft/model.cpp", "// late unmeasured change")
    with pytest.raises(PermissionError):
        proposer.build()
    proposer.write("notes/summary.md", "Changed the fixture. Measured no scientific results.\n")
    summary = proposer.summary()
    assert (tmp_path / summary["path"]).read_text().startswith("Changed")
    proposer.write("notes/summary.md", "late rewrite")
    with pytest.raises(InvalidSnapshot):
        proposer.summary()


def test_reviewer_cannot_edit_or_evaluate(tmp_path):
    reviewer = workspace(tmp_path, role="reviewer")
    reviewer.set_phase("review")
    assert set(reviewer.methods()) == {"files", "read"}
    assert reviewer.grants()["write"] == []
    with pytest.raises(PermissionError):
        reviewer.write("notes/review.md", "edit")
    with pytest.raises(PermissionError):
        reviewer.evaluate_training()
    with pytest.raises(PermissionError):
        reviewer.set_phase("explore")


def test_recovered_summary_preserves_the_original_publication(tmp_path):
    from ramulator_chia.framework.records import CampaignState

    proposer = workspace(tmp_path)
    proposer.set_phase("reflect")
    proposer.write("notes/summary.md", "Original blocked outcome.\n")
    original = proposer.summary()
    CampaignState(tmp_path / "campaign.sqlite").save("operational-recovery:1", {
        "reason": "compiler_identity_changed",
    })
    proposer.write("notes/summary.md", "Recovered round outcome.\n")
    recovered = proposer.summary()
    assert recovered["path"] == "summaries/iteration-1-recovered.md"
    assert (tmp_path / original["path"]).read_text() == "Original blocked outcome.\n"
    assert (tmp_path / recovered["path"]).read_text() == "Recovered round outcome.\n"
    proposer.write("notes/summary.md", "Cannot rewrite the recovered publication.\n")
    with pytest.raises(InvalidSnapshot):
        proposer.summary()


def test_disabled_instruments_are_not_registered(tmp_path):
    view = workspace(tmp_path, features=False)
    view.set_phase("explore")
    assert "synthetic" not in view.methods() and "open_loop" not in view.methods()
    assert "evaluate_training" in view.methods()


def test_shared_mcp_methods_keep_typed_schemas_through_chia_transport(tmp_path, monkeypatch):
    import asyncio

    from ray import cloudpickle

    from ramulator_chia.framework.chia_tools import AuthenticatedJobTool
    from ramulator_chia.framework.tool_server import WorkspaceTool

    view = workspace(tmp_path)
    view.research.node_id = "1" * 56
    view.set_phase("explore")
    monkeypatch.setattr(AuthenticatedJobTool, "__post_init__", lambda self: None)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    tool = WorkspaceTool(view, "fixture-token", 60)
    copied = cloudpickle.loads(cloudpickle.dumps(tool))
    schemas = {row.name: row.inputSchema for row in asyncio.run(copied.mcp.list_tools())}
    assert set(schemas) == set(view.methods()) | {"status"}
    assert "$ref" in schemas["synthetic"]["properties"]["request"]
    assert schemas["write"]["required"] == ["path", "text"]


def test_paths_and_links_cannot_read_outside_the_view(tmp_path):
    view = workspace(tmp_path)
    view.set_phase("explore")
    secret = tmp_path / "held-out.txt"
    secret.write_text("HELD_OUT_CANARY")
    (view.root / "notes/link").symlink_to(secret)
    (view.root / "notes/dir").symlink_to(tmp_path, target_is_directory=True)
    for path in (
        "../held-out.txt",
        str(secret),
        "native-inputs.json",
        "notes/link",
        "notes/dir/held-out.txt",
    ):
        with pytest.raises((ValueError, OSError)):
            view.read(path)
    assert "HELD_OUT_CANARY" not in gzip.open(view.events, "rt").read()
    with pytest.raises((ValueError, OSError)):
        view.write("notes/dir/held-out.txt", "corrupt")
    assert secret.read_text() == "HELD_OUT_CANARY"


def test_trace_pages_use_scoped_case_and_do_not_silently_truncate(tmp_path):
    view = workspace(tmp_path)
    view.set_phase("explore")
    trace = tmp_path / "trace.csv.gz"
    with gzip.open(trace, "wt") as stream:
        stream.write("one\ntwo\nthree\n")
    calls = []
    view.research.training_case = lambda workload: SimpleNamespace(
        observation_names=("controller",)
    )
    view.research.inspect_training = lambda *args: calls.append(args) or {"observation": {}}
    view.research.trace_path = lambda result, name: trace
    first = view.inspect_training("train", "oracle", trace="controller", lines=2)
    assert first == {"text": "one\ntwo\n", "next_offset": 2, "eof": False}
    last = view.inspect_training("train", "oracle", trace="controller", offset=2, lines=2)
    assert last == {"text": "three\n", "next_offset": 3, "eof": True}
    with pytest.raises(PermissionError):
        view.inspect_training("train", "oracle", trace="../../held-out")
    events = [json.loads(line) for line in gzip.open(view.events, "rt")]
    assert len(events) == 3 and events[-1]["error"]["type"] == "PermissionError"
