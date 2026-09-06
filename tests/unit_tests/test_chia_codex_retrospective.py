"""Observable summaries/actions only; fixtures never use a live model."""
import json

import pytest

from tools.chia_loop import artifacts, real_core as P
from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.codex_cli import retrospective as V, runner as B, transport as T


def sse(*events):
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()


def response(summary=None, *, status="completed"):
    item = {"id": "rs_fixture", "type": "reasoning", "encrypted_content": "OPAQUE_PRIVATE_STATE",
            "summary": [] if summary is None else [{"type": "summary_text", "text": summary}]}
    return {"type": "response." + status, "response": {"status": status, "output": [item]}}


def test_summary_snapshots_replace_deltas_without_duplicate_text():
    delta = {"type": "response.reasoning_summary_text.delta", "item_id": "rs_fixture", "summary_index": 0}
    final = response("A complete exposed summary.")
    raw = sse({**delta, "delta": "A complete "}, {**delta, "delta": "exposed summary."},
              {"type": "response.output_item.done", "item": final["response"]["output"][0]}, final)
    result = V.reasoning_summaries(raw)
    assert result["availability"] == "returned" and len(result["summaries"]) == 1
    assert result["summaries"][0]["text"] == "A complete exposed summary."
    assert not result["hidden_reasoning_recorded"]
    assert "OPAQUE_PRIVATE_STATE" not in json.dumps(result)


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
def test_absent_summary_is_not_invented_from_encrypted_content_or_reasoning_tokens(status):
    event = response(status=status)
    event["response"]["usage"] = {"output_tokens_details": {"reasoning_tokens": 1000}}
    result = V.reasoning_summaries(sse(event))
    assert result["availability"] == "not_returned" and not result["summaries"]
    assert result["terminal_status"] == status


def test_interrupted_stream_keeps_readable_partial_summary():
    raw = sse({"type": "response.reasoning_summary_text.delta", "item_id": "rs_fixture",
               "summary_index": 0, "delta": "I will inspect the available interface."}) + b'data: {"type":'
    result = V.reasoning_summaries(raw)
    assert result["availability"] == "partial" and result["terminal_status"] is None
    assert result["malformed_or_truncated_events"] == 1
    assert result["summaries"][0]["text"] == "I will inspect the available interface."


def fixture(root):
    root.mkdir()
    atomic_write_json(root / "codex_config.json", {"run_id": root.name, "model": T.MODEL, "effort": "max"})
    atomic_write_json(root / "state.json", {"run_id": root.name, "history": [
        {"iteration": 1, "status": "evaluated", "promoted": False, "metrics": {"cycle_macro_mae_pct": 40}}]})
    candidate = root / "candidates/astra_001"
    atomic_write_json(candidate / "proposal_state.json", {"parent_id": "seed", "parent_sha256": "parent_hash",
        "conversation": [{"role": "user", "content": "FULL_INPUT_NOT_COPIED"},
                         {"role": "assistant", "content": "inspect action"},
                         {"role": "user", "content": "measured diagnostic feedback"}]})
    atomic_write_json(candidate / "draft_001/rejection.json", {"reason": "source check failed"})
    atomic_write_json(candidate / "draft_002/review.json", {"verdict": "pass", "approved": True})
    (candidate / "draft_002/atomic_controller.cpp").write_text("fixture source")
    operations = [("proposal_001_001", "proposal", 1, {"status": "inspect", "requests": [{"tool": "read_file"}]}),
                  ("review_astra_001_d02_00", "review", 3, {"verdict": "pass"}),
                  ("proposal_001_002", "proposal", 2, {"status": "proposal", "hypothesis": "model claim",
                                                        "regions": {"CODE": "REGION_BODY_NOT_COPIED"}})]
    for name, role, time, answer in operations:
        operation = root / "interactions" / name
        atomic_write_json(operation / "identity.json", {"run_id": root.name, "role": role,
            "effort": "xhigh" if role == "review" else "max", "reasoning_summary": "auto"})
        atomic_write_json(operation / "input.json", {"conversation": "FULL_INPUT_NOT_COPIED"})
        atomic_write_json(operation / "result.json", {"answer": answer})
        attempt = operation / "attempt_001"
        atomic_write_json(attempt / "provider_request.json", {"reasoning": {"effort": "max", "summary": "auto"}})
        raw = sse(response("Exposed fixture summary. ``` <script>"))
        (attempt / "provider_response.sse").write_bytes(raw)
        atomic_write_json(attempt / "receipt.json", {"started_at": time, "response_sha256": P.sha(raw)})
    return root


def test_iteration_timeline_links_actions_feedback_drafts_and_outcomes_without_new_analysis(tmp_path):
    root = fixture(tmp_path / "own_run")
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    payload = V.report(root)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    row = payload["iterations"][0]
    assert row["parent"] == "seed" and row["outcome"]["promoted"] is False
    assert [op["operation"] for op in row["operations"]] == [
        "proposal_001_001", "proposal_001_002", "review_astra_001_d02_00"]
    assert row["operations"][-1]["effort"] == "xhigh"
    assert row["operations"][1]["action_and_stated_rationale"]["hypothesis"] == "model claim"
    assert row["feedback_by_proposal_turn"]["1"]["content"] == "measured diagnostic feedback"
    assert row["drafts"][0]["rejection"]["reason"] == "source check failed"
    assert row["drafts"][1]["review"]["approved"]
    encoded = json.dumps(payload)
    assert "FULL_INPUT_NOT_COPIED" not in encoded and "REGION_BODY_NOT_COPIED" not in encoded
    assert "OPAQUE_PRIVATE_STATE" not in encoded
    assert V.write_report(root) == payload
    markdown = (root / "reports/retrospective/index.md").read_text()
    assert "Exposed fixture summary" in markdown and "<script>" not in markdown
    assert "../../interactions/proposal_001_001/input.json" in markdown
    assert "measured diagnostic feedback" in json.dumps(B.R.read_json(root / "reports/retrospective/index.json"))


def test_retrospective_survives_verified_compression_without_restoring_or_calling_models(tmp_path, monkeypatch):
    root = fixture(tmp_path / "own_run")
    original = V.report(root)
    def forbidden(*args, **kwargs):
        pytest.fail("retrospective must not call a model")
    monkeypatch.setattr(T, "invoke", forbidden)
    artifacts.compress(root, root / "aux_archive_manifest.json", min_bytes=1)
    artifacts.verify(root / "aux_archive_manifest.json")
    payload = V.write_report(root)
    for first, second in zip(original["iterations"][0]["operations"], payload["iterations"][0]["operations"]):
        assert first["attempts"][0]["summaries"] == second["attempts"][0]["summaries"]
        assert second["attempts"][0]["receipt_hash_matches"] is True
        assert second["input"].endswith(".gz")
    assert not list((root / "interactions").rglob("provider_response.sse"))


def test_retrospective_refuses_other_run_symlink_and_owner(tmp_path):
    root = fixture(tmp_path / "own_run")
    outside = tmp_path / "other_run"
    outside.mkdir()
    (outside / "identity.json").write_text('{"run_id": "other_run"}')
    (root / "interactions/proposal_002_001").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="escapes"):
        V.report(root)
    with pytest.raises(RuntimeError, match="owner"):
        atomic_write_json(root / "codex_config.json", {"run_id": "other_run"})
        V.report(root)


def test_retrospective_flags_hash_disagreement_and_missing_response(tmp_path):
    root = fixture(tmp_path / "own_run")
    attempt = root / "interactions/proposal_001_001/attempt_001"
    atomic_write_json(attempt / "receipt.json", {"started_at": 1, "response_sha256": "wrong"})
    empty = root / "interactions/proposal_002_001/attempt_001"
    atomic_write_json(empty / "receipt.json", {"started_at": 5, "stage": "provider_dispatch"})
    payload = V.report(root)
    assert payload["iterations"][0]["operations"][0]["attempts"][0]["receipt_hash_matches"] is False
    missing = payload["iterations"][1]["operations"][0]["attempts"][0]
    assert missing["availability"] == "unavailable" and missing["receipt_hash_matches"] is None
