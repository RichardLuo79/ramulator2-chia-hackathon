"""Offline cache-layout, accounting and immutable-run regression checks."""

import copy
import json
import pathlib
import re

import pytest

from tools.chia_loop import prompt_cache as K
from tools.chia_loop import real_core as P
from tools.chia_loop.core import atomic_write_json


@pytest.mark.parametrize("sort_keys", [False, True])
def test_lossless_stable_history_boundaries_even_with_large_diagnostic(sort_keys):
    original = [{"role": "user", "content": 'source: "\\n" 中文\n' * 4000}]
    blocks, anchors = K.conversation_blocks(original, sort_keys=sort_keys)
    extended = original + [
        {"role": "assistant", "content": "inspect"},
        {"role": "user", "content": "large diagnostic " * 30000},
    ]
    newer, new_anchors = K.conversation_blocks(extended, sort_keys=sort_keys)
    assert "".join(blocks) == json.dumps(original, sort_keys=sort_keys)
    assert "".join(newer) == json.dumps(extended, sort_keys=sort_keys)
    assert newer[: len(blocks) - 1] == blocks[:-1]
    assert len(newer) - len(blocks) > 20  # Exceeds Claude's automatic lookback.
    assert len(blocks) - 2 in anchors & new_anchors
    assert len(anchors) <= 3 and len(new_anchors) <= 3
    assert len(blocks) - 1 not in anchors  # No write through the mutable ']'.


@pytest.mark.parametrize(
    "conversation",
    [[], {}, [{"role": "system", "content": "bad"}], [{"role": "user", "content": {}, "extra": 1}]],
)
def test_unexpected_conversation_shapes_fail_closed(conversation):
    with pytest.raises(ValueError):
        K.conversation_blocks(conversation, sort_keys=True)


def test_namespaces_stable_and_isolated_by_every_experiment_dimension(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    K.install(first)
    K.install(second)
    base = (first, "codex", "gpt-6-astra", "xhigh", "proposal")
    key = K.scope(*base)
    assert key == K.scope(*base)
    assert len(key) < 64
    changes = [
        (second, *base[1:]),
        (first, "claude", *base[2:]),
        (*base[:2], "different-model", *base[3:]),
        (*base[:3], "max", "proposal"),
        (*base[:4], "review"),
    ]
    assert all(K.scope(*args) != key for args in changes)
    assert len({K.scope(*args) for args in changes}) == len(changes)
    assert "first" not in key and "second" not in key


def test_legacy_layout_and_new_policy_are_immutable(tmp_path):
    assert K.load(tmp_path) is None and not K.enabled(tmp_path)
    assert K.policy(tmp_path, {"old": True}) == {"old": True}
    config = K.install(tmp_path)
    atomic_write_json(tmp_path / "run_manifest.json", {"policy": {"prompt_cache": config}})
    with pytest.raises(RuntimeError):
        K.install(tmp_path)
    atomic_write_json(tmp_path / "prompt_cache.json", {**config, "mode": "legacy"})
    with pytest.raises(RuntimeError, match="frozen"):
        K.load(tmp_path)
    (tmp_path / "prompt_cache.json").unlink()
    with pytest.raises(RuntimeError, match="disappeared"):
        K.load(tmp_path)


def test_no_silent_policy_install_after_spending(tmp_path):
    atomic_write_json(tmp_path / "ledger.json", {"calls": [{"id": 0}]})
    with pytest.raises(RuntimeError, match="model call"):
        K.install(tmp_path)


def test_no_silent_policy_install_in_prepared_legacy_gemini(tmp_path):
    atomic_write_json(
        tmp_path / "preparation_manifest.json", {"run_id": tmp_path.name, "status": "ready"}
    )
    with pytest.raises(RuntimeError, match="before freezing"):
        K.install(tmp_path)


def test_continuations_cannot_silently_switch_cache_ablation(tmp_path):
    K.check_continuation(tmp_path, "legacy")
    with pytest.raises(ValueError, match="start fresh"):
        K.check_continuation(tmp_path, K.DEFAULT)
    K.install(tmp_path)
    K.check_continuation(tmp_path, K.DEFAULT)
    with pytest.raises(ValueError, match="start fresh"):
        K.check_continuation(tmp_path, "legacy")


def test_report_is_read_only_and_rejects_foreign_or_unknown_ledgers(tmp_path):
    path = tmp_path / "ledger.json"
    row = {
        "iteration": 1,
        "role": "proposal",
        "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 75}},
    }
    ledger = {"run_id": tmp_path.name, "model": "gpt-6-astra", "calls": [row]}
    atomic_write_json(path, ledger)
    before = path.read_bytes()
    result = K.report(tmp_path)
    assert result["by_iteration"]["1"]["cached_input_fraction"] == 0.75
    assert path.read_bytes() == before and set(tmp_path.iterdir()) == {path}
    atomic_write_json(path, {**ledger, "run_id": "another-run"})
    with pytest.raises(ValueError, match="own run"):
        K.report(tmp_path)
    atomic_write_json(path, {**ledger, "model": "unknown"})
    with pytest.raises(ValueError, match="model"):
        K.report(tmp_path)


def test_initial_fields_preserve_all_values_and_put_stable_content_first(tmp_path):
    fields = {
        "budget": {"used": 2},
        "iteration": 2,
        "training_configuration": {"std": "DDR5"},
        "history": ["own history"],
        "policy": {"constraint": "atomic"},
    }
    assert K.initial_fields(tmp_path, fields) == json.dumps(fields, sort_keys=True)
    K.install(tmp_path)
    encoded = K.initial_fields(tmp_path, fields)
    assert json.loads(encoded) == fields
    next_encoded = K.initial_fields(tmp_path, {**fields, "budget": {"used": 3}, "iteration": 3})
    stable = encoded[: encoded.index(', "budget"')]
    assert next_encoded.startswith(stable)


def test_gemini_template_keeps_every_existing_field_once():
    directory = pathlib.Path(__file__).resolve().parents[2] / "tools/chia_loop/prompts"
    old = (directory / "iteration_v1.md").read_text()
    new = (directory / "iteration_cache_v1.md").read_text()

    def extract(text):
        return sorted(re.findall(r"{{(\w+)}}", text))

    assert extract(old) == extract(new)
    assert len(extract(new)) == len(set(extract(new)))
    assert new.index("{{promotion_and_guardrail_configuration}}") < new.index(
        "{{iteration_number}}"
    )


def test_cache_summary_is_token_weighted_and_missing_is_unknown():
    rows = [
        {"usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 90}}},
        {"usage": {"input_tokens": 900, "input_tokens_details": {"cached_tokens": 90}}},
        {},
    ]
    summary = K.usage_summary(rows, "codex")
    assert summary["cached_input_fraction"] == 0.18
    assert summary["unknown_usage_calls"] == 1
    assert summary["cache_reported_calls"] == 2
    gemini = K.usage_summary(
        [
            {"usage": {"prompt_token_count": 100, "cached_content_token_count": 90}},
            {"usage": {"prompt_token_count": 900}},
        ],
        "gemini",
    )
    assert gemini["cached_input_fraction"] is None
    assert gemini["cached_input_fraction_lower_bound"] == 0.09
    assert gemini["missing_cache_counter_calls"] == 1
    claude = K.usage_summary(
        [
            {
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 70,
                }
            }
        ],
        "claude",
    )
    assert claude["cached_input_fraction"] == 0.7
    with pytest.raises(ValueError):
        K.usage_summary(
            [{"usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 11}}}],
            "codex",
        )


@pytest.mark.parametrize("mode", K.MODES)
def test_new_gemini_receipts_price_known_cache_reads_without_rewriting_legacy(tmp_path, mode):
    old, new = tmp_path / "old", tmp_path / "new"
    K.install(new, mode)
    old_ledger = P.Ledger(old / "ledger.json", "flash")
    new_ledger = P.Ledger(new / "ledger.json", "flash")
    usage = {
        "prompt_token_count": 10000,
        "cached_content_token_count": 9000,
        "candidates_token_count": 100,
        "thoughts_token_count": 50,
        "total_token_count": 10150,
    }
    for ledger in (old_ledger, new_ledger):
        index = ledger.reserve("data", 1, 1, input_tokens=10000)
        ledger.settle(index, usage)
    before = old_ledger.path.read_bytes()
    legacy, revised = old_ledger.calls()[0], new_ledger.calls()[0]
    expected_saving = 9000 * 0.9 * P.PRICING["flash"]["input"] / 1e6
    assert legacy["estimated_standard_usd"] - revised["estimated_standard_usd"] == pytest.approx(
        expected_saving
    )
    assert legacy["cap_charge_usd"] == revised["cap_charge_usd"]
    assert old_ledger.path.read_bytes() == before
    assert "cache_accounting" not in legacy
    new_ledger.settle(0, usage)
    assert len(new_ledger.calls()) == 1


def test_gemini_preserves_native_history_and_thought_signatures(tmp_path, monkeypatch):
    from google.genai import types

    from tests.unit_tests.test_chia_unattended import fake_client, generation_args, response
    from tools.chia_loop import generation as Q

    K.install(tmp_path)
    calls = fake_client(monkeypatch, [response({"status": "no_change"})])
    args = generation_args(tmp_path)
    args["contents"] += [
        types.Content(
            role="model", parts=[types.Part(text="old own answer", thought_signature=b"opaque")]
        ),
        types.Content(role="user", parts=[types.Part(text="new feedback")]),
    ]
    before = [c.model_dump(mode="json", exclude_none=True) for c in args["contents"]]
    Q.generate(**args)
    assert len(calls) == 1
    assert calls[0]["contents"] == args["contents"]
    assert [c.model_dump(mode="json", exclude_none=True) for c in calls[0]["contents"]] == before
    assert "chia_cache_scope" in calls[0]["config"].system_instruction
    assert calls[0]["config"].cached_content is None
    assert Q.generate(**args)["candidates"] and len(calls) == 1


def test_claude_fragmentation_preserves_native_attribution_and_all_text():
    conversation = [{"role": "user", "content": "same context " * 3000}]
    request = {
        "system": [
            {"type": "text", "text": "native attribution"},
            {
                "type": "text",
                "text": "contract",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "native date reminder"},
                    {"type": "text", "text": json.dumps(conversation, sort_keys=True)},
                ],
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "native effort",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
        ],
    }
    original = copy.deepcopy(request)
    value, receipt = K.claude_request(request, conversation=conversation, namespace="scope")
    assert request == original  # Never mutate a captured native request.
    assert value["system"][0] == original["system"][0]
    for before, after in zip(original["messages"], value["messages"]):
        assert before["role"] == after["role"]
        assert "".join(b["text"] for b in before["content"]) == "".join(
            b["text"] for b in after["content"]
        )
    assert receipt["breakpoints"] <= 4
