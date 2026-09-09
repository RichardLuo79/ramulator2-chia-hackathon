"""Versioned prompt-cache layout, isolated by run/model/effort/role.

No model calls, account lookup, shared history, explicit Gemini cache resources,
or rewriting of existing usage receipts. Legacy roots keep their old wire shape.
The CLI wrappers still start fresh, filesystem-isolated native processes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import re
import uuid

from tools.chia_loop.core import atomic_write_json

DEFAULT = "prefix_v1"
MODES = ("legacy", DEFAULT)
CHUNK_CHARS = 16_384
VIRTUAL_WORKSPACE = "/chia/isolated-workspace"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def install(root, mode=DEFAULT):
    root = pathlib.Path(root)
    if mode not in MODES:
        raise ValueError("unknown prompt-cache layout")
    if any(
        (root / name).exists()
        for name in ("prompt_cache.json", "run_manifest.json", "preparation_manifest.json")
    ):
        raise RuntimeError("prompt-cache policy must be installed before freezing a new run")
    if (root / "ledger.json").exists() and json.loads((root / "ledger.json").read_text()).get(
        "calls"
    ):
        raise RuntimeError("cannot change prompt-cache policy after a model call")
    config = {
        "schema_version": 1,
        "run_id": root.name,
        "mode": mode,
        "scope_id": uuid.uuid4().hex,
        "explicit_gemini_cache": False,
        "chatgpt_cache_controls": "native_defaults_with_stable_key",
    }
    atomic_write_json(root / "prompt_cache.json", config)
    return config


def load(root):
    root = pathlib.Path(root)
    path = root / "prompt_cache.json"
    config = json.loads(path.read_text()) if path.exists() else None
    if config is not None:
        if (
            not isinstance(config, dict)
            or set(config)
            != {
                "schema_version",
                "run_id",
                "mode",
                "scope_id",
                "explicit_gemini_cache",
                "chatgpt_cache_controls",
            }
            or type(config["schema_version"]) is not int
            or config["schema_version"] != 1
            or config["run_id"] != root.name
            or config["mode"] not in MODES
            or not isinstance(config["scope_id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", config["scope_id"])
            or config["explicit_gemini_cache"] is not False
            or config["chatgpt_cache_controls"] != "native_defaults_with_stable_key"
        ):
            raise RuntimeError("invalid or foreign prompt-cache policy")
    # Deleting/changing the switch cannot silently downgrade a prepared run.
    for name in (
        "preparation_manifest.json",
        "run_manifest.json",
        "codex_config.json",
        "claude_config.json",
    ):
        p = root / name
        if p.exists():
            saved = json.loads(p.read_text())
            for owner in (
                saved,
                saved.get("policy", {}),
                saved.get("configuration", {}).get("policy", {}),
            ):
                if "prompt_cache" in owner and owner["prompt_cache"] != config:
                    raise RuntimeError("frozen prompt-cache policy changed or disappeared")
    return config


def enabled(root):
    return (load(root) or {}).get("mode") == DEFAULT


def policy(root, base):
    config = load(root)
    return {**base, "prompt_cache": config} if config is not None else base


def check_continuation(root, mode):
    if (load(root) or {}).get("mode", "legacy") != mode:
        raise ValueError(
            "prompt-cache layout ablations must start fresh, not continue existing training"
        )


def scope(root, provider, model, effort, role):
    config = load(root)
    if config is None or config["mode"] != DEFAULT:
        return None
    if role not in ("proposal", "review"):
        raise ValueError("invalid cache role")
    # The root binding also keeps copied roots from inheriting cache namespaces.
    return (
        "chia-"
        + digest(
            [config["scope_id"], str(pathlib.Path(root).resolve()), provider, model, effort, role]
        )[:48]
    )


def scoped_system(system, namespace):
    # An opaque label, not a hint about model design, predecessors, or other runs.
    return f"<chia_cache_scope>{namespace}</chia_cache_scope>\n{system}" if namespace else system


def initial_fields(root, fields):
    """Reorder only JSON object members: same facts, stable material first."""
    if not enabled(root):
        return json.dumps(fields, sort_keys=True)
    stable = (
        "training_configuration",
        "policy",
        "readable_files",
        "tool_manifest",
        "comparisons",
        "human_modeling_hint",
    )
    keys = [k for k in stable if k in fields] + sorted(set(fields) - set(stable))
    # Canonicalize nested objects without sorting away the chosen outer order.
    return (
        "{"
        + ", ".join(json.dumps(k) + ": " + json.dumps(fields[k], sort_keys=True) for k in keys)
        + "}"
    )


def conversation_blocks(conversation, *, sort_keys):
    """Losslessly fragment the existing JSON envelope; never promote data roles.

    Cache ends exclude the mutable closing ']'. The previous user-turn boundary
    is explicitly retained even if large diagnostic replies exceed lookback.
    Fixed-size fragments also expose a reusable beginning of the first message.
    """
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("a nonempty explicit conversation is required")
    blocks, ends = [], []
    for i, item in enumerate(conversation):
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or item["role"] not in ("user", "assistant")
            or not isinstance(item["content"], str)
        ):
            raise ValueError("unexpected CHIA conversation shape")
        encoded = ("[" if i == 0 else ", ") + json.dumps(item, sort_keys=sort_keys)
        blocks.extend(encoded[n : n + CHUNK_CHARS] for n in range(0, len(encoded), CHUNK_CHARS))
        ends.append(len(blocks) - 1)
    blocks.append("]")
    if "".join(blocks) != json.dumps(conversation, sort_keys=sort_keys):
        raise AssertionError("conversation fragmentation changed model input")
    anchors = {0, ends[-1]}
    if len(ends) >= 3:
        anchors.add(ends[-3])
    return blocks, anchors


def _text(message):
    content = message.get("content")
    if not isinstance(content, list) or not all(
        b.get("type") == "input_text" and isinstance(b.get("text"), str) for b in content
    ):
        raise RuntimeError("unexpected native Codex content")
    return "".join(b["text"] for b in content)


def codex_request(request, *, work, conversation, system, namespace, api_mode):
    """Normalize only audited native framing, never source/trace/prompt data.

    Public API cache options are deliberately NOT injected into the subscription
    endpoint: its compatibility with these newer API fields is not established.
    """
    value = copy.deepcopy(request)
    items = value.get("input")
    if (
        not isinstance(items, list)
        or len(items) != 5
        or [(m.get("type"), m.get("role")) for m in items]
        != [
            ("additional_tools", "developer"),
            ("message", "developer"),
            ("message", "developer"),
            ("message", "user"),
            ("message", "user"),
        ]
    ):
        raise RuntimeError("native Codex framing changed; cache preflight must be repeated")
    tools = {k: v for k, v in items[0].items() if k != "id"}
    if (
        hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
        != "8dc4b793b71e86e155c86bac1ab7d667a17556cef30df414f8affad4dde1d036"
    ):
        raise RuntimeError("native Codex tool framing changed")
    if (
        _text(items[1])
        != system + "\nReturn only the requested CHIA JSON object. Native tools are unavailable."
    ):
        raise RuntimeError("native Codex imported extra developer context")
    if _text(items[-1]) != json.dumps(conversation):
        raise RuntimeError("native Codex imported extra conversation")
    hashes = {
        2: "da806f07e9af24ea39165104abca5a8c61ae7f864803a98fc224e9daedd544b6",
        3: "222b7c8d1152fa5e2967dd540910e9def7edb938291afbd21780c1c19c9fb5f5",
    }
    for index in (2, 3):
        # Exact known injected paths only; model data is not regex-rewritten.
        original = _text(items[index])
        normalized = original.replace(str(work), VIRTUAL_WORKSPACE)
        checked = re.sub(
            r"<current_date>\d{4}-\d{2}-\d{2}</current_date>",
            "<current_date>DATE</current_date>",
            normalized,
        )
        if hashlib.sha256(checked.encode()).hexdigest() != hashes[index]:
            raise RuntimeError("native Codex context changed; no broad normalization allowed")
        items[index]["content"] = [{"type": "input_text", "text": normalized}]
    fragments, anchors = conversation_blocks(conversation, sort_keys=False)
    items[-1]["content"] = [{"type": "input_text", "text": text} for text in fragments]
    # Input IDs describe fresh native messages; they are not inherited server
    # response IDs. Omitting optional IDs makes the explicit prefix stable.
    for item in items:
        item.pop("id", None)
    value["prompt_cache_key"] = namespace
    if api_mode:
        value.pop("prompt_cache_retention", None)
        value["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
        items[1]["content"][-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
        for index in anchors:
            items[-1]["content"][index]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return value, {
        "mode": DEFAULT,
        "namespace": namespace,
        "fragments": len(fragments),
        "conversation_text_preserved": True,
        "native_paths_virtualized": True,
        "explicit_breakpoints": api_mode,
        "public_api_options_added": api_mode,
    }


def claude_request(request, *, conversation, namespace):
    # Called ONLY after the unmodified native request passes validate_request.
    value = copy.deepcopy(request)
    fragments, anchors = conversation_blocks(conversation, sort_keys=True)
    original = value["messages"][0]["content"]
    if (
        not isinstance(original, list)
        or len(original) != 2
        or original[1].get("text") != "".join(fragments)
    ):
        raise RuntimeError("unexpected native Claude conversation")
    # Keep the native attribution/billing prefix and authentication untouched.
    # Use one system and at most three data breakpoints, all with the native TTL.
    controls = [b["cache_control"] for b in value["system"] if "cache_control" in b]
    if not controls or any(c != {"type": "ephemeral", "ttl": "1h"} for c in controls):
        raise RuntimeError("native Claude cache policy changed")
    for block in value["system"]:
        block.pop("cache_control", None)
    for message in value["messages"]:
        for block in message["content"]:
            block.pop("cache_control", None)
    value.pop("cache_control", None)
    value["system"][-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    blocks = [{"type": "text", "text": text} for text in fragments]
    for index in anchors:
        blocks[index]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    value["messages"][0]["content"] = [original[0], *blocks]
    return value, {
        "mode": DEFAULT,
        "namespace": namespace,
        "fragments": len(fragments),
        "conversation_text_preserved": True,
        "breakpoints": 1 + len(anchors),
        "ttl": "1h",
    }


def usage_summary(rows, provider):
    """Token-weighted, content-free counters; missing counts are NOT zero hits."""
    if provider not in ("codex", "claude", "gemini"):
        raise ValueError("unknown cache accounting provider")
    total = cached = reported_input = input_calls = cached_calls = 0
    for row in rows:
        usage = row.get("usage") or {}
        if provider == "codex":
            inp, hit = (
                usage.get("input_tokens"),
                (usage.get("input_tokens_details") or {}).get("cached_tokens"),
            )
        elif provider == "claude":
            fields = [
                usage.get(k)
                for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
            ]
            if any(v is not None and (type(v) is not int or v < 0) for v in fields):
                raise ValueError("invalid provider cache usage")
            inp = sum(fields) if all(type(v) is int and v >= 0 for v in fields) else None
            hit = usage.get("cache_read_input_tokens")
        elif provider == "gemini":
            inp, hit = usage.get("prompt_token_count"), usage.get("cached_content_token_count")
        if inp is None:
            continue
        if (
            type(inp) is not int
            or inp < 0
            or (hit is not None and (type(hit) is not int or not 0 <= hit <= inp))
        ):
            raise ValueError("invalid provider cache usage")
        total += inp
        input_calls += 1
        if hit is not None:
            cached_calls += 1
            cached += hit
            reported_input += inp
    return {
        "attempts": len(rows),
        "input_reported_calls": input_calls,
        "cache_reported_calls": cached_calls,
        "unknown_usage_calls": len(rows) - input_calls,
        "missing_cache_counter_calls": input_calls - cached_calls,
        "input_tokens": total,
        "known_cached_input_tokens": cached,
        "cached_input_fraction": cached / total if total and input_calls == cached_calls else None,
        "cached_input_fraction_lower_bound": cached / total if total else None,
        "fraction_on_calls_with_cache_counter": cached / reported_input if reported_input else None,
    }


def report(root):
    from tools.chia_loop.recovery import read_json

    root = pathlib.Path(root)
    ledger = read_json(root / "ledger.json")
    model = ledger.get("model", "")
    if ledger.get("run_id") != root.name:
        raise ValueError("cache report requires the ledger's own run root")
    provider = next(
        (
            p
            for prefix, p in (("gpt-", "codex"), ("claude-", "claude"), ("gemini-", "gemini"))
            if isinstance(model, str) and model.startswith(prefix)
        ),
        None,
    )
    if provider is None:
        raise ValueError("unknown cache accounting model")
    rows = ledger["calls"]
    return {
        "record_type": "individual_run_prompt_cache_report",
        "run_id": root.name,
        "model": model,
        "policy": load(root),
        "provider": provider,
        "account_quota_measured": False,
        "totals": usage_summary(rows, provider),
        "by_role": {
            role: usage_summary(
                [r for r in rows if r.get("role", r.get("purpose")) == role], provider
            )
            for role in ("proposal", "review")
        },
        "by_iteration": {
            str(i): usage_summary([r for r in rows if r.get("iteration") == i], provider)
            for i in sorted({r["iteration"] for r in rows if "iteration" in r})
        },
        "notes": [
            "Ratios are sums of tokens, not averages of call percentages.",
            "Unknown calls/counters are not zero. "
            "Lower bounds refer only to reported input tokens.",
            "Observed cache reuse is not a subscription quota measurement "
            "or a guaranteed future hit rate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read-only cache-usage audit for one CHIA run; no model calls"
    )
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.root), indent=2))


if __name__ == "__main__":
    main()
