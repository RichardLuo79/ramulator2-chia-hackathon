#!/usr/bin/python3
"""Inert native-format CLI fixture. Never imports a provider or reads a login."""

import json
import os
import re
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5


def main():
    if os.environ.get("CHIA_OFFLINE_ADAPTER_FIXTURE") != "1":
        raise RuntimeError("this executable is for offline qualification only")
    args = sys.argv[1:]
    kind = os.environ["CHIA_FIXTURE_KIND"]
    phase = os.environ["CHIA_FIXTURE_PHASE"]
    private = Path(os.environ["CHIA_FIXTURE_STATE"])
    prompt = sys.stdin.read()
    assert "Episode inputs:" in prompt
    assert "CHIA_FIXTURE_AMBIENT_SECRET" not in os.environ
    assert "What I tried" in prompt if phase == "reflect" else True
    if kind == "codex_cli":
        resumed = "resume" in args
        sid = args[-2] if resumed else str(uuid5(NAMESPACE_URL, str(private)))
        path = private / "sessions" / "2099" / ("rollout-fixture-" + sid + ".jsonl")
    else:
        resumed = "--resume" in args
        flag = "--resume" if resumed else "--session-id"
        sid = args[args.index(flag) + 1]
        path = (
            private / "projects" / re.sub(r"[^a-zA-Z0-9]", "-", str(Path.cwd())) / (sid + ".jsonl")
        )
        assert "--append-system-prompt" in args and "--system-prompt" not in args
    if resumed:
        previous = path.read_bytes()
        assert b"opaque-fixture-signature" in previous
    else:
        assert not path.exists(), "new role inherited a native session"
        previous = b""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        previous
        + json.dumps({"phase": phase, "signature": "opaque-fixture-signature"}).encode()
        + b"\n"
    )
    if phase in {"explore", "revise"}:
        source = Path("model.cpp")
        source.write_bytes(
            source.read_bytes() + ("\n// offline native file edit: " + phase + "\n").encode()
        )
    else:
        try:
            with Path("model.cpp").open("ab") as stream:
                stream.write(b"// forbidden mutation")
        except PermissionError:
            pass
        else:
            raise AssertionError("closed candidate phase was writable")
        if phase == "reflect":
            Path("../reflection/reflection.md").write_text(
                "# What I tried\n\nOffline adapter continuity check.\n\n"
                "# What I learned\n\nThe recorded native signature survived.\n"
            )
    failed = os.environ.get("CHIA_FIXTURE_FAIL") == "1"
    reason = "opaque native output\n" * 5000
    if kind == "codex_cli":
        output = Path(args[args.index("--output-last-message") + 1])
        assert output.parent == Path(os.environ["TMPDIR"])
        output.write_text("offline fixture explanation")
        events = [
            {"type": "thread.started", "thread_id": sid},
            {"type": "item.completed", "item": {"type": "reasoning", "text": reason}},
            {
                "type": "turn.failed" if failed else "turn.completed",
                "usage": {"input_tokens": 120, "cached_input_tokens": 90, "output_tokens": 10},
            },
        ]
    else:
        events = [
            {
                "type": "assistant",
                "message": {"content": [{"type": "thinking", "thinking": reason}]},
            },
            {
                "type": "result",
                "result": "offline fixture explanation",
                "usage": {"input_tokens": 30, "cache_read_input_tokens": 90, "output_tokens": 10},
            },
        ]
    for event in events:
        print(json.dumps(event), flush=True)
    if failed:
        print("503 service unavailable: offline fixture", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
