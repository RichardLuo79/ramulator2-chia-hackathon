"""Offline provider replies through real CHIA tools and native CLI boundaries.

No login is read and no provider endpoint is contacted. The installed CLIs run
unchanged; API adapters use fixture SDK responses. This checks wiring and access,
not service availability or model quality.
"""

import argparse
import gzip
import json
import secrets
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import ray

from ramulator_chia.framework.config import CampaignConfig
from ramulator_chia.framework.model_sessions import create_session, upstream_identity
from ramulator_chia.framework.snapshots import publish_bytes, read_file
from ramulator_chia.framework.tool_server import phase_tools
from ramulator_chia.framework.transport import NativeTransport


def codex_reply(request, number):
    """Responses SSE fixture: one actual MCP read, then a final answer."""
    if number == 1:
        # Installed Codex exposes MCP tools through its native code-mode tool.
        # Exercise that route instead of inventing a direct function schema.
        item = dict(
            type="custom_tool_call",
            id="fc_fixture",
            call_id="call_fixture",
            namespace="functions",
            name="exec",
            input='text(await tools.mcp__ramulator__read({path:"references/hello.txt"}));',
        )
    else:
        item = dict(
            type="message",
            id="msg_fixture",
            role="assistant",
            status="completed",
            content=[dict(type="output_text", text="Fixture complete.", annotations=[])],
        )
    response = dict(
        id=f"resp_fixture_{number}",
        object="response",
        model=request["model"],
        status="completed",
        output=[item],
        usage=dict(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=dict(cached_tokens=0),
            total_tokens=15,
        ),
    )
    events = [
        dict(type="response.created", response={**response, "status": "in_progress", "output": []}),
        dict(type="response.output_item.done", output_index=0, item=item),
        dict(type="response.completed", response=response),
    ]
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()


def claude_reply(request, number):
    message = dict(
        id=f"msg_fixture_{number}",
        type="message",
        role="assistant",
        model=request["model"],
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage=dict(input_tokens=10, output_tokens=0),
    )
    if number == 1:
        block = dict(type="tool_use", id="toolu_fixture", name="mcp__ramulator__read", input={})
        delta = dict(type="input_json_delta", partial_json='{"path":"references/hello.txt"}')
    else:
        block, delta = dict(type="text", text=""), dict(type="text_delta", text="Fixture complete.")
    events = [
        dict(type="message_start", message=message),
        dict(type="content_block_start", index=0, content_block=block),
        dict(type="content_block_delta", index=0, delta=delta),
        dict(type="content_block_stop", index=0),
        dict(
            type="message_delta",
            delta=dict(stop_reason="tool_use" if number == 1 else "end_turn", stop_sequence=None),
            usage=dict(output_tokens=5),
        ),
        dict(type="message_stop"),
    ]
    return "".join(
        "event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events
    ).encode()


def check(config: CampaignConfig, root: Path):
    upstream_identity()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    workspace, private = root / "workspace", root / "private"
    private.mkdir(mode=0o700)
    publish_bytes(workspace / "references/hello.txt", b"APPROVED_FIXTURE_CONTENT\n")
    for name in ("draft", "notes"):
        (workspace / name).mkdir()
    publish_bytes(root / "held-out.txt", b"PRIVATE_HELD_OUT_CANARY\n")
    view = SimpleNamespace(
        root=workspace,
        iteration=1,
        role="proposer",
        phase="explore",
        research=SimpleNamespace(
            root=root, configuration=config, node_id=ray.get_runtime_context().get_node_id()
        ),
        grants=lambda: {
            "read": [workspace / n for n in ("references", "draft", "notes")],
            "write": [workspace / n for n in ("draft", "notes")],
        },
    )

    def read(path: str) -> dict:
        """Read approved fixture files, using the production no-follow reader."""
        return {"text": read_file(workspace, path, maximum_bytes=1024).decode()}

    view.methods = lambda: {"read": read}
    replies, events = [], []
    token = secrets.token_urlsafe(32)

    @contextmanager
    def fake_http(method, url, **options):
        expected = (
            "https://chatgpt.com/backend-api/codex/responses"
            if config.backend.kind == "codex_cli"
            else "https://api.anthropic.com/v1/messages?beta=true"
        )
        assert method == "POST" and url == expected
        request = json.loads(options["content"])
        replies.append(request)
        if len(replies) > 3:
            raise ValueError("unexpected fixture loop")
        body = (codex_reply if config.backend.kind == "codex_cli" else claude_reply)(
            request, len(replies)
        )
        yield SimpleNamespace(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            iter_bytes=lambda: iter([body]),
        )

    def google_reply(**request):
        from google.genai import types

        replies.append(
            {
                "model": request["model"],
                "contents": [c.model_dump(mode="json") for c in request["contents"]],
            }
        )
        if len(replies) == 1:
            part = types.Part(
                function_call=types.FunctionCall(
                    name="ramulator__read", args={"path": "references/hello.txt"}
                )
            )
        else:
            part = types.Part(text="Fixture complete.")
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[part]), finish_reason="STOP"
                )
            ]
        )

    async def deepseek_reply(request):
        body = json.loads(request.content)
        replies.append(body)
        message = dict(
            role="assistant", content="Fixture complete.", reasoning_content="Fixture reasoning."
        )
        if len(replies) == 1:
            message.update(
                content=None,
                tool_calls=[
                    dict(
                        id="call_fixture",
                        type="function",
                        function=dict(
                            name="ramulator__read", arguments='{"path":"references/hello.txt"}'
                        ),
                    )
                ],
            )
        return httpx.Response(
            200,
            json=dict(
                id="chat_fixture",
                object="chat.completion",
                created=0,
                model=config.backend.model,
                choices=[
                    dict(
                        index=0,
                        message=message,
                        finish_reason="tool_calls" if len(replies) == 1 else "stop",
                    )
                ],
                usage=dict(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        )

    with ExitStack() as stack:
        offered = stack.enter_context(phase_tools(view, token))
        options = dict(
            vertex_event_callback=lambda e, s: events.append(e),
            chat_event_callback=lambda e, s: events.append(e),
            anthropic_event_callback=lambda e, s: events.append(e),
            tool_http_client_factory=lambda _: httpx.AsyncClient(
                headers={"Authorization": "Bearer " + token}, trust_env=False
            ),
        )
        if config.backend.kind in {"codex_cli", "claude_cli"}:
            stack.enter_context(
                patch(
                    "ramulator_chia.codex_cli.auth.headers",
                    return_value={"Authorization": "Bearer OFFLINE_FIXTURE"},
                )
            )
            stack.enter_context(
                patch(
                    "ramulator_chia.claude_cli.auth.read_setup_token",
                    return_value="OFFLINE_FIXTURE",
                )
            )
            stack.enter_context(
                patch("ramulator_chia.framework.transport.httpx.stream", fake_http)
            )
            options.update(
                stack.enter_context(
                    NativeTransport()(view, private, {"CHIA_WORKSPACE_TOKEN": token})
                )
            )
        elif config.backend.kind == "vertex_gemini":
            stack.enter_context(
                patch(
                    "google.genai.Client",
                    lambda **_: SimpleNamespace(
                        models=SimpleNamespace(generate_content=google_reply), close=lambda: None
                    ),
                )
            )
            options["vertex_client_kwargs"] = {}
        elif config.backend.kind == "vertex_claude":
            import httpx2
            from google.oauth2.credentials import Credentials

            async def vertex_claude_reply(request):
                assert request.url.host == "aiplatform.googleapis.com"
                body = json.loads(request.content)
                replies.append(body)
                if len(replies) > 2:
                    raise ValueError("unexpected fixture loop")
                wire = claude_reply({**body, "model": config.backend.model}, len(replies))
                wire = wire.replace(b"mcp__ramulator__read", b"ramulator__read")
                return httpx2.Response(200, content=wire, headers={"Content-Type": "text/event-stream"})

            options["anthropic_client_kwargs"] = {
                "credentials": Credentials("OFFLINE_FIXTURE"),
                "http_client": httpx2.AsyncClient(
                    transport=httpx2.MockTransport(vertex_claude_reply), trust_env=False
                ),
            }
        elif config.backend.kind == "deepseek_api":
            # The adapter owns/closes the AsyncOpenAI client's lifetime.
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(deepseek_reply), trust_env=False
            )
            options["chat_client_kwargs"] = {"api_key": "OFFLINE_FIXTURE", "http_client": client}
        else:
            raise ValueError("unsupported fixture backend")
        session = create_session(
            config.backend,
            session_id="offline-boundary",
            private_directory=private,
            workspace=workspace,
            system_message="Read only the approved fixture using the supplied tool.",
            tool_token_env=lambda _: "CHIA_WORKSPACE_TOKEN",
            timeout_seconds=60,
            **options,
        )
        result = session.prompt_once("Read references/hello.txt.", offered)
    evidence = json.dumps({"requests": replies, "events": events}, default=str)
    # The next model request must contain the actual protected tool's response.
    passed = result.success and len(replies) == 2 and "APPROVED_FIXTURE_CONTENT" in evidence
    assert "PRIVATE_HELD_OUT_CANARY" not in evidence
    report = dict(
        passed=passed,
        backend=config.backend.model_dump(mode="json"),
        provider_calls=0,
        fixture_responses=len(replies),
        native_success=result.success,
    )
    publish_bytes(root / "result.json", json.dumps(report, indent=2).encode())
    publish_bytes(root / "evidence.json.gz", gzip.compress(evidence.encode(), mtime=0))
    publish_bytes(root / "stdout.jsonl.gz", gzip.compress(result.raw_stdout.encode(), mtime=0))
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError("offline boundary fixture failed; evidence retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = CampaignConfig.model_validate_json(args.config.read_text())
    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
        _temp_dir=tempfile.mkdtemp(prefix="chia-boundary-fixture."),
    )
    try:
        check(config, args.output)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
