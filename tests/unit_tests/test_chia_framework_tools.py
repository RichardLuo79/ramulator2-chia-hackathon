"""Real MCP auth/schema and CHIA async-tool fixtures, without model calls."""

import asyncio
import hashlib
import json
import secrets
import subprocess
import sys
import threading
import time

import httpx
import pytest
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings
from ray import cloudpickle

from tools.chia_loop.framework.chia_tools import AuthenticatedJobTool, RoleTokenVerifier
from tools.chia_loop.framework.diagnostic_cases import SyntheticRequest


def make_tool(name="role_a", *, expired=False, endpoint_lifetime=False):
    token = secrets.token_urlsafe(32)
    verifier = RoleTokenVerifier(
        hashlib.sha256(token.encode()).hexdigest(),
        name,
        "call:" + name,
        None if endpoint_lifetime else (1 if expired else int(time.time()) + 60),
    )
    tool = AuthenticatedJobTool(
        name,
        verifier=verifier,
        auth=AuthSettings(
            issuer_url="http://localhost",
            resource_server_url=None,
            required_scopes=[verifier.scope],
        ),
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["localhost", "localhost:*"],
            allowed_origins=["http://localhost", "http://localhost:*"],
        ),
        task_options={"num_cpus": 0},
    )
    return tool, token


def permitted_file() -> dict:
    """Read the fixture's sole permitted object."""
    return {"content": "explicitly permitted fixture evidence"}


def typed_diagnostic(candidate_id: str, cases: list[SyntheticRequest], trace_rows: int = 2) -> dict:
    return {
        "candidate_id": candidate_id,
        "cases": [case.model_dump(exclude_unset=True) for case in cases],
        "trace_rows": trace_rows,
    }


def test_typed_mcp_arguments_survive_transfer_to_a_fresh_process():
    tool, _ = make_tool()
    tool.register(typed_diagnostic, name="typed_diagnostic")
    program = """
import asyncio, json, sys
sys.path[:] = sys.argv[1:]
from ray import cloudpickle
tool = cloudpickle.loads(sys.stdin.buffer.read())
metadata = tool.mcp._tool_manager.get_tool('typed_diagnostic')
arguments = {'candidate_id': 'own-draft', 'cases': [{'mlp': 128, 'mode_b': {'mlp': 1}}]}
result = asyncio.run(metadata.run(arguments))
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, *sys.path],
        input=cloudpickle.dumps(tool),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    # Exercise the SDK's actual argument validation and typed function call.
    assert b'"candidate_id": "own-draft"' in result.stdout
    assert b'"mlp": 128' in result.stdout


def response_payload(response):
    # Preserve and parse the actual MCP SSE response; do not replace transport
    # behavior with handwritten JSON mocks or change the server to JSON mode.
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    return json.loads(events[-1])


def test_real_mcp_rejects_missing_cross_role_and_expired_credentials():
    a, token_a = make_tool()
    b, token_b = make_tool("role_b", endpoint_lifetime=True)
    expired, expired_token = make_tool("expired", expired=True)
    for tool in (a, b, expired):
        tool.register(permitted_file, name="read_permitted")

    async def check():
        for tool, own, other in (
            (a, token_a, token_b),
            (b, token_b, token_a),
            (expired, expired_token, token_a),
        ):
            app = tool.mcp.streamable_http_app()
            async with tool.mcp.session_manager.run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://localhost"
                ) as client:
                    headers = {"Accept": "application/json, text/event-stream"}
                    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
                    for invalid in (None, other, "unrecognized"):
                        attempted = dict(headers)
                        if invalid is not None:
                            attempted["Authorization"] = "Bearer " + invalid
                        response = await client.post("/mcp", headers=attempted, json=message)
                        assert response.status_code == 401
                        assert "read_permitted" not in response.text
                        assert "explicitly permitted" not in response.text
                    headers["Authorization"] = "Bearer " + own
                    response = await client.post("/mcp", headers=headers, json=message)
                    if tool is expired:
                        assert response.status_code == 401
                        continue
                    assert response.status_code == 200
                    listed = response_payload(response)["result"]["tools"]
                    assert [item["name"] for item in listed] == ["read_permitted"]
                    called = await client.post(
                        "/mcp",
                        headers=headers,
                        json={
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {"name": "read_permitted", "arguments": {}},
                        },
                    )
                    assert called.status_code == 200
                    assert "explicitly permitted fixture evidence" in called.text
                    hidden = await client.post(
                        "/mcp",
                        headers=headers,
                        json={
                            "jsonrpc": "2.0",
                            "id": 3,
                            "method": "tools/call",
                            "params": {"name": "query_campaign_database", "arguments": {}},
                        },
                    )
                    assert response_payload(hidden)["result"]["isError"]
                    rejected = await client.post(
                        "/mcp",
                        headers={**headers, "Origin": "http://elsewhere.invalid"},
                        json=message,
                    )
                    assert rejected.status_code == 403
                    assert "explicitly permitted" not in rejected.text

    asyncio.run(check())


def test_role_token_plaintext_is_not_serialized_with_the_tool():
    tool, token = make_tool()
    tool.register(permitted_file, name="read_permitted")
    data = cloudpickle.dumps(tool)
    assert token.encode() not in data
    copied = cloudpickle.loads(data)
    assert copied._server_actor is None
    assert copied._registered_names == {"read_permitted"}
    assert copied._job_status(0)["running"] is False


def test_job_exception_is_terminal_without_a_retry_thread():
    tool, _ = make_tool()
    calls = []

    def broken():
        calls.append(True)
        raise ValueError("fixture failure with preserved task evidence")

    assert tool._job_start(broken)["started"]
    result = tool._job_status(1)
    assert result == {
        "done": True,
        "running": False,
        "job_status": "failed",
        "error_type": "ValueError",
        "error": "fixture failure with preserved task evidence",
    }
    assert tool._job_status(0) == result
    assert calls == [True]


@pytest.mark.parametrize("value", [None, []])
def test_invalid_job_results_cannot_forge_polling_state(value):
    tool, _ = make_tool()
    tool._job_start(lambda: value)
    result = tool._job_status(1)
    assert result["done"] and not result["running"]
    assert result["job_status"] == "failed"


@pytest.mark.parametrize("value", [{"done": False}, {"running": True}, {"started": True}])
def test_evidence_fields_cannot_override_polling_state(value):
    tool, _ = make_tool()
    tool._job_start(lambda: value)
    assert tool._job_status(1) == {
        "done": True,
        "running": False,
        "job_status": "complete",
        "result": value,
    }


def test_busy_job_uses_chias_existing_single_job_admission():
    tool, _ = make_tool()
    ready, release = threading.Event(), threading.Event()

    def work():
        ready.set()
        assert release.wait(2)
        return {"candidate_id": "fixture"}

    try:
        assert tool._job_start(work)["started"]
        assert ready.wait(1)
        assert tool._job_start(lambda: {"unexpected": True})["started"] is False
    finally:
        release.set()
    assert tool._job_status(1) == {
        "done": True,
        "running": False,
        "job_status": "complete",
        "result": {"candidate_id": "fixture"},
    }


def test_exposed_names_are_positive_unique_and_never_truncated():
    tool, _ = make_tool()
    tool.register(permitted_file, name="read_permitted")
    for name in ("read_permitted", "../read", "x" * 64, "read;unexpected"):
        with pytest.raises(ValueError, match="tool name"):
            tool.register(permitted_file, name=name)
    tool._server_actor = object()
    with pytest.raises(RuntimeError, match="immutable"):
        tool.register(permitted_file, name="another")


def test_authentication_configuration_fails_closed():
    tool, _ = make_tool()
    verifier = tool.mcp._token_verifier
    settings = tool.mcp.settings
    for scopes, cpus, protection in (
        (["other-role"], 0, True),
        ([verifier.scope], 1, True),
        ([verifier.scope], False, True),
        ([verifier.scope], 0, False),
    ):
        with pytest.raises(ValueError):
            AuthenticatedJobTool(
                "invalid",
                verifier=verifier,
                auth=settings.auth.model_copy(update={"required_scopes": scopes}),
                transport_security=settings.transport_security.model_copy(
                    update={"enable_dns_rebinding_protection": protection}
                ),
                task_options={"num_cpus": cpus},
            )
