"""Expose the common workspace through CHIA's MCP server and job polling.

There is one short-lived server per phase. Its grants never change in place.
Long evaluations use CHIA's existing single-job interface so HTTP calls remain
short; closing a phase waits for that job before stopping its server.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import contextmanager
from functools import wraps

import httpx
import ray
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from .chia_tools import AuthenticatedJobTool, RoleTokenVerifier

JOBS = frozenset({"build", "evaluate_training", "synthetic", "open_loop", "inspect_training"})


def result_value(result) -> dict:
    """FastMCP may return a JSON text block instead of structuredContent."""
    if result.isError:
        raise RuntimeError("MCP tool returned an error: " + str(result.content))
    value = result.structuredContent
    if value is None and len(result.content) == 1 and result.content[0].type == "text":
        value = json.loads(result.content[0].text)
    if not isinstance(value, dict):
        raise ValueError("the workspace tool did not return an object")
    return value


class WorkspaceTool(AuthenticatedJobTool):
    def __init__(self, workspace, token: str, lifetime_seconds: int | None):
        host = ray.util.get_node_ip_address()
        scope = f"iteration:{workspace.iteration}:{workspace.role}:{workspace.phase}"
        super().__init__(
            "ramulator",
            verifier=RoleTokenVerifier(
                hashlib.sha256(token.encode()).hexdigest(),
                scope,
                scope,
                None if lifetime_seconds is None else int(time.time()) + lifetime_seconds,
            ),
            auth=AuthSettings(
                issuer_url="http://localhost", resource_server_url=None, required_scopes=[scope]
            ),
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[host, host + ":*"],
                allowed_origins=["http://" + host, "http://" + host + ":*"],
            ),
            task_options={
                "num_cpus": 0,
                "scheduling_strategy": NodeAffinitySchedulingStrategy(
                    workspace.research.node_id, soft=False
                ),
            },
        )
        for name, method in workspace.methods().items():
            self.register(self.background(method) if name in JOBS else method, name=name)
        self.register(self.status, name="status")
        self.__post_init__()

    def background(self, method):
        @wraps(method)
        def start(*args, **kwargs):
            return self._job_start(lambda: method(*args, **kwargs))

        # FastMCP retains the typed argument schema of the workspace method.
        start.__doc__ = (method.__doc__ or "") + " Start a job; poll status until done."
        return start

    def status(self, wait_seconds: int = 10) -> dict:
        """Poll the most recent diagnostic job; no evaluation is started here."""
        return self._job_status(min(30, max(0, wait_seconds)))


async def drain(tool, token):
    """Use the normal MCP client to observe the owning actor, not a local copy."""
    url = f"http://{tool.hostname}:{tool.port}/{tool.name}/mcp"
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + token}, trust_env=False, timeout=60
    ) as client:
        async with streamable_http_client(url, http_client=client) as (reader, writer, _):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                while True:
                    result = await session.call_tool("status", {"wait_seconds": 10})
                    if not result_value(result)["running"]:
                        return


@contextmanager
def phase_tools(workspace, token: str, lifetime_seconds: int | None = None):
    # Ownership of this context bounds the grant. Do not confuse an individual
    # provider request timeout with the duration of an autonomous tool session.
    tool = WorkspaceTool(workspace, token, lifetime_seconds)
    try:
        yield [tool]
    finally:
        try:
            asyncio.run(drain(tool, token))
        finally:
            tool.stop()
