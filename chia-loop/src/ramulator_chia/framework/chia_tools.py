"""Small, source-qualified extensions to CHIA's existing MCP job tool.

CHIA still owns deployment, transport, job threads and polling. This adapter
configures the MCP SDK's bearer authentication and makes a throwing job terminal.
It is not a sandbox, a durable job store or an alternative tool protocol.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Callable

from chia.base.tools.AsyncJobTool import AsyncJobTool
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings


@dataclass(frozen=True)
class RoleTokenVerifier:
    """Verify one preprovisioned role token using the SDK's TokenVerifier hook.

    Only a digest reaches the tool actor. The token belongs to the trusted model
    transport, not the agent workspace or evidence. Provisioning/refresh stays
    with the launch boundary; this class creates no OAuth/login endpoints.
    Roles have separate tokens. Short-lived phase endpoints can bind validity
    to their lifetime instead of expiring during a long diagnostic job.
    """

    token_sha256: str
    client_id: str
    scope: str
    expires_at: int | None

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.token_sha256):
            raise ValueError("role token needs a SHA-256 digest, not a raw credential")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.client_id, self.scope)
        ):
            raise ValueError("role authentication needs explicit client and scope identities")
        if self.expires_at is not None and (
            type(self.expires_at) is not int or self.expires_at <= 0
        ):
            raise ValueError("role token expiry must be a Unix timestamp or endpoint lifetime")

    async def verify_token(self, token: str) -> AccessToken | None:
        if not hmac.compare_digest(hashlib.sha256(token.encode()).hexdigest(), self.token_sha256):
            return None
        return AccessToken(
            token=token, client_id=self.client_id, scopes=[self.scope], expires_at=self.expires_at
        )


class AuthenticatedJobTool(AsyncJobTool):
    """CHIA async tool with explicit authentication and terminal error receipts.

    Register the role's permitted methods with ``register``, then call CHIA's
    ``__post_init__`` to deploy. Replacing the empty FastMCP object before that
    point is necessary because CHIA 1.0.1 does not forward its auth arguments.
    No unauthenticated endpoint is started, even temporarily.
    """

    def __init__(
        self,
        name: str,
        *,
        verifier: RoleTokenVerifier,
        auth: AuthSettings,
        transport_security: TransportSecuritySettings,
        task_options: dict,
    ):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("tool server name must be a simple MCP identifier")
        if auth.required_scopes != [verifier.scope]:
            raise ValueError("the endpoint must require exactly its role's token scope")
        if not transport_security.enable_dns_rebinding_protection:
            raise ValueError("the role endpoint requires an explicit host/origin policy")
        if type(task_options.get("num_cpus")) not in {int, float} or task_options["num_cpus"] != 0:
            raise ValueError("the polling tool must not reserve CPUs needed by its child tasks")
        super().__init__(name, task_options=task_options)
        self._mcp_options = {
            "name": name,
            "stateless_http": True,
            "token_verifier": verifier,
            "auth": auth,
            "transport_security": transport_security,
        }
        self.mcp = FastMCP(**self._mcp_options)
        self._registered_names: set[str] = set()
        self._registered_functions: dict[str, Callable] = {}

    def register(self, function: Callable, *, name: str):
        """Reject names that the inspected Vertex adapter would truncate/collide."""
        if self._server_actor is not None:
            raise RuntimeError("a deployed role's method grant is immutable")
        if (
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
            or len(self.name + "__" + name) > 64
            or name in self._registered_names
        ):
            raise ValueError("duplicate, unsafe or overlong exposed tool name")
        self.mcp.add_tool(function, name=name)
        self._registered_names.add(name)
        self._registered_functions[name] = function

    def __getstate__(self):
        state = super().__getstate__()
        # CHIA moves the tool to its worker using cloudpickle. The MCP SDK's
        # dynamically generated Pydantic argument validators do not round-trip
        # reliably to a fresh process when they contain nested model fields.
        # Transfer the public constructor inputs/callables, not a live server.
        state.pop("mcp", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # Reuse SDK schema construction in the owning worker. CHIA still owns
        # the actor, server lifecycle and async job state; no replacement server
        # or per-instrument schema/argument decoder is introduced here.
        self.mcp = FastMCP(**self._mcp_options)
        for name, function in self._registered_functions.items():
            self.mcp.add_tool(function, name=name)

    def _job_start(self, work: Callable[[], dict]) -> dict:
        def terminal_work():
            try:
                result = work()
                if not isinstance(result, dict):
                    raise ValueError("job result must be a dict")
                # Keep application output nested; an evidence field named
                # 'running' must not overwrite the transport's actual state.
                return {"job_status": "complete", "result": result}
            except Exception as exc:
                # The durable native operation retains full failed/partial
                # evidence. This is its short-lived transport status, not a
                # substitute receipt or an automatic retry.
                return {"job_status": "failed", "error_type": type(exc).__name__, "error": str(exc)}

        return super()._job_start(terminal_work)
