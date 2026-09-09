"""Deploy two scoped CHIA MCP fixture tools and verify their native behavior.

Only a private two-CPU local cluster and its local HTTP endpoints are used. This
qualifies authentication, async failure and nested task admission, not a complete
native CLI sandbox or a real provider. No credential file or model adapter is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import httpx
import ray
from chia.base.ChiaFunction import ChiaFunction, get
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tools.chia_loop.framework.chia_tools import AuthenticatedJobTool, RoleTokenVerifier
from tools.chia_loop.framework.identity import canonical_json
from tools.chia_loop.framework.snapshots import publish_bytes, read_file


@ChiaFunction(num_cpus=1, max_retries=0)
def fixture_child(role: str, fail: bool) -> dict:
    if fail:
        raise ValueError("intentional native CHIA fixture failure")
    return {"role": role, "model_calls": 0, "node_id": ray.get_runtime_context().get_node_id()}


class FixtureTool(AuthenticatedJobTool):
    def __init__(self, name: str, root: Path, node_id: str, token: str, host: str):
        scope = "fixture:" + name
        super().__init__(
            name,
            verifier=RoleTokenVerifier(
                hashlib.sha256(token.encode()).hexdigest(), name, scope, int(time.time()) + 300
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
                "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id, soft=False),
            },
        )
        self.root = root
        self.register(self.read_approved, name="read_approved")
        self.register(self.start_job, name="start_job")
        self.register(self.job_status, name="job_status")
        self.__post_init__()

    def read_approved(self) -> dict:
        """Read the only approved fixture input; no caller-controlled path."""
        return {"content": read_file(self.root, "approved.txt", maximum_bytes=1024).decode()}

    def start_job(self, fail: bool = False) -> dict:
        """Start one fixture child through normal CHIA CPU admission."""
        return self._job_start(lambda: get(fixture_child.chia_remote(self.name, fail)))

    def job_status(self) -> dict:
        """Poll the inherited CHIA job for at most one second in this fixture."""
        return self._job_status(1)


def payload(response: httpx.Response) -> dict:
    response.raise_for_status()
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    value = json.loads(events[-1])
    if "error" in value:
        raise RuntimeError(value["error"])
    return value["result"]


def tool_result(response: httpx.Response) -> dict:
    result = payload(response)
    if result.get("isError"):
        raise RuntimeError(result["content"])
    return result.get("structuredContent") or json.loads(result["content"][0]["text"])


def check(output: Path) -> dict:
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ("role_a", "role_b"):
        publish_bytes(output / name / "approved.txt", (name + " permitted evidence").encode())
    publish_bytes(output / "held-out-canary.txt", b"not part of either tool's positive grant")
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    logs = tempfile.mkdtemp(prefix="chia-tools-ray.")
    repo = Path(__file__).resolve().parents[2]
    ray.init(
        address="local",
        num_cpus=2,
        object_store_memory=128 * 1024**2,
        include_dashboard=False,
        log_to_driver=False,
        namespace="tools-check-" + uuid4().hex,
        _temp_dir=logs,
        runtime_env={"env_vars": {"PYTHONPATH": str(repo)}},
    )
    deployed = []
    try:
        node_id = ray.get_runtime_context().get_node_id()
        host = ray.util.get_node_ip_address()
        tokens = [secrets.token_urlsafe(32), secrets.token_urlsafe(32)]
        for name, token in zip(("role_a", "role_b"), tokens):
            deployed.append(FixtureTool(name, output / name, node_id, token, host))
        urls = [f"http://{tool.hostname}:{tool.port}/{tool.name}/mcp" for tool in deployed]
        with httpx.Client(trust_env=False, timeout=15) as client:

            def request(index, method, params=None, *, credential=None):
                headers = {"Accept": "application/json, text/event-stream"}
                if credential is not None:
                    headers["Authorization"] = "Bearer " + credential
                body = {"jsonrpc": "2.0", "id": 1, "method": method}
                if params is not None:
                    body["params"] = params
                return client.post(urls[index], headers=headers, json=body)

            def call(index, name, **arguments):
                return tool_result(
                    request(
                        index,
                        "tools/call",
                        {"name": name, "arguments": arguments},
                        credential=tokens[index],
                    )
                )

            for index in range(2):
                assert request(index, "tools/list").status_code == 401
                assert request(index, "tools/list", credential=tokens[1 - index]).status_code == 401
                tools = payload(request(index, "tools/list", credential=tokens[index]))["tools"]
                assert {item["name"] for item in tools} == {
                    "read_approved",
                    "start_job",
                    "job_status",
                }
                assert (
                    call(index, "read_approved")["content"]
                    == deployed[index].name + " permitted evidence"
                )
                assert call(index, "start_job", fail=index == 0)["started"]
            results = {}
            deadline = time.monotonic() + 45
            while len(results) < 2 and time.monotonic() < deadline:
                for index in range(2):
                    if index not in results:
                        value = call(index, "job_status")
                        if value["done"]:
                            results[index] = value
            assert len(results) == 2, "nested CHIA fixture tools did not finish under two CPUs"
            assert results[0]["job_status"] == "failed" and not results[0]["running"]
            assert "intentional native CHIA fixture failure" in results[0]["error"]
            assert results[1]["job_status"] == "complete" and not results[1]["running"]
            assert results[1]["result"]["role"] == "role_b"
            deployed[0].stop()
            assert call(1, "read_approved")["content"] == "role_b permitted evidence"
        report = {
            "passed": True,
            "paid_calls": 0,
            "workers": 2,
            "missing_and_cross_role_tokens_rejected": True,
            "nested_jobs_terminal": results,
            "stopping_one_endpoint_preserved_the_other": True,
            "scope": (
                "native CHIA MCP auth/job fixtures; "
                "not native CLI filesystem or control-plane isolation"
            ),
            "ray_logs": logs,
        }
        publish_bytes(output / "authenticated-tools.json", canonical_json(report).encode())
        return report
    finally:
        for tool in deployed:
            tool.stop()
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(args.output.absolute())
    print(json.dumps({key: report[key] for key in ("passed", "paid_calls", "workers")}, indent=2))


if __name__ == "__main__":
    main()
