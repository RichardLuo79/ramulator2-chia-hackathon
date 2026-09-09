"""One common campaign with scripted agents and real full-window DRAM jobs.

No provider is instantiated. The script edits only a comment in the fixed-delay
seed. It qualifies integration, not agent reasoning or an improved DRAM design.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import secrets
import tempfile
from contextlib import contextmanager
from pathlib import Path

import ray
from pydantic_core import to_jsonable_python

from tools.chia_loop.framework.archive import describe_payload
from tools.chia_loop.framework.campaign import Campaign
from tools.chia_loop.framework.config import CampaignConfig, load
from tools.chia_loop.framework.diagnostic_cases import SyntheticRequest
from tools.chia_loop.framework.dram import DramResearch
from tools.chia_loop.framework.export import (
    export_campaign,
    load_frontend_recipes,
    validate_recipes,
)
from tools.chia_loop.framework.identity import canonical_json, file_sha256
from tools.chia_loop.framework.snapshots import publish_bytes
from tools.chia_loop.framework.tool_server import phase_tools, result_value
from tools.chia_loop.framework.workspace import Workspace
from tools.eval import config


class ScriptedSession:
    def __init__(self, view):
        self.view = view

    def snapshot(self):
        return self.view.snapshot()

    def summary(self):
        return self.view.summary()

    def turn(self, phase, inputs):
        view = self.view
        view.set_phase(phase)
        with offered_methods(view) as methods:
            result = self.work(phase, inputs, methods)
        view.record({"phase": phase, "inputs": inputs, "response": result, "paid_calls": 0})
        return result

    def work(self, phase, inputs, methods):
        view = self.view
        if phase == "explore":
            methods["read"]("references/task.md")
            methods["read"]("references/api.h")
            source = methods["read"]("draft/model.cpp")["text"]
            methods["write"]("draft/model.cpp", source + "\n// Scripted integration check.\n")
            assert methods["build"]()["passed"]
            assert methods["evaluate_training"]()["measurement"]["eligible_for_promotion"]
            workload = view.research.configuration.experiment.evaluation.simpleo3.training[0]
            if "synthetic" in methods:
                diagnostic = methods["synthetic"](SyntheticRequest())
                assert diagnostic["diagnostic_only"] and not diagnostic["eligible_for_promotion"]
            if "open_loop" in methods:
                diagnostic = methods["open_loop"](workload)
                assert diagnostic["diagnostic_only"] and not diagnostic["eligible_for_promotion"]
            methods["inspect_training"](workload, "oracle", trace="controller.csv.ch0", lines=10)
            methods["inspect_training"](
                workload, "oracle", trace="controller.csv.ch0", offset=10, lines=10
            )
        elif phase == "review":
            assert set(methods) == {"files", "read"}
            methods["read"]("draft/model.cpp")
        elif phase == "reflect":
            methods["write"](
                "notes/summary.md",
                "# Scripted integration fixture\n\n"
                "Only a source comment changed; no DRAM rule was developed. "
                "Native training and enabled diagnostics completed.\n\n"
                + canonical_json(
                    {
                        "promoted": inputs["promoted"],
                        "metrics": inputs["training"]["measurement"]["aggregate"],
                    }
                )
                + "\n",
            )
        result = {
            "text": "Scripted advisory feedback; not a semantic assessment.",
            "fixture": True,
            "phase": phase,
        }
        return result


@contextmanager
def offered_methods(view):
    if view.research.configuration.backend.scenario != "comment-only-mcp":
        yield view.methods()
        return
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    token = secrets.token_urlsafe(32)
    with phase_tools(view, token, 3600) as tools:
        tool = tools[0]
        url = f"http://{tool.hostname}:{tool.port}/{tool.name}/mcp"

        async def call(name, arguments):
            async with httpx.AsyncClient(
                headers={"Authorization": "Bearer " + token}, trust_env=False, timeout=60
            ) as client:
                async with streamable_http_client(url, http_client=client) as (r, w, _):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        result = await session.call_tool(name, arguments)
                        payload = result_value(result)
                        while payload.get("running"):
                            result = await session.call_tool("status", {"wait_seconds": 10})
                            payload = result_value(result)
                        if "job_status" in payload:
                            assert payload["job_status"] == "complete", payload
                            return payload["result"]
                        return payload

        def wrap(name, method):
            def invoke(*args, **kwargs):
                bound = inspect.signature(method).bind(*args, **kwargs)
                return asyncio.run(call(name, to_jsonable_python(bound.arguments)))

            return invoke

        yield {name: wrap(name, method) for name, method in view.methods().items()}


@contextmanager
def scripted_session(research, iteration, role, candidate, history):
    yield ScriptedSession(Workspace(research, iteration, role, candidate, history))


def settings(workers=None, iterations=None, mcp=None, evaluation_file=None, campaign_config=None):
    """Use an exact campaign configuration, or the small documented defaults."""
    if campaign_config is not None:
        if any(value is not None for value in (workers, iterations, mcp, evaluation_file)):
            raise ValueError("use campaign config or individual settings, not both")
        configuration = load(campaign_config)
        if (
            configuration.backend.kind != "fixture"
            or configuration.backend.scenario not in {"comment-only", "comment-only-mcp"}
            or configuration.execution != "native"
        ):
            raise ValueError("this command requires scripted agents and native evaluation")
        return configuration
    evaluation = (
        json.loads(evaluation_file.read_text())
        if evaluation_file
        else {
            "name": "full-window-integration-qualification",
            "simpleo3": {
                "training": ["429.mcf"],
                "test": ["433.milc"],
                "instructions_per_core": 20_000_000,
                "minimum_oracle_owner_reads": 10_000,
            },
        }
    )
    return CampaignConfig.model_validate(
        {
            "campaign_id": "native-integration-fixture",
            "backend": {
                "kind": "fixture",
                "scenario": "comment-only-mcp" if mcp else "comment-only",
            },
            "run": {
                "maximum_iterations": 2 if iterations is None else iterations,
                "resources": {"cpus": 6 if workers is None else workers},
            },
            "experiment": {
                "semantic_llm_check": True,
                "evaluation": evaluation,
            },
        }
    )


def run(
    runtime,
    output,
    workers=None,
    iterations=None,
    mcp=None,
    evaluation_file=None,
    gem5_programs=None,
    campaign_config=None,
    archive_output=None,
    frontend_recipes=None,
):
    configuration = settings(workers, iterations, mcp, evaluation_file, campaign_config)
    workers = configuration.run.resources.cpus
    transfer = configuration.experiment.evaluation.transfer
    recipes = {}
    if frontend_recipes is not None and archive_output is None:
        raise ValueError("frontend recipes are used with --archive-output")
    if archive_output is not None:
        if frontend_recipes is not None:
            recipes = load_frontend_recipes(frontend_recipes)
        frontends = {name for name in ("champsim", "gem5") if getattr(transfer, name) is not None}
        if set(recipes) != frontends:
            raise ValueError("archive recipes must match the configured transfer frontends")
        if transfer.gem5 is not None:
            if gem5_programs is None:
                raise ValueError("source-only export needs the built guest suite manifest")
            from tools.eval.gem5.build_suite import source_recipes

            guests = source_recipes(gem5_programs)
            recipes.update(
                {
                    "benchmark:" + name: guests["benchmark:" + name]
                    for name in transfer.gem5.workloads
                }
            )
        validate_recipes(recipes, frontends)
    output.mkdir(parents=True, exist_ok=False)
    from tools.chia_loop.framework.inputs import prepare

    cases, hosts = prepare(configuration, output, gem5_programs)
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    ray.init(
        address="local",
        num_cpus=workers,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
        namespace="chia-clean-campaign",
        _temp_dir=tempfile.mkdtemp(prefix="chia-clean-campaign-ray."),
    )
    try:
        research = DramResearch(
            configuration,
            output,
            runtime,
            cases,
            describe_payload(Path(config.MESS_CURVES["DDR5"]), "inputs/mess.txt"),
            ray.get_runtime_context().get_node_id(),
            scripted_session,
            hosts,
        )
        campaign = Campaign(output, configuration, research)
        selection = campaign.run_search()
        evaluation = campaign.evaluate()
        event_hashes = {str(p.relative_to(output)): file_sha256(p) for p in output.glob("events/*")}
        assert campaign.run_search() == selection
        assert campaign.evaluate() == evaluation
        assert event_hashes == {
            str(p.relative_to(output)): file_sha256(p) for p in output.glob("events/*")
        }
        report = {
            "passed": True,
            "paid_calls": 0,
            "scripted_agents": True,
            "native_evaluation": True,
            "workers": workers,
            "scope": "configured full-window cohort with scripted agents; not model optimization",
            "cohorts": {group: [case.workload for case in rows] for group, rows in cases.items()},
            "selection": selection,
            "postrun": evaluation,
            "event_hashes": event_hashes,
        }
        if archive_output is not None:
            sealed = export_campaign(campaign, research, archive_output, external_recipes=recipes)
            report["archive"] = {
                "path": str(sealed.path),
                "sha256": sealed.sha256,
                "bytes": sealed.bytes,
            }
        publish_bytes(output / "qualification.json", canonical_json(report).encode())
        return report
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, help="evaluation CPUs; default 6")
    parser.add_argument("--iterations", type=int, help="scripted iterations; default 2")
    parser.add_argument(
        "--mcp", action="store_true", default=None, help="exercise real CHIA MCP tool dispatch"
    )
    config_options = parser.add_mutually_exclusive_group()
    config_options.add_argument(
        "--evaluation", type=Path, help="explicit DDR5 evaluation cohort JSON"
    )
    config_options.add_argument(
        "--campaign-config", type=Path, help="complete fixture campaign JSON"
    )
    parser.add_argument("--gem5-programs", type=Path, help="source-built guest suite manifest")
    parser.add_argument("--archive-output", type=Path, help="seal source/evidence after evaluation")
    parser.add_argument(
        "--frontend-recipes", type=Path, help="pinned frontend source/build recipes"
    )
    args = parser.parse_args()
    report = run(
        args.runtime.absolute(),
        args.output.absolute(),
        args.workers,
        args.iterations,
        args.mcp,
        args.evaluation,
        args.gem5_programs,
        args.campaign_config,
        args.archive_output,
        args.frontend_recipes,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("passed", "paid_calls", "workers", "scope", "archive")
                if key in report
            }
        )
    )
