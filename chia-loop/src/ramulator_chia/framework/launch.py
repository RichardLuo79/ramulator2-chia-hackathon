"""Run the common research loop. Post-search evaluation is a separate command choice.

This entry point does not tune models, reconstruct previous conversations, or
run an archive/review stage. Provider differences belong to the native transport.
"""

import argparse
import json
import os
import platform
import tempfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import ray

from ramulator_chia.eval import config as evaluation_config

from .agent import NativeSessions
from .archive import describe_payload
from .campaign import Campaign
from .config import (ContextCompaction, ExecutionOverrides, diagnostic_policy, load,
                     preserved_configuration)
from .dram import DramResearch
from .identity import canonical_json
from .inputs import prepare
from .snapshots import publish_bytes, replace_file
from .usage import Tariff


def run(
    configuration,
    runtime,
    output,
    transport,
    *,
    tariff=None,
    gem5_programs=None,
    evaluate=False,
    capacity_cooldown_seconds=3600,
    context_compaction=None,
    execution_overrides=None,
    round_finish_deadline=None,
):
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    overrides = execution_overrides or ExecutionOverrides()
    overrides.validate_backend(configuration)
    workers = overrides.workers(configuration)
    if overrides.vertex_project != getattr(transport, "vertex_execution_project", None):
        raise ValueError("recorded Vertex route differs from the actual transport")
    execution = {
        "schema_version": 1,
        "campaign_id": configuration.campaign_id,
        "started_at": time.time(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "overrides": overrides.model_dump(mode="json"),
        "effective_evaluation_workers": workers,
        "configured_evaluation_workers": configuration.run.resources.cpus,
        "effective_vertex_project": overrides.vertex_project or
            getattr(configuration.backend, "project", None),
        "scientific_configuration_changed": False,
        "round_finish_deadline": round_finish_deadline,
    }
    publish_bytes(output / "operations" / (uuid4().hex + ".json"),
                  canonical_json(execution).encode())
    replace_file(output, "execution-policy.json", canonical_json(execution).encode())
    if context_compaction is not None:
        if configuration.backend.kind not in {"vertex_gemini", "deepseek_api"}:
            raise ValueError("ADK compaction is only for API backends")
        publish_bytes(output / "context-compaction.json",
                      canonical_json(context_compaction.model_dump(mode="json")).encode())
    elif (output / "context-compaction.json").exists():
        raise ValueError("continued campaign requires its recorded context compaction policy")
    config_path = output / "config.json"
    original = preserved_configuration(
        configuration, json.loads(config_path.read_text()) if config_path.exists() else None
    )
    publish_bytes(config_path, canonical_json(original).encode())
    publish_bytes(output / "diagnostic-policy.json",
                  canonical_json(diagnostic_policy(configuration, original)).encode())

    def status(stage, **values):
        value = {"stage": stage, "time": time.time(), "pid": os.getpid(), **values}
        replace_file(output, "status.json", canonical_json(value).encode())
        print(canonical_json(value), flush=True)

    status("preparing_inputs")
    # Operational retry timing does not change the model, prompts or scoring.
    replace_file(
        output,
        "retry-policy.json",
        canonical_json(
            {
                "capacity_cooldown_seconds": capacity_cooldown_seconds,
                "maximum_attempts": configuration.run.maximum_attempts,
            }
        ).encode(),
    )
    cases, hosts = prepare(configuration, output, gem5_programs)
    ray.init(
        address="local",
        num_cpus=workers,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024**2,
        namespace=configuration.campaign_id,
        _temp_dir=tempfile.mkdtemp(prefix="chia-paid-ray."),
    )
    try:
        research = DramResearch(
            configuration,
            output,
            runtime,
            cases,
            describe_payload(Path(evaluation_config.MESS_CURVES["DDR5"]), "inputs/mess.txt"),
            ray.get_runtime_context().get_node_id(),
            NativeSessions(transport, tariff, capacity_cooldown_seconds, context_compaction),
            hosts,
            evaluation_workers=workers,
        )
        campaign = Campaign(output, configuration, research)
        status("search", maximum_iterations=configuration.run.maximum_iterations)
        from .execution import round_admission
        selected = campaign.run_search(before_round=(
            None if round_finish_deadline is None else
            round_admission(campaign.state, round_finish_deadline)
        ))
        status("search_complete", selection=selected)
        if evaluate:
            status("held_out_evaluation")
            campaign.evaluate()
            status("evaluation_complete")
        return selected
    except BaseException as error:
        # Native events/checkpoints carry details. Do not print credential-bearing
        # exception payloads or silently restart an ambiguous paid operation.
        status("stopped", error_type=type(error).__name__)
        raise
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tariff", required=True, type=Path)
    parser.add_argument("--gem5-programs", type=Path)
    parser.add_argument("--evaluation-workers", type=int,
                        help="recorded host-only worker override (1–120); scientific config stays fixed")
    parser.add_argument("--vertex-execution-project",
                        help="recorded Vertex billing/execution route; retain the native session settings")
    parser.add_argument("--round-finish-deadline",
                        help="UTC ISO timestamp: defer new rounds unlikely to finish before this host deadline")
    parser.add_argument("--context-compaction", type=Path,
                        help="API context policy JSON; requires a separate ADK environment")
    parser.add_argument(
        "--capacity-cooldown-seconds",
        type=int,
        default=3600,
        help="wait after a terminal capacity error (default: 3600)",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate frozen source after search"
    )
    parser.add_argument(
        "--allow-paid", action="store_true", help="explicit inference authorization"
    )
    args = parser.parse_args()
    if not args.allow_paid:
        parser.error("paid inference requires --allow-paid")
    if args.capacity_cooldown_seconds < 0:
        parser.error("capacity cooldown must be nonnegative")
    from .transport import NativeTransport

    configuration = load(args.config)
    deadline = None
    if args.round_finish_deadline:
        parsed = datetime.fromisoformat(args.round_finish_deadline.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            parser.error("round finish deadline must include its timezone")
        deadline = parsed.timestamp()
    overrides = ExecutionOverrides(evaluation_workers=args.evaluation_workers,
                                   vertex_project=args.vertex_execution_project)
    overrides.validate_backend(configuration)
    tariff = Tariff.model_validate_json(args.tariff.read_text())
    run(
        configuration,
        args.runtime,
        args.output,
        NativeTransport(vertex_execution_project=overrides.vertex_project),
        tariff=tariff,
        gem5_programs=args.gem5_programs,
        evaluate=args.evaluate,
        capacity_cooldown_seconds=args.capacity_cooldown_seconds,
        context_compaction=(None if args.context_compaction is None else
                            ContextCompaction.model_validate_json(args.context_compaction.read_text())),
        execution_overrides=overrides,
        round_finish_deadline=deadline,
    )


if __name__ == "__main__":
    main()
