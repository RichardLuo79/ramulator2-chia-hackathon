"""Run the common research loop. Post-search evaluation is a separate command choice.

This entry point does not tune models, reconstruct previous conversations, or
run an archive/review stage. Provider differences belong to the native transport.
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import ray

from tools.eval import config as evaluation_config

from .agent import NativeSessions
from .archive import describe_payload
from .campaign import Campaign
from .config import load
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
):
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        output / "config.json", canonical_json(configuration.model_dump(mode="json")).encode()
    )

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
        num_cpus=configuration.run.resources.cpus,
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
            NativeSessions(transport, tariff, capacity_cooldown_seconds),
            hosts,
        )
        campaign = Campaign(output, configuration, research)
        status("search", maximum_iterations=configuration.run.maximum_iterations)
        selected = campaign.run_search()
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
    tariff = Tariff.model_validate_json(args.tariff.read_text())
    run(
        configuration,
        args.runtime,
        args.output,
        NativeTransport(),
        tariff=tariff,
        gem5_programs=args.gem5_programs,
        evaluate=args.evaluate,
        capacity_cooldown_seconds=args.capacity_cooldown_seconds,
    )


if __name__ == "__main__":
    main()
