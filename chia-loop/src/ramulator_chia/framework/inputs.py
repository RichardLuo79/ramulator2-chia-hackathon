"""Bind configured workloads to the existing evaluators; no model or selection policy."""

import shlex
from pathlib import Path

from ramulator_chia.eval import artifacts, config

from .archive import describe_payload
from .evaluation import SimpleO3Case
from .external_frontends import ExternalHost, TransferCase, describe_champsim_trace, prepare_host
from .identity import file_sha256


def _host(directory, frontend, binary, source):
    """A resumed search reuses its checked frontend inputs, not a new installation."""
    receipt = directory / "host.json"
    if receipt.exists():
        host = ExternalHost(directory, file_sha256(receipt))
        if host.record()["frontend"] != frontend:
            raise ValueError("prepared frontend differs from the requested frontend")
        return host
    return prepare_host(directory, frontend, binary, source)


def prepare(configuration, output: Path, gem5_programs: Path | None = None):
    evaluation = configuration.experiment.evaluation
    cases, hosts = {}, {}
    for group in ("training", "test") if evaluation.simpleo3 else ():
        cases[group] = tuple(
            SimpleO3Case(
                name,
                group,
                evaluation.simpleo3.instructions_per_core,
                (
                    describe_payload(
                        path := artifacts.resolve(Path(config.trace_path(name))),
                        "inputs/" + name,
                        codec="gzip" if path.suffix == ".gz" else "none",
                    ),
                ),
            )
            for name in getattr(evaluation.simpleo3, group)
        )
    if evaluation.champsim:
        cohort = evaluation.champsim
        if cohort.cases:
            return _staged_champsim(cohort, output)
        host = _host(
            output / "frontend-builds/champsim", "champsim",
            config.CHAMPSIM_BIN, config.CHAMPSIM_DIR,
        )
        hosts["champsim"] = host
        for group in ("training", "validation", "test"):
            rows = []
            for name in getattr(cohort, group):
                source = cohort.traces[name]
                payload, inventory = describe_champsim_trace(Path(source.path), host)
                if (inventory["stored_sha256"], inventory["decoded_sha256"]) != (
                    source.sha256, source.decoded_sha256,
                ):
                    raise ValueError("configured ChampSim input changed: " + name)
                rows.append(TransferCase(
                    "champsim", name, payload, stage=group,
                    warmup_instructions=cohort.warmup_instructions,
                    roi_instructions=cohort.roi_instructions,
                    instruction_inventory=inventory,
                ))
            cases[group] = tuple(rows)
    transfer = evaluation.transfer
    if transfer.champsim is not None:
        host = _host(
            output / "frontend-builds/champsim",
            "champsim",
            config.CHAMPSIM_BIN,
            config.CHAMPSIM_DIR,
        )
        hosts["champsim"] = host
        rows = []
        for name in transfer.champsim.workloads:
            payload, inventory = describe_champsim_trace(
                config.CHAMPSIM_TRACES / (name + ".champsimtrace.xz"), host
            )
            rows.append(
                TransferCase(
                    "champsim",
                    name,
                    payload,
                    warmup_instructions=transfer.champsim.warmup_instructions,
                    roi_instructions=transfer.champsim.roi_instructions,
                    instruction_inventory=inventory,
                )
            )
        cases["champsim"] = tuple(rows)
    if transfer.gem5 is not None:
        if gem5_programs is not None:
            from ramulator_chia.eval.gem5.build_suite import load_programs

            programs = load_programs(gem5_programs)
        else:
            programs = {
                name: (config.GEM5_BENCH_DIR / name, tuple(shlex.split(arguments)))
                for name, arguments in config.GEM5_BENCH.items()
            }
        missing = set(transfer.gem5.workloads) - set(programs)
        if missing:
            raise ValueError("missing built gem5 programs: " + ", ".join(sorted(missing)))
        binary = Path(config.GEM5_BIN)
        source = next(p for p in binary.parents if (p / "SConstruct").is_file())
        hosts["gem5"] = _host(output / "frontend-builds/gem5", "gem5", binary, source)
        cases["gem5"] = tuple(
            TransferCase(
                "gem5",
                name,
                describe_payload(programs[name][0], "inputs/benchmark", executable=True),
                arguments=programs[name][1],
            )
            for name in transfer.gem5.workloads
        )
    return cases, hosts


def _staged_champsim(cohort, output):
    """Reuse the qualified placement/contention cases with explicit per-core builds."""
    from ramulator_chia.eval.champsim_mixes import MultiProgramCase

    hosts = {}
    for cores, build in cohort.builds.items():
        hosts["champsim:" + cores] = _host(
            output / "frontend-builds" / ("champsim-c" + cores),
            "champsim",
            Path(build.binary),
            Path(build.source),
        )
    hosts["champsim"] = hosts["champsim:1"]
    # Decode/CRC-check each trace once; bind the same bytes to each relevant host.
    verified = {}
    for name, source in cohort.traces.items():
        payload, inventory = describe_champsim_trace(Path(source.path), hosts["champsim"])
        if (inventory["stored_sha256"], inventory["decoded_sha256"]) != (
            source.sha256,
            source.decoded_sha256,
        ):
            raise ValueError("configured ChampSim input changed: " + name)
        verified[name] = payload, inventory
    cases = {}
    for group in ("training", "validation", "test"):
        rows = []
        for name in getattr(cohort, group):
            spec = cohort.cases[name]
            host = hosts[f"champsim:{len(spec.programs)}"]
            members = []
            for program in spec.programs:
                payload, inventory = verified[program]
                members.append(
                    TransferCase(
                        "champsim",
                        program,
                        payload,
                        stage=group,
                        warmup_instructions=cohort.warmup_instructions,
                        roi_instructions=cohort.roi_instructions,
                        instruction_inventory={**inventory, "host_sha256": host.receipt_sha256},
                    )
                )
            placement = describe_payload(Path(spec.placement.path), "placement.txt", codec="gzip")
            if (placement.member.stored_sha256, placement.member.logical_sha256) != (
                spec.placement.sha256,
                spec.placement.decoded_sha256,
            ):
                raise ValueError("configured physical placement changed: " + name)
            first = members[0]
            rows.append(
                MultiProgramCase(
                    frontend="champsim",
                    workload=name,
                    stage=group,
                    payload=first.payload,
                    warmup_instructions=cohort.warmup_instructions,
                    roi_instructions=cohort.roi_instructions,
                    instruction_inventory=first.instruction_inventory,
                    companions=tuple(members[1:]),
                    placement=placement,
                    completion_policy="background-replay",
                )
            )
        cases[group] = tuple(rows)
    return cases, hosts
