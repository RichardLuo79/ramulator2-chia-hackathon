"""Freeze the 10+5 ChampSim study from existing inputs, without launching an LLM.

Only dataset/configuration metadata is read from the earlier studies. Evolved
models, selections, scores and conversations are not copied into a new campaign.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ramulator_chia.eval.champsim_placement import CAPACITY, balanced_mixes, scan_trace

from .archive import describe_payload
from .build import run_build_command
from .config import CampaignConfig, load
from .identity import canonical_json, file_sha256
from .review import PROMPT_FILES, PROMPTS, prompt_inventory
from .snapshots import publish_bytes

BACKENDS = {
    "gemini_flash": {
        "kind": "vertex_gemini",
        "model": "gemini-3.8-flash",
        "reasoning_effort": "high",
    },
    "astra_xhigh": {"kind": "codex_cli", "model": "gpt-6-astra", "reasoning_effort": "xhigh"},
    "deepseek_max": {
        "kind": "deepseek_api",
        "model": "deepseek-v4-flash",
        "reasoning_effort": "max",
    },
    "fable_xhigh": {"kind": "claude_cli", "model": "claude-fable-5-1", "reasoning_effort": "xhigh"},
}


def save(path, data):
    publish_bytes(path, canonical_json(data).encode())


def prepare_map(output, programs, pages, binary):
    """Use ChampSim's existing canonical allocator; do not implement a second mapper."""
    page_file = output.with_suffix(".pages")
    expanded = output.with_suffix("")
    body = f"CHAMPSIM_PAGES_V1 {len(programs)} 4096 {CAPACITY}\n".encode()
    for core, program in enumerate(programs):
        body += b"".join(
            f"{core} ".encode() + row + b"\n"
            for row in gzip.decompress(pages[program].read_bytes()).splitlines()
        )
    publish_bytes(page_file, body)
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CHAMPSIM_PLACEMENT_FILE", "RAMULATOR_CONFIG"}
    }
    with output.with_suffix(".log").open("wb") as log:
        subprocess.run(
            [str(binary), "--prepare-placement", str(page_file), str(expanded)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    publish_bytes(output, gzip.compress(expanded.read_bytes(), compresslevel=3, mtime=0))
    describe_payload(output, "placement.txt", codec="gzip")
    # These two explicitly created intermediates have a checked compressed copy.
    page_file.unlink()
    expanded.unlink()


def qualification_config(config: CampaignConfig) -> CampaignConfig:
    raw = config.model_dump(mode="json")
    raw.update(
        campaign_id="staged-native-qualification",
        backend={"kind": "fixture", "scenario": "comment-only-mcp"},
    )
    raw["run"].update(maximum_iterations=2)
    raw["run"]["resources"]["cpus"] = 16
    for stage in raw["experiment"]["stages"]:
        stage["rounds"] = 1
    cohort = raw["experiment"]["evaluation"]["champsim"]
    for group in ("training", "validation"):
        # Selection is from metadata alone, not scores. Prefer the familiar
        # MCF/PR one-core qualification pair; use the first frozen mix otherwise.
        chosen = []
        for cores in (1, 4, 8):
            options = [
                name for name in cohort[group] if len(cohort["cases"][name]["programs"]) == cores
            ]
            preferred = "mcf" if group == "training" else "gap_pr"
            chosen.append(
                next(
                    (
                        name
                        for name in options
                        if cores == 1 and cohort["cases"][name]["programs"] == [preferred]
                    ),
                    options[0],
                )
            )
        cohort[group] = chosen
    cohort["test"] = []
    names = {*cohort["training"], *cohort["validation"]}
    cohort["cases"] = {name: case for name, case in cohort["cases"].items() if name in names}
    programs = {p for case in cohort["cases"].values() for p in case["programs"]}
    cohort["traces"] = {p: v for p, v in cohort["traces"].items() if p in programs}
    return CampaignConfig.model_validate(raw)


def prepare(args):
    output = args.output.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    previous = load(args.single_core_config)
    old_cohort = previous.experiment.evaluation.champsim
    if old_cohort is None or len(old_cohort.training) != 14 or len(old_cohort.validation) != 14:
        raise ValueError("expected the frozen 14-training/14-validation input assignment")
    study = args.contention_study.absolute()
    protocol = json.loads((study / "protocol.json").read_text())
    if (
        protocol.get("completion_policy") != "background-replay"
        or protocol["capacity_bytes"] != CAPACITY
    ):
        raise ValueError("expected the qualified continuous-contention placement protocol")
    source_root = study.parent
    builds = {
        str(n): {
            "source": str(source_root / f"source{n}"),
            "binary": str(source_root / f"source{n}/bin/champsim-{n}core"),
        }
        for n in (1, 4, 8)
    }
    pins = json.loads((source_root / "build-pins.json").read_text())
    for cores, build in builds.items():
        if file_sha256(Path(build["binary"])) != pins["builds"][cores]["executable_sha256"]:
            raise ValueError("qualified ChampSim binary changed")
    repo = __import__("ramulator_chia.layout", fromlist=["RAMULATOR"]).RAMULATOR
    scanner = output / "trace-pages"
    run_build_command(
        [
            "/usr/bin/c++",
            "-O3",
            "-DNDEBUG",
            "-std=c++20",
            "-I" + str(source_root / "source1/inc"),
            str(repo / "resources/champsim_bridge/trace_pages.cc"),
            "-o",
            str(scanner),
        ],
        output / "scanner-build.log",
        cwd=repo,
        cpus=1,
        timeout_seconds=180,
    )
    training_sources = {
        n: old_cohort.traces[n].model_dump(mode="json") for n in old_cohort.training
    }
    with ThreadPoolExecutor(max_workers=args.scan_workers) as pool:
        futures = {
            n: pool.submit(scan_trace, output, scanner, n, s) for n, s in training_sources.items()
        }
        scanned = {n: future.result() for n, future in futures.items()}
    traces, pages = {}, {}
    for name, row in scanned.items():
        traces[name] = {**training_sources[name], "path": str(output / row["path"])}
        pages[name] = output / "pages" / (name + ".txt.gz")
    # Read only program/input/mapping metadata from the preserved study.
    old_cases = protocol["mixes"]
    for name, row in protocol["inputs"].items():
        if name == "sierra_a_4_0006":
            raise ValueError("excluded slow trace appeared in the preserved input cohort")
        source = study / row["path"]
        if file_sha256(source) != row["inventory"]["stored_sha256"]:
            raise ValueError("retained input changed: " + name)
        traces[name] = {
            "path": str(source),
            "sha256": row["inventory"]["stored_sha256"],
            "decoded_sha256": row["inventory"]["decoded_sha256"],
            "family": row["source"]["family"],
        }
    groups = {"training": [], "validation": [], "test": []}
    cases = {}
    new = [
        dict(name="training-c1-" + n, programs=[n], stage="training", cores=1)
        for n in old_cohort.training
    ]
    for cores in (4, 8):
        new += balanced_mixes(old_cohort.training, "training", cores)
    for row in [*new, *old_cases]:
        name, group = row["name"], row["stage"]
        mapping = output / "maps" / (name + ".txt.gz")
        mapping.parent.mkdir(exist_ok=True)
        if group == "training":
            prepare_map(mapping, row["programs"], pages, Path(builds[str(row["cores"])]["binary"]))
        else:
            original = study / "maps" / (name + ".txt.gz")
            if file_sha256(original) != row["placement"]["stored_sha256"]:
                raise ValueError("retained placement changed: " + name)
            publish_bytes(mapping, original.read_bytes())
        payload = describe_payload(mapping, "placement.txt", codec="gzip")
        cases[name] = dict(
            programs=row["programs"],
            placement=dict(
                path=str(mapping),
                sha256=payload.member.stored_sha256,
                decoded_sha256=payload.member.logical_sha256,
            ),
        )
        groups[group].append(name)
    # Stable aliases enumerate single-core cases first, then four- and eight-core.
    for group in groups:
        groups[group].sort(key=lambda n: (len(cases[n]["programs"]), n))
    if tuple(map(len, groups.values())) != (30, 30, 40):
        raise ValueError("expected 30 training, 30 validation and 40 final-test cases")
    validation_programs = {p for n in groups["validation"] for p in cases[n]["programs"]}
    if validation_programs != set(old_cohort.validation):
        raise ValueError("validation program membership changed")
    evaluation = dict(
        schema_version=2,
        name="dpc4-ddr5-staged-contention-v1",
        champsim=dict(
            **groups,
            traces=traces,
            cases=cases,
            builds=builds,
            warmup_instructions=2_000_000,
            roi_instructions=20_000_000,
            minimum_oracle_owner_reads=10_000,
            request_objective="champsim_exact_physical_filter",
        ),
    )
    experiment = dict(
        evaluation=evaluation,
        promotion_policy="llm_review",
        stages=[
            dict(name="single_core", rounds=10, core_counts=[1]),
            dict(name="multicore", rounds=5, core_counts=[1, 4, 8]),
        ],
        semantic_llm_check=True,
        validation_non_worsening=False,
        features=dict(synthetic_diagnostics=True, open_loop_diagnostics=True),
        prompt_sha256=prompt_inventory(),
    )
    for name in PROMPT_FILES:
        publish_bytes(output / "prompts" / name, (PROMPTS / name).read_bytes())
    configurations = {}
    for name, settings in BACKENDS.items():
        backend = copy.deepcopy(settings)
        if backend["kind"] == "vertex_gemini":
            backend.update(project=previous.backend.project, location=previous.backend.location)
        raw = dict(
            campaign_id=name + "-staged-10plus5-v1",
            backend=backend,
            experiment=experiment,
            run=dict(
                maximum_iterations=15,
                model_timeout_seconds=10_800,
                resources=dict(
                    cpus=4,
                    simulation_timeout_seconds=None,
                    memory_bytes=8 << 30,
                    file_bytes=32 << 30,
                ),
            ),
        )
        config = CampaignConfig.model_validate(raw)
        path = output / "configs" / (name + ".json")
        save(path, config.model_dump(mode="json"))
        configurations[name] = file_sha256(path)
    save(output / "qualification-config.json", qualification_config(config).model_dump(mode="json"))
    save(
        output / "preparation.json",
        dict(
            config_sha256=configurations,
            prompt_sha256=prompt_inventory(),
            original_split_config_sha256=file_sha256(args.single_core_config),
            contention_protocol_sha256=file_sha256(study / "protocol.json"),
            paid_calls=0,
            maximum_combined_workers=16,
            cloud_allocation=False,
            prior_models_or_scores_copied=False,
            cases={g: len(n) for g, n in groups.items()},
            test_status="previously examined historical cohort; hidden during this campaign",
        ),
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-core-config",
        type=Path,
        required=True,
        help="the preceding Gemini campaign config; only input/backend settings are read",
    )
    parser.add_argument("--contention-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-workers", type=int, choices=range(1, 17), default=4)
    args = parser.parse_args()
    print(prepare(args))


if __name__ == "__main__":
    main()
