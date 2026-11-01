"""Experiment data; provider adapters do not own campaign policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from ramulator_chia.evaluation_config import validate as validate_evaluation

Name = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
ModelIdentifier = Annotated[str, Field(min_length=1, pattern=r"^[^\s\x00]+$")]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True)


class FileIdentity(StrictRecord):
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def relative_path(self):
        from .archive import safe_name

        safe_name(self.path)
        return self


class DeclaredLimit(StrictRecord):
    value: PositiveInt
    kind: Literal["provider_constraint", "operator_safety_guard", "experimental_treatment"]
    reason: Annotated[str, Field(min_length=1)]


class ContextCompaction(StrictRecord):
    """Operational API context policy; native CLI compaction remains native."""

    adk_python: str
    input_limit_tokens: PositiveInt
    token_counter: Literal["vertex_count_tokens", "utf8_upper_bound"]
    trigger_fraction: Annotated[float, Field(gt=0.5, lt=1)] = 0.8
    retain_tool_rounds: Annotated[int, Field(ge=1)] = 8

    @model_validator(mode="after")
    def explicit_interpreter(self):
        if not Path(self.adk_python).is_absolute():
            raise ValueError("ADK needs an explicit isolated Python interpreter")
        return self


class CodexBackend(StrictRecord):
    kind: Literal["codex_cli"]
    model: ModelIdentifier
    reasoning_effort: Name
    auto_compact_tokens: DeclaredLimit | None = None
    service_tier: Name | None = None


class ClaudeBackend(StrictRecord):
    kind: Literal["claude_cli"]
    model: ModelIdentifier
    reasoning_effort: Name
    output_tokens: DeclaredLimit | None = None


class VertexBackend(StrictRecord):
    kind: Literal["vertex_gemini"]
    model: ModelIdentifier
    reasoning_effort: Name
    project: Name
    location: Name
    output_tokens: DeclaredLimit | None = None
    include_thoughts: bool = True

    @model_validator(mode="after")
    def preserve_exposed_thoughts(self):
        if not self.include_thoughts:
            raise ValueError("exposed native thoughts must be retained")
        return self


class VertexClaudeBackend(StrictRecord):
    """Anthropic Messages on Vertex; no CLI or subscription credentials."""

    kind: Literal["vertex_claude"]
    model: Literal["claude-fable-5-1"] = "claude-fable-5-1"
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "xhigh"
    project: Name
    location: Literal["global"] = "global"
    output_tokens: DeclaredLimit = DeclaredLimit(
        value=128_000, kind="provider_constraint", reason="Fable 5.1 maximum output tokens"
    )
    compaction_input_tokens: Annotated[int, Field(ge=50_000, le=800_000)] = 800_000

    @model_validator(mode="after")
    def provider_limits(self):
        if self.output_tokens.value > 128_000:
            raise ValueError("Fable 5.1 supports at most 128000 output tokens")
        return self


class FixtureBackend(StrictRecord):
    kind: Literal["fixture"]
    scenario: Name


class DeepSeekBackend(StrictRecord):
    kind: Literal["deepseek_api"]
    # Preserve existing campaign identities; new Flash runs use the current API name.
    model: Literal["deepseek-v4-flash", "deepseek-flash"] = "deepseek-v4-flash"
    reasoning_effort: Literal["low", "high", "max"] = "high"
    output_tokens: DeclaredLimit | None = None


Backend = Annotated[
    CodexBackend | ClaudeBackend | VertexBackend | VertexClaudeBackend | DeepSeekBackend | FixtureBackend,
    Field(discriminator="kind"),
]


def _ordered(value):
    if type(value) not in (tuple, list):
        raise ValueError("expected an ordered array")
    return tuple(value)


Names = Annotated[tuple[Name, ...], BeforeValidator(_ordered)]


class SimpleO3Evaluation(StrictRecord):
    training: Names
    test: Names
    instructions_per_core: Annotated[int, Field(ge=20_000_000)]
    minimum_oracle_owner_reads: PositiveInt


class ChampSimEvaluation(StrictRecord):
    workloads: Names
    warmup_instructions: Annotated[int, Field(ge=2_000_000)]
    roi_instructions: Annotated[int, Field(ge=20_000_000)]


class ChampSimTrace(StrictRecord):
    """Operator-only location and content identity of one compressed input."""

    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    decoded_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    family: Name


class ChampSimPlacement(StrictRecord):
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    decoded_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ChampSimCase(StrictRecord):
    """An ordered program mix; a one-program case uses the same protocol."""

    programs: Names
    placement: ChampSimPlacement

    @model_validator(mode="after")
    def distinct_programs(self):
        if len(self.programs) not in (1, 4, 8) or len(set(self.programs)) != len(self.programs):
            raise ValueError("a staged case needs 1, 4 or 8 distinct programs")
        return self


class ChampSimBuild(StrictRecord):
    binary: str
    source: str


class ChampSimSearchEvaluation(StrictRecord):
    training: Names
    validation: Names
    test: Names = ()
    traces: dict[Name, ChampSimTrace]
    warmup_instructions: Annotated[int, Field(ge=2_000_000)]
    roi_instructions: Annotated[int, Field(ge=20_000_000)]
    minimum_oracle_owner_reads: PositiveInt
    request_objective: Literal["champsim_exact_physical_filter"]
    cases: dict[Name, ChampSimCase] = Field(default_factory=dict)
    builds: dict[Literal["1", "4", "8"], ChampSimBuild] = Field(default_factory=dict)

    @model_validator(mode="after")
    def disjoint_cohorts(self):
        if not self.training or not self.validation:
            raise ValueError("ChampSim search needs training and validation cohorts")
        names = (*self.training, *self.validation, *self.test)
        inventory = self.cases if self.cases else self.traces
        if len(names) != len(set(names)) or set(names) != set(inventory):
            raise ValueError("trace inventory must cover exactly the disjoint cohorts")
        if self.cases:
            programs = {p for case in self.cases.values() for p in case.programs}
            if programs != set(self.traces):
                raise ValueError("case programs must cover exactly the trace inventory")
            cores = {str(len(case.programs)) for case in self.cases.values()}
            if set(self.builds) != cores:
                raise ValueError("staged cases need an explicit build for each core count")
        elif self.builds:
            raise ValueError("per-core builds require explicit staged cases")
        families, hashes = set(), set()
        for group in (self.training, self.validation, self.test):
            programs = (
                {p for name in group for p in self.cases[name].programs} if self.cases else group
            )
            group_families = {self.traces[name].family for name in programs}
            group_hashes = {self.traces[name].decoded_sha256 for name in programs}
            if families & group_families or hashes & group_hashes:
                raise ValueError("evaluation cohorts share a family or decoded input")
            families.update(group_families)
            hashes.update(group_hashes)
        return self


class Gem5Evaluation(StrictRecord):
    workloads: Names
    run_to_exit: bool = True


class TransferEvaluation(StrictRecord):
    champsim: ChampSimEvaluation | None = None
    gem5: Gem5Evaluation | None = None


class Evaluation(StrictRecord):
    schema_version: Literal[1, 2] = 1
    name: Name
    standard: Literal["DDR5"] = "DDR5"
    simpleo3: SimpleO3Evaluation | None = None
    champsim: ChampSimSearchEvaluation | None = None
    transfer: TransferEvaluation = Field(default_factory=TransferEvaluation)

    @model_validator(mode="after")
    def cohorts(self):
        if (self.simpleo3 is None) == (self.champsim is None):
            raise ValueError("select exactly one primary evaluation frontend")
        if self.simpleo3 is not None:
            validate_evaluation(self.model_dump(mode="json", exclude_none=True))
        elif self.schema_version != 2 or self.transfer.champsim is not None:
            raise ValueError("ChampSim search requires schema 2 and no duplicate ChampSim transfer")
        return self

    @property
    def primary(self):
        return self.champsim if self.champsim is not None else self.simpleo3

    @property
    def request_objective(self):
        return self.champsim.request_objective if self.champsim else "complete_stable_id"

    @property
    def validation(self):
        return self.champsim.validation if self.champsim else ()

    @property
    def observation(self):
        return (
            {"scope": "champsim_dram_controller_lifecycle", "timebase": "ramulator_controller_cycles"}
            if self.champsim else
            {"scope": "simpleo3_logical_llc", "timebase": "simpleo3_frontend_cycles"}
        )


class EvidenceFeatures(StrictRecord):
    synthetic_diagnostics: bool = True
    open_loop_diagnostics: bool = True


class SyntheticLimits(StrictRecord):
    requests_per_stream: PositiveInt = 100_000
    maximum_total_reads: PositiveInt = 1_000_000
    maximum_trace_rows: Annotated[int, Field(ge=0)] = 100

    @model_validator(mode="after")
    def population_fits(self):
        if self.requests_per_stream > self.maximum_total_reads:
            raise ValueError("default synthetic population exceeds the case guard")
        return self


class CampaignStage(StrictRecord):
    name: Name
    rounds: PositiveInt
    core_counts: Annotated[tuple[Literal[1, 4, 8], ...], BeforeValidator(_ordered)]

    @model_validator(mode="after")
    def ordered_cores(self):
        if not self.core_counts or tuple(sorted(set(self.core_counts))) != self.core_counts:
            raise ValueError("stage core counts must be nonempty, unique and increasing")
        return self


class Experiment(StrictRecord):
    evaluation: Evaluation
    features: EvidenceFeatures = Field(default_factory=EvidenceFeatures)
    semantic_llm_check: bool = False
    validation_non_worsening: bool = False
    synthetic_limits: SyntheticLimits = Field(default_factory=SyntheticLimits)
    stages: Annotated[tuple[CampaignStage, ...], BeforeValidator(_ordered)] = ()
    promotion_policy: Literal["strict_two_objective_pareto", "llm_review"] = (
        "strict_two_objective_pareto"
    )
    prompt_sha256: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validation_gate(self):
        if self.validation_non_worsening and not self.evaluation.validation:
            raise ValueError("a validation gate requires a validation cohort")
        if self.stages:
            cohort = self.evaluation.champsim
            if not cohort or not cohort.cases or self.promotion_policy != "llm_review":
                raise ValueError("staged search needs explicit ChampSim cases and LLM review")
            if self.evaluation.transfer.champsim or self.evaluation.transfer.gem5:
                raise ValueError("staged ChampSim search does not include frontend-transfer jobs")
            if self.validation_non_worsening:
                raise ValueError("LLM promotion replaces the validation non-worsening gate")
            if len({stage.name for stage in self.stages}) != len(self.stages):
                raise ValueError("stage names must be unique")
            previous = set()
            for stage in self.stages:
                current = set(stage.core_counts)
                if not previous <= current:
                    raise ValueError("later stages must retain earlier core-count groups")
                for group in (cohort.training, cohort.validation):
                    if not current <= {len(cohort.cases[name].programs) for name in group}:
                        raise ValueError("stage has a missing training or validation core group")
                previous = current
        elif self.promotion_policy == "llm_review":
            raise ValueError("LLM promotion requires an explicit stage schedule")
        return self

    def stage_for(self, iteration: int) -> CampaignStage | None:
        if not self.stages:
            return None
        if iteration < 1:
            raise ValueError("iteration must be positive")
        stop = 0
        for stage in self.stages:
            stop += stage.rounds
            if iteration <= stop:
                return stage
        raise ValueError("iteration exceeds the stage schedule")

    def case_names(self, group: str, stage: CampaignStage | None = None) -> tuple:
        names = getattr(self.evaluation.primary, group)
        if stage is None:
            return names
        cases = self.evaluation.champsim.cases
        return tuple(name for name in names if len(cases[name].programs) in stage.core_counts)

    def agent_view(self, stage: CampaignStage | None = None) -> dict:
        """Positive training projection; never pass the whole experiment to an agent."""
        stage = stage or (self.stages[0] if self.stages else None)
        primary = self.evaluation.primary
        result = {
            "standard": self.evaluation.standard,
            "training": list(self.case_names("training", stage)),
            "minimum_oracle_owner_reads": primary.minimum_oracle_owner_reads,
            "features": self.features.model_dump(),
            "semantic_llm_check": self.semantic_llm_check,
            "synthetic_limits": (
                self.synthetic_limits.model_dump() if self.features.synthetic_diagnostics else None
            ),
            "promotion": self.promotion_policy,
        }
        if self.evaluation.champsim:
            result.update(
                frontend="champsim",
                warmup_instructions=primary.warmup_instructions,
                roi_instructions=primary.roi_instructions,
                request_objective=primary.request_objective,
                validation_feedback="two metrics per anonymous workload and two equal-workload means",
                validation_non_worsening=self.validation_non_worsening,
            )
        else:
            result["instructions_per_core"] = primary.instructions_per_core
        if self.stages:
            stage = stage or self.stages[0]
            result.update(
                stage=stage.model_dump(mode="json"),
                schedule=[s.model_dump(mode="json") for s in self.stages],
                semantic_llm_check="one final promotion and contract review; no revision phase",
                validation_feedback="anonymous core/request/tail metrics and reliability counts only",
                completion_policy="background-replay",
                placement="timing-independent, canonical frames in 8 GiB",
                request_window="per-core foreground admissions; no drain; background excluded",
                training_programs={
                    name: list(primary.cases[name].programs)
                    for name in self.case_names("training", stage)
                },
            )
        return result


class EvaluationResources(StrictRecord):
    cpus: Annotated[int, Field(ge=1, le=16)] = 6
    build_timeout_seconds: PositiveInt = 1800
    simulation_timeout_seconds: PositiveInt | None = 1800
    diagnostic_timeout_seconds: PositiveInt = 1800
    memory_bytes: PositiveInt = 4 * 1024**3
    file_bytes: PositiveInt = 8 * 1024**3
    source_bytes: PositiveInt = 1024**2
    gzip_level: Annotated[int, Field(ge=0, le=9)] = 3


class ExecutionOverrides(StrictRecord):
    """Host allocation and billing route, never scientific campaign settings."""

    evaluation_workers: Annotated[int, Field(ge=1, le=120)] | None = None
    vertex_project: str | None = None

    def workers(self, configuration):
        return self.evaluation_workers or configuration.run.resources.cpus

    def validate_backend(self, configuration):
        if self.vertex_project is not None:
            import re

            if configuration.backend.kind != "vertex_gemini":
                raise ValueError("Vertex execution project requires the Vertex backend")
            if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", self.vertex_project):
                raise ValueError("invalid Vertex execution project ID")


def preserved_configuration(configuration, previous: dict | None) -> dict:
    """Keep pre-deadline campaign records intact; allow no other drift."""
    current = configuration.model_dump(mode="json")
    if previous is None:
        return current
    comparable = configuration.model_dump(mode="json")
    if "diagnostic_timeout_seconds" not in previous["run"]["resources"]:
        comparable["run"]["resources"].pop("diagnostic_timeout_seconds")
    if comparable != previous:
        raise ValueError("campaign configuration changed")
    return previous


def diagnostic_policy(configuration, original: dict) -> dict:
    """An operational amendment, not a change to the scientific protocol."""
    resources = configuration.run.resources
    timeout = resources.diagnostic_timeout_seconds
    if resources.simulation_timeout_seconds is not None:
        timeout = min(timeout, resources.simulation_timeout_seconds)
    return {
        "diagnostic_timeout_seconds": resources.diagnostic_timeout_seconds,
        "effective_timeout_seconds": timeout,
        "scope": ["ControllerReplay", "SyntheticPattern"],
        "clock": "wall time from simulator subprocess launch",
        "legacy_configuration_amendment":
            "diagnostic_timeout_seconds" not in original["run"]["resources"],
    }


class Run(StrictRecord):
    maximum_iterations: PositiveInt
    model_timeout_seconds: PositiveInt = 3600
    evaluation_execution: Literal["native", "offline_fixture"] = "native"
    maximum_attempts: Annotated[int, Field(ge=1, le=10)] = 3
    retry_delay_seconds: Annotated[int, Field(ge=0, le=60)] = 5
    resources: EvaluationResources = Field(default_factory=EvaluationResources)


class CampaignConfig(StrictRecord):
    campaign_id: Name
    experiment: Experiment
    backend: Backend
    run: Run

    @model_validator(mode="after")
    def stage_rounds(self):
        if (
            self.experiment.stages
            and sum(s.rounds for s in self.experiment.stages) != self.run.maximum_iterations
        ):
            raise ValueError("iteration guard must equal the sum of stage rounds")
        return self

    @property
    def execution(self) -> str:
        # Scripted proposer responses can exercise real native DRAM evaluation.
        # Conversely, a numerical fixture must never be labeled a native result.
        return self.run.evaluation_execution


def load(path: Path) -> CampaignConfig:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate configuration key: {key}")
            result[key] = value
        return result

    return CampaignConfig.model_validate(json.loads(path.read_text(), object_pairs_hook=unique))
