"""Experiment data; provider adapters do not own campaign policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from tools.chia_loop.evaluation_config import validate as validate_evaluation

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


class FixtureBackend(StrictRecord):
    kind: Literal["fixture"]
    scenario: Name


Backend = Annotated[
    CodexBackend | ClaudeBackend | VertexBackend | FixtureBackend, Field(discriminator="kind")
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


class Gem5Evaluation(StrictRecord):
    workloads: Names
    run_to_exit: bool = True


class TransferEvaluation(StrictRecord):
    champsim: ChampSimEvaluation | None = None
    gem5: Gem5Evaluation | None = None


class Evaluation(StrictRecord):
    schema_version: int = 1
    name: Name
    standard: Literal["DDR5"] = "DDR5"
    simpleo3: SimpleO3Evaluation
    transfer: TransferEvaluation = Field(default_factory=TransferEvaluation)

    @model_validator(mode="after")
    def cohorts(self):
        validate_evaluation(self.model_dump(mode="json", exclude_none=True))
        return self


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


class Experiment(StrictRecord):
    evaluation: Evaluation
    features: EvidenceFeatures = Field(default_factory=EvidenceFeatures)
    semantic_llm_check: bool = False
    synthetic_limits: SyntheticLimits = Field(default_factory=SyntheticLimits)

    def agent_view(self) -> dict:
        """Positive training projection; never pass the whole experiment to an agent."""
        simple = self.evaluation.simpleo3
        return {
            "standard": self.evaluation.standard,
            "training": list(simple.training),
            "instructions_per_core": simple.instructions_per_core,
            "minimum_oracle_owner_reads": simple.minimum_oracle_owner_reads,
            "features": self.features.model_dump(),
            "semantic_llm_check": self.semantic_llm_check,
            "synthetic_limits": (
                self.synthetic_limits.model_dump() if self.features.synthetic_diagnostics else None
            ),
            "promotion": "strict_two_objective_pareto",
        }


class EvaluationResources(StrictRecord):
    cpus: Annotated[int, Field(ge=1, le=12)] = 6
    build_timeout_seconds: PositiveInt = 1800
    simulation_timeout_seconds: PositiveInt = 1800
    memory_bytes: PositiveInt = 4 * 1024**3
    file_bytes: PositiveInt = 8 * 1024**3
    source_bytes: PositiveInt = 1024**2
    gzip_level: Annotated[int, Field(ge=0, le=9)] = 3


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
