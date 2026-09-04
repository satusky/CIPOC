"""Versioned serialization contracts for orchestrator runs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    UUID4,
    field_validator,
    model_validator,
)

from .case import Case
from .notes import NoteCorpusDescriptors, NoteDigest, ProcessedClinicalNote
from .observability import RunObservability
from .variables import TargetGroup


_NonNegativeFloat = Annotated[float, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class _RunModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class OrchestratorConfigFingerprint(_RunModel):
    """Resolved configuration and resource identity that shaped a run."""

    agent_llm_config: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    retry: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    max_extraction_attempts: _PositiveInt
    variable_groups_digest: str = Field(min_length=1)
    data_dictionary_digest: str | None = None
    site_data_dictionary_digest: str | None = None
    prompt_digests: dict[str, str] = Field(default_factory=dict)
    cipoc_version: str | None = None

    @field_validator("agent_llm_config")
    @classmethod
    def reject_excluded_llm_fields(cls, value):
        for agent, config in value.items():
            excluded = {"api_key", "tools"}.intersection(config)
            if excluded:
                names = ", ".join(sorted(excluded))
                raise ValueError(
                    f"Agent {agent!r} config contains excluded field(s): {names}."
                )
        return value


class OrchestratorRunInfo(_RunModel):
    """Identity, timing, completion state, and provenance for one run."""

    run_id: UUID4
    started_at: AwareDatetime
    finished_at: AwareDatetime
    duration_seconds: _NonNegativeFloat
    status: Literal["completed", "failed"]
    config_fingerprint: OrchestratorConfigFingerprint
    contains_phi: Literal[True] = True

    @model_validator(mode="after")
    def validate_timing(self):
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at.")
        return self


class OrchestratorRunInputs(_RunModel):
    """Caller inputs and configured extraction plan for a run."""

    target_variables: list[TargetGroup] = Field(default_factory=list)
    structured_data: dict[int, str] = Field(default_factory=dict)


class OrchestratorRunCorpus(_RunModel):
    """Post-scan corpus and deterministic characterization used by the run."""

    note_corpus: dict[int | str, ProcessedClinicalNote] = Field(default_factory=dict)
    note_digests: dict[int | str, NoteDigest] = Field(default_factory=dict)
    note_corpus_descriptors: NoteCorpusDescriptors | None = None


class OrchestratorRunResult(_RunModel):
    """Canonical completed-run artifact consumed across the JSON boundary."""

    schema_version: Literal["1.0"] = "1.0"
    run: OrchestratorRunInfo
    case: Case
    inputs: OrchestratorRunInputs
    corpus: OrchestratorRunCorpus
    observability: RunObservability

    @model_validator(mode="after")
    def validate_completed_status(self):
        if self.run.status != "completed":
            raise ValueError("A completed run result requires status 'completed'.")
        return self


class OrchestratorRunFailure(_RunModel):
    """Partial artifact retained when orchestration raises before finalization."""

    schema_version: Literal["1.0"] = "1.0"
    run: OrchestratorRunInfo
    inputs: OrchestratorRunInputs
    corpus: OrchestratorRunCorpus | None = None
    observability: RunObservability
    error: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_failed_status(self):
        if self.run.status != "failed":
            raise ValueError("A run failure requires status 'failed'.")
        return self


class OrchestratorRunError(RuntimeError):
    """Raised for an orchestrator failure while retaining its partial artifact."""

    def __init__(self, failure: OrchestratorRunFailure) -> None:
        self.failure = failure
        super().__init__(failure.error)


__all__ = [
    "OrchestratorConfigFingerprint",
    "OrchestratorRunCorpus",
    "OrchestratorRunError",
    "OrchestratorRunFailure",
    "OrchestratorRunInfo",
    "OrchestratorRunInputs",
    "OrchestratorRunResult",
]
