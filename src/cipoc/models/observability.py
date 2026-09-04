"""JSON serialization contracts for run observability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, model_validator


AttemptMode = Literal["group", "individual", "repair"]
LLMAgent = Literal[
    "note_scanner", "note_retriever", "extractor", "orchestrator", "unknown"
]

_NonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
_PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class _ObservabilityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TokenDetails(RootModel[dict[str, _NonNegativeInt]]):
    """Provider token-detail counts, including unknown provider-specific keys.

    Detail counts are breakdowns of the corresponding input or output total,
    not additional tokens to add to that total.
    """

    root: dict[str, _NonNegativeInt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_keys(self):
        if any(not key for key in self.root):
            raise ValueError("Token detail names cannot be empty.")
        return self


class NormalizedTokenUsage(_ObservabilityModel):
    """Provider-reported usage normalized to one JSON-safe shape."""

    input_tokens: _NonNegativeInt = 0
    output_tokens: _NonNegativeInt = 0
    total_tokens: _NonNegativeInt = 0
    input_token_details: TokenDetails = Field(default_factory=TokenDetails)
    output_token_details: TokenDetails = Field(default_factory=TokenDetails)


class LLMUsageBucket(NormalizedTokenUsage):
    """Invocation counts and token usage for one aggregate grouping."""

    logical_calls: _NonNegativeInt = 0
    model_invocations: _NonNegativeInt = 0
    successful_invocations: _NonNegativeInt = 0
    failed_invocations: _NonNegativeInt = 0
    retry_invocations: _NonNegativeInt = 0
    usage_reported_invocations: _NonNegativeInt = 0
    missing_usage_invocations: _NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_invocation_counts(self):
        if (
            self.successful_invocations + self.failed_invocations
            != self.model_invocations
        ):
            raise ValueError(
                "Successful and failed invocations must equal model invocations."
            )
        if (
            self.usage_reported_invocations + self.missing_usage_invocations
            != self.model_invocations
        ):
            raise ValueError(
                "Usage-reported and missing-usage invocations must equal model "
                "invocations."
            )
        if self.logical_calls + self.retry_invocations != self.model_invocations:
            raise ValueError(
                "Logical calls plus retry invocations must equal model invocations."
            )
        if self.usage_reported_invocations == 0 and (
            self.input_tokens
            or self.output_tokens
            or self.total_tokens
            or self.input_token_details.root
            or self.output_token_details.root
        ):
            raise ValueError(
                "Token usage requires at least one usage-reporting invocation."
            )
        return self


class LLMUsageSummary(LLMUsageBucket):
    """Complete run usage totals and the same metrics by stable dimensions."""

    by_agent: dict[str, LLMUsageBucket] = Field(default_factory=dict)
    by_node: dict[str, LLMUsageBucket] = Field(default_factory=dict)
    by_model: dict[str, LLMUsageBucket] = Field(default_factory=dict)


class LLMPromptMessage(_ObservabilityModel):
    """One retained visible prompt message and any truncation metadata."""

    role: str = Field(min_length=1)
    content: str
    truncated: bool = False
    original_char_count: _NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_truncation(self):
        retained_count = len(self.content)
        if (
            self.original_char_count is not None
            and self.original_char_count < retained_count
        ):
            raise ValueError(
                "Original character count cannot be shorter than retained content."
            )
        if self.truncated and (
            self.original_char_count is None
            or self.original_char_count <= retained_count
        ):
            raise ValueError(
                "Truncated content requires an original character count greater "
                "than the retained length."
            )
        if (
            not self.truncated
            and self.original_char_count is not None
            and self.original_char_count != retained_count
        ):
            raise ValueError(
                "Untruncated content must retain its full original character count."
            )
        return self


class LLMExchange(_ObservabilityModel):
    """One completed model callback lifecycle correlated to a graph entity."""

    entity_key: str = Field(min_length=1)
    agent: LLMAgent
    node: str = Field(min_length=1)
    attempt: _PositiveInt = Field(description="Semantic extraction attempt number.")
    retry_ordinal: _PositiveInt | None = Field(
        default=None,
        description="Transport retry ordinal; absent for the first invocation.",
    )
    model: str | None = None
    prompt_messages: list[LLMPromptMessage] | None = None
    response: JsonValue | None = None
    usage: NormalizedTokenUsage | None = None
    error: str | None = None


class VariableAttempt(_ObservabilityModel):
    """Candidate and validation verdict emitted by one validation task."""

    attempt: _PositiveInt
    mode: AttemptMode
    candidate: JsonValue | None = None
    validation_errors: list[str] = Field(default_factory=list)
    is_valid: bool


class RunObservability(_ObservabilityModel):
    """All execution telemetry retained for one orchestrator run."""

    llm_content_captured: bool
    max_content_chars: _NonNegativeInt | None = None
    content_truncated: bool = False
    variable_attempts: dict[str, list[VariableAttempt]] = Field(default_factory=dict)
    llm_exchanges: dict[str, list[LLMExchange]] = Field(default_factory=dict)
    llm_usage_summary: LLMUsageSummary = Field(default_factory=LLMUsageSummary)

    @model_validator(mode="before")
    @classmethod
    def populate_exchange_entity_keys(cls, value: Any):
        """Accept the current keyed artifact shape while making keys explicit."""
        if not isinstance(value, Mapping):
            return value
        raw_exchanges = value.get("llm_exchanges")
        if not isinstance(raw_exchanges, Mapping):
            return value

        changed = False
        exchanges: dict[Any, Any] = {}
        for entity_key, raw_items in raw_exchanges.items():
            if not isinstance(raw_items, list):
                exchanges[entity_key] = raw_items
                continue
            items = []
            for raw_item in raw_items:
                if isinstance(raw_item, Mapping) and "entity_key" not in raw_item:
                    raw_item = {"entity_key": str(entity_key), **raw_item}
                    changed = True
                items.append(raw_item)
            exchanges[entity_key] = items

        if not changed:
            return value
        normalized = dict(value)
        normalized["llm_exchanges"] = exchanges
        return normalized

    @model_validator(mode="after")
    def validate_content_capture(self):
        any_truncated = False
        for entity_key, exchanges in self.llm_exchanges.items():
            if not entity_key:
                raise ValueError("LLM exchange map keys cannot be empty.")
            for exchange in exchanges:
                if exchange.entity_key != entity_key:
                    raise ValueError(
                        f"LLM exchange entity key {exchange.entity_key!r} does not "
                        f"match map key {entity_key!r}."
                    )
                if not self.llm_content_captured and (
                    exchange.prompt_messages is not None or exchange.response is not None
                ):
                    raise ValueError(
                        "Prompt and response content must be absent when LLM content "
                        "capture is disabled."
                    )
                if self.llm_content_captured and exchange.prompt_messages is None:
                    raise ValueError(
                        "Prompt messages must be present when LLM content capture "
                        "is enabled."
                    )
                for message in exchange.prompt_messages or ():
                    if self.max_content_chars is not None and (
                        len(message.content) > self.max_content_chars
                    ):
                        raise ValueError(
                            "Retained prompt content exceeds max_content_chars."
                        )
                    if message.truncated:
                        if self.max_content_chars is None:
                            raise ValueError(
                                "Truncated prompt content requires max_content_chars."
                            )
                        any_truncated = True

        if self.content_truncated != any_truncated:
            raise ValueError(
                "content_truncated must report whether any prompt message was truncated."
            )
        return self


__all__ = [
    "AttemptMode",
    "LLMAgent",
    "LLMExchange",
    "LLMPromptMessage",
    "LLMUsageBucket",
    "LLMUsageSummary",
    "NormalizedTokenUsage",
    "RunObservability",
    "TokenDetails",
    "VariableAttempt",
]
