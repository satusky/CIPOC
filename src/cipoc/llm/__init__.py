"""LLM clients and rate limiting utilities."""

from typing import NamedTuple

from .base import BaseAgentModel, LLMConfig
from .limiter import MaxRetriesExceededError, RateLimiter
from .openai import OpenAIAgentModel, OpenAIConfig, OpenAIReasoning


class ProviderClasses(NamedTuple):
    """The config + agent-model classes that implement a given provider."""
    config: type[LLMConfig]
    model: type[BaseAgentModel]


def provider_classes(provider: str | None) -> ProviderClasses:
    """Resolve a provider string to its (config, model) classes.

    Single source of truth for provider dispatch, replacing a central class
    registry. Providers beyond the default are imported lazily so selecting one
    never forces importing another provider's (possibly-unavailable) SDK —
    important in the airgapped Databricks runtime. Add a provider with one
    ``case`` that lazily imports its module.
    """
    match provider:
        case "openai":
            return ProviderClasses(OpenAIConfig, OpenAIAgentModel)
        # case "anthropic":
        #     from .anthropic import AnthropicConfig, AnthropicAgentModel  # lazy: optional dep
        #     return ProviderClasses(AnthropicConfig, AnthropicAgentModel)
        case _:
            raise ValueError(
                f"Unknown LLM provider {provider!r}. Supported: 'openai'."
            )


def agent_model_for(provider: str | None) -> type[BaseAgentModel]:
    """Resolve a provider string to its ``BaseAgentModel`` subclass."""
    return provider_classes(provider).model


def config_for(provider: str | None) -> type[LLMConfig]:
    """Resolve a provider string to its ``LLMConfig`` subclass."""
    return provider_classes(provider).config


__all__ = [
    "BaseAgentModel",
    "LLMConfig",
    "OpenAIAgentModel",
    "OpenAIConfig",
    "OpenAIReasoning",
    "RateLimiter",
    "MaxRetriesExceededError",
    "ProviderClasses",
    "provider_classes",
    "agent_model_for",
    "config_for",
]
