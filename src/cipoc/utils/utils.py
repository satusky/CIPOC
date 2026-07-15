"""Configuration loading for CIPOC.

Loads a YAML config defining a default LLM configuration plus optional
per-agent overrides, expands ``${VAR}`` environment placeholders, and builds
merged :class:`~cipoc.llm.OpenAIConfig` / :class:`~cipoc.llm.RateLimiter`
objects for each agent.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from cipoc.llm import LLMConfig, RateLimiter, config_for


class DocumentsConfig(BaseModel):
    """Filesystem locations for source documents used by the pipeline.

    Paths are taken as-is from the config (CWD-relative unless absolute),
    consistent with how the config file itself is resolved. ``extra="allow"``
    leaves room for the planned document library to add fields without a
    loader change.
    """

    documents_path: Path = Field(
        default=Path("documents"),
        description="Directory holding the document library.",
    )
    data_dictionary_path: Path | None = Field(
        default=None,
        description="Path to the NAACCR data dictionary JSON.",
    )
    rules_path: Path | None = Field(
        default=None,
        description="Directory holding the compiled manual rule store (manifest.json plus per-manual rule files).",
    )
    model_config = ConfigDict(extra="allow")

# User-edited runtime config; resolved relative to the current working directory.
DEFAULT_CONFIG_PATH = Path("config/config.yaml")

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` placeholders in strings using os.environ."""
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            var = match.group(1)
            if var not in os.environ:
                raise KeyError(f"Environment variable '{var}' referenced in config is not set")
            return os.environ[var]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a deep merge of ``override`` onto ``base`` without mutating inputs."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class CipocConfig:
    """Parsed CIPOC configuration with default LLM settings and agent overrides."""
    def __init__(self, raw: dict):
        self._raw = raw
        self.defaults: dict = raw.get("llm", {}) or {}
        self.agents: dict[str, dict] = raw.get("agents", {}) or {}
        self._documents: dict = raw.get("documents", {}) or {}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CipocConfig":
        """Load and parse a config file (defaults to ``config/config.yaml``)."""
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found at '{path}'. Pass an explicit path to "
                f"load_config(), or create '{DEFAULT_CONFIG_PATH}' relative to the "
                f"working directory."
            )
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(_expand_env(raw))

    def agent_settings(self, agent: str) -> dict:
        """Return the merged settings dict for ``agent`` (defaults + overrides)."""
        override = self.agents.get(agent) or {}
        return _deep_merge(self.defaults, override)

    def llm_config(self, agent: str | None = None) -> LLMConfig:
        """Build the provider-specific :class:`LLMConfig` for ``agent`` (or the
        defaults if None), choosing the config class from the ``provider`` setting."""
        settings = self.agent_settings(agent) if agent is not None else copy.deepcopy(self.defaults)
        settings.pop("limiter", None)
        return config_for(settings.get("provider", "openai"))(**settings)

    def limiter(self, agent: str | None = None) -> RateLimiter:
        """Build the :class:`RateLimiter` for ``agent`` (or the defaults if None)."""
        settings = self.agent_settings(agent) if agent is not None else copy.deepcopy(self.defaults)
        limiter_settings = settings.get("limiter") or {}
        return RateLimiter(**limiter_settings)

    def documents(self) -> DocumentsConfig:
        """Build the :class:`DocumentsConfig` from the ``documents`` section."""
        return DocumentsConfig(**self._documents)


def load_config(path: str | Path | None = None) -> CipocConfig:
    """Convenience wrapper around :meth:`CipocConfig.load`."""
    return CipocConfig.load(path)


__all__ = [
    "CipocConfig",
    "DocumentsConfig",
    "load_config",
    "DEFAULT_CONFIG_PATH",
]
