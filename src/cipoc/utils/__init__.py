"""Utility functions for CIPOC."""

# Config loading is dependency-light, so import it eagerly. Eager names become
# real module globals and never reach __getattr__ below.
from .utils import CipocConfig, DocumentsConfig, DEFAULT_CONFIG_PATH, load_config

# Databricks helpers are imported lazily so importing this package doesn't pull in
# runtime-only dependencies outside a Databricks environment.
_LAZY_DATABRICKS = {
    "get_databricks_token",
    "set_spark_env_variable",
    "get_llm_endpoint_base_url",
}

__all__ = [
    "CipocConfig",
    "DocumentsConfig",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    *sorted(_LAZY_DATABRICKS),
]


def __getattr__(name: str):
    """Lazily expose Databricks utilities to avoid eager runtime imports."""
    if name in _LAZY_DATABRICKS:
        from . import databricks_utils

        return getattr(databricks_utils, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
