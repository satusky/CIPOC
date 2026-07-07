"""CIPOC: NAACCR variable extraction from clinical notes."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cipoc")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
