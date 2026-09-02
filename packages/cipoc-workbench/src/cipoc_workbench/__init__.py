"""Standalone workbench for inspecting one CIPOC extraction run."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = PACKAGE_DIR / "example"
WEB_DIR = PACKAGE_DIR / "web"

__all__ = ["EXAMPLE_DIR", "WEB_DIR"]
