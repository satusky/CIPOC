"""Standalone workbench for inspecting one CIPOC extraction run."""

from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"

__all__ = ["WEB_DIR"]
