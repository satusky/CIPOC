"""Internal workbench for inspecting one CIPOC extraction run.

The frontend is the static directory ``web/``; ``server.py`` adds the endpoints
a static file server cannot provide (ground-truth delivery and feedback
persistence). Nothing here is imported by the runtime package.

``server`` is deliberately not imported at package level: it needs ``fastapi``,
which lives in the optional ``workbench`` extra and is absent from the pinned
DBR-18.2 install. ``import cipoc.workbench`` must stay free of it.
"""

from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"

__all__ = ["WEB_DIR"]
