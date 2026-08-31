"""FastAPI app serving the workbench: static frontend, plus reference and feedback state.

The workbench frontend is a directory of static files that reads one JSON state
dump. Everything it needs to *display* a run is available over plain
``http.server`` — this module exists for the two things a static server cannot
do: hand the page a ground-truth file chosen at launch, and accept the
annotations a reviewer writes back.

``fastapi`` / ``uvicorn`` are workbench-only dependencies (the ``workbench``
extra), so this module is imported only when actually serving — never by
``cipoc.workbench`` itself, and never by the runtime package. That keeps the
airgapped DBR-18.2 install free of a web framework.

The frontend degrades rather than breaks: every endpoint here is optional from
its point of view. Served without ``--ground-truth`` the comparison features
stay hidden; served without this module at all, the feedback form renders
disabled with an explanatory line.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


WEB_DIR = Path(__file__).resolve().parent / "web"

# The entity kinds an annotation can attach to. Fixed rather than open: a typo in
# a URL would otherwise silently create a fourth bucket nothing ever reads.
ANNOTATION_KINDS = ("variable", "group", "note")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write via a temp file in the same directory, then rename.

    A reviewer's annotations are the only thing here that cannot be regenerated
    by re-running the pipeline, and a half-written file is worse than a stale
    one. ``os.replace`` is atomic within a filesystem, and the temp file is
    created alongside the target so it never crosses one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        # Leave no debris if serialization or the rename fails.
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _empty_document(state_path: Path | None) -> dict[str, Any]:
    return {
        "state_file": str(state_path) if state_path else None,
        "updated_at": None,
        "annotations": {kind: {} for kind in ANNOTATION_KINDS},
    }


def _load_feedback(path: Path | None, state_path: Path | None) -> dict[str, Any]:
    """Read the feedback document, tolerating absence and partial shapes."""
    document = _read_json(path)
    if not isinstance(document, dict):
        return _empty_document(state_path)
    annotations = document.get("annotations")
    if not isinstance(annotations, dict):
        annotations = {}
    # Normalize so callers can index any kind unconditionally, including one
    # added to ANNOTATION_KINDS after a file was already written.
    document["annotations"] = {
        kind: annotations.get(kind) if isinstance(annotations.get(kind), dict) else {}
        for kind in ANNOTATION_KINDS
    }
    document.setdefault("state_file", str(state_path) if state_path else None)
    return document


def _is_empty(annotation: dict[str, Any]) -> bool:
    """An annotation with neither a flag nor a comment carries no information.

    Clearing every box is how a reviewer retracts a note, so it deletes the entry
    rather than leaving an empty husk that later reads as "reviewed, no issue".
    """
    return not annotation.get("flags") and not (annotation.get("note") or "").strip()


def build_app(
    *,
    state_path: Path | None = None,
    ground_truth_path: Path | None = None,
    feedback_path: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app serving the workbench frontend and its side files."""
    app = FastAPI(title="CIPOC Workbench", docs_url=None, redoc_url=None)

    @app.get("/api/ground-truth")
    def ground_truth() -> JSONResponse:
        """The reference values, or an empty object when none was supplied.

        Empty rather than 404: absence is the normal case, not an error, and the
        frontend treats both identically anyway.
        """
        data = _read_json(ground_truth_path)
        return JSONResponse(data if isinstance(data, dict) else {})

    @app.get("/api/feedback")
    def feedback() -> JSONResponse:
        return JSONResponse(_load_feedback(feedback_path, state_path))

    @app.put("/api/feedback/{kind}/{entity_id}")
    def put_feedback(kind: str, entity_id: str, annotation: dict[str, Any]) -> JSONResponse:
        if kind not in ANNOTATION_KINDS:
            raise HTTPException(status_code=404, detail=f"Unknown annotation kind {kind!r}.")
        if feedback_path is None:
            raise HTTPException(
                status_code=409,
                detail="This server was started without --feedback, so there is nowhere to save.",
            )

        # Re-read before every write: the document is small, saves are seconds
        # apart at worst, and a reviewer who hand-edits the file between two
        # saves should not have that edit silently overwritten.
        document = _load_feedback(feedback_path, state_path)
        bucket = document["annotations"][kind]

        record = {
            "flags": [str(f) for f in annotation.get("flags") or []],
            "expected": annotation.get("expected") or None,
            "note": (annotation.get("note") or "").strip(),
            "updated_at": _now(),
        }
        if _is_empty(record):
            bucket.pop(entity_id, None)
            record = None
        else:
            bucket[entity_id] = record

        document["updated_at"] = _now()
        _write_json_atomic(feedback_path, document)
        return JSONResponse({"kind": kind, "id": entity_id, "annotation": record})

    if state_path is not None:
        # Served at the path the frontend already fetches, so STATE_URL in
        # app.js — and the whole no-server fallback — stays untouched. Declared
        # before the static mount, which would otherwise serve the committed
        # copy sitting in web/.
        @app.get("/case_state.json")
        def case_state() -> FileResponse:
            if not state_path.is_file():
                raise HTTPException(status_code=404, detail=f"{state_path} does not exist.")
            return FileResponse(state_path, media_type="application/json")

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app
