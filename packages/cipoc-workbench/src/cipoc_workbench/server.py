"""FastAPI app serving the workbench: static frontend, plus reference and feedback state.

The workbench frontend is a directory of static files that reads one canonical
run-result JSON artifact. Everything it needs to *display* a run is available
over plain ``http.server`` — this module exists for the two things a static
server cannot do: hand the page a ground-truth file chosen at launch, and accept
the annotations a reviewer writes back.

The package is independent from the CIPOC runtime and depends only on its web
server libraries. It communicates with CIPOC through the versioned JSON artifact.

The frontend degrades rather than breaks: every endpoint here is optional from
its point of view. Served without ``--ground-truth`` the comparison features
stay hidden; served without this module at all, the feedback form renders
disabled with an explanatory line.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import WEB_DIR

# The entity kinds an annotation can attach to. Fixed rather than open: a typo in
# a URL would otherwise silently create a fourth bucket nothing ever reads.
ANNOTATION_KINDS = ("variable", "group", "note")
_FEEDBACK_LOCK = threading.Lock()


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
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        # Leave no debris if serialization or the rename fails.
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _canonical_run_id(value: Any) -> str:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Expected a canonical run UUID (lowercase, with hyphens).")
    return value


def _empty_document(state_path: Path | None, run_id: str | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "state_file": str(state_path) if state_path else None,
        "updated_at": None,
        "annotations": {kind: {} for kind in ANNOTATION_KINDS},
    }


def _load_feedback(
    path: Path | None,
    state_path: Path | None,
    run_id: str | None,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Only a missing file is an empty review; never repair corrupt saved data."""
    if path is None:
        return _empty_document(state_path, run_id)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate feedback document key.")
            result[key] = value
        return result

    def finite_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite feedback document number.")
        return number

    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(
                stream, object_pairs_hook=unique_object,
                parse_constant=finite_number, parse_float=finite_number,
            )
    except FileNotFoundError:
        return _empty_document(state_path, run_id)
    except (OSError, ValueError, RecursionError) as exc:
        raise HTTPException(status_code=500, detail="Cannot read the feedback document.") from exc
    if not isinstance(document, dict):
        raise HTTPException(status_code=500, detail="The feedback document must be a JSON object.")
    if "run_id" not in document:
        if not allow_legacy:
            raise HTTPException(status_code=409, detail="The feedback document is missing its run_id.")
    else:
        try:
            recorded_id = _canonical_run_id(document["run_id"])
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="The feedback document has an invalid run_id.") from exc
        if recorded_id != run_id:
            raise HTTPException(status_code=409, detail="The feedback document belongs to a different run.")
    annotations = document.get("annotations")
    if not isinstance(annotations, dict):
        raise HTTPException(status_code=500, detail="The feedback annotations must be a JSON object.")
    for kind in ANNOTATION_KINDS:
        annotations.setdefault(kind, {})
    for bucket in annotations.values():
        if not isinstance(bucket, dict) or any(not isinstance(record, dict) for record in bucket.values()):
            raise HTTPException(status_code=500, detail="The feedback annotation buckets contain invalid records.")
        for record in bucket.values():
            flags, note = record.get("flags"), record.get("note")
            expected = record.get("expected")
            if flags is not None and (not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags)):
                raise HTTPException(status_code=500, detail="The feedback annotation flags must be lists of strings.")
            if note is not None and not isinstance(note, str):
                raise HTTPException(status_code=500, detail="The feedback annotation notes must be strings.")
            if expected is not None and not isinstance(expected, str):
                raise HTTPException(status_code=500, detail="The feedback annotation expected values must be strings or null.")
    # These defaults are response-only on GET. A save preserves all other metadata.
    document["run_id"] = run_id
    document.setdefault("state_file", str(state_path) if state_path else None)
    document.setdefault("updated_at", None)
    return document


def _is_empty(annotation: dict[str, Any]) -> bool:
    """An annotation with no flags, comment, or expected value is empty.

    Clearing every box is how a reviewer retracts a note, so it deletes the entry
    rather than leaving an empty husk that later reads as "reviewed, no issue".
    """
    return (
        not annotation.get("flags")
        and not (annotation.get("note") or "").strip()
        and not (annotation.get("expected") or "").strip()
    )


def build_app(
    *,
    state_path: Path | None = None,
    ground_truth_path: Path | None = None,
    feedback_path: Path | None = None,
    feedback_dir: Path | None = None,
) -> FastAPI:
    """Serve the workbench with no startup artifact unless explicitly configured."""
    if feedback_path is not None and feedback_dir is not None:
        raise ValueError("--feedback and --feedback-dir are mutually exclusive.")
    if state_path is None and (feedback_path is not None or ground_truth_path is not None):
        raise ValueError("--feedback and --ground-truth require an explicit --state.")
    app = FastAPI(title="CIPOC Workbench", docs_url=None, redoc_url=None)
    startup_run_id = None
    try:
        artifact = _read_json(state_path) if state_path is not None else None
        if isinstance(artifact, dict) and isinstance(artifact.get("run"), dict):
            startup_run_id = _canonical_run_id(artifact["run"].get("run_id"))
    except (OSError, ValueError, RecursionError):
        # Feedback binding must not prevent raw serving or local-file recovery.
        pass

    def feedback_destination(run_id: str | None) -> tuple[Path | None, str | None]:
        if feedback_path is None and feedback_dir is None:
            return None, "No --feedback-dir or --feedback was configured; annotations are read-only."
        if run_id is None or (feedback_path is not None and startup_run_id is None):
            return None, "The startup artifact has no valid canonical run UUID. Use --feedback-dir for locally loaded runs."
        if feedback_dir is not None:
            return feedback_dir / f"{run_id}.json", None
        if run_id != startup_run_id:
            return None, "The --feedback file is bound to the startup run only. Use --feedback-dir to save other runs."
        return feedback_path, None

    def read_feedback(run_id: str | None) -> JSONResponse:
        path, reason = feedback_destination(run_id)
        with _FEEDBACK_LOCK:
            document = _load_feedback(
                path, state_path if run_id == startup_run_id else None, run_id,
                allow_legacy=feedback_path is not None and path is not None,
            )
        # Normalize legacy fields for readers without changing the stored records,
        # including when a later save edits only one annotation in the document.
        annotations = {
            kind: {
                entity_id: {
                    **record,
                    "flags": record.get("flags") or [],
                    "note": record.get("note") or "",
                    "expected": record.get("expected"),
                }
                for entity_id, record in bucket.items()
            }
            for kind, bucket in document["annotations"].items()
        }
        return JSONResponse({
            **document, "annotations": annotations,
            "writable": path is not None, "read_only_reason": reason,
        })

    def save_feedback(run_id: str | None, kind: str, entity_id: str, annotation: dict[str, Any]) -> JSONResponse:
        if kind not in ANNOTATION_KINDS:
            raise HTTPException(status_code=404, detail=f"Unknown annotation kind {kind!r}.")
        path, reason = feedback_destination(run_id)
        if path is None:
            raise HTTPException(status_code=409, detail=reason)
        flags = annotation.get("flags")
        note = annotation.get("note")
        expected = annotation.get("expected")
        if flags is not None and (not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags)):
            raise HTTPException(status_code=422, detail="Annotation flags must be a list of strings.")
        if note is not None and not isinstance(note, str):
            raise HTTPException(status_code=422, detail="Annotation note must be a string.")
        if expected is not None and not isinstance(expected, str):
            raise HTTPException(status_code=422, detail="Annotation expected value must be a string or null.")

        # Atomic replacement prevents partial files; this process-wide lock also
        # prevents concurrent saves from dropping one another's unrelated edits.
        with _FEEDBACK_LOCK:
            document = _load_feedback(
                path, state_path if run_id == startup_run_id else None, run_id,
                allow_legacy=feedback_path is not None,
            )
            bucket = document["annotations"][kind]
            record = {
                **bucket.get(entity_id, {}),
                "flags": flags or [],
                "expected": expected or None,
                "note": (note or "").strip(),
                "updated_at": _now(),
            }
            if _is_empty(record):
                bucket.pop(entity_id, None)
                record = None
            else:
                bucket[entity_id] = record
            document["updated_at"] = _now()
            try:
                _write_json_atomic(path, document)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="Annotation values must be finite JSON values.") from exc
            except OSError as exc:
                raise HTTPException(status_code=500, detail="Cannot save the feedback document.") from exc
        return JSONResponse({"run_id": run_id, "kind": kind, "id": entity_id, "annotation": record})

    # Storage errors also carry the requested identity, without exposing file
    # contents or logging annotation bodies.
    def scoped_feedback(run_id: str, kind: str | None = None, entity_id: str = "", annotation=None) -> JSONResponse:
        try:
            try:
                _canonical_run_id(run_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Expected a canonical run UUID (lowercase, with hyphens).") from exc
            if kind is None:
                return read_feedback(run_id)
            return save_feedback(run_id, kind, entity_id, annotation)
        except HTTPException as exc:
            return JSONResponse({"run_id": run_id, "detail": exc.detail}, status_code=exc.status_code)

    @app.get("/api/ground-truth")
    def ground_truth() -> JSONResponse:
        """Legacy startup reference values, or empty when no reference was supplied.

        Without an explicit startup artifact, no run can own this reference.
        """
        if state_path is None:
            raise HTTPException(status_code=409, detail="No startup artifact was configured; ground truth cannot be bound.")
        data = _read_json(ground_truth_path)
        return JSONResponse(data if isinstance(data, dict) else {})

    @app.get("/api/runs/{run_id}/ground-truth")
    def run_ground_truth(run_id: str) -> JSONResponse:
        try:
            _canonical_run_id(run_id)
        except ValueError:
            return JSONResponse({"run_id": run_id, "detail": "Expected a canonical run UUID (lowercase, with hyphens)."}, status_code=400)
        if startup_run_id is None:
            return JSONResponse({"run_id": run_id, "detail": "The startup artifact has no valid canonical run UUID; ground truth cannot be bound."}, status_code=409)
        if run_id != startup_run_id:
            return JSONResponse({"run_id": run_id, "detail": "Ground truth is bound to the initial startup run only."}, status_code=409)
        try:
            values = _read_json(ground_truth_path) if ground_truth_path is not None else {}
            if not isinstance(values, dict):
                raise ValueError("Ground truth must be a JSON object.")
            return JSONResponse({"run_id": run_id, "values": values})
        except (OSError, ValueError, RecursionError):
            return JSONResponse({"run_id": run_id, "detail": "Cannot read the ground-truth document."}, status_code=500)

    @app.get("/api/feedback")
    def feedback() -> JSONResponse:
        return read_feedback(startup_run_id)

    @app.put("/api/feedback/{kind}/{entity_id}")
    def put_feedback(kind: str, entity_id: str, annotation: dict[str, Any]) -> JSONResponse:
        return save_feedback(startup_run_id, kind, entity_id, annotation)

    @app.get("/api/runs/{run_id}/feedback")
    def run_feedback(run_id: str) -> JSONResponse:
        return scoped_feedback(run_id)

    @app.put("/api/runs/{run_id}/feedback/{kind}/{entity_id}")
    def put_run_feedback(run_id: str, kind: str, entity_id: str, annotation: dict[str, Any]) -> JSONResponse:
        return scoped_feedback(run_id, kind, entity_id, annotation)

    @app.put("/api/runs/{run_id}/feedback/{kind}")
    def put_run_feedback_query(run_id: str, kind: str, entity_id: str, annotation: dict[str, Any]) -> JSONResponse:
        return scoped_feedback(run_id, kind, entity_id, annotation)

    # No startup artifact is distinct from an explicitly configured missing file.
    @app.get("/case_state.json")
    def case_state() -> Response:
        if state_path is None:
            return Response(status_code=204)
        if not state_path.is_file():
            raise HTTPException(status_code=404, detail=f"{state_path} does not exist.")
        return FileResponse(state_path, media_type="application/json")

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app
