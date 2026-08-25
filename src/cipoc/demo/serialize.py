"""Reduce arbitrary run payloads to JSON-safe values for the trace.

Tap 1 hands us task inputs/results and full ``CaseState`` snapshots as live
Pydantic models (often nested inside plain dicts, e.g. ``{"note_corpus": {id:
ProcessedClinicalNote}}``); Tap 2 hands us already-flat capture dicts. Both have
to survive a JSONL round-trip and come back *identical*, because replay must
drive the same pipeline as the original run.

:func:`to_jsonable` walks the structure and returns only ``dict`` / ``list`` /
``str`` / ``int`` / ``float`` / ``bool`` / ``None`` — the JSON value space. Two
choices make the round-trip exact rather than merely lossless:

* **Dict keys are stringified.** JSON object keys are always strings, so a
  ``dict[int, ...]`` (note corpora are keyed by note id) would come back
  string-keyed and break equality. We stringify keys up front so what we write is
  what we read.
* **Unknown objects fall back to ``str(obj)``** rather than raising, so a trace
  can always be written; a stray un-modelled value degrades to its repr instead
  of aborting the recording.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel


def _key(key: Any) -> str:
    """Coerce a mapping key to the string JSON requires (idempotent for ``str``)."""
    if isinstance(key, str):
        return key
    if isinstance(key, Enum):
        return str(key.value)
    return str(key)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into a JSON-round-trippable value.

    Handles Pydantic models (via ``model_dump(mode="json")``), dataclasses,
    enums, mappings, sequences/sets, and ``date``/``datetime``. Primitives pass
    through untouched; anything else degrades to ``str(obj)``.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseModel):
        # ``mode="json"`` already yields JSON-safe *values*, but a re-walk is what
        # stringifies any non-str dict keys the model carries.
        return to_jsonable(obj.model_dump(mode="json"))
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {_key(k): to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


__all__ = ["to_jsonable"]
