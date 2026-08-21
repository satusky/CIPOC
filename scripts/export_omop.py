"""Run CIPOC extraction and export one patient's notes as OMOP staging files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cipoc.agents import OrchestratorAgent
from cipoc.export import OmopExporter
from cipoc.models import ClinicalNote


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_notes(path: Path) -> list[ClinicalNote]:
    raw_notes = _load_json(path)
    if not isinstance(raw_notes, list):
        raise ValueError(f"{path} must contain a JSON array of clinical notes.")
    return [ClinicalNote.model_validate(note) for note in raw_notes]


def _load_structured_data(value: str | None) -> dict[int, str] | None:
    if value is None:
        return None
    raw = (
        json.loads(value)
        if value.lstrip().startswith("{")
        else _load_json(Path(value))
    )
    if not isinstance(raw, dict):
        raise ValueError("Structured data must be a JSON object keyed by item ID.")
    return {int(item_id): str(item_value) for item_id, item_value in raw.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CIPOC and export NOTE and NOTE_NLP staging CSVs."
    )
    parser.add_argument(
        "notes",
        type=Path,
        help="Path to a JSON array of clinical notes.",
    )
    parser.add_argument(
        "--person-id",
        required=True,
        help="Raw person identifier to place in NOTE rows.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for note.csv, note_nlp.csv, and omop_errors.json.",
    )
    parser.add_argument(
        "--nlp-date",
        default=None,
        help="NLP processing date in YYYY-MM-DD format; defaults to today.",
    )
    parser.add_argument(
        "--nlp-system",
        default="CIPOC",
        help="Raw NLP system identifier placed in NOTE_NLP rows.",
    )
    parser.add_argument(
        "--structured-data",
        default=None,
        help="Inline JSON object or JSON file containing known values by item ID.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent LangGraph tasks.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live orchestration progress display.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    notes = _load_notes(args.notes)
    structured_data = _load_structured_data(args.structured_data)

    case = OrchestratorAgent().run(
        [note.model_dump() for note in notes],
        structured_data=structured_data,
        progress=not args.no_progress,
        max_concurrency=args.max_concurrency,
    )
    result = OmopExporter(
        person_id=args.person_id,
        nlp_system=args.nlp_system,
        nlp_date=args.nlp_date,
    ).export(
        notes=notes,
        case=case,
        output_directory=args.output_directory,
    )

    print(f"NOTE rows: {result.note_count} -> {result.note_path}")
    print(f"NOTE_NLP rows: {result.note_nlp_count} -> {result.note_nlp_path}")
    print(f"Errors: {result.error_count} -> {result.error_path}")


if __name__ == "__main__":
    main()
