"""Utilities for combining per-patient OMOP staging CSVs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from .models import NOTE_FIELDS, NOTE_NLP_FIELDS, OmopMergeResult


def merge_omop_csvs(
    input_directories: Iterable[str | Path],
    output_directory: str | Path,
) -> OmopMergeResult:
    """Merge per-patient NOTE and NOTE_NLP CSVs into one staging export.

    IDs are namespaced by each NOTE row's person ID, and NOTE_NLP references are
    rewritten to the corresponding merged NOTE IDs.
    """
    source_directories = [Path(directory) for directory in input_directories]
    merged_notes: list[dict[str, str]] = []
    merged_note_nlp: list[dict[str, str]] = []
    merged_note_ids: set[str] = set()
    merged_note_nlp_ids: set[str] = set()

    for source_directory in source_directories:
        note_path = source_directory / "note.csv"
        note_nlp_path = source_directory / "note_nlp.csv"
        note_rows = _read_csv(note_path, NOTE_FIELDS)
        note_nlp_rows = _read_csv(note_nlp_path, NOTE_NLP_FIELDS)

        note_id_map: dict[str, tuple[str, str]] = {}
        for row in note_rows:
            source_note_id = _required_value(row, "note_id", note_path)
            person_id = _required_value(row, "person_id", note_path)
            if source_note_id in note_id_map:
                raise ValueError(
                    f"{note_path} contains duplicate note_id '{source_note_id}'."
                )

            merged_note_id = _namespace_id(person_id, source_note_id)
            if merged_note_id in merged_note_ids:
                raise ValueError(
                    "Merging would produce duplicate note_id "
                    f"'{merged_note_id}' from {note_path}."
                )

            note_id_map[source_note_id] = (merged_note_id, person_id)
            merged_note_ids.add(merged_note_id)
            merged_notes.append({**row, "note_id": merged_note_id})

        for row in note_nlp_rows:
            source_note_id = _required_value(row, "note_id", note_nlp_path)
            source_note_nlp_id = _required_value(row, "note_nlp_id", note_nlp_path)
            if source_note_id not in note_id_map:
                raise ValueError(
                    f"{note_nlp_path} references note_id '{source_note_id}', which "
                    "does not exist in its note.csv."
                )

            merged_note_id, person_id = note_id_map[source_note_id]
            merged_note_nlp_id = _namespace_id(person_id, source_note_nlp_id)
            if merged_note_nlp_id in merged_note_nlp_ids:
                raise ValueError(
                    "Merging would produce duplicate note_nlp_id "
                    f"'{merged_note_nlp_id}' from {note_nlp_path}."
                )

            merged_note_nlp_ids.add(merged_note_nlp_id)
            merged_note_nlp.append(
                {
                    **row,
                    "note_nlp_id": merged_note_nlp_id,
                    "note_id": merged_note_id,
                }
            )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    note_path = output_directory / "note.csv"
    note_nlp_path = output_directory / "note_nlp.csv"
    _write_csv(note_path, NOTE_FIELDS, merged_notes)
    _write_csv(note_nlp_path, NOTE_NLP_FIELDS, merged_note_nlp)

    return OmopMergeResult(
        note_path=note_path,
        note_nlp_path=note_nlp_path,
        source_count=len(source_directories),
        note_count=len(merged_notes),
        note_nlp_count=len(merged_note_nlp),
    )


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            actual_fields = reader.fieldnames
            if actual_fields is None:
                raise ValueError(f"{path} does not contain a CSV header.")
            if len(actual_fields) != len(expected_fields) or set(actual_fields) != set(
                expected_fields
            ):
                missing = sorted(set(expected_fields) - set(actual_fields))
                unexpected = sorted(set(actual_fields) - set(expected_fields))
                raise ValueError(
                    f"{path} has an incompatible schema; missing={missing}, "
                    f"unexpected={unexpected}."
                )
            return list(reader)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required OMOP staging file not found: {path}") from error


def _required_value(row: dict[str, str], field: str, path: Path) -> str:
    value = row.get(field, "")
    if not value.strip():
        raise ValueError(f"{path} contains a row with an empty required field '{field}'.")
    return value


def _namespace_id(person_id: str, source_id: str) -> str:
    return f"{quote(person_id, safe='')}:{quote(source_id, safe='')}"


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["merge_omop_csvs"]
