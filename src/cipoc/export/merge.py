"""Utilities for combining per-patient OMOP staging CSVs."""

from __future__ import annotations

import csv
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterable, Iterator
from urllib.parse import quote

from pydantic import ValidationError

from ._files import staged_output_paths
from .models import (
    NOTE_FIELDS,
    NOTE_NLP_FIELDS,
    OmopMergeResult,
    OmopNoteNlpRow,
    OmopNoteRow,
)


_CSV_READER_LOCK = Lock()


def merge_omop_csvs(
    input_directories: Iterable[str | Path],
    output_directory: str | Path,
) -> OmopMergeResult:
    """Merge per-patient NOTE and NOTE_NLP CSVs into one staging export.

    IDs are namespaced by each NOTE row's person ID, and NOTE_NLP references are
    rewritten to the corresponding merged NOTE IDs. All inputs are validated
    before staging either output. Closed staged files are replaced individually,
    not as a crash-atomic bundle; replacement failures can leave mixed outputs.
    """
    source_directories = [Path(directory) for directory in input_directories]
    merged_notes: list[dict[str, str]] = []
    merged_note_nlp: list[dict[str, str]] = []
    merged_note_ids: set[str] = set()
    merged_note_nlp_ids: set[str] = set()

    for source_index, source_directory in enumerate(source_directories, start=1):
        note_path = source_directory / "note.csv"
        note_nlp_path = source_directory / "note_nlp.csv"
        note_rows = _read_csv(note_path, OmopNoteRow, source_index)
        note_nlp_rows = _read_csv(note_nlp_path, OmopNoteNlpRow, source_index)

        note_id_map: dict[str, tuple[str, str]] = {}
        for row_number, row in enumerate(note_rows, start=2):
            source_note_id = row["note_id"]
            person_id = row["person_id"]
            if source_note_id in note_id_map:
                raise ValueError(
                    f"Source {source_index} note.csv record {row_number} "
                    "contains duplicate note_id."
                )

            merged_note_id = _namespace_id(person_id, source_note_id)
            if merged_note_id in merged_note_ids:
                raise ValueError(
                    f"Source {source_index} note.csv record {row_number} "
                    "would produce duplicate note_id."
                )

            note_id_map[source_note_id] = (merged_note_id, person_id)
            merged_note_ids.add(merged_note_id)
            merged_notes.append({**row, "note_id": merged_note_id})

        source_note_nlp_ids: set[str] = set()
        for row_number, row in enumerate(note_nlp_rows, start=2):
            source_note_id = row["note_id"]
            source_note_nlp_id = row["note_nlp_id"]
            if source_note_nlp_id in source_note_nlp_ids:
                raise ValueError(
                    f"Source {source_index} note_nlp.csv record {row_number} "
                    "contains duplicate note_nlp_id."
                )
            source_note_nlp_ids.add(source_note_nlp_id)
            if source_note_id not in note_id_map:
                raise ValueError(
                    f"Source {source_index} note_nlp.csv record {row_number} "
                    "references a note_id that does not exist in its note.csv."
                )

            merged_note_id, person_id = note_id_map[source_note_id]
            merged_note_nlp_id = _namespace_id(person_id, source_note_nlp_id)
            if merged_note_nlp_id in merged_note_nlp_ids:
                raise ValueError(
                    f"Source {source_index} note_nlp.csv record {row_number} "
                    "would produce duplicate note_nlp_id."
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
    note_path = output_directory / "note.csv"
    note_nlp_path = output_directory / "note_nlp.csv"
    with staged_output_paths(output_directory, ("note.csv", "note_nlp.csv")) as (
        staged_note,
        staged_note_nlp,
    ):
        _write_csv(staged_note, NOTE_FIELDS, merged_notes)
        _write_csv(staged_note_nlp, NOTE_NLP_FIELDS, merged_note_nlp)

    return OmopMergeResult(
        note_path=note_path,
        note_nlp_path=note_nlp_path,
        source_count=len(source_directories),
        note_count=len(merged_notes),
        note_nlp_count=len(merged_note_nlp),
    )


@contextmanager
def _csv_field_limit() -> Iterator[None]:
    # field_size_limit is process-global. Only readers using this lock cooperate;
    # unrelated csv users must arrange their own coordination.
    with _CSV_READER_LOCK:
        previous_limit = csv.field_size_limit()
        try:
            limit = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(limit)
                    break
                except OverflowError:
                    # Some platforms use a narrower signed C integer.
                    limit //= 2
            yield
        finally:
            csv.field_size_limit(previous_limit)


def _read_csv(
    path: Path,
    model: type[OmopNoteRow] | type[OmopNoteNlpRow],
    source_index: int,
) -> list[dict[str, str]]:
    expected_fields = tuple(model.model_fields)
    label = f"Source {source_index} {'note.csv' if model is OmopNoteRow else 'note_nlp.csv'}"
    record_number = 1
    try:
        with _csv_field_limit(), path.open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            actual_fields = next(reader, None)
            if actual_fields is None:
                raise ValueError(f"{label} does not contain a CSV header.")
            if len(actual_fields) != len(expected_fields) or set(actual_fields) != set(
                expected_fields
            ):
                raise ValueError(f"{label} has an incompatible schema.")
            rows = []
            while True:
                record_number += 1
                values = next(reader, None)
                if values is None:
                    return rows
                if len(values) != len(expected_fields):
                    raise ValueError(
                        f"{label} CSV record {record_number} has incorrect row width; "
                        f"expected {len(expected_fields)} cells, got {len(values)}."
                    )
                row = dict(zip(actual_fields, values))
                try:
                    model.model_validate(row)
                except ValidationError as error:
                    fields = sorted({
                        issue["loc"][0]
                        for issue in error.errors(
                            include_url=False, include_input=False, include_context=False
                        )
                    })
                    raise ValueError(
                        f"{label} CSV record {record_number} has invalid OMOP fields: "
                        f"{', '.join(fields)}."
                    ) from None
                rows.append(row)
    except FileNotFoundError:
        raise FileNotFoundError(f"Required OMOP staging file not found: {label}.") from None
    except (csv.Error, UnicodeError):
        # Parser/decoder and Pydantic exception bodies can include clinical text.
        raise ValueError(
            f"{label} has malformed CSV or UTF-8 near record {record_number}."
        ) from None


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
