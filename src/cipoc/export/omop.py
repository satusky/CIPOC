"""Flat-file export of CIPOC results to OMOP NOTE and NOTE_NLP rows."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ValidationError

from cipoc.models import Case, ClinicalNote

from .models import (
    NOTE_FIELDS,
    NOTE_NLP_FIELDS,
    OmopErrorReport,
    OmopExportResult,
    OmopNoteNlpRow,
    OmopNoteRow,
    OmopRowError,
    OmopValidationIssue,
)


class OmopExporter:
    """Export source notes and validated evidence as OMOP-shaped staging files.

    Concept fields deliberately contain raw CIPOC values. A downstream conversion
    step is expected to map them to OMOP concept identifiers.
    """

    def __init__(
        self,
        *,
        person_id: int | str | None,
        nlp_system: str = "CIPOC RapCID-E",
        nlp_date: date | str | None = None,
        nlp_datetime: datetime | str | None = None,
        note_type: str = "EHR",
        encoding: str = "UTF-8",
        language: str = "English",
    ) -> None:
        self.person_id = person_id
        self.nlp_system = nlp_system
        self.nlp_datetime = _format_datetime(nlp_datetime)
        self.nlp_date = _format_date(nlp_date) or _date_from_datetime(
            self.nlp_datetime
        ) or date.today().isoformat()
        self.note_type = note_type
        self.encoding = encoding
        self.language = language

    def export(
        self,
        *,
        notes: Iterable[ClinicalNote],
        case: Case,
        output_directory: str | Path,
    ) -> OmopExportResult:
        """Write ``note.csv``, ``note_nlp.csv``, then ``omop_errors.json``.

        Incomplete rows are retained in the error file with their populated fields
        instead of being mixed into the loadable OMOP files.
        """
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        source_notes = list(notes)
        note_rows, note_errors = self._build_note_rows(source_notes)
        valid_note_ids = {str(row.note_id) for row in note_rows}
        note_nlp_rows, note_nlp_errors = self._build_note_nlp_rows(
            source_notes,
            valid_note_ids,
            case,
        )

        note_path = output_directory / "note.csv"
        note_nlp_path = output_directory / "note_nlp.csv"
        error_path = output_directory / "omop_errors.json"

        # NOTE is intentionally materialized first because NOTE_NLP references it.
        _write_csv(note_path, NOTE_FIELDS, note_rows)
        _write_csv(note_nlp_path, NOTE_NLP_FIELDS, note_nlp_rows)
        errors = note_errors + note_nlp_errors
        error_path.write_text(
            OmopErrorReport(errors=errors).model_dump_json(indent=2),
            encoding="utf-8",
        )

        return OmopExportResult(
            note_path=note_path,
            note_nlp_path=note_nlp_path,
            error_path=error_path,
            note_count=len(note_rows),
            note_nlp_count=len(note_nlp_rows),
            error_count=len(errors),
        )

    def _build_note_rows(
        self,
        notes: list[ClinicalNote],
    ) -> tuple[list[OmopNoteRow], list[OmopRowError]]:
        rows: list[OmopNoteRow] = []
        errors: list[OmopRowError] = []

        serialized_ids: dict[str, int] = {}
        for note in notes:
            note_id = str(note.note_id)
            serialized_ids[note_id] = serialized_ids.get(note_id, 0) + 1

        for note in notes:
            raw_row = {
                "note_id": note.note_id,
                "person_id": self.person_id,
                "note_date": note.date,
                "note_datetime": "",
                "note_type_concept_id": self.note_type,
                "note_class_concept_id": note.note_type,
                "note_title": note.note_type,
                "note_text": note.content,
                "encoding_concept_id": self.encoding,
                "language_concept_id": self.language,
                "provider_id": "",
                "visit_occurrence_id": "",
                "visit_detail_id": "",
                "note_source_value": note.note_type,
            }
            issues: list[OmopValidationIssue] = []
            try:
                row = OmopNoteRow.model_validate(raw_row)
            except ValidationError as error:
                issues.extend(_validation_issues(error))
            if serialized_ids[str(note.note_id)] > 1:
                issues.append(
                    OmopValidationIssue(
                        field="note_id",
                        type="duplicate",
                        message=(
                            "The note ID is duplicated after conversion to its CSV "
                            "representation."
                        ),
                    )
                )

            if issues:
                errors.append(
                    OmopRowError(
                        table_name="note",
                        source_id=str(note.note_id),
                        issues=issues,
                        row_data=raw_row,
                    )
                )
            else:
                rows.append(row)

        return rows, errors

    def _build_note_nlp_rows(
        self,
        notes: list[ClinicalNote],
        valid_note_ids: set[str],
        case: Case,
    ) -> tuple[list[OmopNoteNlpRow], list[OmopRowError]]:
        rows: list[OmopNoteNlpRow] = []
        errors: list[OmopRowError] = []
        notes_by_id: dict[str, list[ClinicalNote]] = {}
        for note in notes:
            notes_by_id.setdefault(str(note.note_id), []).append(note)

        for item_id, result in case.variable_results.items():
            extraction = result.extraction
            if (
                result.value is None
                or extraction is None
                or not extraction.is_valid
                or extraction.value is None
            ):
                continue

            spans = extraction.spans or [None]
            for span_index, span in enumerate(spans, start=1):
                span_note_id = span.note_id if span is not None else ""
                lexical_variant = span.text if span is not None else ""
                source_id = f"{item_id}:{span_index}"
                raw_row = {
                    "note_nlp_id": f"{span_note_id}:{item_id}:{span_index}",
                    "note_id": span_note_id,
                    "section_concept_id": "",
                    "snippet": lexical_variant,
                    "offset": "",
                    "lexical_variant": lexical_variant,
                    "note_nlp_concept_id": result.value,
                    "note_nlp_source_concept_id": item_id,
                    "nlp_system": self.nlp_system,
                    "nlp_date": self.nlp_date,
                    "nlp_datetime": self.nlp_datetime,
                    "term_exists": "Y",
                    "term_temporal": "",
                    "term_modifiers": json.dumps(
                        {
                            "item_id": item_id,
                            "value": result.value,
                            "confidence": extraction.presence_confidence.value,
                        },
                        separators=(",", ":"),
                    ),
                }

                normalized_note_id = str(span_note_id)
                cited_notes = notes_by_id.get(normalized_note_id, [])
                issues: list[OmopValidationIssue] = []
                if span_note_id != "" and normalized_note_id not in valid_note_ids:
                    issues.append(
                        OmopValidationIssue(
                            field="note_id",
                            type="invalid_reference",
                            message="The cited note does not have a complete NOTE row.",
                        )
                    )
                elif len(cited_notes) == 1 and lexical_variant:
                    offsets = _find_offsets(cited_notes[0].content, lexical_variant)
                    if not offsets:
                        issues.append(
                            OmopValidationIssue(
                                field="lexical_variant",
                                type="value_error",
                                message=(
                                    "The evidence text is not present verbatim in the "
                                    "cited note."
                                ),
                            )
                        )
                    elif len(offsets) == 1:
                        offset = offsets[0]
                        raw_row["offset"] = str(offset)
                        raw_row["snippet"] = _build_snippet(
                            cited_notes[0].content,
                            offset,
                            len(lexical_variant),
                        )

                try:
                    row = OmopNoteNlpRow.model_validate(raw_row)
                except ValidationError as error:
                    issues.extend(_validation_issues(error))

                if issues:
                    errors.append(
                        OmopRowError(
                            table_name="note_nlp",
                            source_id=source_id,
                            issues=issues,
                            row_data=raw_row,
                        )
                    )
                else:
                    rows.append(row)

        return rows, errors


def _validation_issues(error: ValidationError) -> list[OmopValidationIssue]:
    return [
        OmopValidationIssue(
            field=".".join(str(part) for part in issue["loc"]),
            type=issue["type"],
            message=issue["msg"],
        )
        for issue in error.errors(include_url=False)
    ]


def _format_date(value: date | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _format_datetime(value: datetime | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    return value


def _date_from_datetime(value: str) -> str:
    if not value:
        return ""
    return value.split("T", maxsplit=1)[0].split(" ", maxsplit=1)[0]


def _build_snippet(
    content: str,
    offset: int,
    lexical_length: int,
    *,
    maximum_length: int = 250,
) -> str:
    if len(content) <= maximum_length:
        return content
    context = max(maximum_length - lexical_length, 0)
    start = max(offset - context // 2, 0)
    end = min(start + maximum_length, len(content))
    start = max(end - maximum_length, 0)
    return content[start:end]


def _find_offsets(content: str, text: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while (offset := content.find(text, start)) >= 0:
        offsets.append(offset)
        start = offset + 1
    return offsets


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[BaseModel],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.model_dump() for row in rows)


__all__ = [
    "OmopExporter",
]
