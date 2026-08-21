"""Pydantic schemas for OMOP staging rows and export errors."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _required_csv_value(value: Any) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Value must be a string or integer.")
    if isinstance(value, str) and not value.strip():
        raise ValueError("Value must not be blank.")
    return value


def _required_string(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Value must be a string.")
    if not value.strip():
        raise ValueError("Value must not be blank.")
    return value


RequiredCsvValue = Annotated[int | str, BeforeValidator(_required_csv_value)]
RequiredString = Annotated[str, BeforeValidator(_required_string)]
OptionalCsvValue = int | str | None
OptionalStringValue = str | None


class OmopNoteRow(BaseModel):
    """One complete OMOP NOTE-shaped staging row."""

    model_config = ConfigDict(extra="forbid")

    note_id: RequiredCsvValue
    person_id: RequiredCsvValue
    note_date: RequiredString
    note_datetime: OptionalStringValue = None
    note_type_concept_id: RequiredCsvValue
    note_class_concept_id: RequiredCsvValue
    note_title: OptionalStringValue = None
    note_text: RequiredString
    encoding_concept_id: RequiredCsvValue
    language_concept_id: RequiredCsvValue
    provider_id: OptionalCsvValue = None
    visit_occurrence_id: OptionalCsvValue = None
    visit_detail_id: OptionalCsvValue = None
    note_source_value: OptionalStringValue = None


class OmopNoteNlpRow(BaseModel):
    """One complete OMOP NOTE_NLP-shaped staging row."""

    model_config = ConfigDict(extra="forbid")

    note_nlp_id: RequiredCsvValue
    note_id: RequiredCsvValue
    section_concept_id: OptionalCsvValue = None
    snippet: OptionalStringValue = None
    offset: OptionalStringValue = None
    lexical_variant: RequiredString
    note_nlp_concept_id: OptionalCsvValue = None
    note_nlp_source_concept_id: OptionalCsvValue = None
    nlp_system: OptionalStringValue = None
    nlp_date: RequiredString
    nlp_datetime: OptionalStringValue = None
    term_exists: OptionalStringValue = None
    term_temporal: OptionalStringValue = None
    term_modifiers: OptionalStringValue = None


NOTE_FIELDS = tuple(OmopNoteRow.model_fields)
NOTE_NLP_FIELDS = tuple(OmopNoteNlpRow.model_fields)


class OmopValidationIssue(BaseModel):
    """One field-level reason a staging row could not be exported."""

    field: str
    type: str
    message: str


class OmopRowError(BaseModel):
    """A partial staging row and the issues that prevented its export."""

    table_name: Literal["note", "note_nlp"]
    source_id: str
    issues: list[OmopValidationIssue] = Field(min_length=1)
    row_data: dict[str, Any]


class OmopErrorReport(BaseModel):
    """All rejected rows from one OMOP export."""

    errors: list[OmopRowError] = Field(default_factory=list)


class OmopExportResult(BaseModel):
    """Paths and row counts produced by one export."""

    model_config = ConfigDict(frozen=True)

    note_path: Path
    note_nlp_path: Path
    error_path: Path
    note_count: int
    note_nlp_count: int
    error_count: int


class OmopMergeResult(BaseModel):
    """Paths and row counts produced by merging patient export CSVs."""

    model_config = ConfigDict(frozen=True)

    note_path: Path
    note_nlp_path: Path
    source_count: int
    note_count: int
    note_nlp_count: int


__all__ = [
    "NOTE_FIELDS",
    "NOTE_NLP_FIELDS",
    "OmopErrorReport",
    "OmopExportResult",
    "OmopMergeResult",
    "OmopNoteNlpRow",
    "OmopNoteRow",
    "OmopRowError",
    "OmopValidationIssue",
]
