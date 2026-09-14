from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .notes import NoteSelectionProvenance
from .rules import CaseFacts
from .variables import ValidatedVariableOutput


class VariableStatus(str, Enum):
    PENDING = "pending"
    STRUCTURED_DATA = "structured_data"
    EXTRACTED = "extracted"
    NOT_FOUND = "not_found"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"
    BLOCKED = "blocked"


COMPLETED_VARIABLE_STATUSES = {
    VariableStatus.STRUCTURED_DATA,
    VariableStatus.EXTRACTED,
    VariableStatus.NOT_FOUND,
    VariableStatus.NOT_APPLICABLE,
    VariableStatus.ERROR,
}
TERMINAL_VARIABLE_STATUSES = COMPLETED_VARIABLE_STATUSES | {VariableStatus.BLOCKED}


class CaseVariableResult(BaseModel):
    """Orchestration state for one requested variable."""

    item_id: int = Field(description="NAACCR item ID.")
    status: VariableStatus = Field(default=VariableStatus.PENDING)
    value: str | None = Field(
        default=None,
        description="Final coded value when the variable has been populated.",
    )
    extraction: ValidatedVariableOutput | None = Field(
        default=None,
        description="Extractor output retained as evidence for the orchestration status.",
    )
    reason: str | None = Field(
        default=None,
        description="Reason for an error, applicability decision, or blocker.",
    )
    blocking_item_ids: list[int] = Field(
        default_factory=list,
        description="Unresolved variable dependencies when status is blocked.",
    )

    @model_validator(mode="after")
    def validate_status(self):
        if self.extraction is not None and self.extraction.item_id != self.item_id:
            raise ValueError("Extraction item ID must match the case variable item ID.")

        if self.status == VariableStatus.PENDING:
            if (
                self.value is not None
                or self.extraction is not None
                or self.reason
                or self.blocking_item_ids
            ):
                raise ValueError("Pending variables cannot carry a result or terminal reason.")

        elif self.status == VariableStatus.STRUCTURED_DATA:
            if self.value is None:
                raise ValueError("Structured-data variables require a value.")
            if self.extraction is not None:
                raise ValueError("Structured-data variables cannot carry an extraction.")

        elif self.status == VariableStatus.EXTRACTED:
            if self.value is None:
                raise ValueError("Extracted variables require a value.")
            if self.extraction is None or not self.extraction.is_valid:
                raise ValueError("Extracted variables require a valid extraction.")
            if self.extraction.value != self.value:
                raise ValueError(
                    "The case value must match the validated extraction value."
                )

        elif self.status == VariableStatus.NOT_FOUND:
            if (
                self.extraction is None
                or not self.extraction.is_valid
                or self.extraction.value is not None
            ):
                raise ValueError(
                    "Variables not found require a valid extraction with no value."
                )

        elif self.status == VariableStatus.NOT_APPLICABLE:
            if self.extraction is not None:
                raise ValueError("Not-applicable variables cannot carry an extraction.")
            if not self.reason:
                raise ValueError("Not-applicable variables require a reason.")

        elif self.status == VariableStatus.ERROR:
            if self.extraction is not None and self.extraction.is_valid:
                raise ValueError("Error variables cannot carry a valid extraction.")
            if self.extraction is None and not self.reason:
                raise ValueError("Errors without an extraction require a reason.")

        elif self.status == VariableStatus.BLOCKED:
            if self.extraction is not None:
                raise ValueError("Blocked variables cannot carry an extraction.")
            if not self.reason and not self.blocking_item_ids:
                raise ValueError(
                    "Blocked variables require a reason or blocking variable IDs."
                )

        if self.status != VariableStatus.BLOCKED and self.blocking_item_ids:
            raise ValueError("Only blocked variables may identify blocking variable IDs.")
        if self.status not in {
            VariableStatus.STRUCTURED_DATA,
            VariableStatus.EXTRACTED,
        }:
            if self.value is not None:
                raise ValueError(
                    "Only structured-data or extracted variables may carry a value."
                )

        return self


class ReviewFlagType(str, Enum):
    """Controlled reasons a coded variable is surfaced for human review."""

    ERROR = "error"
    INVALID_EXTRACTION = "invalid_extraction"
    LOW_CONFIDENCE = "low_confidence"


class ReviewFlag(BaseModel):
    """One reviewable finding about a single variable in the finished case."""

    item_id: int = Field(description="NAACCR item ID the flag concerns.")
    flag_type: ReviewFlagType = Field(description="Why the variable was flagged.")
    detail: str = Field(description="Human-readable explanation of the flag.")


class CaseReport(BaseModel):
    """Auditability roll-up produced when a case is finalized.

    Carries the variables a human should look at — errors, extractions that
    failed validation, and accepted-but-low-confidence values — so the report is
    empty exactly when nothing needs review.
    """

    flags: list[ReviewFlag] = Field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


class Case(BaseModel):
    """Durable snapshot of a finished extraction.

    Holds the final coded values plus the explanations needed to interpret them
    — blocked/not-applicable reasons, errors, and any fatal blocker. Live
    orchestration bookkeeping (what's still pending, whether the run may
    continue) belongs to ``CaseState``, not here.
    """

    case_facts: CaseFacts | None = Field(
        default=None,
        description="Coding-rule scoping facts, if any were derived for the case.",
    )
    variable_results: dict[int, CaseVariableResult] = Field(default_factory=dict)
    note_selection: dict[str, NoteSelectionProvenance] = Field(
        default_factory=dict,
        description="Per-group note-selection provenance keyed as group:<group_id>.",
    )
    fatal_blocker: str | None = Field(
        default=None,
        description="Reason no further extraction could be attempted for this case.",
    )
    report: CaseReport | None = Field(
        default=None,
        description="Review roll-up flagging errors, invalid extractions, and low-confidence values.",
    )

    @model_validator(mode="after")
    def check_result_keys(self):
        """The dict key must equal each result's own item ID — the one integrity
        invariant a final snapshot still needs."""
        for item_id, result in self.variable_results.items():
            if result.item_id != item_id:
                raise ValueError(
                    f"Variable result key {item_id} does not match item ID "
                    f"{result.item_id}."
                )
        for key, selection in self.note_selection.items():
            expected_key = f"group:{selection.group_id}"
            if key != expected_key:
                raise ValueError(
                    f"Note selection key {key!r} does not match group ID "
                    f"{selection.group_id!r}; expected {expected_key!r}."
                )
        return self
