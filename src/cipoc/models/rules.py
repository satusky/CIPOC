"""Models for the compiled manual rule store and runtime context scoping.

One ``RuleUnit`` is one retrievable piece of coding guidance compiled offline
from a source manual (see ``documents/rules/``). Runtime scoping filters units
by variable (``item_ids``), case applicability (``applies_to`` vs ``CaseFacts``),
and manual precedence, then returns per-variable ``ScopedVariableContext``.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, RootModel, model_validator


RuleKind = Literal["instruction", "code_table", "definition", "priority_rule", "example"]


class RuleApplicability(BaseModel):
    """Predicate object restricting a rule to specific cases.

    An absent/None field means "applies to all". A condition only excludes a
    rule when the corresponding case fact is known and fails to match
    (unknown widens).
    """
    sites: list[str] | None = Field(default=None, description="ICD-O-3 topography codes or ranges (e.g., 'C509', 'C500-C509') the rule applies to.")
    histologies: list[str] | None = Field(default=None, description="ICD-O-3 morphology codes or ranges (e.g., '8500', '8500-8549') the rule applies to.")
    behaviors: list[str] | None = Field(default=None, description="ICD-O-3 behavior codes (e.g., ['2', '3']) the rule applies to.")
    sex: str | None = Field(default=None, description="Sex the rule applies to, when sex-specific.")
    dx_date_min: str | None = Field(default=None, description="Earliest diagnosis date (ISO format) the rule applies to; overrides the manual-level publication date.")
    dx_date_max: str | None = Field(default=None, description="Latest diagnosis date (ISO format) the rule applies to; overrides the manual-level publication date.")


class RuleUnit(BaseModel):
    """One retrievable piece of coding guidance compiled from a source manual."""
    rule_id: str = Field(description="Stable identifier, e.g. 'solid_tumor_rules:breast:h4'.")
    source_doc: str = Field(description="Manifest key of the manual this unit was compiled from.")
    section_path: list[str] = Field(description="Heading trail locating the unit in the source markdown.")
    anchor: str | None = Field(default=None, description="Line range or heading slug in the source markdown, for audit.")
    kind: RuleKind = Field(description="Kind of guidance this unit carries.")
    item_ids: list[int] = Field(default_factory=list, description="NAACCR item IDs governed by this unit. Empty list means a general principle applying to all items in its manual.")
    applies_to: RuleApplicability | None = Field(default=None, description="Case applicability predicate. None means the unit applies to all cases.")
    text: str = Field(description="Near-verbatim instruction text from the source manual.")
    codes: dict[str, str] | None = Field(default=None, description="Code-to-description table for code_table units.")
    notes: str | None = Field(default=None, description="Compiler or reviewer notes; never injected into prompts.")

    @model_validator(mode="after")
    def _code_table_requires_codes(self):
        if self.kind == "code_table" and not self.codes:
            raise ValueError("code_table units must carry a non-empty codes table.")
        return self


class CaseFacts(BaseModel):
    """Known facts about the current case, fed from orchestrator state.

    Every field is optional; unknown facts widen scope, never narrow it.
    """
    primary_site: str | None = Field(default=None, description="ICD-O-3 topography code of the primary site, when known.")
    histology: str | None = Field(default=None, description="ICD-O-3 morphology code, when known.")
    behavior: str | None = Field(default=None, description="ICD-O-3 behavior code, when known.")
    sex: str | None = Field(default=None, description="Patient sex, when known.")
    date_of_diagnosis: str | None = Field(default=None, description="Date of diagnosis as recorded; parse leniently for comparisons.")


class ScopingReviewReason(str, Enum):
    """Controlled reasons a scoped context should be flagged for human review."""
    EMPTY_CODE_REDUCTION_FALLBACK = "empty_code_reduction_fallback"
    UNKNOWN_DX_DATE_WIDE_SCOPE = "unknown_dx_date_wide_scope"
    INSTRUCTION_CONTEXT_TRUNCATED = "instruction_context_truncated"


class ScopedVariableContext(BaseModel):
    """Scoped coding context for one variable, produced by deterministic runtime scoping."""
    item_id: int = Field(description="NAACCR item ID this context applies to.")
    units: list[RuleUnit] = Field(default_factory=list, description="Applicable rule units after filtering and precedence resolution, ordered most specific first.")
    reduced_codes: dict[str, str] | None = Field(default=None, description="Valid codes reduced to the case specifics. None means no reduction was applied and the full data-dictionary set stands.")
    review_reasons: list[ScopingReviewReason] = Field(default_factory=list, description="Review flags raised while scoping, deduplicated in order of first occurrence.")


class ManualSource(BaseModel):
    """Manifest entry describing one compiled source manual."""
    title: str = Field(description="Human-readable manual title.")
    family: str = Field(description="Standard-setter family, e.g. 'SEER' or 'CoC'. Used for precedence resolution.")
    publication_date: str = Field(description="Publication/effective date (ISO format). The manual applies to cases diagnosed on or after this date unless a unit's applies_to overrides it.")
    effective_note: str | None = Field(default=None, description="Free-text note on effective-date handling, e.g. per-site-group ranges.")
    source_markdown: str | None = Field(default=None, description="Repository path of the source markdown the manual was compiled from.")
    compiled_at: str | None = Field(default=None, description="Date the manual was last compiled into rule units.")


class RuleStoreManifest(RootModel[dict[str, ManualSource]]):
    """Registry of compiled manuals, keyed by source_doc, mirroring documents/rules/manifest.json."""

    def __getitem__(self, source_doc: str) -> ManualSource:
        return self.root[source_doc]

    def get(self, source_doc: str) -> ManualSource | None:
        return self.root.get(source_doc)

    def items(self):
        return self.root.items()
