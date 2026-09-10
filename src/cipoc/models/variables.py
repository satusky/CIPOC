from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from .base import ConfidenceLevel, confidence_field
from .notes import TextSpan, CancerStatus


class VariableInfo(BaseModel):
    """ Information about a variable """
    item_id: int = Field(description="Item ID number.")
    name: str | None = Field(default=None, description="Variable name.")
    description: str | None = Field(default=None, description="Variable description.")
    data_type: str | None = Field(default=None, description="Data type defined by the data dictionary.")
    length: int | None = Field(default=None, description="Maximum field length defined by the data dictionary.")
    allowable_values: str | None = Field(default=None, description="Allowable values defined by the data dictionary.")
    format: str | None = Field(default=None, description="Format for coded value as defined by the data dictionary and/or instructions.")
    valid_codes: str | dict | list[dict] | None = Field(default=None, description="Valid codes from the data dictionary, optionally scoped by gross primary site.")
    coding_instructions: str | None = Field(default=None, description="Coding instructions from the NAACCR data dictionary, when present.")
    model_config = ConfigDict(protected_namespaces=())


class VariableGroupInfo(BaseModel):
    name: str | None = Field(default=None, description="Variable group name (optional).")
    group_id: str | None = Field(default=None, description="Variable group ID (optional).")
    variables: list[VariableInfo] = Field(description="List of variables in group with variable-level information.")
    extract_as_group: bool = Field(default=False, description="Extract the entire group together (True) or individually (False).")


class CorpusGate(str, Enum):
    """Deterministic gate conditions evaluated against note-corpus characteristics."""
    METASTASIS_PRESENT = "metastasis_present"
    TREATMENT_PRESENT = "treatment_present"
    LYMPH_NODES_REMOVED = "lymph_nodes_removed"


class SiteApplicability(BaseModel):
    """A leaf's restrictions are OR alternatives; use all_of for conjunction.

    Composite nodes use either any_of or all_of, without leaf restrictions.
    Unknown facts keep scope open unless the expression is definitively false.
    """
    gross_primary_sites: list[str] = Field(default_factory=list, description="Tissue-level primary sites the group applies to.")
    primary_sites: list[Annotated[str, Field(pattern=r"^C[0-9]{3}(-C[0-9]{3})?$")]] = Field(default_factory=list, description="ICD-O-3 topography codes or inclusive ranges, e.g. C440-C449.")
    histology_families: list[str] = Field(default_factory=list, description="Histology family labels, preserved as serialized. Runtime configuration supports melanoma (8720-8790), independent of site.")
    any_of: list["SiteApplicability"] = Field(default_factory=list, description="OR alternatives; excludes only when every alternative is definitively false.")
    all_of: list["SiteApplicability"] = Field(default_factory=list, description="AND conditions; any definitively false condition excludes this alternative.")

    @field_validator("primary_sites")
    @classmethod
    def _ordered_site_ranges(cls, values: list[str]) -> list[str]:
        for value in values:
            low, _, high = value.partition("-")
            if high and low > high:
                raise ValueError(f"Primary-site range is reversed: {value}")
        return values

    @model_validator(mode="after")
    def _unambiguous_expression(self):
        leaf = bool(self.gross_primary_sites or self.primary_sites or self.histology_families)
        if sum((leaf, bool(self.any_of), bool(self.all_of))) > 1:
            raise ValueError("Use leaf restrictions, any_of, or all_of, not a mixture.")
        return self


class NoteFilter(BaseModel):
    """Deterministic note-level hard filter narrowing a group's candidate notes."""
    note_types: list[str] = Field(default_factory=list, description="Allowed note types; a note passes when its type case-insensitively equals one of these. Empty means any type.")
    keywords: list[str] = Field(default_factory=list, description="Keyword stems; a note passes when any stem is a case-insensitive substring of one of its flags or of its summary. Empty means no keyword restriction.")
    cancer_status: list[CancerStatus] = Field(default_factory=list, description="Allowed cancer temporality statuses; a note passes when its cancer_status set intersects these. Empty means no status restriction.")
    within_days: int | None = Field(default=None, description="Maximum absolute distance in days between the note date and the case temporal anchor. Requires an anchor at evaluation time and is skipped when none is supplied. None means no date restriction.")


class TargetGroup(VariableGroupInfo):
    """A planned extraction group: a variable group plus deterministic orchestration gating."""
    stage: str | None = Field(default=None, description="'initial' runs first and scopes later groups; 'dependent' runs afterward.")
    gate: list[CorpusGate] | None = Field(default=None, description="Corpus conditions that must all hold for the group to be extracted. None or empty means ungated.")
    applies_to: SiteApplicability | None = Field(default=None, description="Site/histology restriction; None means the group applies to all cases.")
    note_filter: NoteFilter | None = Field(default=None, description="Deterministic note-level hard filter narrowing the group's candidate notes before retrieval; None means no per-note filtering.")

    def to_variable_group(self) -> VariableGroupInfo:
        """Project down to the plain extractor/snapshot view, dropping gating fields."""
        return VariableGroupInfo(
            group_id=self.group_id,
            name=self.name,
            variables=self.variables,
            extract_as_group=self.extract_as_group,
        )


class VariableOutput(BaseModel):
    """ Structured output for an extracted variable """
    item_id: int = Field(description="Item ID number.")
    value: str | None = Field(description="Coded value for the variable. Must be selected from the valid codes and in the appropriate format. Return `None` if no value can be determined.")
    explanation: str = Field(description="Reasoning used for assigning the selected value.")
    most_important_note: int | str | None = Field(description="Exact note_id of the supplied note containing the strongest evidence for the extracted value. Return the identifier, never quoted text, a note title, or an explanation. Preserve string IDs and leading zeroes. Return null if no value could be determined.")
    spans: list[TextSpan] = Field(description="List of text span(s) in the clinical note that provide evidence for this claim. A span containing newline characters should be split into multiple spans at the newlines.")
    presence_confidence: ConfidenceLevel = confidence_field()


class VariableGroupOutput(BaseModel):
    variables: list[VariableOutput] = Field(description="List of coded values for each variable in group with explanations and confidence level.")


class ValidatedVariableOutput(VariableOutput):
    """A VariableOutput carrying the pipeline's validation verdict."""
    is_valid: bool = Field(description="Whether the emitted result passed validation; False means it exhausted its repair attempts still failing.")
    validation_errors: list[str] = Field(default_factory=list)
    extraction_attempts: int = 0


class ValidatedVariableGroupOutput(VariableGroupOutput):
    variables: list[ValidatedVariableOutput] = Field(description="List of coded values for each variable in group, with validation verdicts.")
