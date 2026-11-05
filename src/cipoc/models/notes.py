from datetime import datetime
from enum import Enum

from typing import Literal
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ConfigDict, field_serializer

from .base import ConfidenceLevel, confidence_field


CONCEPT_DESCRIPTIONS: dict[str, str] = {
    "cancer": "Any malignant neoplasm — current, recent, or historical — including explicit diagnoses, pathology findings, cancer-directed treatment, or clear reference to a prior malignancy.",
    "neoplasm": "Any neoplasm — benign, borderline, or malignant; current, recent, or historical — or any mass, nodule, or lesion that could be one, even without a diagnosis.",
    "metastasis": "Any indication that cancer has spread beyond its primary site (distant or regional metastatic disease).",
    "surgery": "Any cancer-directed surgical procedure (e.g. resection, excision, mastectomy, lobectomy).",
    "chemotherapy": "Any systemic cytotoxic chemotherapy that was administered or planned.",
    "hormonal_therapy": "Any cancer-directed hormonal or endocrine therapy that was administered or planned (e.g. tamoxifen, aromatase inhibitors, androgen deprivation). Exclude non-cancer hormone replacement or other non-cancer endocrine treatment.",
    "immunotherapy": "Any cancer-directed immunotherapy or biologic response modifier that was administered or planned (e.g. immune checkpoint inhibitors, CAR T-cell therapy). Exclude non-cancer immunotherapy such as allergy desensitization or autoimmune treatment.",
    "radiation": "Any radiation therapy that was administered or planned.",
    "lymph_nodes_removed": "Any removal or surgical sampling of lymph nodes (e.g. lymphadenectomy, sentinel node biopsy, regional node dissection).",
}

CONCEPTS = list(CONCEPT_DESCRIPTIONS)

CancerStatus = Literal["historical", "recent", "current"]
MentionBehavior = Literal["benign", "malignant", "unknown"]
DiagnosticCertainty = Literal["definite", "probable", "possible"]


class NoteSelectionRejectionCode(str, Enum):
    """Controlled reasons a note failed deterministic group filtering."""

    NOTE_TYPE_MISMATCH = "note_type_mismatch"
    CANCER_STATUS_MISMATCH = "cancer_status_mismatch"
    MISSING_OR_INVALID_DATE = "missing_or_invalid_date"
    OUTSIDE_DATE_WINDOW = "outside_date_window"


class NoteSelectionUnevaluatedCode(str, Enum):
    """Configured note checks that could not be applied during selection."""

    KEYWORD_FILTER_DISABLED = "keyword_filter_disabled"
    TEMPORAL_ANCHOR_UNAVAILABLE = "temporal_anchor_unavailable"


class NoteFilterEvaluation(BaseModel):
    """Explainable result of applying one deterministic note filter."""

    passes: bool = Field(description="Whether the note passed every evaluated check.")
    rejection_reasons: list[NoteSelectionRejectionCode] = Field(
        default_factory=list,
        description="All deterministic reasons the note was rejected.",
    )
    unevaluated_checks: list[NoteSelectionUnevaluatedCode] = Field(
        default_factory=list,
        description="Configured checks that were skipped during evaluation.",
    )


class NoteSelectionProvenance(BaseModel):
    """Durable evidence-selection funnel for one variable group."""

    group_id: str = Field(description="Variable group whose notes were selected.")
    requested_item_ids: list[int] = Field(
        default_factory=list,
        description="NAACCR item IDs requested from the group branch.",
    )
    candidate_note_ids: list[int | str] = Field(
        default_factory=list,
        description="Note IDs that passed deterministic filtering.",
    )
    rejected_note_ids: dict[
        int | str, list[NoteSelectionRejectionCode]
    ] = Field(
        default_factory=dict,
        description="Rejected note IDs mapped to all deterministic rejection reasons.",
    )
    selected_note_ids: list[int | str] = Field(
        default_factory=list,
        description="Candidate note IDs accepted after retriever output validation.",
    )
    discarded_note_ids: list[int | str] = Field(
        default_factory=list,
        description="Note IDs proposed by the retriever that were not offered to it.",
    )
    unevaluated_checks: list[NoteSelectionUnevaluatedCode] = Field(
        default_factory=list,
        description="Configured deterministic checks that were not evaluated.",
    )


class TextSpan(BaseModel):
    note_id: int | str = Field(description="The ID value of the note this text was copied from. Must exactly match the ID value for one of the provided notes — not a field name (e.g. 'content').")
    text: str = Field(description="Verbatim text snippet from a document that provides evidence for a claim.")


class ConceptWithEvidence(BaseModel):
    presence: bool = Field(default=False, description="Presence (`True`) or absence (`False`) of a concept in a document.")
    confidence: ConfidenceLevel | None = confidence_field(default=None)
    evidence: list[TextSpan] | None = Field(default=None, description="List of text span(s) in the clinical note that provide evidence for this claim. A span containing newline characters should be split into multiple spans.")
    model_config = ConfigDict(protected_namespaces=())


def behavior_field(**kwargs):
    """Pydantic field for a mention's behavior, with the category definitions in its description."""
    return Field(
        description="Behavior of the neoplasm or finding as the note establishes it. {'malignant': malignant, including in situ; 'benign': benign, borderline, or uncertain whether benign or malignant; 'unknown': the note doesn't establish behavior, including a mass, nodule, or lesion with no diagnosis yet}",
        **kwargs,
    )


def diagnostic_certainty_field(**kwargs):
    """Pydantic field for a mention's diagnostic certainty, with the wording rules in its description."""
    return Field(
        description="How certain the note's wording makes the diagnosis. This describes the note, not your confidence in this answer, which belongs in `confidence`: a note that says 'possible lung cancer' is 'possible' here even though the note states it plainly. {'definite': the diagnosis is stated, for example on pathology or as the clinician's diagnosis; 'probable': wording such as 'probable', 'likely', 'suspicious for' or 'consistent with'; 'possible': wording such as 'possible', 'cannot rule out' or 'questionable'}",
        **kwargs,
    )


class CancerMention(BaseModel):
    presence: Literal[True] = Field(description="Every reported mention is present in the note.")
    confidence: ConfidenceLevel = confidence_field()
    evidence: list[TextSpan] = Field(min_length=1, description="Verbatim text spans supporting the mention.")
    status: CancerStatus = Field(description="Approximate timeframe of the neoplasm or finding. {'current': ongoing case, 'recent': case resolved <10 years prior, 'historical': case resolved 10+ years prior}")
    affected_tissue: str = Field(description="Primary organ or tissue affected.")
    metastasis: bool = Field(description="Metastases mentioned in the note.")
    # The defaults describe mentions saved before these fields existed, when the
    # scanner captured only diagnosed malignancies. The scanner must answer both.
    behavior: MentionBehavior = behavior_field(default="malignant")
    diagnostic_certainty: DiagnosticCertainty = diagnostic_certainty_field(default="definite")
    model_config = ConfigDict(protected_namespaces=())
    

class ClinicalNote(BaseModel):
    note_id: int | str = Field(description="ID value for note.")
    date: str = Field(description="Date note was written in 'YYYY-MM-DD' format.")
    note_type: str = Field(description="Type of note.")
    content: str = Field(description="Text contents of note.")
    model_config = ConfigDict(protected_namespaces=())


def build_concept_presence_dict(
    concepts: list[str] | dict[str, dict] = CONCEPTS,
) -> dict[str, ConceptWithEvidence]:
    if isinstance(concepts, list):
        concepts = {concept: {} for concept in concepts}
    return {concept: ConceptWithEvidence(**vals) for concept, vals in concepts.items()}


class ProcessedClinicalNote(ClinicalNote):
    summary: str | None = Field(default=None, description="Summary of clinical note.")
    concepts: dict[str, ConceptWithEvidence] = Field(
        default_factory=build_concept_presence_dict
    ) # type: ignore
    cancer_status: set[CancerStatus] | None = Field(default=None, description="Distinct temporality statuses across the note's definite malignant mentions. `None` when the note has none.")
    cancer_mentions: list[CancerMention] | None = Field(default=None, description="List of cancer mentions.")
    flags: list[str] | None = Field(default=None, description="Keywords associated with the note contents for search.")
    model_config = ConfigDict(protected_namespaces=())

    @field_serializer("cancer_status")
    def _serialize_cancer_status(self, value: set[CancerStatus] | None) -> list[CancerStatus] | None:
        """Emit a deterministic sorted list so JSON output is stable (sets are unordered)."""
        return sorted(value) if value else None


class NoteDigest(BaseModel):
    note_id: int | str = Field(description="ID value for note.")
    note_type: str = Field(description="Type of note.")
    summary: str | None = Field(default=None, description="Summary of clinical note.")
    flags: list[str] | None = Field(default=None, description="Keywords associated with the note contents for search.")


class NoteCorpusDescriptors(BaseModel):
    note_count: int = Field(default=0, description="Number of unique notes.")
    date_range: tuple[str, str] | None = Field(
        default_factory=tuple,
        description="Range of note dates converted to 'YYYY-MM-DD' format."
    )
    types: set[str] | None = Field(
        default_factory=set,
        description="Set of unique note types."
    )
    affected_tissues: dict[CancerStatus, set[str]] | None = Field(
        default_factory=dict,
        description="Dictionary of affected tissues from definite malignant mentions, keyed by the status (current, recent, historical) associated with tumors in that tissue. `None` if no cancer is present."
    )
    concepts: dict[str, ConceptWithEvidence] = Field(default_factory=build_concept_presence_dict) # type: ignore
    unique_flags: set[str]
    model_config = ConfigDict(protected_namespaces=())
