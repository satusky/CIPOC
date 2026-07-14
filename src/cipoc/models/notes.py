from typing import Literal
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ConfigDict, field_serializer

from .base import ConfidenceLevel, confidence_field, TextSpan


CancerStatus = Literal["historical", "recent", "current"]


class CancerMention(BaseModel):
    spans: list[TextSpan] = Field(description="List of text span(s) in the clinical note that provide evidence for this claim. A span containing newline characters should be split into multiple spans.")
    status: CancerStatus = Field(description="Approximate timeframe of cancer case. {'current': ongoing case, 'recent': case resolved <10 years prior, 'historical': case resolved 10+ years prior}")
    affected_tissue: str = Field(description="Primary organ or tissue affected.")
    metastasis: bool = Field(description="Metastases mentioned in the note.")
    confidence: ConfidenceLevel = confidence_field()


class CancerMentionList(BaseModel):
    mentions: list[CancerMention] = Field(description="List of cancer mentions.")


class ClinicalNote(BaseModel):
    note_id: int | str = Field(description="ID value for note.")
    date: str = Field(description="Date note was written in 'YYYY-MM-DD' format.")
    type: str = Field(description="Type of note.")
    content: str = Field(description="Text contents of note.")
    model_config = ConfigDict(protected_namespaces=())


class ProcessedClinicalNote(ClinicalNote):
    cancer_present: bool | None = Field(default=None, description="Cancer is mentioned in the note. If uncertain, default to `True`.")
    presence_confidence: ConfidenceLevel | None = confidence_field(default=None)
    cancer_status: set[CancerStatus] | None = Field(default=None, description="Distinct temporality statuses across all cancer mentions in the note. `None` when no cancer is present.")
    summary: str | None = Field(default=None, description="Summary of clinical note.")
    cancer_mentions: CancerMentionList | None = Field(default=None, description="List of cancer mentions.")
    flags: list[str] | None = Field(default=None, description="Keywords associated with the note contents for search.")
    model_config = ConfigDict(protected_namespaces=())

    @field_serializer("cancer_status")
    def _serialize_cancer_status(self, value: set[CancerStatus] | None) -> list[CancerStatus] | None:
        """Emit a deterministic sorted list so JSON output is stable (sets are unordered)."""
        return sorted(value) if value else None
