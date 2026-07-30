"""Pydantic data models used by CIPOC."""

from .base import (
    ConfidenceLevel,
    confidence_instructions,
    confidence_field,
)

from .notes import (
    CancerStatus,
    CONCEPTS,
    CONCEPT_DESCRIPTIONS,
    TextSpan,
    ConceptWithEvidence,
    build_concept_presence_dict,
    ClinicalNote,
    ProcessedClinicalNote,
    CancerMention,
    NoteDigest,
    NoteCorpusDescriptors,
)

from .variables import (
    VariableInfo,
    VariableOutput,
    VariableGroupInfo,
    VariableGroupOutput,
    ValidatedVariableOutput,
    ValidatedVariableGroupOutput,
    CorpusGate,
    SiteApplicability,
    NoteFilter,
    TargetGroup,
)

from .rules import (
    RuleKind,
    RuleApplicability,
    RuleUnit,
    CaseFacts,
    ScopingReviewReason,
    ScopedVariableContext,
    ManualSource,
    RuleStoreManifest,
)

from .case import (
    VariableStatus,
    CaseVariableResult,
    ReviewFlagType,
    ReviewFlag,
    CaseReport,
    Case,
)


__all__ = [
    "ConfidenceLevel",
    "confidence_instructions",
    "confidence_field",
    "CancerStatus",
    "CONCEPTS",
    "CONCEPT_DESCRIPTIONS",
    "TextSpan",
    "ConceptWithEvidence",
    "build_concept_presence_dict",
    "ClinicalNote",
    "ProcessedClinicalNote",
    "CancerMention",
    "NoteDigest",
    "NoteCorpusDescriptors",
    "VariableInfo",
    "VariableOutput",
    "VariableGroupInfo",
    "VariableGroupOutput",
    "ValidatedVariableOutput",
    "ValidatedVariableGroupOutput",
    "CorpusGate",
    "SiteApplicability",
    "NoteFilter",
    "TargetGroup",
    "RuleKind",
    "RuleApplicability",
    "RuleUnit",
    "CaseFacts",
    "ScopingReviewReason",
    "ScopedVariableContext",
    "ManualSource",
    "RuleStoreManifest",
    "VariableStatus",
    "CaseVariableResult",
    "ReviewFlagType",
    "ReviewFlag",
    "CaseReport",
    "Case",
]
