"""Deterministic reportability checks that can stop a case before extraction."""

import re
from typing import Mapping

from cipoc.models import (
    CancerMention,
    CaseFacts,
    CaseVariableResult,
    ClinicalNote,
    ExtractionStop,
    ProcessedClinicalNote,
    VariableStatus,
)


# Tumors of any behavior are reportable in these tissues (C70, C71, C72, C75),
# so an undiagnosed finding there still continues. C75 also holds non-CNS
# glands; is_reportable applies the exact CNS ranges once the site is coded.
CNS_TISSUES = frozenset({
    "meninges",
    "brain",
    "spinal_cord_and_cranial_nerves",
    "other_endocrine_gland",
})
_CNS_SITE_RANGES = (("C700", "C729"), ("C751", "C753"))


def _known_site(primary_site: str | None) -> str | None:
    """A coded topography without its decimal; None for C809, malformed or missing."""
    site = (primary_site or "").strip().upper()
    if not re.fullmatch(r"C[0-9]{2}\.?[0-9]", site):
        return None
    site = site.replace(".", "")
    return None if site >= "C809" else site


def is_reportable(facts: CaseFacts | None) -> bool | None:
    """Whether the coded site and behavior make the case reportable.

    Behavior 2 (in situ) and 3 (malignant) are always reportable. Behavior 0
    (benign) and 1 (borderline) are reportable only at CNS sites, C700-C729 and
    C751-C753. No other exclusions are applied. None means the facts can't
    decide: behavior is unknown, or a benign or borderline tumor has no known
    site (C809 counts as unknown).
    """
    behavior = ((facts.behavior if facts else None) or "").strip()
    if behavior in ("2", "3"):
        return True
    if behavior not in ("0", "1"):
        return None
    site = _known_site(facts.primary_site)
    if site is None:
        return None
    return any(low <= site <= high for low, high in _CNS_SITE_RANGES)


def mention_continues(mention: CancerMention) -> bool:
    """Whether one mention lets the case past checkpoint 1.

    It must be definite or probable, the wording SEER counts as a diagnosis,
    and have a known behavior. A mention with unknown behavior continues only
    in a CNS tissue.
    """
    if mention.diagnostic_certainty not in ("definite", "probable"):
        return False
    return mention.behavior != "unknown" or mention.affected_tissue in CNS_TISSUES


def _describe_mention(note_id: int | str, mention: CancerMention) -> str:
    evidence = " / ".join(span.text for span in mention.evidence)
    return (
        f"note {note_id} ({mention.affected_tissue}, behavior {mention.behavior}, "
        f"{mention.diagnostic_certainty}): \"{evidence}\""
    )


def checkpoint_one_stop_reason(
    facts: CaseFacts | None,
    note_corpus: Mapping[int | str, ClinicalNote | ProcessedClinicalNote],
) -> str | None:
    """Why checkpoint 1 stops the case, or None to continue.

    Caller-supplied facts come first: if primary site, histology or behavior is
    known, the case stops only when ``is_reportable`` is False, and the mentions
    aren't consulted. Otherwise some mention must pass ``mention_continues``.
    The reason cites the supplied values, or every mention with its note,
    behavior, certainty and evidence.
    """
    if facts is not None and (facts.primary_site or facts.histology or facts.behavior):
        if is_reportable(facts) is not False:
            return None
        supplied = ", ".join(
            f"{label} {value}"
            for label, value in (
                ("primary site", facts.primary_site),
                ("histology", facts.histology),
                ("behavior", facts.behavior),
            )
            if value
        )
        return (
            f"Not reportable: supplied {supplied}. Benign and borderline tumors "
            "are reportable only at CNS sites (C700-C729, C751-C753)."
        )

    mentions = [
        (note_id, mention)
        for note_id, note in note_corpus.items()
        for mention in getattr(note, "cancer_mentions", None) or []
    ]
    if any(mention_continues(mention) for _, mention in mentions):
        return None
    if not mentions:
        return "Not reportable: no note mentions a neoplasm."
    cited = "; ".join(_describe_mention(note_id, mention) for note_id, mention in mentions)
    return (
        "Not reportable: no mention is a definite or probable neoplasm with a "
        "known behavior, or a definite or probable finding in a CNS tissue. "
        f"Mentions: {cited}."
    )


def stop_pending_results(
    results: dict[int, CaseVariableResult], stop: ExtractionStop
) -> dict[int, CaseVariableResult]:
    """Move every PENDING variable to NOT_APPLICABLE, flagged with ``stop``.

    The variables carry no reason text; the caller records the evidence once,
    on the case. Returns only the changed results, for the ``variable_results``
    reducer. Structured-data seeds and other terminal results are left alone.
    """
    return {
        item_id: CaseVariableResult(
            item_id=item_id, status=VariableStatus.NOT_APPLICABLE, stopped_by=stop
        )
        for item_id, result in results.items()
        if result.status == VariableStatus.PENDING
    }
