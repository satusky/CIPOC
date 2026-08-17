"""Deterministic orchestration helpers.

Pure functions used to set up and drive orchestrator state. Keep these free of
LLM calls; bounded model usage belongs in the scanner/extractor subagents.
"""
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable

from cipoc.models import (
    CaseFacts,
    CaseReport,
    CaseVariableResult,
    ConfidenceLevel,
    CorpusGate,
    NoteCorpusDescriptors,
    NoteDigest,
    NoteFilter,
    ProcessedClinicalNote,
    ReviewFlag,
    ReviewFlagType,
    SiteApplicability,
    TargetGroup,
    ValidatedVariableGroupOutput,
    ValidatedVariableOutput,
    VariableGroupInfo,
    VariableInfo,
    VariableStatus,
    ConceptWithEvidence,
)
from cipoc.models.case import TERMINAL_VARIABLE_STATUSES
from .coding_context import resolve_gross_site, site_in_ranges


def _concept_present(corpus: NoteCorpusDescriptors, concept: str) -> bool:
    """Return whether a single concept is flagged present in the corpus roll-up."""
    return corpus.concepts.get(concept, ConceptWithEvidence(presence=False)).presence


# Treatment is modeled as granular modalities rather than an aggregate concept,
# so the treatment gate is derived by OR-ing those modality concepts.
TREATMENT_CONCEPTS = ("surgery", "chemotherapy", "radiation")


CORPUS_GATE_PREDICATES: dict[CorpusGate, Callable[[NoteCorpusDescriptors], bool]] = {
    CorpusGate.METASTASIS_PRESENT: lambda corpus: _concept_present(corpus, "metastasis"),
    CorpusGate.TREATMENT_PRESENT: lambda corpus: any(
        _concept_present(corpus, concept) for concept in TREATMENT_CONCEPTS
    ),
    CorpusGate.LYMPH_NODES_REMOVED: lambda corpus: _concept_present(corpus, "lymph_nodes_removed"),
}


def corpus_gate_passes(
    gates: list[CorpusGate] | None, corpus: NoteCorpusDescriptors
) -> bool:
    """Return True when every gate condition holds for the corpus.

    An empty or ``None`` gate list is ungated and always passes. Evaluation is
    deterministic; each condition reads one boolean characteristic of the corpus.
    """
    return all(CORPUS_GATE_PREDICATES[gate](corpus) for gate in (gates or []))


def load_variable_groups(path: str | Path) -> list[TargetGroup]:
    """Parse the variable-group config into flat ``TargetGroup`` plan entries.

    Each JSON group becomes a ``TargetGroup`` built from its loose ``variables``;
    each nested ``subgroup`` becomes its own ``TargetGroup`` that inherits the
    parent's ``gate`` and ``stage`` (unless it sets its own) so every emitted
    group is self-contained for gating. A group carrying only subgroups (no loose
    variables of its own) yields just those subgroups.

    Only ``item_id`` and ``name`` are read per variable; richer data-dictionary
    and case-scoped metadata is filled later by
    :func:`cipoc.tools.build_variable_group` once case facts are known.
    """
    with open(path, "r") as f:
        config = json.load(f)

    def _to_group(node: dict, *, gate=None, stage=None, note_filter=None) -> TargetGroup:
        return TargetGroup(
            group_id=node.get("group_id"),
            name=node.get("name"),
            extract_as_group=node.get("extract_as_group", False),
            stage=node.get("stage", stage),
            gate=node.get("gate", gate),
            applies_to=node.get("applies_to"),
            note_filter=node.get("note_filter", note_filter),
            variables=[
                VariableInfo(item_id=variable["item_id"], name=variable.get("name"))
                for variable in node.get("variables", [])
            ],
        )

    groups: list[TargetGroup] = []
    for group in config.get("groups", []):
        if group.get("variables"):
            groups.append(_to_group(group))
        for subgroup in group.get("subgroups", []):
            if subgroup.get("variables"):
                groups.append(
                    _to_group(
                        subgroup,
                        gate=group.get("gate"),
                        stage=group.get("stage"),
                        note_filter=group.get("note_filter"),
                    )
                )
    return groups


@dataclass(frozen=True)
class GroupNode:
    """One node of the variable-group config tree, in display order."""

    group_id: str
    name: str
    parent_id: str | None
    item_ids: tuple[int, ...]


def load_group_hierarchy(path: str | Path) -> list[GroupNode]:
    """Parse the variable-group config preserving its parent/subgroup nesting.

    :func:`load_variable_groups` deliberately flattens subgroups into peer
    ``TargetGroup`` plan entries, which is what the orchestrator needs but loses
    the grouping a reader recognizes. This returns the same config as a
    display-ordered tree instead: each top-level group followed by its subgroups.

    A group carrying only subgroups still yields a node (with no ``item_ids``) so
    it can be rendered as a header over its children.
    """
    with open(path, "r") as f:
        config = json.load(f)

    def _item_ids(node: dict) -> tuple[int, ...]:
        return tuple(variable["item_id"] for variable in node.get("variables", []))

    nodes: list[GroupNode] = []
    for group in config.get("groups", []):
        group_id = group.get("group_id")
        nodes.append(
            GroupNode(
                group_id=group_id,
                name=group.get("name") or group_id,
                parent_id=None,
                item_ids=_item_ids(group),
            )
        )
        for subgroup in group.get("subgroups", []):
            nodes.append(
                GroupNode(
                    group_id=subgroup.get("group_id"),
                    name=subgroup.get("name") or subgroup.get("group_id"),
                    parent_id=group_id,
                    item_ids=_item_ids(subgroup),
                )
            )
    return nodes


def _parse_note_date(value: str | None) -> date | None:
    """Parse a ``ClinicalNote.date`` string ('YYYY-MM-DD') into a date, or None."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def note_matches_filter(
    note: ProcessedClinicalNote,
    note_filter: NoteFilter | None,
    *,
    anchor: date | None = None,
) -> bool:
    """Return True when a note satisfies every constraint set on the filter.

    Each dimension is evaluated independently and combined with AND; a dimension
    left empty/``None`` imposes no restriction. A ``None`` filter passes every
    note. The ``within_days`` date dimension is skipped when no ``anchor`` is
    supplied, since it cannot be evaluated without one.
    """
    if note_filter is None:
        return True

    # Note type: case-insensitive exact match against the allowed set.
    if note_filter.note_types:
        allowed = {t.strip().casefold() for t in note_filter.note_types}
        if (note.note_type or "").strip().casefold() not in allowed:
            return False

    ##### Commented out for now because the keywords are LLM generated and will usually not pass
    # Keywords: any stem is a case-insensitive substring of a flag or the summary.
    # if note_filter.keywords:
    #     haystack = [f.casefold() for f in (note.flags or [])]
    #     if note.summary:
    #         haystack.append(note.summary.casefold())
    #     stems = [k.strip().casefold() for k in note_filter.keywords if k.strip()]
    #     if not any(stem in hay for stem in stems for hay in haystack):
    #         return False

    # Cancer status: note's temporality set must intersect the allowed statuses.
    if note_filter.cancer_status:
        if not (note.cancer_status or set()).intersection(note_filter.cancer_status):
            return False

    # Date window: note within `within_days` of the anchor (skipped without one).
    if note_filter.within_days is not None and anchor is not None:
        note_date = _parse_note_date(note.date)
        if note_date is None or abs((note_date - anchor).days) > note_filter.within_days:
            return False

    return True


def prefilter_notes(
    notes: Iterable[ProcessedClinicalNote],
    note_filter: NoteFilter | None,
    *,
    anchor: date | None = None,
) -> list[ProcessedClinicalNote]:
    """Apply a group's deterministic hard filter to the candidate corpus.

    The ``prefilter_notes`` stage of the selection funnel: returns, in input
    order, the notes that survive ``note_filter`` so only they get projected to
    digests for the retriever. A ``None`` filter passes every note through.
    """
    return [note for note in notes if note_matches_filter(note, note_filter, anchor=anchor)]



def build_corpus_descriptors(note_corpus: dict[int | str, ProcessedClinicalNote]) -> NoteCorpusDescriptors:
    notes = list(note_corpus.values())
    dates = sorted([note.date for note in notes])
    types = {note.note_type for note in notes}

    affected_tissues = defaultdict(set)
    unique_flags = set([])
    for note in notes:
        if note.cancer_mentions is not None:
            for mention in note.cancer_mentions:
                affected_tissues[mention.status].update(mention.affected_tissue)

        if note.flags is not None:
            unique_flags.update(note.flags)

    def merge_concept_dicts(
        right: dict[str, ConceptWithEvidence],
        left: dict[str, ConceptWithEvidence],
    ) -> dict[str, ConceptWithEvidence]:
        for concept, update in right.items():
            current = left.get(concept)
            if current is None:
                left[concept] = ConceptWithEvidence(
                    presence=update.presence,
                    confidence=update.confidence,
                )
                continue

            if current.presence and current.confidence == "max":
                continue
            
            left[concept] = ConceptWithEvidence(
                presence=current.presence or update.presence,
                confidence=max(current.confidence, update.confidence) if current.confidence and update.confidence else None,
            )

        return left

    note_concepts = [note.concepts for note in notes]
    all_concepts: dict[str, ConceptWithEvidence] = {}
    for concept_dict in note_concepts:
        all_concepts = merge_concept_dicts(concept_dict, all_concepts)

    return NoteCorpusDescriptors(
        note_count=len(notes),
        date_range=(dates[0], dates[-1]),
        types=types,
        affected_tissues=affected_tissues,
        concepts=all_concepts,
        unique_flags=unique_flags
    )


def build_corpus_digests(
    note_corpus: dict[int | str, ProcessedClinicalNote],
) -> dict[int | str, NoteDigest]:
    return {
        note_id: NoteDigest(
            note_id=note.note_id,
            note_type=note.note_type,
            summary=note.summary,
            flags=note.flags,
        )
        for note_id, note in note_corpus.items()
    }


# --- Extraction planning (flow Step 5) ---
# Stage names on ``TargetGroup.stage``: initials run first and scope later groups;
# dependents wait until every initial-stage group is terminal.
INITIAL_STAGE = "initial"
DEPENDENT_STAGE = "dependent"


def site_applies(applies_to: SiteApplicability | None, facts: CaseFacts | None) -> bool:
    """Whether a site-limited group applies to the case.

    Follows the ``CaseFacts`` principle that *unknown facts widen scope, never
    narrow it*: a group with no restriction, or a case whose site/histology is
    still unknown, passes. A group is ruled out only when a site/histology is
    positively known and none of them match the restriction.
    """
    if applies_to is None:
        return True  # not site-limited

    gross_site = facts.gross_primary_site if facts else None
    primary_site = facts.primary_site if facts else None
    histology = facts.histology if facts else None
    if not gross_site and not primary_site and not histology:
        return True  # nothing known that could exclude it

    if gross_site and any(
        s.casefold() in gross_site.casefold() or gross_site.casefold() in s.casefold()
        for s in applies_to.gross_primary_sites
    ):
        return True
    if primary_site:
        for site in applies_to.gross_primary_sites:
            ranges = resolve_gross_site(site)
            if ranges is not None and site_in_ranges(primary_site, ranges):
                return True
    if histology and histology in applies_to.histology_families:
        return True
    return False


def group_item_ids(group: TargetGroup) -> list[int]:
    """Item IDs of every variable in a group."""
    return [variable.item_id for variable in group.variables]


def pending_item_ids(
    group: TargetGroup, results: dict[int, CaseVariableResult]
) -> list[int]:
    """Item IDs in the group still awaiting a result (PENDING).

    Structured-data seeding can leave a group partly done, so eligibility and
    extraction work off the pending subset rather than assuming the whole group
    is untouched.
    """
    return [
        item_id
        for item_id in group_item_ids(group)
        if item_id in results and results[item_id].status == VariableStatus.PENDING
    ]


def group_has_pending(group: TargetGroup, results: dict[int, CaseVariableResult]) -> bool:
    """The group has at least one variable still awaiting a result."""
    return bool(pending_item_ids(group, results))


def pending_group(
    group: TargetGroup, results: dict[int, CaseVariableResult]
) -> TargetGroup:
    """Project a group down to only its still-pending variables, preserving its
    gating/filter fields. Extraction runs on this so already-seeded (or already
    coded) variables are never re-extracted. In the common all-pending case the
    result is the group unchanged."""
    pending = set(pending_item_ids(group, results))
    return group.model_copy(
        update={"variables": [v for v in group.variables if v.item_id in pending]}
    )


def group_is_terminal(group: TargetGroup, results: dict[int, CaseVariableResult]) -> bool:
    """Every variable in the group has reached a terminal status."""
    return all(
        results[item_id].status in TERMINAL_VARIABLE_STATUSES
        for item_id in group_item_ids(group)
        if item_id in results
    )


def stage_is_ready(
    group: TargetGroup,
    groups: list[TargetGroup],
    results: dict[int, CaseVariableResult],
) -> bool:
    """Initial groups are always stage-ready; dependents wait until every
    initial-stage group has reached a terminal status."""
    if group.stage != DEPENDENT_STAGE:
        return True
    return all(
        group_is_terminal(other, results)
        for other in groups
        if other.stage == INITIAL_STAGE
    )


def eligible_groups(
    groups: list[TargetGroup],
    results: dict[int, CaseVariableResult],
    descriptors: NoteCorpusDescriptors,
    facts: CaseFacts | None,
) -> list[TargetGroup]:
    """Groups that can run right now: unstarted, stage-ready, and passing both the
    corpus gate and the site restriction.

    May be several at once when an initial fact opens more than one dependent
    group — matching the flow doc's "can be more than one if the gate is passed
    by existing facts." Note-level filtering is deliberately absent here: it is
    per-group and happens on the extract branch, not during planning.
    """
    return [
        group
        for group in groups
        if group_has_pending(group, results)
        and stage_is_ready(group, groups, results)
        and corpus_gate_passes(group.gate, descriptors)
        and site_applies(group.applies_to, facts)
    ]


def resolve_leftovers(
    groups: list[TargetGroup],
    results: dict[int, CaseVariableResult],
    descriptors: NoteCorpusDescriptors,
    facts: CaseFacts | None,
) -> dict[int, CaseVariableResult]:
    """At the fixed point (nothing eligible), attribute why each group with
    pending work cannot run and turn its PENDING variables terminal so the loop
    ends. Already-terminal variables (e.g. structured-data seeds) are left alone.

    Reason precedence is deliberate — site, then gate, then deps:
      * site does not match  -> NOT_APPLICABLE (a definitive exclusion)
      * corpus gate unmet    -> NOT_APPLICABLE (a definitive exclusion)
      * otherwise            -> BLOCKED, citing the still-pending initial item IDs
        (the only "dependency" the models express is the initial->dependent stage
        ordering, so a would-be-eligible group left unrun was blocked on it).
    """
    pending_initials = [
        item_id
        for group in groups
        if group.stage == INITIAL_STAGE
        for item_id in pending_item_ids(group, results)
    ]

    updates: dict[int, CaseVariableResult] = {}
    for group in groups:
        pending = pending_item_ids(group, results)
        if not pending:
            continue

        if not site_applies(group.applies_to, facts):
            status = VariableStatus.NOT_APPLICABLE
            reason = "Group does not apply to the case's primary site/histology."
            blockers: list[int] = []
        elif not corpus_gate_passes(group.gate, descriptors):
            status = VariableStatus.NOT_APPLICABLE
            reason = f"Corpus gate not met: {[gate.value for gate in group.gate or []]}."
            blockers = []
        else:
            status = VariableStatus.BLOCKED
            reason = "Prerequisite initial-stage extraction did not complete."
            blockers = pending_initials

        for item_id in pending:
            updates[item_id] = CaseVariableResult(
                item_id=item_id,
                status=status,
                reason=reason,
                blocking_item_ids=blockers if status == VariableStatus.BLOCKED else [],
            )
    return updates


def _result_from_extraction(extraction: ValidatedVariableOutput) -> CaseVariableResult:
    """Map one validated extractor output to its terminal orchestration status.

    Three outcomes, mirroring ``CaseVariableResult``'s validators:
      * invalid (exhausted repair)          -> ERROR, extraction kept as evidence
      * valid with a value                  -> EXTRACTED
      * valid with no value (a clean miss)  -> NOT_FOUND
    """
    if not extraction.is_valid:
        return CaseVariableResult(
            item_id=extraction.item_id,
            status=VariableStatus.ERROR,
            extraction=extraction,
            reason=(
                "; ".join(extraction.validation_errors)
                or "Extraction failed validation."
            ),
        )
    if extraction.value is not None:
        return CaseVariableResult(
            item_id=extraction.item_id,
            status=VariableStatus.EXTRACTED,
            value=extraction.value,
            extraction=extraction,
        )
    return CaseVariableResult(
        item_id=extraction.item_id,
        status=VariableStatus.NOT_FOUND,
        extraction=extraction,
    )


# NAACCR item IDs whose coded value is also a case-scoping fact. Populating these
# after an initial-stage group extracts lets dependent groups be scoped (site
# applicability, rule reduction) against real coded values instead of only the
# coarse characterization facts. Laterality (410) has no CaseFacts field, so it is
# deliberately absent.
ITEM_ID_TO_CASE_FACT: dict[int, str] = {
    390: "date_of_diagnosis",
    400: "primary_site",
    522: "histology",
    523: "behavior",
    220: "sex",
}


def derive_case_facts(
    facts: CaseFacts | None,
    results: dict[int, CaseVariableResult],
) -> CaseFacts | None:
    """Fold newly coded values into the case facts (flow Step 7).

    Only fills a fact that is currently unknown — unknown widens scope, and a
    known fact (pre-seeded, from characterization, or from an earlier group) is
    never clobbered by a later extraction. ``value`` is non-None only for
    extracted/structured-data results, so no status check is needed. Returns the
    original ``facts`` object unchanged when nothing new was learned.
    """
    updates: dict[str, str] = {}
    for item_id, field in ITEM_ID_TO_CASE_FACT.items():
        result = results.get(item_id)
        if result is None or result.value is None:
            continue
        if getattr(facts, field, None) is None:
            updates[field] = result.value
    if not updates:
        return facts
    base = facts.model_dump() if facts is not None else {}
    base.update(updates)
    return CaseFacts(**base)


def not_found_results(
    requested_variables: VariableGroupInfo, reason: str
) -> dict[int, CaseVariableResult]:
    """Terminal NOT_FOUND for every requested variable when no notes were read.

    Used when selection leaves a group with zero notes: the variable applies but
    no evidence exists to read, which is a clean miss (NOT_FOUND), not an error or
    a non-applicability. ``CaseVariableResult`` requires a valid, value-less
    extraction for NOT_FOUND, so a deterministic empty one is synthesized here
    instead of spending an LLM call on an empty note set.
    """
    results: dict[int, CaseVariableResult] = {}
    for variable in requested_variables.variables:
        extraction = ValidatedVariableOutput(
            item_id=variable.item_id,
            value=None,
            explanation=reason,
            most_important_note=None,
            spans=[],
            presence_confidence=ConfidenceLevel.LOW,
            is_valid=True,
            validation_errors=[],
            extraction_attempts=0,
        )
        results[variable.item_id] = CaseVariableResult(
            item_id=variable.item_id,
            status=VariableStatus.NOT_FOUND,
            extraction=extraction,
        )
    return results


def to_case_results(
    requested_variables: VariableGroupInfo,
    extracted_values: ValidatedVariableGroupOutput | None,
) -> dict[int, CaseVariableResult]:
    """Fold an extractor group output into per-item orchestration results.

    Iterates the *requested* variables so every requested item_id gets a result
    even when the extractor dropped one; a missing (or wholly absent) extraction
    becomes an ERROR rather than silently staying PENDING and re-planning forever.
    """
    by_id = {
        extraction.item_id: extraction
        for extraction in (extracted_values.variables if extracted_values else [])
    }
    results: dict[int, CaseVariableResult] = {}
    for variable in requested_variables.variables:
        extraction = by_id.get(variable.item_id)
        if extraction is None:
            results[variable.item_id] = CaseVariableResult(
                item_id=variable.item_id,
                status=VariableStatus.ERROR,
                reason="Extractor produced no result for this variable.",
            )
        else:
            results[variable.item_id] = _result_from_extraction(extraction)
    return results


def build_report(results: dict[int, CaseVariableResult]) -> CaseReport:
    """Roll finished variable results up into a review report (flow Step 8).

    Emits at most one flag per variable, in priority order so the report stays
    scannable and non-redundant:
      * ERROR status               -> the reason already carries any validation
        detail, so no separate invalid-extraction flag is added;
      * a retained invalid extraction on a non-error result (a defensive guard;
        the current mapping funnels invalid extractions into ERROR);
      * an accepted value coded at LOW presence confidence.
    Variables that are clean (extracted/structured with adequate confidence,
    not-found, not-applicable, blocked) raise no flag. Ordered by item ID.
    """
    flags: list[ReviewFlag] = []
    for item_id in sorted(results):
        result = results[item_id]
        if result.status == VariableStatus.ERROR:
            flags.append(ReviewFlag(
                item_id=item_id,
                flag_type=ReviewFlagType.ERROR,
                detail=result.reason or "Extraction ended in error.",
            ))
        elif result.extraction is not None and not result.extraction.is_valid:
            flags.append(ReviewFlag(
                item_id=item_id,
                flag_type=ReviewFlagType.INVALID_EXTRACTION,
                detail=(
                    "; ".join(result.extraction.validation_errors)
                    or "Retained extraction failed validation."
                ),
            ))
        elif (
            result.status == VariableStatus.EXTRACTED
            and result.extraction is not None
            and result.extraction.presence_confidence == ConfidenceLevel.LOW
        ):
            flags.append(ReviewFlag(
                item_id=item_id,
                flag_type=ReviewFlagType.LOW_CONFIDENCE,
                detail=f"Value {result.value!r} was coded at low confidence.",
            ))
    return CaseReport(flags=flags)
