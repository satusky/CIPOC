"""Deterministic orchestration helpers.

Pure functions used to set up and drive orchestrator state. Keep these free of
LLM calls; bounded model usage belongs in the scanner/extractor subagents.
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Collection, Iterable, Mapping, Sequence, get_args

from cipoc.models import (
    CancerMention,
    CaseFacts,
    CaseReport,
    CaseVariableResult,
    ConfidenceLevel,
    CorpusGate,
    NoteCorpusDescriptors,
    NoteDigest,
    NoteFilter,
    NoteFilterEvaluation,
    NoteSelectionRejectionCode,
    NoteSelectionUnevaluatedCode,
    ProcessedClinicalNote,
    ReviewFlag,
    ReviewFlagType,
    ScopingDictionary,
    SiteApplicability,
    SsfEquivalent,
    TargetGroup,
    ValidatedVariableGroupOutput,
    ValidatedVariableOutput,
    VariableGroupInfo,
    VariableInfo,
    VariableStatus,
    ConceptWithEvidence,
    GrossSite,
)
from cipoc.models.case import TERMINAL_VARIABLE_STATUSES
from .coding_context import ranges_overlap, site_in_ranges
from .scoping import (
    BEHAVIOR_ITEM,
    DISCRIMINATOR_FACTS,
    DISCRIMINATOR_KEYS,
    HISTOLOGY_ITEM,
    PRIMARY_SITE_ITEM,
    discriminator_choice,
    discriminator_not_needed_reason,
    discriminator_variable,
    discriminators_separating,
    gross_site_candidates,
    item_applies_in_year,
    item_label,
    out_of_year_reason,
    ssf_equivalent,
    ssf_items,
    ssf_variable,
)


def _concept_present(corpus: NoteCorpusDescriptors, concept: str) -> bool:
    """Return whether a single concept is flagged present in the corpus roll-up."""
    return corpus.concepts.get(concept, ConceptWithEvidence(presence=False)).presence


# Treatment is modeled as granular modalities rather than an aggregate concept,
# so the treatment gate is derived by OR-ing those modality concepts.
TREATMENT_CONCEPTS = (
    "surgery", "chemotherapy", "radiation", "hormonal_therapy", "immunotherapy"
)


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

    Only ``item_id``, ``name`` and ``replaces`` are read per variable; richer
    data-dictionary and case-scoped metadata is filled later by
    :func:`cipoc.tools.build_variable_group` once case facts are known.
    ``replaces`` marks an older equivalent, such as Grade (440), with the items
    in its group it stands in for (``TargetGroup.replaces``).

    A ``stage`` outside the file's ``stages`` list (see :func:`load_stages`)
    fails here, at load, and so does a ``replaces`` naming the variable itself
    or an item that isn't in its group.
    """
    with open(path, "r") as f:
        config = json.load(f)
    stages = _config_stages(config)

    def _to_group(node: dict, *, gate=None, stage=None, note_filter=None) -> TargetGroup:
        group = TargetGroup(
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
            replaces={
                variable["item_id"]: variable["replaces"]
                for variable in node.get("variables", [])
                if "replaces" in variable
            },
        )
        site_applies(group.applies_to, None)
        _stage_index(group, stages)
        _check_replaces(group)
        return group

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


def _check_replaces(group: TargetGroup) -> None:
    item_ids = group_item_ids(group)
    for older, replaced in group.replaces.items():
        if not replaced:
            raise ValueError(f"Group {group.group_id!r}: item {older}'s 'replaces' is empty.")
        if older in replaced:
            raise ValueError(f"Group {group.group_id!r}: item {older} can't replace itself.")
        missing = [item_id for item_id in replaced if item_id not in item_ids]
        if missing:
            raise ValueError(
                f"Group {group.group_id!r}: item {older} replaces {missing}, which aren't in the group."
            )


def load_stages(path: str | Path) -> tuple[str, ...]:
    """Read the variable-group config's stage order, earliest first.

    A config without ``stages`` uses :data:`DEFAULT_STAGES`.
    """
    with open(path, "r") as f:
        return _config_stages(json.load(f))


def _config_stages(config: dict) -> tuple[str, ...]:
    stages = config.get("stages", list(DEFAULT_STAGES))
    if (
        not isinstance(stages, list)
        or not stages
        or not all(isinstance(stage, str) and stage for stage in stages)
        or len(set(stages)) != len(stages)
    ):
        raise ValueError(f"'stages' must be a non-empty list of unique stage names, got {stages!r}.")
    return tuple(stages)


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


def evaluate_note_filter(
    note: ProcessedClinicalNote,
    note_filter: NoteFilter | None,
    *,
    anchor: date | None = None,
) -> NoteFilterEvaluation:
    """Evaluate every configured note-filter dimension and explain the result.

    Each dimension is evaluated independently and combined with AND; a dimension
    left empty/``None`` imposes no restriction. A ``None`` filter passes every
    note. Configured checks that are currently disabled or lack required context
    are reported as unevaluated and do not reject the note.
    """
    if note_filter is None:
        return NoteFilterEvaluation(passes=True)

    rejection_reasons: list[NoteSelectionRejectionCode] = []
    unevaluated_checks: list[NoteSelectionUnevaluatedCode] = []

    # Note type: case-insensitive exact match against the allowed set.
    if note_filter.note_types:
        allowed = {t.strip().casefold() for t in note_filter.note_types}
        if (note.note_type or "").strip().casefold() not in allowed:
            rejection_reasons.append(NoteSelectionRejectionCode.NOTE_TYPE_MISMATCH)

    # Keyword filtering remains disabled because these values are LLM-generated.
    if note_filter.keywords:
        unevaluated_checks.append(
            NoteSelectionUnevaluatedCode.KEYWORD_FILTER_DISABLED
        )

    # Cancer status: note's temporality set must intersect the allowed statuses.
    if note_filter.cancer_status:
        if not (note.cancer_status or set()).intersection(note_filter.cancer_status):
            rejection_reasons.append(
                NoteSelectionRejectionCode.CANCER_STATUS_MISMATCH
            )

    # Date window: note within `within_days` of the anchor (skipped without one).
    if note_filter.within_days is not None:
        if anchor is None:
            unevaluated_checks.append(
                NoteSelectionUnevaluatedCode.TEMPORAL_ANCHOR_UNAVAILABLE
            )
        else:
            note_date = _parse_note_date(note.date)
            if note_date is None:
                rejection_reasons.append(
                    NoteSelectionRejectionCode.MISSING_OR_INVALID_DATE
                )
            elif abs((note_date - anchor).days) > note_filter.within_days:
                rejection_reasons.append(
                    NoteSelectionRejectionCode.OUTSIDE_DATE_WINDOW
                )

    return NoteFilterEvaluation(
        passes=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        unevaluated_checks=unevaluated_checks,
    )


def note_matches_filter(
    note: ProcessedClinicalNote,
    note_filter: NoteFilter | None,
    *,
    anchor: date | None = None,
) -> bool:
    """Return whether a note passes its deterministic filter."""

    return evaluate_note_filter(note, note_filter, anchor=anchor).passes


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


def is_diagnosed_malignant(mention: CancerMention) -> bool:
    """Whether a mention counts toward note status and affected tissues.

    Probable wording is a SEER ambiguous term that counts as a diagnosis, so a
    probable malignant mention counts like a definite one. Benign,
    unknown-behavior and possible mentions are kept on the note for review but
    don't establish a cancer.
    """
    return mention.behavior == "malignant" and mention.diagnostic_certainty in ("definite", "probable")


def candidate_gross_sites(note_corpus: dict[int | str, ProcessedClinicalNote]) -> list[str]:
    """The tissues Primary Site is narrowed to, read from the notes' mentions.

    The first status, in the order current, recent, historical, with any
    definite or probable mention of any behavior decides. A current benign
    tumor therefore blocks falling back to an older cancer, while a current
    mention the notes only call possible doesn't. Its tissues come from its
    definite or probable malignant mentions, and a name outside ``GrossSite``
    counts as "unspecified". No tissues, or any "unspecified", gives [], which
    doesn't narrow.
    """
    mentions = [
        mention
        for note in note_corpus.values()
        for mention in note.cancer_mentions or []
    ]
    for status in ("current", "recent", "historical"):
        diagnosed = [
            mention
            for mention in mentions
            if mention.status == status
            and mention.diagnostic_certainty in ("definite", "probable")
        ]
        if not diagnosed:
            continue
        tissues = {
            tissue if tissue in _RESTRICTABLE_GROSS_SITES else "unspecified"
            for tissue in (
                mention.affected_tissue.strip()
                for mention in diagnosed
                if is_diagnosed_malignant(mention)
            )
        }
        return sorted(tissues) if tissues and "unspecified" not in tissues else []
    return []


def build_corpus_descriptors(note_corpus: dict[int | str, ProcessedClinicalNote]) -> NoteCorpusDescriptors:
    notes = list(note_corpus.values())
    dates = sorted([note.date for note in notes])
    types = {note.note_type for note in notes}

    affected_tissues = defaultdict(set)
    unique_flags = set([])
    for note in notes:
        if note.cancer_mentions is not None:
            for mention in note.cancer_mentions:
                if is_diagnosed_malignant(mention):
                    affected_tissues[mention.status].add(mention.affected_tissue)

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
# Stage order comes from the variable-group config's ``stages`` list. Each stage
# waits until every group in the stages before it is terminal. A group with no
# stage runs right away and holds no other stage.
DEFAULT_STAGES = ("initial", "dependent")


def _stage_index(group: TargetGroup, stages: Sequence[str]) -> int | None:
    """Position of the group's stage in ``stages``, or None when it has no stage."""
    if group.stage is None:
        return None
    if group.stage not in stages:
        raise ValueError(
            f"Group {group.group_id!r} has unknown stage {group.stage!r}; "
            f"expected one of {list(stages)}."
        )
    return stages.index(group.stage)


# "unspecified" never narrows, so a group can't be restricted to it.
_RESTRICTABLE_GROSS_SITES = frozenset(get_args(GrossSite)) - {"unspecified"}


def site_applies(
    applies_to: SiteApplicability | None,
    facts: CaseFacts | None,
    *,
    tissue_codes: Mapping[str, Collection[str]] | None = None,
) -> bool:
    """Exclude only definitively false restrictions, evaluating relevant facts.

    Internally None means unknown. OR excludes only when all alternatives are
    false; AND is false when even one condition is false. An informative coded
    primary outranks gross-site inference, including when they conflict. An
    explicit unknown/malformed primary also suppresses gross-site fallback.
    Candidate gross sites are OR: a restriction matches if any candidate does.
    Histology families are 3-digit prefixes of a coded histology. When histology
    isn't coded, the candidate histology families stand in for it, again as OR.
    Gross primary sites are ``GrossSite`` tissues other than "unspecified";
    anything else raises here, not during historical artifact parsing.

    ``tissue_codes`` gives each tissue's Primary Site codes (``tissue_site_codes``).
    A tissue missing from it is unknown, so it never excludes.
    """
    if applies_to is None:
        return True

    def tissue_ranges(tissue: str | None) -> list[str] | None:
        codes = (tissue_codes or {}).get(tissue) if tissue else None
        return list(codes) if codes else None

    candidates = gross_site_candidates(facts) if facts and not facts.primary_site else []
    candidate_ranges = [tissue_ranges(tissue) for tissue in candidates]
    # Candidates are OR, so their union stands for them. One unknown candidate
    # makes the whole gross site unknown.
    gross_ranges = (
        sorted(set().union(*candidate_ranges))
        if candidate_ranges and None not in candidate_ranges
        else None
    )
    primary_site = (facts.primary_site or "").strip().upper() if facts else ""
    # C809 is unknown primary; malformed/out-of-range codes cannot exclude sites.
    if (
        not re.fullmatch(r"C[0-9]{2}\.?[0-9]", primary_site)
        or primary_site.replace(".", "") >= "C809"
    ):
        primary_site = ""
    histology = (facts.histology or "").strip() if facts else ""
    # The families the histology could be in: a coded histology's own, or else
    # the candidates. As with Primary Site, a malformed coded histology is unknown
    # and stops the fallback, and so does a malformed candidate.
    if re.fullmatch(r"[0-9]{4}", histology):
        histology_families: set[str] | None = {histology[:3]}
    elif histology:
        histology_families = None
    else:
        candidate_families = list(facts.candidate_histology_families) if facts else []
        histology_families = (
            set(candidate_families)
            if candidate_families and all(re.fullmatch(r"[0-9]{3}", family) for family in candidate_families)
            else None
        )

    def matches_sites(ranges: list[str] | None) -> bool | None:
        if ranges is None:
            return None
        if primary_site:
            return site_in_ranges(primary_site, ranges)
        if gross_ranges:
            return ranges_overlap(gross_ranges, ranges)
        return None

    def evaluate(restriction: SiteApplicability) -> bool | None:
        for family in restriction.histology_families:
            if not re.fullmatch(r"[0-9]{3}", family):
                raise ValueError(
                    f"Unsupported histology family: {family!r}. "
                    "Expected a 3-digit histology prefix, such as '872'."
                )
        for site in restriction.gross_primary_sites:
            if site not in _RESTRICTABLE_GROSS_SITES:
                raise ValueError(
                    f"Unsupported gross primary site: {site!r}. "
                    "Expected a GrossSite tissue name, such as 'breast'."
                )
        if restriction.all_of:
            conditions = [evaluate(child) for child in restriction.all_of]
            return False if False in conditions else None if None in conditions else True
        if restriction.any_of:
            alternatives = [evaluate(child) for child in restriction.any_of]
        else:
            alternatives: list[bool | None] = []
            for site in restriction.gross_primary_sites:
                match = matches_sites(tissue_ranges(site))
                if not primary_site and any(site.casefold() == tissue.casefold() for tissue in candidates):
                    match = True
                alternatives.append(match)
            if restriction.primary_sites:
                alternatives.append(matches_sites(restriction.primary_sites))
            if restriction.histology_families:
                alternatives.append(
                    bool(histology_families & set(restriction.histology_families))
                    if histology_families is not None else None
                )
        if not alternatives:
            return True
        return True if True in alternatives else None if None in alternatives else False

    return evaluate(applies_to) is not False


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
    *,
    stages: Sequence[str] = DEFAULT_STAGES,
) -> bool:
    """First-stage and unstaged groups are always stage-ready; a later stage waits
    until every group in the stages before it has reached a terminal status."""
    index = _stage_index(group, stages)
    if index is None or index == 0:
        return True
    return all(
        group_is_terminal(other, results)
        for other in groups
        if (other_index := _stage_index(other, stages)) is not None and other_index < index
    )


def eligible_groups(
    groups: list[TargetGroup],
    results: dict[int, CaseVariableResult],
    descriptors: NoteCorpusDescriptors,
    facts: CaseFacts | None,
    *,
    stages: Sequence[str] = DEFAULT_STAGES,
    tissue_codes: Mapping[str, Collection[str]] | None = None,
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
        and stage_is_ready(group, groups, results, stages=stages)
        and corpus_gate_passes(group.gate, descriptors)
        and site_applies(group.applies_to, facts, tissue_codes=tissue_codes)
    ]


def resolve_leftovers(
    groups: list[TargetGroup],
    results: dict[int, CaseVariableResult],
    descriptors: NoteCorpusDescriptors,
    facts: CaseFacts | None,
    *,
    stages: Sequence[str] = DEFAULT_STAGES,
    tissue_codes: Mapping[str, Collection[str]] | None = None,
) -> dict[int, CaseVariableResult]:
    """When nothing is eligible, close pending work ruled out by site or gate.

    Resolve only the earliest stage with pending work: making it terminal can
    release the next stage on the next planner pass. Unstaged groups close with
    the last stage. Already-terminal variables (e.g. structured-data seeds) are
    left alone.

    Reason precedence is deliberate — site, then gate, then deps:
      * site does not match  -> NOT_APPLICABLE (a definitive exclusion)
      * corpus gate unmet    -> NOT_APPLICABLE (a definitive exclusion)
      * otherwise            -> BLOCKED (defensive fallback for unexplained work)
    """
    def leftover_index(group: TargetGroup) -> int:
        index = _stage_index(group, stages)
        return len(stages) - 1 if index is None else index

    pending_by_stage: dict[int, list[int]] = defaultdict(list)
    for group in groups:
        pending_by_stage[leftover_index(group)].extend(pending_item_ids(group, results))
    pending_stages = [index for index, item_ids in pending_by_stage.items() if item_ids]
    if not pending_stages:
        return {}
    closing = min(pending_stages)
    pending_in_stage = pending_by_stage[closing]

    updates: dict[int, CaseVariableResult] = {}
    for group in groups:
        if leftover_index(group) != closing:
            continue
        pending = pending_item_ids(group, results)
        if not pending:
            continue

        if not site_applies(group.applies_to, facts, tissue_codes=tissue_codes):
            status = VariableStatus.NOT_APPLICABLE
            reason = "Group does not apply to the case's primary site/histology."
            blockers: list[int] = []
        elif not corpus_gate_passes(group.gate, descriptors):
            status = VariableStatus.NOT_APPLICABLE
            reason = f"Corpus gate not met: {[gate.value for gate in group.gate or []]}."
            blockers = []
        else:
            status = VariableStatus.BLOCKED
            reason = f"Prerequisite {stages[closing]}-stage extraction did not complete."
            blockers = pending_in_stage

        for item_id in pending:
            updates[item_id] = CaseVariableResult(
                item_id=item_id,
                status=status,
                reason=reason,
                blocking_item_ids=blockers if status == VariableStatus.BLOCKED else [],
            )
    return updates


def _equivalent_names(
    groups: Sequence[TargetGroup],
) -> tuple[dict[int, str | None], dict[int, list[int]]]:
    """Every grouped item's name, and each older equivalent's ``replaces``."""
    names = {variable.item_id: variable.name for group in groups for variable in group.variables}
    replaces = {older: replaced for group in groups for older, replaced in group.replaces.items()}
    return names, replaces


def _year_reason(
    scoping: ScopingDictionary,
    variable: VariableInfo,
    date_of_diagnosis: str | None,
    names: Mapping[int, str | None],
    replaces: Mapping[int, list[int]],
    **ssf,
) -> str | None:
    """``out_of_year_reason`` with the variable's fixed equivalents from the groups."""
    return out_of_year_reason(
        scoping, variable.item_id, date_of_diagnosis, variable.name,
        held_by=[
            (older, names[older])
            for older, replaced in replaces.items()
            if variable.item_id in replaced
        ],
        replaced_by=[
            (newer, names[newer])
            for newer in replaces.get(variable.item_id, [])
            if newer in names
        ],
        **ssf,
    )


def out_of_year_results(
    groups: Sequence[TargetGroup],
    results: Mapping[int, CaseVariableResult],
    scoping: ScopingDictionary | None,
    date_of_diagnosis: str | None,
) -> dict[int, CaseVariableResult]:
    """NOT_APPLICABLE for every PENDING item the diagnosis year is outside of.

    Decided per item, not per group, since a group can mix them: one holds both
    surgery items, 670 (through 2022) and 671 (from 2023). The reason names the
    year and the item's years (``out_of_year_reason``), and any equivalent that
    is collected that year (``TargetGroup.replaces``): the older item for a
    newer one, or the newer items for an older one. Terminal results,
    structured-data seeds among them, are left alone, and so is everything when
    the year is unknown or there's no site dictionary. Returns only the changed
    results, for the ``variable_results`` reducer.

    In a CS or TNM year, an item some schema holds in a Site-Specific Factor
    (``ssf_items``) is left PENDING: which SSF depends on the schema, and that
    needs the site and histology, which a later stage may code. It's settled
    when its group is planned (``ssf_equivalent_results``).
    """
    if scoping is None:
        return {}
    names, replaces = _equivalent_names(groups)
    deferred = ssf_items(scoping, date_of_diagnosis)
    closed: dict[int, CaseVariableResult] = {}
    for group in groups:
        for variable in group.variables:
            result = results.get(variable.item_id)
            if result is None or result.status != VariableStatus.PENDING or variable.item_id in deferred:
                continue
            reason = _year_reason(scoping, variable, date_of_diagnosis, names, replaces)
            if reason is not None:
                closed[variable.item_id] = CaseVariableResult(
                    item_id=variable.item_id, status=VariableStatus.NOT_APPLICABLE, reason=reason
                )
    return closed


def with_ssf_equivalents(
    groups: Sequence[TargetGroup],
    equivalents: Mapping[int, SsfEquivalent],
    scoping: ScopingDictionary | None,
) -> list[TargetGroup]:
    """The groups with each SSF equivalent added to the group it's extracted with.

    The SSF joins as ``ssf_variable`` gives it, so planning, extraction and
    stage release treat it as one of the group's variables. The configured
    groups, and with them the run's inputs, never list it. An SSF the config
    requests itself stays in its own group.
    """
    if not equivalents or scoping is None:
        return list(groups)
    configured = {variable.item_id for group in groups for variable in group.variables}
    added: dict[str | None, list[VariableInfo]] = defaultdict(list)
    for equivalent in equivalents.values():
        if equivalent.item_id in configured:
            continue
        # The candidates agree on the SSF, so any of them describes it.
        mapping = scoping.schemas[equivalent.schemas[0]].ssf[equivalent.replaces[0]]
        added[equivalent.group_id].append(ssf_variable(mapping))
    return [
        group.model_copy(update={"variables": [*group.variables, *added[group.group_id]]})
        if group.group_id in added else group
        for group in groups
    ]


def ssf_equivalent_results(
    ready: Sequence[TargetGroup],
    groups: Sequence[TargetGroup],
    results: Mapping[int, CaseVariableResult],
    scoping: ScopingDictionary | None,
    facts: CaseFacts | None,
    equivalents: Mapping[int, SsfEquivalent],
    structured_data: Mapping[int, str] | None = None,
) -> tuple[dict[int, CaseVariableResult], dict[int, SsfEquivalent]]:
    """Settle the items ``out_of_year_results`` left for their group's planning.

    For each PENDING item in a ``ready`` group that some schema holds in a CS
    Site-Specific Factor (``ssf_items``) and the year doesn't collect, by
    ``ssf_equivalent`` over the case facts:

    * substitute: the item ends NOT_APPLICABLE, its reason naming the SSF with
      any fixed equivalent, and the SSF is recorded to be extracted with the
      group (``with_ssf_equivalents``). Its result starts PENDING, or from the
      caller's structured data, unless it already has one. Two items held in
      one SSF, like Grade Clinical and Pathological, share it.
    * disagree: the item ends BLOCKED, the reason saying what each candidate
      schema holds it in. Nothing is extracted for it. When the run requests a
      discriminator the candidates differ by and it ended without a value, as
      with the discriminator variable groups file, ``blocking_item_ids`` names
      it (``_unsettled_discriminators``).
    * none: the item ends NOT_APPLICABLE with the plain year reason.

    ``groups`` supplies the items' names and fixed equivalents. Returns the
    changed results and the new or updated SSF equivalents, each keyed by item.
    """
    date_of_diagnosis = facts.date_of_diagnosis if facts else None
    deferred = ssf_items(scoping, date_of_diagnosis) if scoping is not None else frozenset()
    if not deferred:
        return {}, {}
    names, replaces = _equivalent_names(groups)
    updates: dict[int, CaseVariableResult] = {}
    recorded: dict[int, SsfEquivalent] = {}
    for group in ready:
        for variable in group.variables:
            item_id = variable.item_id
            result = results.get(item_id)
            if (
                item_id not in deferred
                or result is None
                or result.status != VariableStatus.PENDING
                or item_applies_in_year(scoping, item_id, date_of_diagnosis)
            ):
                continue
            choice = ssf_equivalent(scoping, item_id, facts)
            mapping = choice.mapping
            existing = None
            if mapping is not None:
                existing = recorded.get(mapping.item)
                if existing is None:
                    existing = equivalents.get(mapping.item)
            if choice.outcome == "substitute" and existing is not None and existing.name != mapping.name:
                # Another requested item already holds this SSF number under a
                # different name, so it can't stand in for both.
                choice = choice.model_copy(update={
                    "outcome": "disagree",
                    "reason": f"{mapping.label} ({mapping.item}) already holds {existing.name}",
                })
            if choice.outcome == "substitute":
                recorded[mapping.item] = (existing or SsfEquivalent(
                    item_id=mapping.item, label=mapping.label, name=mapping.name,
                    replaces=[], schemas=choice.schemas, group_id=group.group_id,
                )).model_copy(update={"replaces": [*(existing.replaces if existing else []), item_id]})
                updates[item_id] = CaseVariableResult(
                    item_id=item_id, status=VariableStatus.NOT_APPLICABLE,
                    reason=_year_reason(scoping, variable, date_of_diagnosis, names, replaces, ssf=mapping),
                )
                if mapping.item not in results and mapping.item not in updates:
                    value = (structured_data or {}).get(mapping.item)
                    updates[mapping.item] = (
                        CaseVariableResult(item_id=mapping.item)
                        if value is None else
                        CaseVariableResult(
                            item_id=mapping.item, status=VariableStatus.STRUCTURED_DATA, value=str(value)
                        )
                    )
            elif choice.outcome == "disagree":
                blocking = _unsettled_discriminators(scoping, facts, choice.schemas, names, results)
                disagreement = choice.reason
                if blocking:
                    labels = " and ".join(item_label(blocker, names[blocker]) for blocker in blocking)
                    verb = "has" if len(blocking) == 1 else "have"
                    disagreement += f"; {labels}, which would tell them apart, {verb} no value"
                updates[item_id] = CaseVariableResult(
                    item_id=item_id, status=VariableStatus.BLOCKED,
                    reason=_year_reason(
                        scoping, variable, date_of_diagnosis, names, replaces,
                        ssf_disagreement=disagreement,
                    ),
                    blocking_item_ids=blocking,
                )
            else:
                updates[item_id] = CaseVariableResult(
                    item_id=item_id, status=VariableStatus.NOT_APPLICABLE,
                    reason=_year_reason(scoping, variable, date_of_diagnosis, names, replaces),
                )
    return updates, recorded


# A discriminator that ended this way can't have narrowed the schemas.
_UNSETTLED = frozenset({VariableStatus.NOT_FOUND, VariableStatus.ERROR})


def _unsettled_discriminators(
    scoping: ScopingDictionary,
    facts: CaseFacts | None,
    schemas: Sequence[str],
    names: Mapping[int, str | None],
    results: Mapping[int, CaseVariableResult],
) -> list[int]:
    """The requested discriminators ``schemas`` differ by that ended without a value."""
    requested = [item_id for item_id in DISCRIMINATOR_KEYS if item_id in names]
    return [
        item_id
        for item_id in discriminators_separating(scoping, facts, schemas, requested)
        if (result := results.get(item_id)) is not None and result.status in _UNSETTLED
    ]


def _pending_discriminators(
    groups: Sequence[TargetGroup], results: Mapping[int, CaseVariableResult]
):
    for group in groups:
        for variable in group.variables:
            result = results.get(variable.item_id)
            if variable.item_id in DISCRIMINATOR_KEYS and result is not None and result.status == VariableStatus.PENDING:
                yield group, variable


def discriminator_results(
    ready: Sequence[TargetGroup],
    results: Mapping[int, CaseVariableResult],
    scoping: ScopingDictionary | None,
    facts: CaseFacts | None,
) -> dict[int, CaseVariableResult]:
    """Settle each PENDING discriminator in a ``ready`` group the case can't use.

    The discriminator variable groups file runs them in their own stage, after
    histology, so site, histology and year are coded by then
    (``discriminator_choice``):

    * the candidate schemas don't differ by it: NOT_APPLICABLE, naming them.
    * they differ by it but read it from different tables, so there's no one
      set of codes to offer: BLOCKED. The generator gives schemas a case can
      match together the same table, so this shouldn't happen.
    * no site dictionary: NOT_APPLICABLE, since there are no schemas to pick
      between, and v25 gives the discriminators no codes of their own.

    The rest stay PENDING and are extracted with their table
    (``with_discriminator_variables``).
    """
    closed: dict[int, CaseVariableResult] = {}
    for _, variable in _pending_discriminators(ready, results):
        label = item_label(variable.item_id, variable.name)
        if scoping is None:
            closed[variable.item_id] = CaseVariableResult(
                item_id=variable.item_id, status=VariableStatus.NOT_APPLICABLE,
                reason=f"No site dictionary is loaded, so there are no staging schemas for {label} to pick between.",
            )
            continue
        choice = discriminator_choice(scoping, variable.item_id, facts)
        if not choice.needed:
            closed[variable.item_id] = CaseVariableResult(
                item_id=variable.item_id, status=VariableStatus.NOT_APPLICABLE,
                reason=discriminator_not_needed_reason(choice, label),
            )
        elif choice.discriminator is None:
            closed[variable.item_id] = CaseVariableResult(
                item_id=variable.item_id, status=VariableStatus.BLOCKED,
                reason=f"The schemas the case matches differ by {label} but read it from different tables, "
                       "so there's no one set of codes to offer.",
            )
    return closed


def with_discriminator_variables(
    groups: Sequence[TargetGroup],
    results: Mapping[int, CaseVariableResult],
    scoping: ScopingDictionary | None,
    facts: CaseFacts | None,
) -> list[TargetGroup]:
    """The groups with each PENDING discriminator the candidates differ by
    replaced by its table's variable (``discriminator_variable``), so it's
    offered the table's codes under its name and notes. One whose candidates
    don't share a table keeps the NAACCR entry."""
    if scoping is None:
        return list(groups)
    replaced: dict[int, VariableInfo] = {}
    for _, variable in _pending_discriminators(groups, results):
        choice = discriminator_choice(scoping, variable.item_id, facts)
        if choice.needed and choice.discriminator is not None:
            replaced[variable.item_id] = discriminator_variable(choice.discriminator, variable.item_id)
    if not replaced:
        return list(groups)
    return [
        group.model_copy(update={
            "variables": [replaced.get(variable.item_id, variable) for variable in group.variables],
        })
        if any(variable.item_id in replaced for variable in group.variables) else group
        for group in groups
    ]


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
# after an earlier-stage group extracts lets later groups be scoped (site
# applicability, dictionary selection) against real coded values instead of only the
# coarse characterization facts. Laterality (410) has no CaseFacts field, so it is
# deliberately absent.
ITEM_ID_TO_CASE_FACT: dict[int, str] = {
    390: "date_of_diagnosis",
    400: "primary_site",
    522: "histology",
    523: "behavior",
    220: "sex",
    **DISCRIMINATOR_FACTS,
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

    The histology families the extractor's family step chose fill
    ``candidate_histology_families`` the same way, from the 522 result's
    extraction, even when it ended NOT_FOUND. The block fallback chooses no
    families, so it adds none.
    """
    updates: dict[str, str | list[str]] = {}
    for item_id, field in ITEM_ID_TO_CASE_FACT.items():
        result = results.get(item_id)
        if result is None or result.value is None:
            continue
        if getattr(facts, field, None) is None:
            updates[field] = result.value
    histology = results.get(HISTOLOGY_ITEM)
    selection = histology.extraction.histology if histology and histology.extraction else None
    if selection is not None and selection.families and not (facts and facts.candidate_histology_families):
        updates["candidate_histology_families"] = list(selection.families)
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


def _result_flag(item_id: int, result: CaseVariableResult) -> ReviewFlag | None:
    if result.status == VariableStatus.ERROR:
        return ReviewFlag(
            item_id=item_id,
            flag_type=ReviewFlagType.ERROR,
            detail=result.reason or "Extraction ended in error.",
        )
    if result.extraction is not None and not result.extraction.is_valid:
        return ReviewFlag(
            item_id=item_id,
            flag_type=ReviewFlagType.INVALID_EXTRACTION,
            detail=(
                "; ".join(result.extraction.validation_errors)
                or "Retained extraction failed validation."
            ),
        )
    if (
        result.status == VariableStatus.EXTRACTED
        and result.extraction is not None
        and result.extraction.presence_confidence == ConfidenceLevel.LOW
    ):
        return ReviewFlag(
            item_id=item_id,
            flag_type=ReviewFlagType.LOW_CONFIDENCE,
            detail=f"Value {result.value!r} was coded at low confidence.",
        )
    return None


def build_report(
    results: dict[int, CaseVariableResult],
    case_facts: CaseFacts | None = None,
) -> CaseReport:
    """Roll finished variable results up into a review report (flow Step 8).

    Emits at most one result flag per variable, in priority order so the report
    stays scannable and non-redundant:
      * ERROR status               -> the reason already carries any validation
        detail, so no separate invalid-extraction flag is added;
      * a retained invalid extraction on a non-error result (a defensive guard;
        the current mapping funnels invalid extractions into ERROR);
      * an accepted value coded at LOW presence confidence.
    Variables that are clean (extracted/structured with adequate confidence,
    not-found, not-applicable, blocked) raise no result flag. Separately, more
    than one candidate gross site flags Primary Site (400), after any result
    flag of its own and whether or not 400 was requested, since a run should
    cover one tumor. A coded histology the extractor found off the coded site's
    list flags Histology (522) after its own result flag, since SEER requires
    review of those combinations. Ordered by item ID.
    """
    candidates = case_facts.candidate_gross_sites if case_facts is not None else []
    multiple_sites = len(candidates) > 1
    item_ids = set(results) | ({PRIMARY_SITE_ITEM} if multiple_sites else set())
    flags: list[ReviewFlag] = []
    for item_id in sorted(item_ids):
        result = results.get(item_id)
        flag = _result_flag(item_id, result) if result is not None else None
        if flag is not None:
            flags.append(flag)
        if item_id == PRIMARY_SITE_ITEM and multiple_sites:
            flags.append(ReviewFlag(
                item_id=item_id,
                flag_type=ReviewFlagType.MULTIPLE_CANDIDATE_SITES,
                detail=(
                    f"The notes point to more than one tissue ({', '.join(candidates)}); "
                    "a run should cover one tumor."
                ),
            ))
        if item_id == HISTOLOGY_ITEM and _unlisted_histology(result):
            behavior = results.get(BEHAVIOR_ITEM)
            pair = f"{result.value}/{behavior.value}" if behavior and behavior.value else result.value
            flags.append(ReviewFlag(
                item_id=item_id,
                flag_type=ReviewFlagType.UNLISTED_SITE_HISTOLOGY,
                detail=(
                    f"Histology {pair} isn't on the coded site's SEER Site/Histology "
                    "Validation List; SEER requires review of combinations off the list."
                ),
            ))
    return CaseReport(flags=flags)


def _unlisted_histology(result: CaseVariableResult | None) -> bool:
    """Whether a coded 522 is a pair the extractor recorded as off the site's list."""
    selection = result.extraction.histology if result and result.extraction else None
    return selection is not None and result.value is not None and selection.listed is False
