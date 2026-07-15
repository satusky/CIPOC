"""Deterministic runtime scoping of compiled manual rules into coding context.

Pure functions only — no LLM calls, no I/O beyond loading the committed rule
store under ``documents/rules/`` (or a fixture directory). Scoping narrows
along three axes: the variable being extracted (``item_ids``), case
applicability (``applies_to`` evaluated against ``CaseFacts``, where unknown
facts widen scope), and valid-code reduction (which never empties; an empty
intersection falls back to the unreduced set and flags review).
"""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import TypeAdapter

from cipoc.models import (
    CaseFacts,
    RuleApplicability,
    RuleStoreManifest,
    RuleUnit,
    ScopedVariableContext,
    ScopingReviewReason,
)


_RULE_LIST_ADAPTER = TypeAdapter(list[RuleUnit])

_STORE_CACHE: dict[Path, "RuleStore"] = {}


@dataclass(frozen=True)
class RuleStore:
    """In-memory rule store: manifest plus item-indexed rule units."""
    manifest: RuleStoreManifest
    units: tuple[RuleUnit, ...]
    units_by_item: dict[int, tuple[RuleUnit, ...]]
    general_units: tuple[RuleUnit, ...]


def load_rule_store(rules_dir: str | Path, *, use_cache: bool = True) -> RuleStore:
    """Parse manifest + rule files under rules_dir and index units by item ID."""
    path = Path(rules_dir).resolve()
    if use_cache and path in _STORE_CACHE:
        return _STORE_CACHE[path]

    manifest = RuleStoreManifest.model_validate_json((path / "manifest.json").read_text())

    units: list[RuleUnit] = []
    for file in sorted(p for p in path.rglob("*.json") if p.name != "manifest.json"):
        units.extend(_RULE_LIST_ADAPTER.validate_json(file.read_text()))

    unknown_sources = sorted({u.source_doc for u in units if manifest.get(u.source_doc) is None})
    if unknown_sources:
        raise ValueError(f"Rule units reference manuals missing from the manifest: {unknown_sources}")
    rule_ids = [u.rule_id for u in units]
    duplicates = sorted({r for r in rule_ids if rule_ids.count(r) > 1})
    if duplicates:
        raise ValueError(f"Duplicate rule_ids in rule store: {duplicates}")

    units_by_item: dict[int, list[RuleUnit]] = {}
    general_units: list[RuleUnit] = []
    for unit in units:
        if unit.item_ids:
            for item_id in unit.item_ids:
                units_by_item.setdefault(item_id, []).append(unit)
        else:
            general_units.append(unit)

    store = RuleStore(
        manifest=manifest,
        units=tuple(units),
        units_by_item={item: tuple(item_units) for item, item_units in units_by_item.items()},
        general_units=tuple(general_units),
    )
    if use_cache:
        _STORE_CACHE[path] = store
    return store


def parse_lenient_date(value: str | date | None) -> date | None:
    """Parse a date string leniently (ISO, YYYYMMDD, MM/DD/YYYY, partial dates).

    Returns None when the value is missing or unparseable; callers treat None
    as unknown. Unknown month/day resolve to the earliest possible date.
    """
    if value is None or isinstance(value, date):
        return value
    text = value.strip()
    us_format = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if us_format:
        digits = us_format.group(3) + us_format.group(1).zfill(2) + us_format.group(2).zfill(2)
    else:
        digits = re.sub(r"[^0-9]", "", text)
    if len(digits) == 4:
        digits += "0101"
    elif len(digits) == 6:
        digits += "01"
    if len(digits) != 8:
        return None
    year, month, day = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
    try:
        return date(year, month or 1, day or 1)
    except ValueError:
        return None


def normalize_site(code: str | None) -> str | None:
    """Normalize an ICD-O-3 topography code to 'Cxxx' form; None if malformed."""
    if not code:
        return None
    text = code.strip().upper().replace(".", "")
    return text if re.fullmatch(r"C\d{3}", text) else None


def normalize_histology(code: str | None) -> str | None:
    """Normalize an ICD-O-3 morphology code to its 4-digit form; None if malformed."""
    if not code:
        return None
    text = code.strip().split("/")[0]
    return text if re.fullmatch(r"\d{4}", text) else None


def _code_in_ranges(numeric: int, ranges: list[str], normalize) -> bool:
    for entry in ranges:
        low, _, high = entry.partition("-")
        low_norm = normalize(low)
        if low_norm is None:
            continue
        high_norm = normalize(high) if high else low_norm
        if high_norm is None:
            continue
        low_num = int(re.sub(r"[^0-9]", "", low_norm))
        high_num = int(re.sub(r"[^0-9]", "", high_norm))
        if low_num <= numeric <= high_num:
            return True
    return False


def site_in_ranges(site: str, ranges: list[str]) -> bool:
    """True when a normalized site code falls in any code or 'C500-C509' range."""
    norm = normalize_site(site)
    if norm is None:
        return False
    return _code_in_ranges(int(norm[1:]), ranges, normalize_site)


def histology_in_ranges(histology: str, ranges: list[str]) -> bool:
    """True when a normalized histology code falls in any code or '8500-8549' range."""
    norm = normalize_histology(histology)
    if norm is None:
        return False
    return _code_in_ranges(int(norm), ranges, normalize_histology)


def matches_case(applies_to: RuleApplicability | None, case_facts: CaseFacts) -> bool:
    """Evaluate an applicability predicate against known case facts.

    Unknown widens: a condition only excludes a rule when the corresponding
    fact is known (and parseable) and fails to match.
    """
    if applies_to is None:
        return True

    site = normalize_site(case_facts.primary_site)
    if applies_to.sites and site is not None and not site_in_ranges(site, applies_to.sites):
        return False

    histology = normalize_histology(case_facts.histology)
    if applies_to.histologies and histology is not None and not histology_in_ranges(histology, applies_to.histologies):
        return False

    behavior = case_facts.behavior.strip().split("/")[-1] if case_facts.behavior else None
    if applies_to.behaviors and behavior and behavior not in applies_to.behaviors:
        return False

    if applies_to.sex and case_facts.sex and applies_to.sex.casefold() != case_facts.sex.casefold():
        return False

    dx_date = parse_lenient_date(case_facts.date_of_diagnosis)
    if dx_date is not None:
        dx_min = parse_lenient_date(applies_to.dx_date_min)
        if dx_min is not None and dx_date < dx_min:
            return False
        dx_max = parse_lenient_date(applies_to.dx_date_max)
        if dx_max is not None and dx_date > dx_max:
            return False

    return True


def applicable_sources(
    manifest: RuleStoreManifest, dx_date: str | date | None
) -> tuple[set[str], list[ScopingReviewReason]]:
    """Temporal filter: manuals applying to cases diagnosed on/after publication.

    Unknown diagnosis date passes all sources (unknown widens), with a review
    flag when any family holds multiple editions that a date would have
    resolved by recency.
    """
    dx = parse_lenient_date(dx_date)
    if dx is None:
        family_counts: dict[str, int] = {}
        for _, source in manifest.items():
            family_counts[source.family] = family_counts.get(source.family, 0) + 1
        flags = (
            [ScopingReviewReason.UNKNOWN_DX_DATE_WIDE_SCOPE]
            if any(count > 1 for count in family_counts.values())
            else []
        )
        return set(manifest.root), flags

    passing = {
        key
        for key, source in manifest.items()
        if (published := parse_lenient_date(source.publication_date)) is None or published <= dx
    }
    return passing, []


def rules_for_items(store: RuleStore, item_ids: list[int]) -> dict[int, list[RuleUnit]]:
    """Item-indexed units plus general principles from the same manuals."""
    result: dict[int, list[RuleUnit]] = {}
    for item_id in item_ids:
        specific = list(store.units_by_item.get(item_id, ()))
        sources = {unit.source_doc for unit in specific}
        generals = [unit for unit in store.general_units if unit.source_doc in sources]
        result[item_id] = specific + generals
    return result


def resolve_precedence(units: list[RuleUnit], manifest: RuleStoreManifest) -> list[RuleUnit]:
    """Resolve overlapping guidance for one variable.

    Group units by kind; within a family keep only the most recent applicable
    edition; across families prefer SEER, keeping other-family units only for
    kinds SEER does not cover. Input order is preserved.
    """
    by_kind: dict[str, list[RuleUnit]] = {}
    for unit in units:
        by_kind.setdefault(unit.kind, []).append(unit)

    kept_ids: set[str] = set()
    for group in by_kind.values():
        by_family: dict[str, list[RuleUnit]] = {}
        for unit in group:
            by_family.setdefault(manifest[unit.source_doc].family, []).append(unit)

        survivors: list[RuleUnit] = []
        for family_units in by_family.values():
            latest = max(manifest[unit.source_doc].publication_date for unit in family_units)
            survivors.extend(
                unit for unit in family_units
                if manifest[unit.source_doc].publication_date == latest
            )

        families = {manifest[unit.source_doc].family for unit in survivors}
        if "SEER" in families and len(families) > 1:
            survivors = [unit for unit in survivors if manifest[unit.source_doc].family == "SEER"]
        kept_ids.update(unit.rule_id for unit in survivors)

    return [unit for unit in units if unit.rule_id in kept_ids]


def specificity_rank(unit: RuleUnit) -> tuple:
    """Sort key ordering units most specific first (deterministic tie-break)."""
    applies_to = unit.applies_to
    conditions = 0
    if applies_to is not None:
        conditions = sum(
            1
            for value in (
                applies_to.sites,
                applies_to.histologies,
                applies_to.behaviors,
                applies_to.sex,
                applies_to.dx_date_min,
                applies_to.dx_date_max,
            )
            if value
        )
    return (0 if unit.item_ids else 1, -conditions, unit.rule_id)


def reduce_valid_codes(
    full_codes: dict[str, str] | None, code_table_units: list[RuleUnit]
) -> tuple[dict[str, str] | None, list[ScopingReviewReason]]:
    """Intersect the data-dictionary code set with applicable code_table units.

    Never empties: an empty intersection falls back to the unreduced set
    (returned as None) and emits a review reason. With no full set available,
    the union of the applicable tables stands in as the reduced set.
    """
    if not code_table_units:
        return None, []

    allowed: dict[str, str] = {}
    for unit in code_table_units:
        allowed.update(unit.codes or {})

    if not full_codes:
        return allowed, []

    # Manuals cite histologies as 'xxxx/x' morphology/behavior pairs while the
    # stored field (and hence the dictionary set after lookup) uses the 4-digit
    # base, so membership is checked on the behavior-stripped form as well.
    allowed_bases = {code.partition("/")[0] for code in allowed}
    reduced = {
        code: description
        for code, description in full_codes.items()
        if code in allowed or code.partition("/")[0] in allowed_bases
    }
    if not reduced:
        return None, [ScopingReviewReason.EMPTY_CODE_REDUCTION_FALLBACK]
    return reduced, []


def _dedupe_reasons(reasons: list[ScopingReviewReason]) -> list[ScopingReviewReason]:
    seen: list[ScopingReviewReason] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return seen


def _has_same_family_overlap(units: list[RuleUnit], manifest: RuleStoreManifest) -> bool:
    """True when any (kind, family) group spans multiple manuals — an overlap
    that a known diagnosis date would have resolved by recency."""
    groups: dict[tuple[str, str], set[str]] = {}
    for unit in units:
        key = (unit.kind, manifest[unit.source_doc].family)
        groups.setdefault(key, set()).add(unit.source_doc)
    return any(len(sources) > 1 for sources in groups.values())


def scope_coding_context(
    item_ids: list[int],
    case_facts: CaseFacts,
    store: RuleStore,
    full_codes_by_item: dict[int, dict[str, str]] | None = None,
) -> dict[int, ScopedVariableContext]:
    """Scope the rule store to the requested variables and known case facts.

    Pure over (item_ids, case_facts, store contents). Applies the temporal
    filter, case-applicability matching, precedence resolution, specificity
    ordering, and valid-code reduction; returns one ScopedVariableContext per
    requested item ID.
    """
    dx_unknown = parse_lenient_date(case_facts.date_of_diagnosis) is None
    passing_sources, _ = applicable_sources(store.manifest, case_facts.date_of_diagnosis)

    contexts: dict[int, ScopedVariableContext] = {}
    for item_id, candidates in rules_for_items(store, item_ids).items():
        reasons: list[ScopingReviewReason] = []

        matched: list[RuleUnit] = []
        for unit in candidates:
            applies_to = unit.applies_to
            has_own_dates = applies_to is not None and (applies_to.dx_date_min or applies_to.dx_date_max)
            if not has_own_dates and unit.source_doc not in passing_sources:
                continue
            if not matches_case(applies_to, case_facts):
                continue
            matched.append(unit)

        if dx_unknown and _has_same_family_overlap(matched, store.manifest):
            reasons.append(ScopingReviewReason.UNKNOWN_DX_DATE_WIDE_SCOPE)

        selected = resolve_precedence(matched, store.manifest)
        selected.sort(key=specificity_rank)

        code_tables = [unit for unit in selected if unit.kind == "code_table"]
        full_codes = (full_codes_by_item or {}).get(item_id)
        reduced_codes, code_reasons = reduce_valid_codes(full_codes, code_tables)
        reasons.extend(code_reasons)

        contexts[item_id] = ScopedVariableContext(
            item_id=item_id,
            units=selected,
            reduced_codes=reduced_codes,
            review_reasons=_dedupe_reasons(reasons),
        )
    return contexts


def unit_citation(unit: RuleUnit, manifest: RuleStoreManifest) -> str:
    """Source citation for one unit, e.g. '[Solid Tumor Rules – Breast, H4]'."""
    title = manifest[unit.source_doc].title
    tail = unit.rule_id.split(":")[-1].upper().replace("_", " ")
    section = unit.section_path[0] if unit.section_path else None
    if section and section.casefold() not in title.casefold():
        return f"[{title} – {section}, {tail}]"
    return f"[{title}, {tail}]"


def assemble_coding_instructions(
    context: ScopedVariableContext,
    manifest: RuleStoreManifest,
    *,
    max_chars: int = 4000,
) -> tuple[str | None, list[ScopingReviewReason]]:
    """Render a scoped context into the prompt instruction block.

    Units render most specific first, each prefixed with its source citation.
    code_table units are excluded (their content flows through the reduced
    valid-code set) and example units are excluded by default. The size cap
    drops the least-specific units first and flags truncation; the most
    specific unit is always kept.
    """
    lines: list[str] = []
    seen_texts: set[str] = set()
    for unit in context.units:
        if unit.kind in ("code_table", "example"):
            continue
        text = " ".join(unit.text.split())
        if text in seen_texts:
            continue
        seen_texts.add(text)
        lines.append(f"- {unit_citation(unit, manifest)} {text}")

    if not lines:
        return None, []

    truncated = False
    while len(lines) > 1 and sum(len(line) + 1 for line in lines) - 1 > max_chars:
        lines.pop()
        truncated = True

    reasons = [ScopingReviewReason.INSTRUCTION_CONTEXT_TRUNCATED] if truncated else []
    return "\n".join(lines), reasons
