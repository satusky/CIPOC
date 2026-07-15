"""Deterministic validation of compiled rule units.

Pure code, no LLM. Every unit the tagging pass emits must pass these checks
before it is promoted into ``documents/rules/``; failures are quarantined with a
reason and surfaced in the review report. The checks mirror the plan's §5:

- schema validity (guaranteed upstream by ``RuleUnit`` construction);
- provenance: the anchor resolves into the source markdown and the unit text
  fuzzy-matches the anchored region (the LLM classifies, it must not paraphrase);
- code-table codes exist in the data dictionary set for the unit's item_ids;
- applicability site/histology entries are well-formed ICD-O-3 codes or ranges;
- item_ids exist in the data dictionary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from cipoc.models import RuleUnit
from cipoc.tools.coding_context import normalize_histology, normalize_site

_ANCHOR = re.compile(r"^L(\d+)-L(\d+):")

# Fraction of the unit's text that must be covered by matching blocks in the
# anchored source region for the fidelity check to pass. Near-verbatim excerpts
# score ~1.0; paraphrases fall well below.
FIDELITY_THRESHOLD = 0.85


@dataclass
class UnitValidation:
    """Outcome of validating one rule unit."""
    rule_id: str
    errors: list[str] = field(default_factory=list)
    fidelity: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalize_ws(text: str) -> str:
    return " ".join(text.split()).casefold()


def fidelity_score(unit_text: str, source_region: str) -> float:
    """Fraction of the unit text covered by matching blocks in the source region.

    1.0 means the unit text is a verbatim contiguous excerpt; near-verbatim
    excerpts (whitespace/emphasis differences) stay close to 1.0; paraphrases
    drop off sharply.
    """
    unit_norm = _normalize_ws(unit_text)
    if not unit_norm:
        return 0.0
    matcher = SequenceMatcher(None, unit_norm, _normalize_ws(source_region), autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(unit_norm)


_MORPHOLOGY_PAIR = re.compile(r"\d{4}/\d")
_BARE_MORPHOLOGY = re.compile(r"\d{4}")


def _code_in_item_set(code: str, valid: dict) -> bool:
    """Behavior-aware membership of a manual code in a dictionary code set.

    Manuals cite histologies as 'xxxx/x' morphology/behavior pairs; the
    dictionary enumerates item 522 the same way but the stored field holds only
    the 4-digit base, and item 523 holds only the single behavior digit. Match
    whichever component the item's set actually encodes; a bare morphology
    checked against a single-digit behavior set carries no information for that
    item and cannot fail it.
    """
    if code in valid:
        return True
    base, sep, behavior = code.partition("/")
    single_digit_set = all(len(key) == 1 for key in valid)
    if sep and _MORPHOLOGY_PAIR.fullmatch(code):
        if single_digit_set:
            return behavior in valid
        return any(key.partition("/")[0] == base for key in valid)
    if _BARE_MORPHOLOGY.fullmatch(code):
        if single_digit_set:
            return True
        return any(key.partition("/")[0] == code for key in valid)
    return False


def _well_formed_code_or_range(entry: str, normalize) -> bool:
    low, sep, high = entry.partition("-")
    if normalize(low) is None:
        return False
    if sep and normalize(high) is None:
        return False
    return True


def validate_unit(
    unit: RuleUnit,
    *,
    source_lines: list[str],
    data_dictionary: dict,
) -> UnitValidation:
    """Run every deterministic check on one unit and collect failures."""
    result = UnitValidation(rule_id=unit.rule_id)

    # --- Provenance: anchor resolves, text fuzzy-matches the region ---
    match = _ANCHOR.match(unit.anchor or "")
    if not match:
        result.errors.append(f"Anchor {unit.anchor!r} is not in 'L<start>-L<end>:slug' form.")
    else:
        start, end = int(match.group(1)), int(match.group(2))
        if not (1 <= start <= end <= len(source_lines)):
            result.errors.append(f"Anchor line range L{start}-L{end} is out of bounds.")
        else:
            region = "\n".join(source_lines[start - 1 : end])
            result.fidelity = fidelity_score(unit.text, region)
            if result.fidelity < FIDELITY_THRESHOLD:
                result.errors.append(
                    f"Unit text fidelity {result.fidelity:.2f} below {FIDELITY_THRESHOLD:.2f}; "
                    "text may be paraphrased rather than excerpted."
                )

    if not unit.section_path:
        result.errors.append("Empty section_path; provenance trail missing.")

    # --- item_ids exist in the data dictionary ---
    for item_id in unit.item_ids:
        if str(item_id) not in data_dictionary:
            result.errors.append(f"item_id {item_id} not found in the data dictionary.")

    # --- code_table codes exist in the item's enumerated set ---
    if unit.kind == "code_table":
        if not unit.codes:
            result.errors.append("code_table unit carries no codes.")
        for item_id in unit.item_ids:
            entry = data_dictionary.get(str(item_id), {})
            valid = entry.get("Code Descriptions")
            if not isinstance(valid, dict) or not valid:
                continue  # item has no enumerated set to check against
            missing = [code for code in (unit.codes or {}) if not _code_in_item_set(code, valid)]
            if missing:
                result.errors.append(
                    f"code_table codes not in item {item_id} set: {missing[:10]}"
                )

    # --- applicability site/histology entries are well-formed ---
    applies_to = unit.applies_to
    if applies_to is not None:
        for site in applies_to.sites or []:
            if not _well_formed_code_or_range(site, normalize_site):
                result.errors.append(f"Malformed site code or range: {site!r}.")
        for histology in applies_to.histologies or []:
            if not _well_formed_code_or_range(histology, normalize_histology):
                result.errors.append(f"Malformed histology code or range: {histology!r}.")

    return result


def validate_units(
    units: list[RuleUnit],
    *,
    source_markdown_path: str | Path,
    data_dictionary_path: str | Path,
) -> list[UnitValidation]:
    """Validate a batch of units against their source and the data dictionary."""
    source_lines = Path(source_markdown_path).read_text().splitlines()
    data_dictionary = json.loads(Path(data_dictionary_path).read_text())

    results = [
        validate_unit(unit, source_lines=source_lines, data_dictionary=data_dictionary)
        for unit in units
    ]

    rule_ids = [unit.rule_id for unit in units]
    duplicates = {rid for rid in rule_ids if rule_ids.count(rid) > 1}
    for result in results:
        if result.rule_id in duplicates:
            result.errors.append("Duplicate rule_id within this batch.")
    return results
