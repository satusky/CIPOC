"""Item index for the CoC STORE 2024 manual.

Pure code, no LLM. STORE is not a site-partitioned manual: it is ~200 NAACCR
data items, each documented by its own ``#`` heading followed by a fixed
``## Description`` / ``## Rationale`` / ``## Coding Instructions`` / ``## Examples``
skeleton. There is no site group to carve, so this module indexes by *item*
instead, and every unit compiled from an item's subtree is forced to that item.

Only the items this project extracts that no already-compiled manual covers are
listed in ``TARGETS`` — compiling STORE whole would mostly duplicate SPCSM and
Solid Tumor Rules guidance that already wins on precedence. See ``ABSENT`` for
targeted items STORE 2024 does not document at all.

Sections are resolved by *exact* level-1 heading, never by substring: STORE
carries near-miss headings that a substring search hits first ('AJCC TNM Clin T'
is a prefix of 'AJCC TNM Clin T Suffix'; 'Scope of Regional Lymph Node Surgery'
is a prefix of the '... at this Facility' item). Each resolution is then
cross-checked against the item number printed in the section's own metadata
block, so a heading that silently moves between editions fails loudly rather
than compiling the wrong item's rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cipoc.models import RuleApplicability

from .compile_manual import DEFAULT_DX_DATE_MIN
from .segment import Section, segment_markdown

MANUAL = "store_2024"
SOURCE = Path("documents/markdown/store-manual-2024.md")

MANIFEST_ENTRY = {
    "title": "STORE 2024 (CoC Standards for Oncology Registry Entry)",
    # STORE is the Commission on Cancer's manual, not SEER's. The family matters:
    # resolve_precedence() prefers SEER over every other family for any rule kind
    # SEER also covers, so tagging this 'SEER' would put it in direct competition
    # with SPCSM on publication_date alone. As 'CoC' it supplies the items SEER
    # does not cover -- which is exactly the gap it is being compiled for -- and
    # correctly yields to SEER anywhere the two later overlap.
    "family": "CoC",
    # Edition recency, NOT the effective date. Every item here predates the 2024
    # edition (AJCC TNM and Grade Clinical/Pathological go back to 2018, the
    # RX Hosp treatment items further still), so the manual-level temporal filter
    # in scope_coding_context() would drop all of them for any case diagnosed
    # before 2024. The real effective date rides on each unit's
    # applies_to.dx_date_min, which that function honours ahead of the
    # manual-level filter. Same trap documented for summary_stage_2018.
    "publication_date": "2024-01-01",
    "effective_note": (
        "2024 edition; each compiled item carries its own effective date on "
        "applies_to.dx_date_min (the item's NAACCR implementation year). "
        "publication_date records edition recency for precedence only."
    ),
}


@dataclass(frozen=True)
class StoreItem:
    """One NAACCR item to compile out of STORE, and the scope to compile it with.

    ``heading`` must match the item's level-1 heading exactly. ``dx_date_min`` is
    the item's NAACCR implementation date, not the manual's: it is what keeps a
    2018-implemented item applicable to a 2019 case compiled from the 2024
    edition. ``tests/test_store_compile.py`` cross-checks it against the data
    dictionary's `year_implemented`.
    """

    item_id: int
    stem: str
    heading: str
    dx_date_min: str
    group_id: str

    @property
    def applicability(self) -> RuleApplicability:
        """Compile-time default applicability: temporal only.

        These items are site-agnostic — Grade Clinical and the AJCC categories are
        coded for every primary — so sites/histologies stay unset and the model
        narrows per unit where a rule genuinely is site-specific.
        """
        return RuleApplicability(dx_date_min=self.dx_date_min)


# Targeted items with no coverage in the compiled store, in variable-group order.
# dx_date_min is the item's NAACCR implementation year; items STORE marks
# "All Years" and the dictionary leaves unset fall back to the project floor.
TARGETS: tuple[StoreItem, ...] = (
    # config/variable_groups.json: first_course_treatment
    StoreItem(700, "rx_hosp_chemo", "Chemotherapy at this Facility", DEFAULT_DX_DATE_MIN, "first_course_treatment"),
    StoreItem(710, "rx_hosp_hormone", "Hormone Therapy at this Facility (Hormone/Steroid Therapy)", DEFAULT_DX_DATE_MIN, "first_course_treatment"),
    StoreItem(720, "rx_hosp_brm", "Immunotherapy at this Facility", DEFAULT_DX_DATE_MIN, "first_course_treatment"),
    StoreItem(740, "rx_hosp_dx_stg_proc", "Surgical Diagnostic and Staging Procedure at This Facility", DEFAULT_DX_DATE_MIN, "first_course_treatment"),
    StoreItem(1280, "rx_date_dx_stg_proc", "Date of Surgical Diagnostic and Staging Procedure", DEFAULT_DX_DATE_MIN, "first_course_treatment"),
    # config/variable_groups.json: lymph_node_removal
    StoreItem(672, "rx_hosp_scope_reg_ln_sur", "Scope of Regional Lymph Node Surgery at this Facility", "1997-01-01", "lymph_node_removal"),
    StoreItem(674, "rx_hosp_surg_oth_reg_dis", "Surgical Procedure/Other Site at this Facility", "1997-01-01", "lymph_node_removal"),
    # config/variable_groups.json: site_specific_codes
    StoreItem(671, "rx_hosp_surg_prim_site_2023", "Rx Hosp-- Surg 2023", "2023-01-01", "site_specific_codes"),
    StoreItem(3843, "grade_clinical", "Grade Clinical", "2018-01-01", "site_specific_codes"),
    StoreItem(3844, "grade_pathological", "Grade Pathological", "2018-01-01", "site_specific_codes"),
    # config/variable_groups.json: tnm_staging
    StoreItem(1001, "ajcc_tnm_clin_t", "AJCC TNM Clin T", "2018-01-01", "tnm_staging"),
    StoreItem(1002, "ajcc_tnm_clin_n", "AJCC TNM Clin N", "2018-01-01", "tnm_staging"),
    StoreItem(1003, "ajcc_tnm_clin_m", "AJCC TNM Clin M", "2018-01-01", "tnm_staging"),
    StoreItem(1004, "ajcc_tnm_clin_stage_group", "AJCC TNM Clin Stage Group", "2018-01-01", "tnm_staging"),
    StoreItem(1011, "ajcc_tnm_path_t", "AJCC TNM Path T", "2018-01-01", "tnm_staging"),
    StoreItem(1012, "ajcc_tnm_path_n", "AJCC TNM Path N", "2018-01-01", "tnm_staging"),
    StoreItem(1013, "ajcc_tnm_path_m", "AJCC TNM Path M", "2018-01-01", "tnm_staging"),
    StoreItem(1014, "ajcc_tnm_path_stage_group", "AJCC TNM Path Stage Group", "2018-01-01", "tnm_staging"),
    # config/variable_groups.json: other
    StoreItem(3280, "rx_hosp_palliative_proc", "Palliative Care at this Facility (Palliative Procedure at this Facility)", "2003-01-01", "other"),
)

# Targeted items STORE 2024 does not document, and why. All three are pre-2023
# CoC items the 2024 edition retired; they need a STORE 2022 (or earlier) edition
# or the NAACCR dictionary, and cannot be recovered from this source.
ABSENT: dict[int, str] = {
    690: "RX Hosp--Radiation: retired; superseded by the phase-based radiation items.",
    676: "RX Hosp--Reg LN Removed: retired; no successor item in STORE 2024.",
    670: "RX Hosp--Surg Prim Site 03-2022: retired, superseded by 671 (Rx Hosp--Surg 2023), "
         "which STORE documents and this index compiles. 670 still governs pre-2023 cases.",
}

# The item number as STORE prints it in each item's own metadata block, either as
# a bare table cell (`| 1001 | 15 | ...`) or bracketed in prose (`[3280]`).
_ITEM_NUMBER = re.compile(r"(?<![\d.])(\d{3,4})(?![\d])")


def sections_for(item: StoreItem, sections: list[Section]) -> list[Section]:
    """Return the item's level-1 section plus its ``##`` subsections.

    Raises when the heading does not resolve to exactly one level-1 section, or
    when the resolved subtree does not print the expected item number — either
    means the source markdown moved under us and compiling would silently emit
    another item's rules.
    """
    matches = [s for s in sections if s.level == 1 and s.heading == item.heading]
    if len(matches) != 1:
        raise ValueError(
            f"Item {item.item_id}: expected exactly one level-1 section titled "
            f"{item.heading!r}, found {len(matches)}."
        )

    start = sections.index(matches[0])
    subtree = [matches[0]]
    for section in sections[start + 1 :]:
        if section.level <= 1:
            break
        subtree.append(section)

    body = "\n".join(s.text for s in subtree)
    if str(item.item_id) not in set(_ITEM_NUMBER.findall(body)):
        raise ValueError(
            f"Item {item.item_id}: section {item.heading!r} does not print item number "
            f"{item.item_id}; the heading may now belong to a different item."
        )
    return subtree


def build_index(
    source: Path = SOURCE, *, items: list[int] | None = None
) -> list[tuple[StoreItem, list[Section]]]:
    """Resolve every target (or the given subset) to its sections.

    Segmentation stops at level 2 so the ``###``/``######`` fragments STORE's PDF
    conversion leaves inside metadata blocks fold into their item's body instead
    of splitting it into noise sections.
    """
    sections = segment_markdown(source.read_text(), max_heading_level=2)
    targets = TARGETS
    if items is not None:
        wanted = set(items)
        targets = tuple(t for t in TARGETS if t.item_id in wanted)
        missing = wanted - {t.item_id for t in targets}
        if missing:
            unavailable = sorted(missing & set(ABSENT))
            detail = "".join(f"\n  {i}: {ABSENT[i]}" for i in unavailable)
            raise ValueError(f"No STORE target for item(s) {sorted(missing)}.{detail}")
    return [(target, sections_for(target, sections)) for target in targets]
