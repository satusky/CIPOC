"""LLM metadata assignment for the rule-compilation pipeline.

This is the one LLM-backed step, run offline. For each deterministically
segmented ``Section`` it asks the model to split the section into normative
coding-guidance units and classify each — ``kind``, ``item_ids``,
``applies_to``, and any ``codes`` table — while keeping ``text`` a near-verbatim
excerpt of the source. Provenance fields (``rule_id``, ``source_doc``,
``section_path``, ``anchor``) are assigned deterministically here, never by the
model, so the downstream fidelity check has a fixed region to match against.

The model output is intentionally a *subset* of ``RuleUnit``; merging in
provenance yields the full unit.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from cipoc.llm import BaseAgentModel
from cipoc.models import RuleApplicability, RuleKind, RuleUnit

from .segment import Section, slugify


class TaggedUnit(BaseModel):
    """The classification fields the model assigns to one normative unit.

    Excludes all provenance fields — those are merged in deterministically.
    """
    kind: RuleKind = Field(description="Kind of guidance this unit carries.")
    item_ids: list[int] = Field(
        default_factory=list,
        description="NAACCR item IDs this unit governs (e.g. 522 histology, 523 behavior, "
        "400 primary site). Empty list ONLY for a general principle that applies to every "
        "item in the manual.",
    )
    applies_to: RuleApplicability | None = Field(
        default=None,
        description="Case applicability narrower than the site-group default (e.g. a "
        "specific histology the rule is about). Null when the unit applies to the whole "
        "site group.",
    )
    text: str = Field(
        description="The rule's instruction text, copied NEAR-VERBATIM from the section. Do "
        "not paraphrase, summarize, or reword; copy the exact sentence(s). Strip only "
        "markdown emphasis markers.",
    )
    codes: dict[str, str] | None = Field(
        default=None,
        description="For a code_table unit only: a code-to-description mapping (e.g. "
        "{'8500/3': 'carcinoma NST'}). Null otherwise.",
    )
    notes: str | None = Field(
        default=None, description="Optional compiler note; never injected into prompts."
    )


class SectionTagging(BaseModel):
    """The model's tagging of one section: zero or more normative units.

    An empty list means the section is non-normative (foreword, changelog,
    rationale prose) and contributes no rule units.
    """
    units: list[TaggedUnit] = Field(
        default_factory=list,
        description="Normative coding-guidance units found in this section, in document "
        "order. Empty when the section carries no coding instructions.",
    )


TAGGING_SYSTEM_PROMPT = """\
You are compiling a cancer-registry coding manual into a structured rule store. \
You are given one section of the manual. Split it into individual normative coding-guidance \
units and classify each one. A "unit" is a single self-contained instruction, definition, \
priority rule, code table, or worked example.

Rules:
- Copy each unit's `text` NEAR-VERBATIM from the section. Classify and tag; never paraphrase.
- Assign `kind`: "instruction" (how to code), "priority_rule" (a hierarchical/ordered rule, \
often numbered like "Rule H12"), "definition" (a term definition), "code_table" (an \
enumerated code-to-meaning mapping), or "example" (a worked example illustrating a rule).
- Assign `item_ids`: the NAACCR data items the unit governs. Use an empty list ONLY for a \
general principle that governs every item in the manual.
- Set `applies_to` only to NARROW below the section's default applicability (e.g. a specific \
histology). Leave it null otherwise.
- For a `code_table`, fill `codes` with the exact codes and their descriptions.
- If the section is non-normative (foreword, "New for 2024" changelog, navigation, \
rationale-only prose), return an empty `units` list.
"""


def _tagging_user_prompt(section: Section, default_applicability: RuleApplicability | None) -> str:
    parts = [
        f"Section heading trail: {' > '.join(section.section_path)}",
    ]
    if default_applicability is not None:
        applies = default_applicability.model_dump(exclude_none=True)
        parts.append(f"Section-group default applicability (already assumed): {applies}")
    parts.append("\nSection text:\n" + section.text)
    return "\n".join(parts)


def _merge_applicability(
    tagged: RuleApplicability | None, default: RuleApplicability | None
) -> RuleApplicability | None:
    """Overlay a unit's narrowing applicability onto the site-group default."""
    if default is None:
        return tagged
    if tagged is None:
        return default.model_copy(deep=True)
    merged = default.model_dump()
    merged.update(tagged.model_dump(exclude_none=True))
    return RuleApplicability(**merged)


def _kind_tag(kind: RuleKind, index: int) -> str:
    """Short rule_id suffix, e.g. 'h1' for the first priority rule in a section."""
    letter = {"priority_rule": "p", "instruction": "i", "definition": "d",
              "code_table": "t", "example": "e"}.get(kind, "u")
    return f"{letter}{index + 1}"


def tag_section(
    section: Section,
    llm: BaseAgentModel,
    *,
    source_doc: str,
    site_group: str,
    default_applicability: RuleApplicability | None = None,
) -> list[RuleUnit]:
    """Tag one section into zero or more fully-provenanced ``RuleUnit`` objects."""
    from langchain.messages import HumanMessage, SystemMessage

    tagging: SectionTagging = llm.model.with_structured_output(SectionTagging).invoke(
        [
            SystemMessage(TAGGING_SYSTEM_PROMPT),
            HumanMessage(_tagging_user_prompt(section, default_applicability)),
        ]
    )

    section_slug = slugify(section.heading, max_words=4).replace("-", "_") or "sec"
    units: list[RuleUnit] = []
    for index, tagged in enumerate(tagging.units):
        units.append(
            RuleUnit(
                rule_id=f"{source_doc}:{site_group}:{section_slug}:{_kind_tag(tagged.kind, index)}",
                source_doc=source_doc,
                section_path=list(section.section_path),
                anchor=section.anchor,
                kind=tagged.kind,
                item_ids=tagged.item_ids,
                applies_to=_merge_applicability(tagged.applies_to, default_applicability),
                text=re.sub(r"\*+", "", tagged.text).strip(),
                codes=tagged.codes,
                notes=tagged.notes,
            )
        )
    return units
