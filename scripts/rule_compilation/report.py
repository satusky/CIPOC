"""Human spot-check report for a rule-compilation run.

Renders a plain-text report the compiler operator reviews before promoting a
manual's output into ``documents/rules/``: counts by kind, every quarantined
unit with its failure reasons, and a sample of accepted units shown
side-by-side with their anchored source region for a fidelity eyeball.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

from cipoc.models import RuleUnit

from .validate import UnitValidation


def _anchor_region(unit: RuleUnit, source_lines: list[str]) -> str:
    import re

    match = re.match(r"^L(\d+)-L(\d+):", unit.anchor or "")
    if not match:
        return "(anchor unresolved)"
    start, end = int(match.group(1)), int(match.group(2))
    return "\n".join(source_lines[start - 1 : end])


def build_report(
    units: list[RuleUnit],
    results: list[UnitValidation],
    *,
    source_markdown_path: str | Path,
    sample_size: int = 5,
) -> str:
    """Assemble the review report string for a compilation run."""
    source_lines = Path(source_markdown_path).read_text().splitlines()
    by_id = {u.rule_id: u for u in units}
    accepted = [r for r in results if r.ok]
    quarantined = [r for r in results if not r.ok]

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("RULE COMPILATION REVIEW REPORT")
    lines.append("=" * 72)
    lines.append(f"Source: {source_markdown_path}")
    lines.append(f"Units emitted: {len(units)}  |  accepted: {len(accepted)}  "
                 f"|  quarantined: {len(quarantined)}")

    kind_counts = Counter(by_id[r.rule_id].kind for r in accepted if r.rule_id in by_id)
    lines.append("\nAccepted units by kind:")
    for kind, count in sorted(kind_counts.items()):
        lines.append(f"  {kind:14s} {count}")

    fidelities = [r.fidelity for r in accepted if r.fidelity is not None]
    if fidelities:
        lines.append(f"\nAccepted fidelity: min {min(fidelities):.2f}  "
                     f"mean {sum(fidelities) / len(fidelities):.2f}")

    lines.append("\n" + "-" * 72)
    lines.append(f"QUARANTINED ({len(quarantined)})")
    lines.append("-" * 72)
    if not quarantined:
        lines.append("(none)")
    for result in quarantined:
        lines.append(f"\n[{result.rule_id}]")
        for error in result.errors:
            lines.append(f"  - {error}")

    lines.append("\n" + "-" * 72)
    lines.append(f"SAMPLED ACCEPTED UNITS (source vs. compiled, up to {sample_size})")
    lines.append("-" * 72)
    step = max(1, len(accepted) // sample_size) if accepted else 1
    for result in accepted[::step][:sample_size]:
        unit = by_id.get(result.rule_id)
        if unit is None:
            continue
        lines.append(f"\n[{unit.rule_id}]  kind={unit.kind}  items={unit.item_ids}  "
                     f"fidelity={result.fidelity:.2f}" if result.fidelity is not None
                     else f"\n[{unit.rule_id}]  kind={unit.kind}  items={unit.item_ids}")
        lines.append(f"  applies_to: {unit.applies_to.model_dump(exclude_none=True) if unit.applies_to else None}")
        lines.append("  SOURCE REGION:")
        lines.append(textwrap.indent(_anchor_region(unit, source_lines), "    | "))
        lines.append("  COMPILED TEXT:")
        lines.append(textwrap.indent(unit.text, "    > "))

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)
