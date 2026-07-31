"""Deterministic validation of SEER*RSA staging units.

Reuses ``validate.validate_unit`` rather than forking its checks: item ids must
exist in the data dictionary, code_table codes must be members of the item's
enumerated set (behavior-aware), a code_table must not target a date item, and
applicability sites/histologies must be well-formed ICD-O-3 codes or ranges.
Those are exactly as load-bearing here as for an LLM compile.

What is dropped is the anchor+fidelity provenance check, via
``require_provenance=False``. It is markdown-specific in both halves: the anchor
regex expects ``L<start>-L<end>:slug`` into a source file, and the fidelity score
exists to catch a model that paraphrased instead of excerpting. This ingest has
neither a markdown source nor a model — ``staging_units`` copies strings out of
one named ZIP member, which is what the anchor records.
"""

from __future__ import annotations

import json
from pathlib import Path

from cipoc.models import RuleUnit

from .validate import UnitValidation, validate_unit


def validate_staging_units(
    units: list[RuleUnit], *, data_dictionary_path: str | Path
) -> list[UnitValidation]:
    """Validate a batch of ingested units against the data dictionary."""
    data_dictionary = json.loads(Path(data_dictionary_path).read_text())

    results = [
        validate_unit(
            unit,
            source_lines=[],
            data_dictionary=data_dictionary,
            require_provenance=False,
        )
        for unit in units
    ]

    rule_ids = [unit.rule_id for unit in units]
    duplicates = {rid for rid in rule_ids if rule_ids.count(rid) > 1}
    for result in results:
        if result.rule_id in duplicates:
            result.errors.append("Duplicate rule_id within this batch.")
    return results
