"""Re-run the deterministic validation checks over the committed rule store.

The compilers promote only units that passed ``validate_unit`` *at the time they
were compiled*. When a check is added later, units already in
``documents/rules/`` predate it and stay there. This script closes that gap: it
re-validates every committed unit and, with ``--prune``, drops the failures.

Provenance is not re-checked (``require_provenance=False``). The anchor/fidelity
pair is meaningful only against the exact source markdown revision a unit was
compiled from, which may have moved or, for ``seer_rsa_*``, never existed. What
is re-checked are the invariants that hold for a unit forever: its item ids exist
in the data dictionary, its code_table codes are members of the item's set, its
descriptions carry information, and its applicability codes are well-formed.

Pruning rather than recompiling is deliberate for a store built by an LLM: a
recompile rewrites every unit in the affected files non-deterministically, so
fixing five bad units would churn several hundred good ones. A later
``--force`` recompile picks up the new check at the source anyway.

    python -m scripts.rule_compilation.revalidate_store            # report only
    python -m scripts.rule_compilation.revalidate_store --prune    # rewrite files
    python -m scripts.rule_compilation.revalidate_store --manual store_2024
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter

from cipoc.models import RuleUnit

from .validate import validate_unit

RULES_DIR = Path("documents/rules")
DATA_DICTIONARY = Path("documents/manuals/naaccr_data_dictionary_v25.json")

_RULE_LIST_ADAPTER = TypeAdapter(list[RuleUnit])


def rule_files(rules_dir: Path, manuals: set[str] | None = None) -> list[Path]:
    """Every committed rule list, skipping the manifest and compile sidecars."""
    return sorted(
        path
        for path in rules_dir.rglob("*.json")
        if path.name != "manifest.json"
        and path.suffixes == [".json"]
        and (manuals is None or path.parent.name in manuals)
    )


def revalidate(
    rules_dir: Path, data_dictionary_path: Path, *, manuals: set[str] | None = None
) -> list[tuple[Path, RuleUnit, list[str]]]:
    """Return every committed unit that fails validation, with its reasons."""
    data_dictionary = json.loads(data_dictionary_path.read_text())
    failures: list[tuple[Path, RuleUnit, list[str]]] = []
    for path in rule_files(rules_dir, manuals):
        for unit in _RULE_LIST_ADAPTER.validate_json(path.read_text()):
            result = validate_unit(
                unit,
                source_lines=[],
                data_dictionary=data_dictionary,
                require_provenance=False,
            )
            if not result.ok:
                failures.append((path, unit, result.errors))
    return failures


def prune(rules_dir: Path, failures: list[tuple[Path, RuleUnit, list[str]]]) -> list[Path]:
    """Rewrite each affected file without its failing units."""
    drop_by_file: dict[Path, set[str]] = {}
    for path, unit, _ in failures:
        drop_by_file.setdefault(path, set()).add(unit.rule_id)

    for path, drop in drop_by_file.items():
        units = _RULE_LIST_ADAPTER.validate_json(path.read_text())
        kept = [unit for unit in units if unit.rule_id not in drop]
        path.write_bytes(_RULE_LIST_ADAPTER.dump_json(kept, indent=2) + b"\n")
    return sorted(drop_by_file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rules-dir", type=Path, default=RULES_DIR)
    parser.add_argument("--data-dictionary", type=Path, default=DATA_DICTIONARY)
    parser.add_argument("--manual", nargs="*", default=None,
                        help="Restrict to these manifest keys, e.g. store_2024.")
    parser.add_argument("--prune", action="store_true",
                        help="Rewrite the affected files without the failing units.")
    args = parser.parse_args(argv)

    if not args.data_dictionary.exists():
        raise SystemExit(f"Data dictionary {args.data_dictionary} does not exist.")

    manuals = set(args.manual) if args.manual else None
    files = rule_files(args.rules_dir, manuals)
    failures = revalidate(args.rules_dir, args.data_dictionary, manuals=manuals)

    print(f"{len(files)} rule file(s) under {args.rules_dir}"
          f"{' (' + ', '.join(sorted(manuals)) + ')' if manuals else ''}\n")
    if not failures:
        print("Every committed unit passes. Nothing to prune.")
        return 0

    by_manual = Counter(path.parent.name for path, _, _ in failures)
    print(f"{len(failures)} failing unit(s): "
          + ", ".join(f"{manual} {count}" for manual, count in sorted(by_manual.items())))
    for path, unit, errors in failures:
        print(f"\n  [{unit.rule_id}]  kind={unit.kind}  items={unit.item_ids}")
        print(f"    {path}")
        for error in errors:
            print(f"    - {error}")

    if not args.prune:
        print("\nRe-run with --prune to drop these units from the store.")
        return 1

    written = prune(args.rules_dir, failures)
    print(f"\nPruned {len(failures)} unit(s) from {len(written)} file(s):")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
