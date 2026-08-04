"""Inspect the compiled coding rules applied to a case and its variables."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from cipoc.models import CaseFacts, RuleUnit, ScopedVariableContext, VariableGroupInfo
from cipoc.tools import build_variable_group, load_rule_store, scope_coding_context
from cipoc.tools.coding_context import RuleStore


DEFAULT_RULES_DIR = Path("documents/rules")
DEFAULT_DATA_DICTIONARY = Path("documents/manuals/naaccr_data_dictionary_v25.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show the final compiled coding rules and scoped variable metadata "
            "for one or more NAACCR item IDs."
        )
    )
    parser.add_argument(
        "item_ids",
        nargs="+",
        type=int,
        metavar="ITEM_ID",
        help="NAACCR variable item ID (one or more)",
    )
    parser.add_argument("--primary-site", help="ICD-O-3 primary-site code")
    parser.add_argument("--gross-primary-site", help="Tissue-level primary site")
    parser.add_argument("--histology", help="ICD-O-3 histology code")
    parser.add_argument("--behavior", help="ICD-O-3 behavior code")
    parser.add_argument("--sex", help="Patient sex")
    parser.add_argument("--date-of-diagnosis", help="Diagnosis date")
    parser.add_argument(
        "--rules-dir",
        type=Path,
        default=DEFAULT_RULES_DIR,
        help=f"Compiled rule-store directory (default: {DEFAULT_RULES_DIR})",
    )
    parser.add_argument(
        "--data-dictionary",
        type=Path,
        default=DEFAULT_DATA_DICTIONARY,
        help=f"NAACCR data-dictionary JSON (default: {DEFAULT_DATA_DICTIONARY})",
    )
    return parser


def case_facts_from_args(args: argparse.Namespace) -> CaseFacts:
    return CaseFacts(
        primary_site=args.primary_site,
        gross_primary_site=args.gross_primary_site,
        histology=args.histology,
        behavior=args.behavior,
        sex=args.sex,
        date_of_diagnosis=args.date_of_diagnosis,
    )


def _render_rule(unit: RuleUnit, store: RuleStore) -> list[str]:
    source = store.manifest[unit.source_doc]
    compiled_file = store.source_files_by_rule_id[unit.rule_id]
    lines = [
        f"  Rule: {unit.rule_id} ({unit.kind})",
        f"    Text: {unit.text}",
    ]
    if unit.kind == "code_table":
        lines.append("    Code table:")
        lines.extend(
            f"      {code}: {description}"
            for code, description in (unit.codes or {}).items()
        )
    if unit.applies_to is not None:
        predicate = json.dumps(
            unit.applies_to.model_dump(exclude_none=True), sort_keys=True
        )
        lines.append(f"    Applicability: {predicate}")
    section_path = " > ".join(unit.section_path) or "(none)"
    lines.extend(
        [
            f"    Section path: {section_path}",
            f"    Compiled JSON: {compiled_file} (rule_id: {unit.rule_id})",
            (
                f"    Original manual: {source.title}; "
                f"source_markdown: {source.source_markdown or '(not recorded)'}; "
                f"anchor: {unit.anchor or '(none)'}"
            ),
        ]
    )
    return lines


def render_report(
    item_ids: list[int],
    case_facts: CaseFacts,
    contexts: dict[int, ScopedVariableContext],
    variable_group: VariableGroupInfo,
    store: RuleStore,
) -> str:
    lines = [
        "Requested item IDs: " + ", ".join(str(item_id) for item_id in item_ids),
        "Supplied case facts:",
    ]
    supplied_facts = case_facts.model_dump(exclude_none=True)
    lines.append(json.dumps(supplied_facts, indent=2) if supplied_facts else "(none)")
    lines.extend(["", "Applied rules by item ID:"])

    for item_id in item_ids:
        lines.append(f"Item {item_id}:")
        units = contexts[item_id].units
        if not units:
            lines.append("  (none)")
            continue
        for unit in units:
            lines.extend(_render_rule(unit, store))

    reasons_by_item = {
        item_id: contexts[item_id].review_reasons
        for item_id in item_ids
        if contexts[item_id].review_reasons
    }
    if reasons_by_item:
        lines.extend(["", "Scoping review reasons:"])
        for item_id, reasons in reasons_by_item.items():
            lines.append(
                f"  Item {item_id}: " + ", ".join(reason.value for reason in reasons)
            )

    lines.extend(
        [
            "",
            "VariableGroupInfo:",
            variable_group.model_dump_json(indent=2),
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest_path = args.rules_dir / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"rule manifest does not exist: {manifest_path}")
    if not args.data_dictionary.is_file():
        parser.error(f"data dictionary does not exist: {args.data_dictionary}")

    try:
        data_dictionary = json.loads(args.data_dictionary.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"could not read data dictionary {args.data_dictionary}: {exc}")
    if not isinstance(data_dictionary, dict):
        parser.error(f"data dictionary must be a JSON object: {args.data_dictionary}")

    item_ids = list(dict.fromkeys(args.item_ids))
    missing_item_ids = [
        item_id for item_id in item_ids if str(item_id) not in data_dictionary
    ]
    if missing_item_ids:
        parser.error(
            "item IDs absent from the data dictionary: "
            + ", ".join(str(item_id) for item_id in missing_item_ids)
        )

    case_facts = case_facts_from_args(args)
    try:
        store = load_rule_store(args.rules_dir)
        unscoped_group = build_variable_group(item_ids, args.data_dictionary)
        full_codes_by_item = {
            variable.item_id: variable.valid_codes
            for variable in unscoped_group.variables
            if isinstance(variable.valid_codes, dict) and variable.valid_codes
        }
        contexts = scope_coding_context(
            item_ids,
            case_facts,
            store,
            full_codes_by_item=full_codes_by_item,
        )
        variable_group = build_variable_group(
            item_ids,
            args.data_dictionary,
            case_facts=case_facts,
            rule_store=store,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(render_report(item_ids, case_facts, contexts, variable_group, store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
