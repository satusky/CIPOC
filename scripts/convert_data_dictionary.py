"""Convert flat or tissue-keyed NAACCR dictionaries to the snake_case schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_CODE_COLUMNS = ("code", "Code", "SS2018", "LVI")
_DESCRIPTION_COLUMNS = ("description", "Description", "Tumor Size Description")

_FIELD_MAP = {
    "item_number": "Data Item Number",
    "item_name": "Data Item Name",
    "item_data_type": "Data Type",
    "item_length": "Length",
    "year_implemented": "Year Implemented",
    "version_implemented": "Version Implemented",
    "year_retired": "Year Retired",
    "version_retired": "Version Retired",
    "xml_naaccr_id": "XML NAACCR ID",
    "xml_parent_id": "Parent XML Element",
    "record_types": "Record Type",
    "section": "Section Name",
    "source_of_standard": "Source of Standard",
    "description": "Description",
    "rationale": "Rationale",
    "clarification": "Clarification",
    "general_notes": "General Notes",
    "instructions_for_coding": "Instructions for Coding",
    "npcr_collect": "NPCR Collect",
    "coc_collect": "CoC Collect",
    "seer_collect": "SEER Collect",
    "cccr_collect": "CCCR Collect",
    "alternate_names": "Alternate Name",
    "format": "Format",
    "code_description": "Data Dictionary Code Note",
    "code_note": "Code Notes",
    "allowable_values": "Allowable Values",
    "record_layout_table_note": "Record Layout Table Note",
    "required_status_table_note": "Required Status Table Note",
    "data_descriptor_table_note": "Data Descriptor Table Note",
    "record_layout_note": "Record Layout Note",
    "required_status_note": "Required Status Note",
    "data_descriptor_note": "Data Descriptor Note",
    "data_dictionary_description_note": "Data Dictionary Description Note",
    "data_dictionary_rationale_note": "Data Dictionary Rationale Note",
}

_TEXT_FIELDS = {
    "description",
    "rationale",
    "clarification",
    "general_notes",
    "instructions_for_coding",
    "code_description",
    "code_note",
    "record_layout_table_note",
    "required_status_table_note",
    "data_descriptor_table_note",
    "record_layout_note",
    "required_status_note",
    "data_descriptor_note",
    "data_dictionary_description_note",
    "data_dictionary_rationale_note",
}


def _normalize_text(value: Any) -> Any:
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return value


def _allowed_codes(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [
            {"code": str(code), "description": str(description)}
            for code, description in value.items()
        ]
    if not isinstance(value, list):
        raise ValueError("Code Descriptions must be an object or a list of rows")

    codes: list[dict[str, str]] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("Code Descriptions rows must be objects")
        code_column = next((column for column in _CODE_COLUMNS if column in row), None)
        description_column = next(
            (column for column in _DESCRIPTION_COLUMNS if column in row), None
        )
        if code_column is None or description_column is None:
            raise ValueError(
                "Code Descriptions row has no supported code/description columns"
            )
        codes.append(
            {
                "code": str(row[code_column]),
                "description": str(row[description_column]),
            }
        )
    return codes


def _value(entry: dict[str, Any], target: str, source: str) -> Any:
    if target in entry:
        return entry[target]
    return entry.get(source)


def convert_item(entry: dict[str, Any], *, naaccr_version: str) -> dict[str, Any]:
    if _value(entry, "item_number", _FIELD_MAP["item_number"]) is None:
        converted = {
            target: _value(entry, target, source)
            for target, source in _FIELD_MAP.items()
            if target in entry or source in entry
        }
        for field in _TEXT_FIELDS & converted.keys():
            converted[field] = _normalize_text(converted[field])
        for target, source in (
            ("date_created", "Date Created"),
            ("date_modified", "Date Modified"),
            ("retired", "Retired"),
        ):
            if target in entry or source in entry:
                converted[target] = _value(entry, target, source)
        if "allowed_codes" in entry or "Code Descriptions" in entry:
            converted["allowed_codes"] = _allowed_codes(
                entry.get("allowed_codes", entry.get("Code Descriptions"))
            )
        return converted

    converted = {
        target: _value(entry, target, source)
        for target, source in _FIELD_MAP.items()
    }
    for field in _TEXT_FIELDS:
        converted[field] = _normalize_text(converted[field])

    year_retired = converted["year_retired"]
    version_retired = converted["version_retired"]
    codes = entry.get("allowed_codes", entry.get("Code Descriptions"))
    return {
        "naaccr_version": str(entry.get("naaccr_version", naaccr_version)),
        **dict(list(converted.items())[:13]),
        "date_created": entry.get("date_created", entry.get("Date Created")),
        "date_modified": entry.get("date_modified", entry.get("Date Modified")),
        **dict(list(converted.items())[13:]),
        "retired": entry.get(
            "retired", "Yes" if year_retired or version_retired else "No"
        ),
        "allowed_codes": _allowed_codes(codes),
    }


def convert_dictionary(
    source: dict[str, Any], *, naaccr_version: str = "24"
) -> dict[str, Any]:
    if not source:
        return {}

    first_value = next(iter(source.values()))
    flat = isinstance(first_value, dict) and any(
        field in first_value for field in ("Data Item Number", "item_number")
    )
    groups = {None: source} if flat else source
    converted: dict[str, Any] = {} if flat else {str(group): {} for group in groups}

    for group, items in groups.items():
        if not isinstance(items, dict):
            raise ValueError(f"Group {group!r} must contain an object of items")
        destination = converted if flat else converted[str(group)]
        for item_id, entry in items.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Item {item_id!r} in group {group!r} must be an object")
            converted_item = convert_item(entry, naaccr_version=naaccr_version)
            if converted_item.get("item_number") is not None and str(
                converted_item["item_number"]
            ) != str(item_id):
                raise ValueError(
                    f"Item key {item_id!r} does not match item_number "
                    f"{converted_item['item_number']!r} in group {group!r}"
                )
            destination[str(item_id)] = converted_item
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a flat or tissue-keyed data dictionary to snake_case."
    )
    parser.add_argument("input", type=Path, help="Source CIPOC dictionary JSON")
    parser.add_argument("output", type=Path, help="Converted dictionary JSON")
    parser.add_argument(
        "--naaccr-version",
        default="24",
        help="Value written to each naaccr_version field (default: 24)",
    )
    args = parser.parse_args()

    with open(args.input, "r") as source_file:
        source = json.load(source_file)
    if not isinstance(source, dict):
        raise ValueError("The source dictionary must be a JSON object")

    converted = convert_dictionary(source, naaccr_version=args.naaccr_version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as output_file:
        json.dump(converted, output_file, indent=2)
        output_file.write("\n")

    first_value = next(iter(converted.values()), None)
    grouped = isinstance(first_value, dict) and "item_number" not in first_value
    item_count = sum(len(items) for items in converted.values()) if grouped else len(converted)
    scope = f" across {len(converted)} groups" if grouped else ""
    print(f"Converted {item_count} entries{scope} to {args.output}")


if __name__ == "__main__":
    main()
