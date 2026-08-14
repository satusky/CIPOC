import json
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING
from langchain.tools import tool

from cipoc.models import VariableInfo, VariableGroupInfo, VariableOutput

if TYPE_CHECKING:
    from cipoc.models import CaseFacts


_ENTRY_FIELD_MAP = {
    "item_name": "name",
    "description": "description",
    "item_data_type": "data_type",
    "item_length": "length",
    "allowable_values": "allowable_values",
    "format": "format",
    "instructions_for_coding": "coding_instructions",
}

_CODE_COLUMN_NAMES = ("code",)
_DESCRIPTION_COLUMN_NAMES = ("description",)
_MISSING = object()

class VariableValueValidator:
    """Deterministically validate an extracted value against variable metadata."""

    def __init__(self, *, allow_unknown_date_day: bool = True) -> None:
        self.allow_unknown_date_day = allow_unknown_date_day

    def validate(self, variable: VariableInfo, candidate: VariableOutput) -> list[str]:
        errors: list[str] = []

        if candidate.item_id != variable.item_id:
            errors.append(
                f"Expected item ID {variable.item_id}, received {candidate.item_id}."
            )

        value = candidate.value
        if value is None:
            errors.append("No value was returned.")
            return errors

        if any(character.isspace() for character in value):
            errors.append("Value contains whitespace or line breaks.")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            errors.append("Value contains control characters.")

        if variable.length is not None and len(value) > variable.length:
            errors.append(
                f"Value exceeds the maximum length of {variable.length} characters."
            )

        # Date syntax takes precedence over scoped code tables. A malformed table
        # must not turn a format token such as "CCYYMMDD" into an allowable value.
        if self._is_date_variable(variable):
            errors.extend(self._validate_date(value))
        elif isinstance(variable.valid_codes, dict) and variable.valid_codes:
            if not self._matches_valid_code(variable, value):
                errors.append("Value is not one of the variable's allowable codes.")

        return errors

    def _matches_valid_code(self, variable: VariableInfo, value: str) -> bool:
        valid_codes = variable.valid_codes
        if not isinstance(valid_codes, dict):
            return False
        return value in valid_codes

    @staticmethod
    def _is_date_variable(variable: VariableInfo) -> bool:
        if variable.data_type and variable.data_type.casefold() == "date":
            return True
        return bool(
            variable.format
            and variable.format.strip().upper().startswith("YYYYMMDD")
        )

    def _validate_date(self, value: str) -> list[str]:
        if re.fullmatch(r"[0-9]{8}", value) is None:
            return ["Date must contain exactly eight ASCII digits in YYYYMMDD form."]

        year = int(value[:4])
        month = int(value[4:6])
        day = int(value[6:8])

        if year == 0:
            return ["Date year must be between 0001 and 9999."]
        if not 1 <= month <= 12:
            return ["Date month must be between 01 and 12."]
        if day == 0 and self.allow_unknown_date_day:
            return []

        try:
            date(year, month, day)
        except ValueError:
            return ["Date does not represent a valid calendar date."]
        return []


_MORPHOLOGY_KEY = re.compile(r"(\d{4})/(\d)")


def _collapse_morphology_valid_codes(codes: dict, length) -> dict:
    """Collapse ICD-O-3 'xxxx/x' morphology/behavior keys to the stored 4-digit base.

    The data dictionary enumerates histology codes with their behavior suffix
    (e.g. '8500/2', '8500/3'), but the NAACCR field stores only the 4-digit
    morphology (behavior is its own item). Left as-is, every key exceeds the
    field length and value validation can never pass. Only applies when the
    field length is 4 and every key is in 'xxxx/x' form; per-behavior
    descriptions are merged so the distinction stays visible to the model.
    """
    if str(length) != "4" or not codes or not all(_MORPHOLOGY_KEY.fullmatch(k) for k in codes):
        return codes
    collapsed: dict[str, str] = {}
    for key, description in codes.items():
        base = key.partition("/")[0]
        entry = f"{key} {description}"
        collapsed[base] = f"{collapsed[base]}; {entry}" if base in collapsed else entry
    return collapsed


def _normalize_code_descriptions(codes):
    """Convert row-oriented site tables to the validator's code dictionary."""
    if not isinstance(codes, list):
        return codes

    normalized: dict[str, str] = {}
    for row in codes:
        if not isinstance(row, dict):
            raise ValueError("allowed_codes rows must be JSON objects.")
        code_column = next((key for key in _CODE_COLUMN_NAMES if key in row), None)
        description_column = next(
            (key for key in _DESCRIPTION_COLUMN_NAMES if key in row), None
        )
        if code_column is None or description_column is None:
            raise ValueError(
                "allowed_codes rows must contain code and description fields."
            )
        normalized[str(row[code_column])] = str(row[description_column])
    return normalized


def _normalize_text(value):
    return "".join(str(part) for part in value) if isinstance(value, list) else value


def _entry_codes(entry: dict | None):
    return entry["allowed_codes"] if entry and "allowed_codes" in entry else _MISSING


def _overlay_site_codes(item_entry: dict | None, site_entry: dict | None) -> dict | None:
    site_codes = _entry_codes(site_entry)
    if site_codes is _MISSING:
        return item_entry
    merged = dict(item_entry or {})
    merged["allowed_codes"] = site_codes
    return merged


def resolve_site_key(case_facts: "CaseFacts | None", site_dictionary: dict) -> str | None:
    """Resolve case facts to a top-level key present in the site dictionary."""
    if case_facts is None:
        return None

    if case_facts.gross_primary_site:
        gross_site = " ".join(
            re.sub(r"[^a-z0-9]+", " ", case_facts.gross_primary_site.casefold()).split()
        )
        matches = [
            key
            for key in site_dictionary
            if " ".join(key.casefold().replace("_", " ").split()) in gross_site
        ]
        if matches:
            return max(matches, key=len)

    if case_facts.primary_site:
        primary_site = case_facts.primary_site.strip().upper().replace(".", "")
        for key, entries in site_dictionary.items():
            site_codes = _entry_codes(entries.get("400"))
            if site_codes is not _MISSING:
                site_codes = _normalize_code_descriptions(site_codes)
            if isinstance(site_codes, dict) and primary_site in site_codes:
                return key
    return None


def _variable_info(item_id: int, item_entry: dict | None) -> VariableInfo | None:
    if not item_entry:
        print(f"No entry exists in the data dictionary for item {item_id}")
        return None

    fields = {
        field: item_entry.get(column) for column, field in _ENTRY_FIELD_MAP.items()
    }
    codes = _entry_codes(item_entry)
    fields["valid_codes"] = None if codes is _MISSING else codes
    fields["description"] = _normalize_text(fields["description"])
    fields["coding_instructions"] = _normalize_text(fields["coding_instructions"])
    fields["valid_codes"] = _normalize_code_descriptions(fields["valid_codes"])
    if isinstance(fields["valid_codes"], dict):
        fields["valid_codes"] = _collapse_morphology_valid_codes(
            fields["valid_codes"], fields.get("length")
        )
    return VariableInfo(item_id=item_id, **fields)


def lookup_variable_info(
    item_id: int,
    data_dictionary_path: str | Path,
    *,
    site_data_dictionary_path: str | Path | None = None,
    case_facts: "CaseFacts | None" = None,
) -> VariableInfo | None:
    """Look up NAACCR variable metadata by item ID from a JSON data dictionary.

    Use this tool when you need the name, description, required value format,
    and valid coding values for a specific NAACCR data item. The data dictionary
    must be a JSON object keyed by item ID as a string.

    Args:
        item_id: NAACCR item ID number to look up.
        data_dictionary_path: Path to the NAACCR data dictionary JSON file.
        site_data_dictionary_path: Optional tissue-keyed dictionary whose code
            descriptions override the NAACCR entry.
        case_facts: Facts used to select a tissue from the site dictionary.

    Returns:
        A string representation of a VariableInfo object containing the variable metadata if the item exists,
        otherwise a message explaining that no entry was found.
    """
    with open(data_dictionary_path, "r") as f:
        data_dictionary = json.load(f)

    item_entry = data_dictionary.get(str(item_id))
    if site_data_dictionary_path is not None:
        with open(site_data_dictionary_path, "r") as f:
            site_dictionary = json.load(f)
        site = resolve_site_key(case_facts, site_dictionary)
        site_entry = site_dictionary.get(site, {}).get(str(item_id)) if site else None
        item_entry = _overlay_site_codes(item_entry, site_entry)

    return _variable_info(item_id, item_entry)


def build_variable_group(
    item_ids: int | list[int],
    data_dictionary_path: str | Path | None,
    *,
    case_facts: "CaseFacts | None" = None,
    site_data_dictionary_path: str | Path | None = None,
) -> VariableGroupInfo:
    """Build NAACCR variable metadata with optional tissue-specific code tables."""
    if data_dictionary_path is None:
        raise ValueError("Cannot retrieve variable information. Please supply a data dictionary path.")

    if isinstance(item_ids, int):
        item_ids = [item_ids]

    with open(data_dictionary_path, "r") as f:
        data_dictionary = json.load(f)

    site_dictionary: dict = {}
    if site_data_dictionary_path is not None:
        with open(site_data_dictionary_path, "r") as f:
            site_dictionary = json.load(f)
    site = resolve_site_key(case_facts, site_dictionary)

    item_info = []
    for item_id in sorted(set(item_ids)):
        item_entry = data_dictionary.get(str(item_id))
        site_entry = site_dictionary.get(site, {}).get(str(item_id)) if site else None
        item_entry = _overlay_site_codes(item_entry, site_entry)
        item_info.append(_variable_info(item_id, item_entry))
    variables = [item for item in item_info if item is not None]

    return VariableGroupInfo(variables=variables)
