import json
import re
from datetime import date
from pathlib import Path
from langchain.tools import tool

from cipoc.models import VariableInfo, VariableGroupInfo, VariableOutput


_ENTRY_FIELD_MAP = {
      "Data Item Name": "name",
      "Description": "description",
      "Data Type": "data_type",
      "Length": "length",
      "Allowable Values": "allowable_values",
      "Format": "format",
      "Code Descriptions": "valid_codes",
  }

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

        if isinstance(variable.valid_codes, dict) and variable.valid_codes:
            if not self._matches_valid_code(variable, value):
                errors.append("Value is not one of the variable's allowable codes.")
        elif self._is_date_variable(variable):
            errors.extend(self._validate_date(value))

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


def lookup_variable_info(item_id: int, data_dictionary_path: str | Path) -> VariableInfo | None:
    """Look up NAACCR variable metadata by item ID from a JSON data dictionary.

    Use this tool when you need the name, description, required value format,
    and valid coding values for a specific NAACCR data item. The data dictionary
    must be a JSON object keyed by item ID as a string.

    Args:
        item_id: NAACCR item ID number to look up.
        data_dictionary_path: Path to the NAACCR data dictionary JSON file.

    Returns:
        A string representation of a VariableInfo object containing the variable metadata if the item exists,
        otherwise a message explaining that no entry was found.
    """
    with open(data_dictionary_path, "r") as f:
        data_dictionary = json.load(f)

    item_entry = data_dictionary.get(str(item_id))
    if not item_entry:
        print(f"No entry exists in the data dictionary for item {item_id}")
        return

    fields = {field: item_entry.get(col) for col, field in _ENTRY_FIELD_MAP.items()}
    if isinstance(fields["description"], list):
        fields["description"] = "".join(fields["description"])
    return VariableInfo(item_id=item_id, **fields)


def build_variable_group(item_ids: int | list[int], data_dictionary_path: str | Path | None) -> VariableGroupInfo:
    if data_dictionary_path is None:
        raise ValueError("Cannot retrieve variable information. Please supply a data dictionary path.")
    
    if isinstance(item_ids, int):
        item_ids = [item_ids]

    item_info = [lookup_variable_info(item, data_dictionary_path) for item in sorted(set(item_ids))]
    return VariableGroupInfo(variables=[item for item in item_info if item is not None])
