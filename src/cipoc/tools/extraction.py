import json
from pathlib import Path
from langchain.tools import tool

from cipoc.models import VariableInfo, VariableGroupInfo


_ENTRY_FIELD_MAP = {
      "Data Item Name": "name",
      "Description": "description",
      "Format": "format",
      "Code Descriptions": "valid_codes",
  }


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
    return VariableInfo(item_id=item_id, **fields)


def build_variable_group(item_ids: int | list[int], data_dictionary_path: str | Path | None) -> VariableGroupInfo | str:
    if data_dictionary_path is None:
        raise ValueError("Cannot retrieve variable information. Please supply a data dictionary path.")
    
    if isinstance(item_ids, int):
        item_ids = [item_ids]

    item_info = [lookup_variable_info(item, data_dictionary_path) for item in item_ids]
    return VariableGroupInfo(variables=[item for item in item_info if item is not None])

