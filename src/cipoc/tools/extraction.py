import json
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from cipoc.models import VariableInfo, VariableGroupInfo, VariableOutput

if TYPE_CHECKING:
    from cipoc.models import CaseFacts


_ENTRY_FIELD_MAP = {
    "name": ("item_name", "Data Item Name"),
    "description": ("description", "Description"),
    "data_type": ("item_data_type", "Data Type"),
    "length": ("item_length", "Length"),
    "allowable_values": ("allowable_values", "Allowable Values"),
    "format": ("format", "Format"),
    "coding_instructions": ("instructions_for_coding", "Instructions for Coding"),
}

_CODE_FIELD_NAMES = ("allowed_codes", "Code Descriptions")
_CODE_COLUMN_NAMES = ("code",)
_DESCRIPTION_COLUMN_NAMES = ("description",)
_MISSING = object()


def _parse_code_domain(tokens: list[str]) -> tuple[set[str], list[tuple[str, str, str]]]:
    """Parse only literal codes and equal-width, equal-prefix numeric intervals."""
    literals: set[str] = set()
    ranges: list[tuple[str, str, str]] = []
    for token in tokens:
        if token.casefold() in {"", "blank", "blanks", "ccyymmdd", "yyyymmdd", "alphanumeric", "numeric", "digits", "free text", "text"}:
            continue
        interval = re.fullmatch(r"([A-Za-z]*)([0-9]+)\s*-\s*([A-Za-z]*)([0-9]+)", token)
        if interval:
            prefix, lower, upper_prefix, upper = interval.groups()
            if prefix != upper_prefix or len(lower) != len(upper) or lower > upper:
                raise ValueError(f"Unsupported code range {token!r}.")
            ranges.append((prefix, lower, upper))
        elif re.fullmatch(r"[A-Za-z0-9]+(?:[./][A-Za-z0-9]+)*", token):
            literals.add(token)
        else:
            raise ValueError(f"Unsupported code domain token {token!r}.")
    return literals, ranges


def _domain_contains(domain: tuple[set[str], list[tuple[str, str, str]]], value: str) -> bool:
    literals, ranges = domain
    return value in literals or any(
        re.fullmatch(re.escape(prefix) + rf"[0-9]{{{len(lower)}}}", value)
        and lower <= value[len(prefix):] <= upper
        for prefix, lower, upper in ranges
    )


class VariableValueValidator:
    """Deterministically validate an extracted value against variable metadata."""

    def __init__(self, *, allow_unknown_date_day: bool = True) -> None:
        self.allow_unknown_date_day = allow_unknown_date_day

    def preflight(self, variable: VariableInfo) -> None:
        """Reject unusable metadata before a model can return even a null answer."""
        try:
            if variable.length is not None and variable.length <= 0:
                raise ValueError("Maximum length must be positive.")
            if self._is_date_variable(variable):
                if variable.length is not None and variable.length < 8:
                    raise ValueError("Date length cannot be less than eight.")
                return
            domain = self._code_domain(variable)
            patterns = self._format_patterns(variable)
            if domain is None:
                format_value = (variable.format or "").strip().casefold()
                if format_value not in {
                    "", "numeric", "digits", "alphanumeric", "alphanumeric blank",
                    "free text", "left justified", "right justified", "right justified, zero filled",
                }:
                    raise ValueError(f"Unsupported format {variable.format!r} without a code domain.")
                if not self._is_free_text(variable) and not any(pattern in {r"[0-9]+", r"[A-Za-z0-9]+"} for pattern in patterns):
                    raise ValueError("No usable code domain or sufficient format constraint.")
            else:
                literals, ranges = domain
                representatives = list(literals) + [prefix + lower for prefix, lower, _ in ranges]
                if not any(
                    (variable.length is None or len(value) <= variable.length)
                    and all(re.fullmatch(pattern, value) for pattern in patterns)
                    for value in representatives
                ):
                    raise ValueError("Code domain has no values satisfying length and type/format constraints.")
        except ValueError as error:
            raise ValueError(f"Invalid validation metadata for item {variable.item_id}: {error}") from error

    def validate(self, variable: VariableInfo, candidate: VariableOutput) -> list[str]:
        self.preflight(variable)
        errors: list[str] = []

        if candidate.item_id != variable.item_id:
            errors.append(
                f"Expected item ID {variable.item_id}, received {candidate.item_id}."
            )

        value = candidate.value
        if value is None:
            return errors

        if not value.strip():
            errors.append("Value must not be empty; use null for an undetermined value.")
        if not self._is_free_text(variable) and any(character.isspace() for character in value):
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
        else:
            domain = self._code_domain(variable)
            if domain is not None and not _domain_contains(domain, value):
                errors.append("Value is not one of the variable's allowable codes.")
            for pattern in self._format_patterns(variable):
                if re.fullmatch(pattern, value) is None:
                    errors.append("Value does not satisfy the variable's type/format constraints.")
                    break

        return errors

    @staticmethod
    def _code_domain(variable: VariableInfo) -> tuple[set[str], list[tuple[str, str, str]]] | None:
        codes = _normalize_code_descriptions(variable.valid_codes)
        if codes and not isinstance(codes, dict):
            raise ValueError("Code table must be an object or code/description rows.")
        if codes:
            if not all(isinstance(code, str) for code in codes):
                raise ValueError("Code table keys must be strings to preserve leading zeros.")
            # NAACCR item 676 abbreviates this specific table with '..'. This is
            # not a general ellipsis expansion, nor permission to widen scoped tables.
            if (
                variable.item_id == 676
                and set(codes) == {"00", "01", "02", "..", "90", "95", "96", "97", "98", "99"}
                and re.sub(r"\s+", "", variable.allowable_values or "") == "00-90,95-99"
            ):
                return _parse_code_domain(["00-90", "95-99"])
            domain = _parse_code_domain(list(codes))
            if any(domain):
                return domain
        declaration = (variable.allowable_values or "").strip()
        if not declaration:
            return None
        if (
            VariableValueValidator._is_free_text(variable)
            and declaration.casefold() == "neither carriage return nor line feed characters allowed"
        ):
            return None
        tokens = [token.strip() for token in declaration.split(",")]
        # These describe representation, not extra allowable literals. A broad
        # descriptor must never override an accompanying concrete code domain.
        descriptors = {"alphanumeric", "numeric", "digits", "free text", "text"}
        tokens = [token for token in tokens if token.casefold() not in descriptors]
        domain = _parse_code_domain(tokens)
        return domain if any(domain) else None

    @staticmethod
    def _is_free_text(variable: VariableInfo) -> bool:
        return any(
            (value or "").strip().casefold() == "free text"
            for value in (variable.format, variable.allowable_values)
        )

    @staticmethod
    def _format_patterns(variable: VariableInfo) -> list[str]:
        patterns = []
        data_type = (variable.data_type or "").strip().casefold()
        format_value = (variable.format or "").strip().casefold()
        declaration = (variable.allowable_values or "").strip().casefold()
        if data_type == "digits" or format_value in {"numeric", "digits"} or declaration in {"numeric", "digits"}:
            patterns.append(r"[0-9]+")
        if format_value in {"alphanumeric", "alphanumeric blank"} or declaration == "alphanumeric":
            patterns.append(r"[A-Za-z0-9]+")
        if format_value == "right justified, zero filled":
            if variable.length is None:
                raise ValueError("Zero-filled format requires a field length.")
            patterns.append(rf".{{{variable.length}}}")
        return patterns

    @staticmethod
    def _is_date_variable(variable: VariableInfo) -> bool:
        if variable.data_type and variable.data_type.casefold() == "date":
            return True
        return bool(
            variable.format
            and re.match(r"(?:YYYYMMDD|CCYYMMDD)(?:\b|\\r|\\n)", variable.format.strip(), re.IGNORECASE)
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
        code = row[code_column]
        if not isinstance(code, str):
            raise ValueError("allowed_codes code fields must be strings to preserve leading zeros.")
        normalized[code] = str(row[description_column])
    return normalized


def _normalize_text(value):
    return "".join(str(part) for part in value) if isinstance(value, list) else value


def _entry_codes(entry: dict | None):
    if entry:
        for field in _CODE_FIELD_NAMES:
            if field in entry:
                return entry[field]
    return _MISSING


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

    raw_primary_site = (case_facts.primary_site or "").strip().upper()
    primary_site = raw_primary_site.replace(".", "")
    if re.fullmatch(r"C[0-9]{2}\.?[0-9]", raw_primary_site) and primary_site < "C809":
        for key, entries in site_dictionary.items():
            site_codes = _entry_codes(entries.get("400"))
            if site_codes is not _MISSING:
                site_codes = _normalize_code_descriptions(site_codes)
            if isinstance(site_codes, dict) and _domain_contains(_parse_code_domain(list(site_codes)), primary_site):
                return key
        # An informative coded primary outranks conflicting gross-site inference,
        # even when only base metadata is available for that primary.
        return None

    if case_facts.primary_site:
        # An explicit unknown or malformed primary must not be narrowed by gross site.
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

    return None


def _variable_info(item_id: int, item_entry: dict | None) -> VariableInfo:
    if not item_entry:
        raise ValueError(f"No entry exists in the data dictionary for item {item_id}.")

    fields = {
        field: next(
            (item_entry[column] for column in columns if column in item_entry),
            None,
        )
        for field, columns in _ENTRY_FIELD_MAP.items()
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
) -> VariableInfo:
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
        Variable metadata. Unknown item IDs raise ValueError.
    """
    with open(data_dictionary_path, "r") as f:
        data_dictionary = json.load(f)

    item_entry = data_dictionary.get(str(item_id))
    if not item_entry:
        raise ValueError(f"No entry exists in the data dictionary for item {item_id}.")
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

    if "items" in data_dictionary:
        data_dictionary = {str(item["item_number"]): item for item in data_dictionary["items"]}

    site_dictionary: dict = {}
    if site_data_dictionary_path is not None:
        with open(site_data_dictionary_path, "r") as f:
            site_dictionary = json.load(f)
    site = resolve_site_key(case_facts, site_dictionary)

    item_info = []
    for item_id in sorted(set(item_ids)):
        item_entry = data_dictionary.get(str(item_id))
        if not item_entry:
            raise ValueError(f"No entry exists in the data dictionary for item {item_id}.")
        site_entry = site_dictionary.get(site, {}).get(str(item_id)) if site else None
        item_entry = _overlay_site_codes(item_entry, site_entry)
        variable = _variable_info(item_id, item_entry)
        VariableValueValidator().preflight(variable)
        item_info.append(variable)

    return VariableGroupInfo(variables=item_info)
