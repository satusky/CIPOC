import json
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from cipoc.models import VariableInfo, VariableGroupInfo, VariableOutput

from .scoping import HISTOLOGY_ITEM, scoped_item_codes

if TYPE_CHECKING:
    from cipoc.models import CaseFacts, ScopingDictionary


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

# One-to-one typographic substitutions keep offsets into the original note valid.
_EVIDENCE_PUNCTUATION = str.maketrans({
    "\u2010": "-", "\u2011": "-",  # Hyphen and non-breaking hyphen, not mathematical minus.
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
})


def resolve_evidence_text(text: str, note_content: str) -> str | None:
    """Return exact source text, allowing only an unambiguous punctuation equivalent.

    Existing literal matches take precedence. Otherwise only typographic hyphens
    and curly quotes are equivalent; do not fold whitespace, case, or other Unicode.
    """
    if not text.strip() or "\n" in text or "\r" in text:
        return None
    if text in note_content:
        return text
    normalized_text = text.translate(_EVIDENCE_PUNCTUATION)
    normalized_note = note_content.translate(_EVIDENCE_PUNCTUATION)
    start = normalized_note.find(normalized_text)
    if start < 0 or normalized_note.find(normalized_text, start + 1) >= 0:
        return None
    return note_content[start:start + len(text)]


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


def _overlay_site_codes(item_entry: dict | None, site_codes: dict | None) -> dict | None:
    if site_codes is None:
        return item_entry
    merged = dict(item_entry or {})
    merged["allowed_codes"] = site_codes
    return merged


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


def load_data_dictionary(data_dictionary_path: str | Path | None) -> dict:
    """Read the NAACCR data dictionary as a JSON object keyed by item ID as a string.

    An ``{"items": [...]}`` export is re-keyed by each item's ``item_number``.
    """
    if data_dictionary_path is None:
        raise ValueError("Cannot retrieve variable information. Please supply a data dictionary path.")

    with open(data_dictionary_path, "r") as f:
        data_dictionary = json.load(f)

    if "items" in data_dictionary:
        data_dictionary = {str(item["item_number"]): item for item in data_dictionary["items"]}
    return data_dictionary


def histology_pairs(data_dictionary: dict) -> dict[str, str]:
    """Histology (522)'s code/behavior pairs and descriptions, uncollapsed.

    The NAACCR entry lists each 4-digit code once per behavior ('8500/2',
    '8500/3'). Validation collapses them to the stored 4 digits, but histology
    scoping offers pairs, since one pair codes both 522 and Behavior (523).
    """
    codes = _normalize_code_descriptions(_entry_codes(data_dictionary.get(str(HISTOLOGY_ITEM))))
    if not isinstance(codes, dict) or not codes or not all(_MORPHOLOGY_KEY.fullmatch(code) for code in codes):
        raise ValueError(
            f"The data dictionary's item {HISTOLOGY_ITEM} has no code/behavior pairs like '8500/3'."
        )
    return dict(codes)


def lookup_variable_info(
    item_id: int,
    data_dictionary: dict,
    *,
    site_dictionary: "ScopingDictionary | None" = None,
    case_facts: "CaseFacts | None" = None,
) -> VariableInfo:
    """Look up NAACCR variable metadata by item ID from a loaded data dictionary.

    Use this tool when you need the name, description, required value format,
    and valid coding values for a specific NAACCR data item.

    Args:
        item_id: NAACCR item ID number to look up.
        data_dictionary: NAACCR data dictionary from ``load_data_dictionary``.
        site_dictionary: Optional scoping dictionary whose codes for the case
            replace the NAACCR entry's.
        case_facts: Facts that pick the tissue and staging schemas.

    Returns:
        Variable metadata. Unknown item IDs raise ValueError.
    """
    item_entry = data_dictionary.get(str(item_id))
    if not item_entry:
        raise ValueError(f"No entry exists in the data dictionary for item {item_id}.")
    if site_dictionary is not None:
        scoped = scoped_item_codes(site_dictionary, case_facts, [item_id])
        item_entry = _overlay_site_codes(item_entry, scoped.get(item_id))

    return _variable_info(item_id, item_entry)


def build_variable_group(
    item_ids: int | list[int],
    data_dictionary: dict,
    *,
    case_facts: "CaseFacts | None" = None,
    site_dictionary: "ScopingDictionary | None" = None,
) -> VariableGroupInfo:
    """Build NAACCR variable metadata, with codes scoped to the case when a
    scoping dictionary is given."""
    if isinstance(item_ids, int):
        item_ids = [item_ids]

    item_ids = sorted(set(item_ids))
    scoped = (
        scoped_item_codes(site_dictionary, case_facts, item_ids)
        if site_dictionary is not None
        else {}
    )

    item_info = []
    for item_id in item_ids:
        item_entry = data_dictionary.get(str(item_id))
        if not item_entry:
            raise ValueError(f"No entry exists in the data dictionary for item {item_id}.")
        item_entry = _overlay_site_codes(item_entry, scoped.get(item_id))
        variable = _variable_info(item_id, item_entry)
        VariableValueValidator().preflight(variable)
        item_info.append(variable)

    return VariableGroupInfo(variables=item_info)


# What a scoped Histology (522) shows in prompts in place of its code list. The
# full list alone is about 33k tokens; the family and code steps offer the part
# that fits the case.
HISTOLOGY_STEPS_NOTE = (
    "Chosen through the histology family and code steps; the full code list isn't shown."
)
_PROMPT_EXCLUDE = {"histology_scope"}


def prompt_variable(variable: VariableInfo, offered: dict[str, str] | None = None) -> VariableInfo:
    """The variable as prompts show it, without its histology scope.

    A variable with a histology scope shows ``offered``, the codes the code step
    offered, as its valid codes, or a note when nothing has been offered yet. Any
    other variable is returned unchanged, so it renders as it always has.
    """
    if variable.histology_scope is None:
        return variable
    return variable.model_copy(update={
        "histology_scope": None,
        "valid_codes": dict(offered) if offered else HISTOLOGY_STEPS_NOTE,
    })


def variable_prompt_json(variable: VariableInfo, offered: dict[str, str] | None = None) -> str:
    """``prompt_variable`` serialized for a prompt."""
    return prompt_variable(variable, offered).model_dump_json(exclude=_PROMPT_EXCLUDE)


def variable_prompt_dict(variable: VariableInfo, offered: dict[str, str] | None = None) -> dict:
    """``prompt_variable`` as a dict, for prompts that embed it in a larger object."""
    return prompt_variable(variable, offered).model_dump(exclude=_PROMPT_EXCLUDE)


def variable_group_prompt_json(group: VariableGroupInfo) -> str:
    """The group serialized for a prompt, each variable through ``prompt_variable``."""
    shown = group.model_copy(
        update={"variables": [prompt_variable(variable) for variable in group.variables]}
    )
    return shown.model_dump_json(exclude={"variables": {"__all__": _PROMPT_EXCLUDE}})
