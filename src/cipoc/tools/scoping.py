"""Load and check the scoping-only site dictionary, and match cases to its schemas."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, get_args

from cipoc.models import CaseFacts, GrossSite, ScopingDictionary


PRIMARY_SITE_ITEM = 400
_GENERATOR = "`python -m canonical_naaccr.cipoc_scoping` in canonical-naaccr"
_TISSUES = frozenset(get_args(GrossSite)) - {"unspecified"}
# The select keys the generator writes. CIPOC doesn't extract the discriminators
# (3926, 3927) or CS Site-Specific Factor 25 (2879), so those always match.
SELECT_KEYS = (
    "site", "hist", "behavior", "year_dx", "sex_at_birth",
    "discriminator_1", "discriminator_2", "ssf25",
)
_SITE_CODE = re.compile(r"C[0-9]{3}")
# NAACCR's YYYYMMDD or ISO YYYY-MM-DD, either one possibly cut to the year or month.
# Day 00 is NAACCR's unknown day.
_DIAGNOSIS_DATE = re.compile(r"([0-9]{4})(?:(-?)(0[1-9]|1[0-2])(?:\2(0[0-9]|[12][0-9]|3[01]))?)?")


def _normalize_code(code: str) -> str:
    return code.strip().upper().replace(".", "")


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _is_old_format(data: Mapping[str, Any]) -> bool:
    # The old file is keyed by tissue, each holding tables keyed by item number.
    return "tissues" not in data and any(
        isinstance(entries, Mapping) and entries and all(str(key).isdigit() for key in entries)
        for entries in data.values()
    )


@lru_cache(maxsize=None)
def _select_ranges(key: str, value: str) -> tuple[tuple, ...]:
    """Parse a select value's comma-separated codes and ``low-high`` ranges.

    Sites stay normalized codes, compared as strings. Every other key is numeric,
    and a range may be open-ended (``2018-``).
    """
    ranges = []
    for token in value.split(","):
        low, dash, high = (part.strip() for part in token.partition("-"))
        if not low and not dash:
            continue
        if key == "site":
            low, high = _normalize_code(low), _normalize_code(high if dash else low)
            if not (_SITE_CODE.fullmatch(low) and _SITE_CODE.fullmatch(high)):
                raise ValueError(f"Unsupported {key} value {value!r}.")
            ranges.append((low, high))
        else:
            if not low.isdigit() or (high and not high.isdigit()):
                raise ValueError(f"Unsupported {key} value {value!r}.")
            upper = int(high) if high else None if dash else int(low)
            ranges.append((int(low), upper))
    if not ranges:
        raise ValueError(f"Unsupported {key} value {value!r}.")
    return tuple(ranges)


def _check_schemas(scoping: ScopingDictionary) -> None:
    for key, schema in scoping.schemas.items():
        for row in schema.select:
            unknown = sorted(set(row) - set(SELECT_KEYS))
            if unknown:
                raise ValueError(
                    f"Schema {key} selects on keys CIPOC doesn't read: {unknown}. "
                    f"It reads the select format written by {_GENERATOR}."
                )
            for select_key, value in row.items():
                try:
                    _select_ranges(select_key, value)
                except ValueError as error:
                    raise ValueError(
                        f"Schema {key}: {error} It reads the select format written by {_GENERATOR}."
                    ) from None
        for item_id, override in schema.codes.items():
            if override.source not in scoping.sources:
                raise ValueError(
                    f"Schema {key} item {item_id} names source {override.source!r}, which isn't in sources. "
                    f"Regenerate the file with {_GENERATOR}."
                )


def parse_scoping_dictionary(data: Mapping[str, Any]) -> ScopingDictionary:
    """Validate a parsed scoping dictionary and check its tissue table and schemas.

    The tissue keys must equal ``GrossSite`` minus "unspecified", and no Primary
    Site code may be in two tissues. Every schema's select keys must be ones the
    matcher reads, with values it can parse, and every override's source must be
    in ``sources``. Coverage of item 400 is checked separately by
    ``check_primary_site_coverage``, since it needs the NAACCR dictionary.
    """
    if _is_old_format(data):
        raise ValueError(
            "The site dictionary is in the old tissue-keyed format. "
            f"Regenerate it as a scoping dictionary with {_GENERATOR}."
        )
    scoping = ScopingDictionary.model_validate(data)

    missing = sorted(_TISSUES - set(scoping.tissues))
    extra = sorted(set(scoping.tissues) - _TISSUES)
    if missing or extra:
        problems = [f"missing {missing}"] if missing else []
        problems += [f"not in GrossSite {extra}"] if extra else []
        raise ValueError(
            f"Scoping dictionary tissues don't match GrossSite: {'; '.join(problems)}. "
            f"Regenerate the file or GrossSite with {_GENERATOR}."
        )

    owner: dict[str, str] = {}
    duplicates: list[str] = []
    for tissue, entry in scoping.tissues.items():
        for code in entry.codes.get(PRIMARY_SITE_ITEM, {}):
            code = _normalize_code(code)
            if code in owner:
                duplicates.append(f"{code} in {owner[code]} and {tissue}")
            else:
                owner[code] = tissue
    if duplicates:
        raise ValueError(
            f"Primary Site codes are in more than one tissue: {'; '.join(duplicates)}."
        )
    _check_schemas(scoping)
    return scoping


def load_scoping_dictionary(path: str | Path) -> ScopingDictionary:
    """Read the scoping dictionary file and check it with ``parse_scoping_dictionary``."""
    with open(path, "r") as f:
        return parse_scoping_dictionary(json.load(f))


def check_primary_site_coverage(
    scoping: ScopingDictionary, primary_site_codes: Iterable[str]
) -> None:
    """Fail unless the tissues together hold exactly item 400's codes."""
    expected = {_normalize_code(code) for code in primary_site_codes}
    covered = {
        _normalize_code(code)
        for entry in scoping.tissues.values()
        for code in entry.codes.get(PRIMARY_SITE_ITEM, {})
    }
    missing = sorted(expected - covered)
    extra = sorted(covered - expected)
    if missing or extra:
        problems = [f"in item 400 but no tissue {missing}"] if missing else []
        problems += [f"in a tissue but not item 400 {extra}"] if extra else []
        raise ValueError(
            f"Scoping dictionary tissues don't cover item 400: {'; '.join(problems)}."
        )


def _diagnosis_year(value: str | None) -> int | None:
    match = _DIAGNOSIS_DATE.fullmatch((value or "").strip())
    if match is None or int(match.group(1)) == 0:
        return None
    return int(match.group(1))


def _case_values(scoping: ScopingDictionary, facts: CaseFacts | None) -> dict[str, frozenset]:
    """The values each known fact could take, by select key. A missing key is unknown."""
    if facts is None:
        return {}
    values: dict[str, frozenset] = {}

    if facts.primary_site:
        # Unlike site_applies, C809 is a real code here: it selects the
        # unknown-primary schemas. A malformed coded primary is unknown, and as in
        # site_applies it also stops the gross-site fallback.
        code = _normalize_code(facts.primary_site)
        if _SITE_CODE.fullmatch(code):
            values["site"] = frozenset({code})
    else:
        tissue = scoping.tissues.get(facts.gross_primary_site or "")
        codes = tissue.codes.get(PRIMARY_SITE_ITEM) if tissue else None
        if codes:
            values["site"] = frozenset(_normalize_code(code) for code in codes)

    histology = (facts.histology or "").strip()
    if re.match(r"[0-9]{4}", histology):
        values["hist"] = frozenset({int(histology[:4])})

    behavior = (facts.behavior or "").strip()
    suffix = re.search(r"/([0-9])", histology)
    if re.fullmatch(r"[0-9]", behavior):
        values["behavior"] = frozenset({int(behavior)})
    elif suffix:
        values["behavior"] = frozenset({int(suffix.group(1))})

    year = _diagnosis_year(facts.date_of_diagnosis)
    if year is not None:
        values["year_dx"] = frozenset({year})

    # Sex (220) and Sex Assigned at Birth (225), which the schemas select on,
    # agree only on 1 (male) and 2 (female).
    if (facts.sex or "").strip() in ("1", "2"):
        values["sex_at_birth"] = frozenset({int(facts.sex.strip())})
    return values


def _row_matches(row: Mapping[str, str], values: Mapping[str, frozenset]) -> bool:
    for key, value in row.items():
        known = values.get(key)
        if known is None:
            continue
        if not any(
            low <= fact and (high is None or fact <= high)
            for fact in known
            for low, high in _select_ranges(key, value)
        ):
            return False
    return True


def matching_schemas(scoping: ScopingDictionary, facts: CaseFacts | None) -> list[str]:
    """Keys of the schemas the case could belong to, in file order.

    A known fact filters and an unknown one matches anything. The site is the
    coded Primary Site, or else every code of the gross-site tissue, which
    matches a row whose sites overlap it. Histology is the coded morphology's 4
    digits, behavior comes from the behavior fact or the histology's ``/n``
    suffix, and the year is parsed from the date of diagnosis.
    """
    values = _case_values(scoping, facts)
    return [
        key
        for key, schema in scoping.schemas.items()
        if any(_row_matches(row, values) for row in schema.select)
    ]


def scoped_codes(
    scoping: ScopingDictionary, schema_keys: Sequence[str], item_id: int
) -> dict[str, str] | None:
    """The item's codes over the matched schemas, or None to keep NAACCR's.

    The item is scoped only when every schema overrides it: a schema without an
    override stands for the NAACCR codes. The result is the union of the
    overrides, and a code two schemas describe differently (ignoring whitespace
    and case) leaves the item unscoped. No matched schema leaves it unscoped too.
    """
    merged: dict[str, str] = {}
    for key in schema_keys:
        override = scoping.schemas[key].codes.get(item_id)
        if override is None:
            return None
        for code, description in override.codes.items():
            if code in merged and _normalize_text(merged[code]) != _normalize_text(description):
                return None
            merged.setdefault(code, description)
    return merged or None
