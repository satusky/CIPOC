"""Load and check the scoping-only site dictionary."""

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, get_args

from cipoc.models import GrossSite, ScopingDictionary


PRIMARY_SITE_ITEM = 400
_GENERATOR = "`python -m canonical_naaccr.cipoc_scoping` in canonical-naaccr"
_TISSUES = frozenset(get_args(GrossSite)) - {"unspecified"}


def _normalize_code(code: str) -> str:
    return code.strip().upper().replace(".", "")


def _is_old_format(data: Mapping[str, Any]) -> bool:
    # The old file is keyed by tissue, each holding tables keyed by item number.
    return "tissues" not in data and any(
        isinstance(entries, Mapping) and entries and all(str(key).isdigit() for key in entries)
        for entries in data.values()
    )


def parse_scoping_dictionary(data: Mapping[str, Any]) -> ScopingDictionary:
    """Validate a parsed scoping dictionary and check its tissue table.

    The tissue keys must equal ``GrossSite`` minus "unspecified", and no Primary
    Site code may be in two tissues. Coverage of item 400 is checked separately
    by ``check_primary_site_coverage``, since it needs the NAACCR dictionary.
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
