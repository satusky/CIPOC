"""Load and check the scoping-only site dictionary, and scope a case's codes by it."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence, get_args

from cipoc.models import (
    CaseFacts,
    DiscriminatorChoice,
    GrossSite,
    HistologyScope,
    SchemaDiscriminator,
    ScopingDictionary,
    SsfChoice,
    SsfMapping,
    StagingSchema,
    VariableInfo,
)


PRIMARY_SITE_ITEM = 400
HISTOLOGY_ITEM = 522
BEHAVIOR_ITEM = 523
UNKNOWN_PRIMARY_SITE = "C809"
_GENERATOR = "`python -m canonical_naaccr.cipoc_scoping` in canonical-naaccr"
# The sections CIPOC reads, each added by a later version of the generator.
_SECTIONS = ("sources", "tissues", "schemas", "histology", "blocks", "algorithms", "items")
_ALGORITHMS = ("cs", "tnm", "eod")
# The algorithms whose schemas held some requested items in Site-Specific
# Factors before 2018 (the schemas' ``ssf``).
_SSF_ALGORITHMS = ("cs", "tnm")
# An SSF table keeps the rows a later CS version converted away, for old data.
# Each starts "OBSOLETE DATA CONVERTED" or "OBSOLETE DATA RETAINED".
_OBSOLETE_CODE = re.compile(r"\s*OBSOLETE\b", re.IGNORECASE)
_TISSUES = frozenset(get_args(GrossSite)) - {"unspecified"}
# The select keys the generator writes. The discriminators match anything until
# they're coded, which only happens with the discriminator variable groups file.
SELECT_KEYS = (
    "site", "hist", "behavior", "year_dx", "sex_at_birth",
    "discriminator_1", "discriminator_2", "ssf25",
)
# The discriminator items, with the select key each fills and its case fact:
# Schema Discriminators 1 and 2 (EOD) and CS Site-Specific Factor 25 (CS, TNM).
DISCRIMINATOR_KEYS = {3926: "discriminator_1", 3927: "discriminator_2", 2879: "ssf25"}
DISCRIMINATOR_FACTS = {
    3926: "schema_discriminator_1",
    3927: "schema_discriminator_2",
    2879: "cs_site_specific_factor_25",
}
# A line of a discriminator table's blank code that only says it isn't needed.
_NOT_NEEDED = re.compile(r"not necessary|does not apply", re.IGNORECASE)
_SITE_CODE = re.compile(r"C[0-9]{3}")
# A coded Primary Site as site_applies accepts it: upper case, with or without the dot.
_CODED_PRIMARY = re.compile(r"C[0-9]{2}\.?[0-9]")
_FAMILY = re.compile(r"[0-9]{3}")
_PAIR = re.compile(r"[0-9]{4}/[0-9]")
_HISTOLOGY_CODE = re.compile(r"[0-9]{4}")
_BLOCK = re.compile(r"([0-9]{3})(?:-([0-9]{3}))?")
# An item's diagnosis years. Unlike a select's year_dx, either end may be open.
_ITEM_YEARS = re.compile(r"([0-9]{4})?-([0-9]{4})?")
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


@lru_cache(maxsize=None)
def _histology_codes(value: str) -> frozenset[int]:
    """Parse an algorithm's accepted histology: comma-separated 4-digit codes and ``low-high`` runs."""
    codes: set[int] = set()
    for token in value.split(","):
        low, dash, high = (part.strip() for part in token.partition("-"))
        high = high if dash else low
        if not (_HISTOLOGY_CODE.fullmatch(low) and _HISTOLOGY_CODE.fullmatch(high)) or low > high:
            raise ValueError(f"Unsupported histology code or range {token!r}.")
        codes.update(range(int(low), int(high) + 1))
    return frozenset(codes)


@lru_cache(maxsize=None)
def _item_years(value: str) -> tuple[int | None, int | None]:
    """Parse an item's years: ``YYYY-YYYY``, ``YYYY-``, ``-YYYY`` or ``-``, an empty end open."""
    match = _ITEM_YEARS.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Unsupported item years {value!r}.")
    first, last = (int(year) if year else None for year in match.groups())
    if first is not None and last is not None and last < first:
        raise ValueError(f"Item years {value!r} run backwards.")
    return first, last


def _all_site_codes(scoping: ScopingDictionary) -> set[str]:
    return {
        _normalize_code(code)
        for entry in scoping.tissues.values()
        for code in entry.codes.get(PRIMARY_SITE_ITEM, {})
    }


def _in_sites(code: str, sites: str) -> bool:
    return any(low <= code <= high for low, high in _select_ranges("site", sites))


def _block_range(key: str) -> tuple[int, int]:
    match = _BLOCK.fullmatch(key)
    if match is None or int(match.group(2) or match.group(1)) < int(match.group(1)):
        raise ValueError(
            f"Morphology block {key!r} isn't a 3-digit prefix or range like '814-838'. "
            f"Regenerate the file with {_GENERATOR}."
        )
    return int(match.group(1)), int(match.group(2) or match.group(1))


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


def _check_algorithms(scoping: ScopingDictionary) -> None:
    if sorted(scoping.algorithms) != sorted(_ALGORITHMS):
        raise ValueError(
            f"The site dictionary's algorithms are {sorted(scoping.algorithms)}, not {list(_ALGORITHMS)}. "
            f"Regenerate the file with {_GENERATOR}."
        )
    spans = []
    for key, algorithm in scoping.algorithms.items():
        try:
            years = _select_ranges("year_dx", algorithm.years)
            if any(high is not None and high < low for low, high in years):
                raise ValueError(f"Unsupported year_dx value {algorithm.years!r}.")
            _histology_codes(algorithm.histology)
        except ValueError as error:
            raise ValueError(f"Algorithm {key}: {error} Regenerate the file with {_GENERATOR}.") from None
        spans += [(low, high, key) for low, high in years]
    spans.sort(key=lambda span: span[0])
    for (_, high, key), (low, _, later) in zip(spans, spans[1:]):
        if high is None or low <= high:
            raise ValueError(f"Algorithms {key} and {later} cover overlapping diagnosis years.")


def _check_items(scoping: ScopingDictionary) -> None:
    for item_id, entry in scoping.items.items():
        try:
            _item_years(entry.years)
        except ValueError as error:
            raise ValueError(f"Item {item_id}: {error} Regenerate the file with {_GENERATOR}.") from None


def _check_ssf(scoping: ScopingDictionary) -> None:
    for key, schema in scoping.schemas.items():
        if schema.ssf and key.partition("/")[0] not in _SSF_ALGORITHMS:
            raise ValueError(
                f"Schema {key} has SSF mappings, which only CS and TNM schemas hold. "
                f"Regenerate the file with {_GENERATOR}."
            )
        for item_id, mapping in schema.ssf.items():
            if mapping.years is None:
                continue
            try:
                _item_years(mapping.years)
            except ValueError as error:
                raise ValueError(
                    f"Schema {key}, item {item_id}'s SSF: {error} Regenerate the file with {_GENERATOR}."
                ) from None


def _check_discriminators(scoping: ScopingDictionary) -> None:
    for key, schema in scoping.schemas.items():
        unknown = sorted(set(schema.discriminators) - set(DISCRIMINATOR_KEYS))
        if unknown:
            raise ValueError(
                f"Schema {key} has discriminators for items {unknown}; CIPOC reads "
                f"{sorted(DISCRIMINATOR_KEYS)}. Regenerate the file with {_GENERATOR}."
            )


def _check_histology(scoping: ScopingDictionary) -> None:
    codes = _all_site_codes(scoping)
    owner: dict[str, str] = {}
    duplicates: list[str] = []
    for key, group in scoping.histology.items():
        try:
            covered = sorted(code for code in codes if _in_sites(code, group.sites))
        except ValueError as error:
            raise ValueError(
                f"Histology group {key}: {error} Regenerate the file with {_GENERATOR}."
            ) from None
        for code in covered:
            if code in owner:
                duplicates.append(f"{code} in {owner[code]} and {key}")
            else:
                owner[code] = key
        for number, family in group.families.items():
            if not _FAMILY.fullmatch(number):
                raise ValueError(f"Histology group {key} has family {number!r}, which isn't 3 digits.")
            malformed = [pair for pair in family.codes if not (_PAIR.fullmatch(pair) and pair.startswith(number))]
            if malformed:
                raise ValueError(
                    f"Histology group {key} family {number} holds {malformed}, "
                    f"which aren't code/behavior pairs starting with {number}."
                )
    if duplicates:
        raise ValueError(
            f"Primary Site codes are in more than one histology group: {'; '.join(duplicates)}."
        )
    uncovered = sorted(codes - set(owner))
    if uncovered:
        raise ValueError(f"Primary Site codes are in no histology group: {uncovered}.")
    if UNKNOWN_PRIMARY_SITE not in owner:
        raise ValueError(f"No histology group holds {UNKNOWN_PRIMARY_SITE}, the unknown primary.")


def _check_blocks(scoping: ScopingDictionary) -> None:
    ranges = sorted((*_block_range(key), key) for key in scoping.blocks)
    for (_, high, key), (low, _, later) in zip(ranges, ranges[1:]):
        if low <= high:
            raise ValueError(f"Morphology blocks {key} and {later} overlap.")


def parse_scoping_dictionary(data: Mapping[str, Any]) -> ScopingDictionary:
    """Validate a parsed scoping dictionary and check its tissue table, schemas,
    histology site groups, morphology blocks, staging algorithms and item years.

    The tissue keys must equal ``GrossSite`` minus "unspecified", and no Primary
    Site code may be in two tissues. Every schema's select keys must be ones the
    matcher reads, with values it can parse, and every override's source must be
    in ``sources``. The histology site groups must hold each of the tissues'
    Primary Site codes exactly once, C809 included, and their families' pairs
    must start with the family's 3 digits. Morphology blocks mustn't overlap.
    The algorithms must be cs, tnm and eod, with diagnosis years that don't
    overlap and histology codes that parse. Every item's years must parse and
    run forwards, and so must the years of every SSF mapping, which only CS and
    TNM schemas may hold. Schemas may only hold discriminators for 3926, 3927
    and 2879.
    Coverage of item 400, and of item 522's pairs, is checked separately
    (``check_primary_site_coverage``, ``check_histology_pairs``), since it needs
    the NAACCR dictionary.
    """
    if _is_old_format(data):
        raise ValueError(
            "The site dictionary is in the old tissue-keyed format. "
            f"Regenerate it as a scoping dictionary with {_GENERATOR}."
        )
    missing_sections = [section for section in _SECTIONS if section not in data]
    if missing_sections:
        raise ValueError(
            f"The site dictionary has no {' or '.join(missing_sections)} section, so it's "
            f"from an older build. Regenerate it with {_GENERATOR}."
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
    _check_histology(scoping)
    _check_blocks(scoping)
    _check_algorithms(scoping)
    _check_items(scoping)
    _check_ssf(scoping)
    _check_discriminators(scoping)
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


def tissue_site_codes(scoping: ScopingDictionary | None) -> dict[str, frozenset[str]]:
    """Map each tissue to its Primary Site (400) codes, normalized like ``C504``.

    A tissue with no 400 codes is left out, so ``site_applies`` reads it as unknown.
    """
    table: dict[str, frozenset[str]] = {}
    for tissue, entry in (scoping.tissues if scoping else {}).items():
        codes = entry.codes.get(PRIMARY_SITE_ITEM)
        if codes:
            table[tissue] = frozenset(_normalize_code(code) for code in codes)
    return table


def gross_site_candidates(facts: CaseFacts | None) -> list[str]:
    """The tissues the notes point to, which count as OR.

    These are ``candidate_gross_sites``, or else the lone ``gross_primary_site``,
    so facts from callers or older artifacts that only set the gross site read
    as one candidate. A coded primary outranks them; callers check it first.
    """
    if facts is None:
        return []
    if facts.candidate_gross_sites:
        return list(facts.candidate_gross_sites)
    return [facts.gross_primary_site] if facts.gross_primary_site else []


def _candidate_tissues(scoping: ScopingDictionary, facts: CaseFacts | None) -> list[str]:
    # One candidate that isn't a tissue in the file, such as "unspecified" or a
    # free-text name, makes the whole gross-site fact unknown.
    tissues = gross_site_candidates(facts)
    return tissues if all(tissue in scoping.tissues for tissue in tissues) else []


def primary_site_tissues(scoping: ScopingDictionary, facts: CaseFacts | None) -> list[str]:
    """The tissues whose codes scope Primary Site, or [] to leave it unscoped.

    A coded primary picks the tissue that holds it, outranking the candidates.
    C809 or a malformed coded primary picks none and, as in ``site_applies``,
    also stops the gross-site fallback. Otherwise every candidate must be a
    tissue in the file, so "unspecified" and names outside the vocabulary never
    narrow.
    """
    if facts is None:
        return []
    if facts.primary_site:
        coded = facts.primary_site.strip().upper()
        code = coded.replace(".", "")
        if not _CODED_PRIMARY.fullmatch(coded) or code == UNKNOWN_PRIMARY_SITE:
            return []
        return [
            tissue
            for tissue, entry in scoping.tissues.items()
            if any(_normalize_code(site) == code for site in entry.codes.get(PRIMARY_SITE_ITEM, {}))
        ][:1]
    return _candidate_tissues(scoping, facts)


def _primary_site_codes(scoping: ScopingDictionary, tissues: Sequence[str]) -> dict[str, str]:
    """The union of the tissues' Primary Site codes, always allowing C809."""
    codes: dict[str, str] = {}
    for tissue in tissues:
        codes.update(scoping.tissues[tissue].codes.get(PRIMARY_SITE_ITEM, {}))
    for entry in scoping.tissues.values():
        for code, description in entry.codes.get(PRIMARY_SITE_ITEM, {}).items():
            if _normalize_code(code) == UNKNOWN_PRIMARY_SITE:
                codes.setdefault(code, description)
    return codes


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
        # The candidates are OR, so the site could be any of their codes.
        codes = {
            _normalize_code(code)
            for tissue in _candidate_tissues(scoping, facts)
            for code in scoping.tissues[tissue].codes.get(PRIMARY_SITE_ITEM, {})
        }
        if codes:
            values["site"] = frozenset(codes)

    histology = (facts.histology or "").strip()
    if re.match(r"[0-9]{4}", histology):
        values["hist"] = frozenset({int(histology[:4])})
    elif not histology and facts.candidate_histology_families:
        # The candidate families stand in for an uncoded histology, as OR, so it
        # could be any code under them. A malformed one leaves it unknown.
        families = facts.candidate_histology_families
        if all(_FAMILY.fullmatch(family) for family in families):
            values["hist"] = frozenset(
                int(family) * 10 + digit for family in families for digit in range(10)
            )

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

    # A coded discriminator picks among the schemas its select key separates.
    for item_id, key in DISCRIMINATOR_KEYS.items():
        coded = (getattr(facts, DISCRIMINATOR_FACTS[item_id]) or "").strip()
        if coded.isdigit():
            values[key] = frozenset({int(coded)})
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
    coded Primary Site, or else every code of the candidate tissues, which
    matches a row whose sites overlap any of them. Histology is the coded
    morphology's 4 digits, or else every code under the candidate histology
    families. Behavior comes from the behavior fact or the histology's ``/n``
    suffix, and the year is parsed from the date of diagnosis. A coded
    discriminator (3926, 3927, 2879) filters the rows that read it.
    """
    return _matching(scoping, _case_values(scoping, facts))


def _matching(scoping: ScopingDictionary, values: Mapping[str, frozenset]) -> list[str]:
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


def scoped_item_codes(
    scoping: ScopingDictionary, facts: CaseFacts | None, item_ids: Iterable[int]
) -> dict[int, dict[str, str]]:
    """The codes that replace NAACCR's for each of the items the case narrows.

    Primary Site (400) takes its tissues' codes (``primary_site_tissues``) plus
    C809. Histology (522) isn't narrowed. Every other item takes
    ``scoped_codes`` over the schemas the case matches. Items left out keep
    NAACCR's codes.
    """
    scoped: dict[int, dict[str, str]] = {}
    schema_keys: list[str] | None = None
    for item_id in item_ids:
        if item_id == PRIMARY_SITE_ITEM:
            tissues = primary_site_tissues(scoping, facts)
            codes = _primary_site_codes(scoping, tissues) if tissues else None
        elif item_id == HISTOLOGY_ITEM:
            codes = None
        else:
            if schema_keys is None:
                schema_keys = matching_schemas(scoping, facts)
            codes = scoped_codes(scoping, schema_keys, item_id)
        if codes:
            scoped[item_id] = dict(codes)
    return scoped


def diagnosis_algorithm(scoping: ScopingDictionary, date_of_diagnosis: str | None) -> str | None:
    """The staging algorithm (cs, tnm or eod) whose years hold the diagnosis year.

    ``date_of_diagnosis`` is parsed as ``matching_schemas`` parses it, so a bare
    year works too. An unknown year, or one before CS's first (2004), has none.
    """
    year = _diagnosis_year(date_of_diagnosis)
    if year is None:
        return None
    for key, algorithm in scoping.algorithms.items():
        if any(
            low <= year and (high is None or year <= high)
            for low, high in _select_ranges("year_dx", algorithm.years)
        ):
            return key
    return None


def accepted_histology_codes(
    scoping: ScopingDictionary, date_of_diagnosis: str | None
) -> frozenset[int] | None:
    """The 4-digit histology codes the diagnosis year's algorithm accepts, or None
    to accept every code when there's no algorithm (``diagnosis_algorithm``)."""
    algorithm = diagnosis_algorithm(scoping, date_of_diagnosis)
    return None if algorithm is None else _histology_codes(scoping.algorithms[algorithm].histology)


def item_years(scoping: ScopingDictionary, item_id: int) -> tuple[int | None, int | None]:
    """The first and last diagnosis year the item is collected for, None for an
    open end. An item the file doesn't list is collected in every year."""
    entry = scoping.items.get(item_id)
    return (None, None) if entry is None else _item_years(entry.years)


def item_applies_in_year(
    scoping: ScopingDictionary, item_id: int, date_of_diagnosis: str | None
) -> bool:
    """Whether the item is collected for the diagnosis year (``item_years``).

    As with the histology codes, an unknown year, or one before CS's first
    (2004), has no staging algorithm (``diagnosis_algorithm``), so every item
    applies: those years are out of scope, and some items' first years, such as
    Histology's 2001, would otherwise close them.
    """
    if diagnosis_algorithm(scoping, date_of_diagnosis) is None:
        return True
    year = _diagnosis_year(date_of_diagnosis)
    first, last = item_years(scoping, item_id)
    return (first is None or first <= year) and (last is None or year <= last)


def item_label(item_id: int, name: str | None) -> str:
    """An item as reasons name it: "Grade (440)", or "Item 440" without a name."""
    return f"{name} ({item_id})" if name else f"Item {item_id}"



def _join_labels(labels: Sequence[str]) -> str:
    return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"


def out_of_year_reason(
    scoping: ScopingDictionary,
    item_id: int,
    date_of_diagnosis: str | None,
    name: str | None = None,
    *,
    held_by: Sequence[tuple[int, str | None]] = (),
    replaced_by: Sequence[tuple[int, str | None]] = (),
    ssf: SsfMapping | None = None,
    ssf_disagreement: str | None = None,
) -> str | None:
    """Why the item isn't collected for the diagnosis year, or None when it is
    (``item_applies_in_year``).

    For example "Not collected for 2015 diagnoses: Grade Clinical (3843) applies
    from 2018." ``held_by`` lists the older equivalents that stand in for the
    item, and ``replaced_by`` the newer items an older equivalent stands in for,
    each as ``(item_id, name)``. The ones collected that year are named: "…;
    Grade (440) holds it for this year." or "…; Grade Clinical (3843) and Grade
    Pathological (3844) replace it."

    ``ssf`` is the case's CS Site-Specific Factor for the item
    (``ssf_equivalent``), named with them: "…; CS SSF2 (2890, FIGO Stage) holds
    it for this year." ``ssf_disagreement`` says instead why no SSF could be
    chosen.
    """
    if item_applies_in_year(scoping, item_id, date_of_diagnosis):
        return None
    first, last = item_years(scoping, item_id)
    if first is not None and last is not None:
        span = f"from {first} through {last}"
    else:
        span = f"from {first}" if first is not None else f"through {last}"
    reason = (
        f"Not collected for {_diagnosis_year(date_of_diagnosis)} diagnoses: "
        f"{item_label(item_id, name)} applies {span}"
    )
    held = [
        item_label(equivalent, equivalent_name)
        for equivalent, equivalent_name in held_by
        if item_applies_in_year(scoping, equivalent, date_of_diagnosis)
    ]
    if ssf is not None:
        held.append(_ssf_label(ssf))
    replaced = [
        item_label(equivalent, equivalent_name)
        for equivalent, equivalent_name in replaced_by
        if item_applies_in_year(scoping, equivalent, date_of_diagnosis)
    ]
    for labels, one, several in (
        (held, "holds it for this year", "hold it for this year"),
        (replaced, "replaces it", "replace it"),
    ):
        if labels:
            reason += f"; {_join_labels(labels)} {one if len(labels) == 1 else several}"
    if ssf_disagreement:
        reason += f"; the candidate schemas disagree on the CS Site-Specific Factor that holds it: {ssf_disagreement}"
    return reason + "."


def _ssf_label(mapping: SsfMapping) -> str:
    return f"{mapping.label} ({mapping.item}, {mapping.name})"


def ssf_items(scoping: ScopingDictionary, date_of_diagnosis: str | None) -> frozenset[int]:
    """The requested items that some schema of the diagnosis year's algorithm
    holds in a CS Site-Specific Factor (the schemas' ``ssf``).

    Empty unless the year picks CS or TNM (``diagnosis_algorithm``).
    """
    algorithm = diagnosis_algorithm(scoping, date_of_diagnosis)
    if algorithm not in _SSF_ALGORITHMS:
        return frozenset()
    return frozenset(
        item_id
        for key, schema in scoping.schemas.items()
        if key.startswith(f"{algorithm}/")
        for item_id in schema.ssf
    )


def ssf_offered_codes(mapping: SsfMapping) -> dict[str, str]:
    """The SSF's codes, without the rows CS kept only for converted data.

    Those rows, like ovary's "888: OBSOLETE DATA CONVERTED V0200 / See code 988",
    describe codes a later CS version replaced, so new abstraction never uses
    them.
    """
    return {
        code: description
        for code, description in mapping.codes.items()
        if not _OBSOLETE_CODE.match(description)
    }


def _ssf_signature(mapping: SsfMapping) -> tuple:
    codes = ssf_offered_codes(mapping)
    return (
        mapping.item,
        _normalize_text(mapping.name),
        tuple(sorted((code, _normalize_text(description)) for code, description in codes.items())),
    )


def _schema_list(keys: Sequence[str], shown: int = 4) -> str:
    if len(keys) <= shown:
        return _join_labels(list(keys))
    return f"{', '.join(keys[:shown])} and {len(keys) - shown} others"


def ssf_equivalent(
    scoping: ScopingDictionary, item_id: int, facts: CaseFacts | None
) -> SsfChoice:
    """The CS Site-Specific Factor that holds a requested item for the case.

    The candidates are the schemas of the diagnosis year's algorithm that the
    case matches (``matching_schemas``). A schema holds the item when its
    ``ssf`` maps it and the mapping's years, if it has them, include the year.

    * substitute: every candidate holds it in the same SSF item, with the same
      name and offered codes (``ssf_offered_codes``). ``mapping`` is that SSF.
    * disagree: candidates hold it in different SSFs, or some hold it and some
      don't. ``reason`` says which does what. A discriminator usually settles
      it.
    * none: no candidate holds it that year. So does an EOD, unknown or
      pre-2004 year, which never substitutes.
    """
    date_of_diagnosis = facts.date_of_diagnosis if facts else None
    algorithm = diagnosis_algorithm(scoping, date_of_diagnosis)
    if algorithm not in _SSF_ALGORITHMS:
        return SsfChoice(outcome="none")
    year = _diagnosis_year(date_of_diagnosis)
    candidates = [key for key in matching_schemas(scoping, facts) if key.startswith(f"{algorithm}/")]
    held: dict[str, SsfMapping | None] = {}
    for key in candidates:
        mapping = scoping.schemas[key].ssf.get(item_id)
        if mapping is not None and mapping.years is not None:
            first, last = _item_years(mapping.years)
            if (first is not None and year < first) or (last is not None and year > last):
                mapping = None
        held[key] = mapping
    mappings = [mapping for mapping in held.values() if mapping is not None]
    if not mappings:
        return SsfChoice(outcome="none", schemas=candidates)
    signatures = {None if mapping is None else _ssf_signature(mapping) for mapping in held.values()}
    if len(signatures) == 1:
        return SsfChoice(outcome="substitute", schemas=candidates, mapping=mappings[0])

    by_signature: dict[tuple | None, list[str]] = {}
    labels: dict[tuple | None, str] = {}
    for key, mapping in held.items():
        signature = None if mapping is None else _ssf_signature(mapping)
        by_signature.setdefault(signature, []).append(key)
        if mapping is not None:
            labels.setdefault(signature, _ssf_label(mapping))
    parts = []
    for signature, keys in by_signature.items():
        verb = "holds" if len(keys) == 1 else "hold"
        held_in = f"it in {labels[signature]}" if signature is not None else "no SSF for it"
        parts.append(f"{_schema_list(keys)} {verb} {held_in}")
    return SsfChoice(outcome="disagree", schemas=candidates, reason="; ".join(parts))


def ssf_variable(mapping: SsfMapping) -> VariableInfo:
    """The SSF as the extractor is offered it: the schema's own name, notes and
    codes (``ssf_offered_codes``), never the NAACCR dictionary's generic "CS
    Site-Specific Factor N". The orchestrator adds the dictionary's format and
    length when it scopes the group."""
    return VariableInfo(
        item_id=mapping.item,
        name=mapping.name,
        description=mapping.definition.strip() or f"{mapping.name}, recorded in {mapping.label}.",
        valid_codes=ssf_offered_codes(mapping),
    )


def _discriminator_values(
    schema: StagingSchema, key: str, values: Mapping[str, frozenset]
) -> frozenset[int] | None:
    """The discriminator values the schema's rows matching the case accept, or
    None when one of those rows doesn't constrain it."""
    accepted: set[int] = set()
    for row in schema.select:
        if not _row_matches(row, values):
            continue
        if key not in row:
            return None
        for low, high in _select_ranges(key, row[key]):
            if high is None:
                return None
            accepted.update(range(low, high + 1))
    return frozenset(accepted)


def _shared_discriminator(
    scoping: ScopingDictionary, item_id: int, schemas: Sequence[str]
) -> SchemaDiscriminator | None:
    """The table the schemas that read the discriminator hold, or None if they
    hold different ones. The generator gives schemas a case can match together
    the same table."""
    tables = {
        scoping.schemas[key].discriminators[item_id].model_dump_json(): scoping.schemas[key].discriminators[item_id]
        for key in schemas
        if item_id in scoping.schemas[key].discriminators
    }
    return next(iter(tables.values())) if len(tables) == 1 else None


def discriminator_choice(
    scoping: ScopingDictionary,
    item_id: int,
    facts: CaseFacts | None,
    *,
    schemas: Sequence[str] | None = None,
) -> DiscriminatorChoice:
    """Whether the case's candidate schemas differ by a discriminator (3926,
    3927 or 2879, ``DISCRIMINATOR_KEYS``).

    The candidates are the schemas the case matches (``matching_schemas``) with
    the discriminator itself unknown, or ``schemas`` when given. They differ
    when the values their matching rows accept for it aren't all the same,
    counting a row that leaves it open as accepting anything, so some value
    would leave out a candidate. One candidate, or none, never needs it. Year is
    one of the facts, so a 2012 case's candidates are CS and TNM schemas, which
    never read Schema Discriminator 1 or 2, and a 2020 case's are EOD schemas,
    which never read CS Site-Specific Factor 25.
    """
    key = DISCRIMINATOR_KEYS[item_id]
    values = {name: known for name, known in _case_values(scoping, facts).items() if name != key}
    candidates = list(schemas) if schemas is not None else _matching(scoping, values)
    accepted = {
        schema_key: _discriminator_values(scoping.schemas[schema_key], key, values)
        for schema_key in candidates
    }
    if len(set(accepted.values())) < 2:
        return DiscriminatorChoice(needed=False, schemas=candidates)
    return DiscriminatorChoice(
        needed=True,
        schemas=candidates,
        discriminator=_shared_discriminator(scoping, item_id, candidates),
    )


def discriminator_not_needed_reason(choice: DiscriminatorChoice, label: str) -> str:
    """Why a discriminator the candidates don't differ by isn't extracted."""
    if not choice.schemas:
        return f"No staging schema matches the case, so {label} isn't needed."
    if len(choice.schemas) == 1:
        return f"Only {choice.schemas[0]} matches the case, so {label} isn't needed."
    return (
        f"The schemas the case matches ({_schema_list(choice.schemas)}) don't differ "
        f"by {label}, so it isn't needed."
    )


def discriminator_variable(discriminator: SchemaDiscriminator, item_id: int) -> VariableInfo:
    """The discriminator as the extractor is offered it: the table's name, notes
    and codes, overlaid on the NAACCR entry like an SSF (``ssf_variable``).

    EOD names its input just "Schema Discriminator 1" and titles the table with
    what it separates; CS does the reverse, so the name uses whichever says
    more. The blank code says the discriminator isn't needed, which can't hold
    when it's extracted, so it isn't offered. Lines of its text that say more,
    like the occult head and neck table's site guidance, join the description.
    """
    if discriminator.title.startswith(discriminator.name):
        name = discriminator.title
    else:
        name = f"{discriminator.name} ({discriminator.title})"
    description = discriminator.definition.strip() or f"{name}, which picks the staging schema."
    extra = [
        line.strip()
        for line in discriminator.codes.get("", "").splitlines()
        if line.strip() and not _NOT_NEEDED.search(line)
    ]
    if extra:
        description += "\n\nWhen it isn't needed, the table also says:\n" + "\n".join(extra)
    return VariableInfo(
        item_id=item_id,
        name=name,
        description=description,
        valid_codes={code: text for code, text in discriminator.codes.items() if code.strip()},
    )


def discriminators_separating(
    scoping: ScopingDictionary,
    facts: CaseFacts | None,
    schemas: Sequence[str],
    item_ids: Iterable[int],
) -> list[int]:
    """Which of the discriminator ``item_ids`` the ``schemas`` differ by
    (``discriminator_choice``), in the order given."""
    return [
        item_id
        for item_id in item_ids
        if item_id in DISCRIMINATOR_KEYS
        and discriminator_choice(scoping, item_id, facts, schemas=schemas).needed
    ]


def _accepted(pair: str, accepted: Collection[int] | None) -> bool:
    return accepted is None or int(pair[:4]) in accepted


def histology_site_group(scoping: ScopingDictionary, primary_site: str | None) -> str:
    """The key of the histology site group for a coded Primary Site.

    A Primary Site that isn't coded (NOT_FOUND and ERROR leave it None), is
    malformed, or isn't one of the tissues' codes gets the group that holds C809,
    the unknown primary.
    """
    coded = (primary_site or "").strip().upper()
    code = coded.replace(".", "")
    if not _CODED_PRIMARY.fullmatch(coded) or code not in _all_site_codes(scoping):
        code = UNKNOWN_PRIMARY_SITE
    for key, group in scoping.histology.items():
        if _in_sites(code, group.sites):
            return key
    raise ValueError(f"No histology group holds {code}.")


def histology_family_listing(
    scoping: ScopingDictionary, group: str, *, accepted: Collection[int] | None = None
) -> dict[str, str]:
    """The site group's families for the family step, keyed by their 3 digits.

    Each is the list's label and its members' abbreviated terms, since the
    labels alone are terse. The full v25 descriptions wait for the code step.
    With ``accepted`` (``accepted_histology_codes``), only the terms of accepted
    codes are shown, and a family left with none is dropped.
    """
    listing: dict[str, str] = {}
    for number, family in scoping.histology[group].families.items():
        terms = dict.fromkeys(
            code.term for pair, code in family.codes.items() if _accepted(pair, accepted)
        )
        if terms:
            listing[number] = f"{family.label}: {'; '.join(terms)}"
    return listing


def histology_code_listing(
    scoping: ScopingDictionary,
    group: str,
    families: Iterable[str],
    *,
    accepted: Collection[int] | None = None,
) -> dict[str, str]:
    """The chosen families' code/behavior pairs with their v25 descriptions, in code order.

    Several families give the union of their pairs, keeping only accepted codes
    when ``accepted`` is given. A family the group doesn't have raises, and so
    does choosing none, which the fallback handles instead.
    """
    site_group = scoping.histology[group]
    chosen = list(dict.fromkeys(families))
    if not chosen:
        raise ValueError("Choose at least one histology family.")
    unknown = [number for number in chosen if number not in site_group.families]
    if unknown:
        raise ValueError(f"Histology group {group} has no families {unknown}.")
    return dict(sorted(
        (pair, code.description)
        for number in chosen
        for pair, code in site_group.families[number].codes.items()
        if _accepted(pair, accepted)
    ))


def histology_pair_listed(scoping: ScopingDictionary, group: str, pair: str) -> bool:
    """Whether the site group lists the code/behavior pair, like '8500/3'."""
    pair = pair.strip()
    family = scoping.histology[group].families.get(pair[:3])
    return family is not None and pair in family.codes


def histology_block_listing(
    scoping: ScopingDictionary,
    pairs: Mapping[str, str],
    *,
    accepted: Collection[int] | None = None,
) -> dict[str, dict[str, str]]:
    """Every pair grouped by ICD-O-3 morphology block, for the fallback.

    ``pairs`` maps item 522's code/behavior pairs to their descriptions
    (``histology_pairs``). With ``accepted``, only pairs whose code it holds are
    kept. Blocks keep the file's order, and one no kept pair falls in is left
    out. A malformed pair, or one not in exactly one block, raises.
    """
    ranges = [(key, *_block_range(key)) for key in scoping.blocks]
    grouped: dict[str, dict[str, str]] = {key: {} for key in scoping.blocks}
    for pair, description in sorted(pairs.items()):
        if not _PAIR.fullmatch(pair):
            raise ValueError(f"{pair!r} isn't a code/behavior pair like '8500/3'.")
        holding = [key for key, low, high in ranges if low <= int(pair[:3]) <= high]
        if len(holding) != 1:
            raise ValueError(f"{pair} is in morphology blocks {holding}, not exactly one.")
        if _accepted(pair, accepted):
            grouped[holding[0]][pair] = description
    return {key: codes for key, codes in grouped.items() if codes}


def check_histology_pairs(scoping: ScopingDictionary, pairs: Iterable[str]) -> None:
    """Fail unless every pair the site groups list is one of item 522's pairs, and
    each of item 522's pairs is in exactly one morphology block.

    Offering only item 522's pairs keeps every offered value valid; the list has
    pairs v25 doesn't, which the generator drops.
    """
    pairs = set(pairs)
    unlisted = sorted(
        {
            pair
            for group in scoping.histology.values()
            for family in group.families.values()
            for pair in family.codes
        }
        - pairs
    )
    if unlisted:
        raise ValueError(f"Histology groups list pairs that aren't in item 522: {unlisted}.")
    histology_block_listing(scoping, dict.fromkeys(pairs, ""))


def build_histology_scope(
    scoping: ScopingDictionary,
    primary_site: str | None,
    pairs: Mapping[str, str],
    date_of_diagnosis: str | None = None,
) -> HistologyScope:
    """Everything the extractor's histology steps offer for a coded Primary Site.

    The site group's families, with the list's terms for the family step and
    their v25 descriptions for the code step, plus every one of ``pairs``
    (``histology_pairs``) grouped by ICD-O-3 block for the fallback. An uncoded
    Primary Site gets the unknown primary's group (``histology_site_group``).

    Both paths offer only the codes the diagnosis year's algorithm accepts
    (``accepted_histology_codes``), dropping families and blocks left empty; a
    group left with no families goes straight to the blocks. An unknown year,
    or one before 2004, offers every code. Codes are kept or dropped whole, so
    whether an offered code is on the site's list doesn't change.
    """
    accepted = accepted_histology_codes(scoping, date_of_diagnosis)
    group = histology_site_group(scoping, primary_site)
    site_group = scoping.histology[group]
    families = histology_family_listing(scoping, group, accepted=accepted)
    block_codes = histology_block_listing(scoping, pairs, accepted=accepted)
    return HistologyScope(
        site_group=group,
        site_group_label=site_group.label,
        families=families,
        family_codes={
            number: histology_code_listing(scoping, group, [number], accepted=accepted)
            for number in families
        },
        blocks={key: scoping.blocks[key].label for key in block_codes},
        block_codes=block_codes,
    )
