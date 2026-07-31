"""Turn one parsed SEER*RSA schema into ``RuleUnit`` records.

Pure: takes already-parsed dicts, returns units, touches no network and no ZIP,
so it is unit-testable against a trimmed fixture. ``staging_fetch`` supplies the
dicts; ``compile_staging`` validates and writes what comes back.

Three things are emitted per schema:

===========================================  ====================================
source                                       unit
===========================================  ====================================
an input's code table                        ``code_table``, ``item_ids=[naaccr_item]``
that table's registrar ``notes``             ``instruction``, ``item_ids=[naaccr_item]``
the schema's own ``notes``                   ``instruction``, ``item_ids=[]``
===========================================  ====================================

and every one of them carries an ``applies_to`` built from the schema-selection
table — the ICD-O-3 site and histology ranges that pick this schema in the first
place. That predicate is the whole point of the ingest: the data dictionary's
code sets are site-agnostic, and this is what narrows them to the case. A schema
whose selection table has several rows yields several predicates, and each unit
is emitted once per predicate; see ``schema_applicabilities`` for why one
unioned predicate is not equivalent.
"""

from __future__ import annotations

import itertools

from cipoc.models import RuleApplicability, RuleUnit

from .staging_index import StagingAlgorithm, anchor_for, rule_id

# Rendering cap for a code_table's ``text``. The codes themselves ride on
# ``codes`` and reach the extractor through reduce_valid_codes; a code_table's
# text is skipped entirely by assemble_coding_instructions, so it exists for the
# review report and the audit trail, and a full re-render would only bloat the
# store.
TEXT_MAX_CHARS = 1200
TEXT_MAX_CODES = 40

# Sentinel years the selection tables use for "unknown" (9998) and "blank"
# (9999). Either means the range has no upper bound worth encoding.
_YEAR_SENTINEL = 9998

_WILDCARD = "*"


def _split_entries(cell: str | None) -> list[str] | None:
    """Split a comma-delimited selection cell; None when it constrains nothing."""
    if cell is None:
        return None
    text = cell.strip()
    if not text or text == _WILDCARD:
        return None
    return [entry.strip() for entry in text.split(",") if entry.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _selection_rows(selection_table: dict) -> list[dict[str, str]]:
    keys = [column.get("key") for column in selection_table.get("definition") or []]
    return [dict(zip(keys, row)) for row in selection_table.get("rows") or []]


def _column(rows: list[dict[str, str]], key: str) -> list[str] | None:
    """Union of one selection column across rows; None if any row leaves it open.

    Only ever called over rows that already agree on every column but ``site``
    (see ``schema_applicabilities``), so the union is exact. Unioning across rows
    that differ on histology is not: it manufactures site×histology pairs the
    selection table never pairs, and a case matching one row's site and another
    row's histology would wrongly pick up the schema.
    """
    if not any(key in row for row in rows):
        return None
    collected: list[str] = []
    for row in rows:
        entries = _split_entries(row.get(key))
        if entries is None:
            return None  # a wildcard row makes the whole column unconstrained
        collected.extend(entries)
    return _dedupe(collected) or None


def _year_window(rows: list[dict[str, str]]) -> tuple[int | None, int | None]:
    """Fold the selection table's ``year_dx`` column into (min_year, max_year).

    Cells look like ``2018-2022``, ``2023-9998, 9999,``, ``2018-2020, 9999,`` or
    ``*``. A missing or wildcard cell leaves both bounds unset and the
    algorithm's own effective date stands.

    The two sentinels are not the same thing, and conflating them widens a closed
    schema into an open one. A *standalone* 9998 or 9999 token means the schema
    also selects a case whose diagnosis year is unknown or blank — it carries no
    upper bound and is dropped. A sentinel as the *high end of a range*
    (``2023-9998``) is what means "and everything after", i.e. no upper bound.
    """
    if not any("year_dx" in row for row in rows):
        return None, None

    lows: list[int] = []
    highs: list[int] = []
    open_ended = False
    for row in rows:
        tokens = _split_entries(row.get("year_dx"))
        if tokens is None:
            return None, None
        for token in tokens:
            low_text, separator, high_text = token.partition("-")
            if not low_text.isdigit():
                continue
            low = int(low_text)
            if low >= _YEAR_SENTINEL:
                continue  # 'unknown'/'blank' also selects; not a bound
            lows.append(low)
            if not separator:
                highs.append(low)
            elif high_text.isdigit() and int(high_text) < _YEAR_SENTINEL:
                highs.append(int(high_text))
            else:
                open_ended = True

    if not lows:
        return None, None
    return min(lows), (None if open_ended or not highs else max(highs))


def _predicate(rows: list[dict[str, str]], algorithm: StagingAlgorithm) -> RuleApplicability:
    """One ``RuleApplicability`` over rows that differ only in their site column."""
    min_year, max_year = _year_window(rows)
    return RuleApplicability(
        sites=_column(rows, "site"),
        histologies=_column(rows, "hist"),
        behaviors=_column(rows, "behavior"),
        dx_date_min=f"{min_year}-01-01" if min_year else algorithm.dx_date_min,
        dx_date_max=f"{max_year}-12-31" if max_year else None,
    )


def schema_applicabilities(
    schema: dict, tables: dict[str, dict], algorithm: StagingAlgorithm
) -> list[RuleApplicability]:
    """Build the case predicates for one schema — one per selection-row group.

    A schema-selection table is a disjunction of site×histology(×behavior×year)
    rows, and ``RuleApplicability`` is a conjunction, so a multi-row schema needs
    more than one predicate. Collapsing them into a single unioned predicate is
    what a conjunction cannot express faithfully: ``soft_tissue_other`` has 14
    rows, one supplying sites C470-C529 and another supplying histologies
    8000-8803, and the union therefore claims breast C509 + 8500 (ductal
    carcinoma) — a pair no row makes. Every emitted unit is repeated once per
    predicate so each carries an exact one.

    Rows that agree on everything but ``site`` *are* unioned, since a disjunction
    over one column is exactly what a list of ranges encodes. 104 of the 141 EOD
    schemas have a single row and are unaffected.

    Sex is read but deliberately not mapped: the selection tables encode it as a
    NAACCR code ('1', '2'), while ``CaseFacts.sex`` holds whatever the notes said
    ('female'), and ``matches_case`` compares the two as strings. A mapping would
    turn every sexed schema into a false exclusion. Only three selection rows in
    EOD and TNM combined use the column, and dropping it widens rather than
    narrows, so the safe direction is to leave it unset.
    """
    selection_id = schema.get("schema_selection_table")
    selection = tables.get(selection_id) if selection_id else None
    if selection is None:
        return [RuleApplicability(dx_date_min=algorithm.dx_date_min)]

    rows = _selection_rows(selection)
    if not rows:
        return [RuleApplicability(dx_date_min=algorithm.dx_date_min)]

    grouped: dict[tuple, list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in ("hist", "behavior", "year_dx"))
        grouped.setdefault(key, []).append(row)

    predicates: list[RuleApplicability] = []
    seen: set[str] = set()
    for group in grouped.values():
        predicate = _predicate(group, algorithm)
        fingerprint = predicate.model_dump_json()
        if fingerprint not in seen:
            seen.add(fingerprint)
            predicates.append(predicate)
    return predicates


def extract_code_table(table: dict, input_key: str) -> dict[str, str] | None:
    """Read a table as a code-to-description set, or None if it is not one.

    A SEER*RSA value set is a table whose single INPUT column is the schema
    input's own key and which carries at least one DESCRIPTION column. Everything
    else in ``tables/`` is machinery — ENDPOINT match tables (schema selection,
    year validation), multi-input mapping and JUMP tables the staging computation
    walks, and the bare range list that stands in for histology. None of those
    are value sets and none may become a ``code_table``.
    """
    definition = table.get("definition") or []
    if not definition:
        return None
    if definition[0].get("type") != "INPUT" or definition[0].get("key") != input_key:
        return None
    if sum(1 for column in definition if column.get("type") == "INPUT") != 1:
        return None
    if any(column.get("type") == "ENDPOINT" for column in definition):
        return None

    description_indices = [
        index for index, column in enumerate(definition)
        if column.get("type") == "DESCRIPTION"
    ]
    if not description_indices:
        return None

    codes: dict[str, str] = {}
    for row in table.get("rows") or []:
        if not row:
            continue
        code = (row[0] or "").strip()
        if not code or code == _WILDCARD or code in codes:
            continue
        parts = [
            (row[index] or "").strip()
            for index in description_indices
            if index < len(row) and (row[index] or "").strip()
        ]
        if not parts:
            continue
        codes[code] = "\n\n".join(parts)
    return codes or None


def _code_table_text(table: dict, codes: dict[str, str]) -> str:
    """Capped rendering of a code table, for the review report and the audit trail."""
    title = table.get("title") or table.get("name") or table.get("id") or "Code table"
    lines = [f"{title} ({len(codes)} codes)"]
    for code, description in itertools.islice(codes.items(), TEXT_MAX_CODES):
        summary = " ".join(description.split())
        lines.append(f"{code}: {summary}")
    body = "\n".join(lines)
    if len(body) > TEXT_MAX_CHARS:
        body = body[:TEXT_MAX_CHARS].rstrip() + " …"
    if len(codes) > TEXT_MAX_CODES:
        body += f"\n… {len(codes) - TEXT_MAX_CODES} further code(s); see the codes table."
    return body


def build_schema_units(
    schema: dict,
    tables: dict[str, dict],
    algorithm: StagingAlgorithm,
    *,
    items: frozenset[int] | set[int],
) -> list[RuleUnit]:
    """Compile one schema into its rule units, in schema-input order.

    ``tables`` must hold every table the schema's inputs name plus its selection
    table (``StagingRelease.tables_for_schema`` assembles exactly that). Inputs
    whose NAACCR item is outside ``items``, and tables that are not value sets,
    contribute nothing.

    Every unit is emitted once per predicate from ``schema_applicabilities`` —
    one for a single-row schema, several for a multi-row one. The duplicates
    differ only in ``applies_to``, and ``assemble_coding_instructions`` dedupes
    instruction text before it reaches a prompt, so the cost is store size only.
    """
    schema_id = schema["id"]
    manual = algorithm.manual
    applicabilities = schema_applicabilities(schema, tables, algorithm)
    schema_title = schema.get("title") or schema.get("name") or schema_id
    ordinals = itertools.count(1)

    units: list[RuleUnit] = []

    schema_notes = (schema.get("notes") or "").strip()
    if schema_notes:
        units.extend(
            RuleUnit(
                rule_id=rule_id(manual, schema_id, "schema", "instruction", next(ordinals)),
                source_doc=manual,
                section_path=[schema_title, "Schema notes"],
                anchor=anchor_for(algorithm, f"schemas/{schema_id}.json"),
                kind="instruction",
                item_ids=[],  # general principle for this schema, not one item
                applies_to=applicability,
                text=schema_notes,
            )
            for applicability in applicabilities
        )

    for schema_input in schema.get("inputs") or []:
        item_id = schema_input.get("naaccr_item")
        table_id = schema_input.get("table")
        if item_id not in items or not table_id:
            continue
        table = tables.get(table_id)
        if table is None:
            continue

        member = f"tables/{table_id}.json"
        table_title = table.get("title") or table.get("name") or schema_input.get("name") or table_id
        section_path = [schema_title, table_title]

        codes = extract_code_table(table, schema_input["key"])
        if codes:
            units.extend(
                RuleUnit(
                    rule_id=rule_id(manual, schema_id, table_id, "code_table", next(ordinals)),
                    source_doc=manual,
                    section_path=section_path,
                    anchor=anchor_for(algorithm, member),
                    kind="code_table",
                    item_ids=[item_id],
                    applies_to=applicability,
                    text=_code_table_text(table, codes),
                    codes=codes,
                )
                for applicability in applicabilities
            )

        notes = (table.get("notes") or "").strip()
        if notes:
            units.extend(
                RuleUnit(
                    rule_id=rule_id(manual, schema_id, table_id, "instruction", next(ordinals)),
                    source_doc=manual,
                    section_path=section_path,
                    anchor=anchor_for(algorithm, member),
                    kind="instruction",
                    item_ids=[item_id],
                    applies_to=applicability,
                    text=notes,
                )
                for applicability in applicabilities
            )

    return units
