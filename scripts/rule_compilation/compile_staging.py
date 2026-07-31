"""Ingest SEER*RSA staging data into the rule store, schema by schema.

Unlike the other drivers in this package there is no LLM in this pipeline:

    fetch (network)  →  build units (pure)  →  validate (pure)  →  report + write

SEER*RSA publishes per-schema — i.e. per site/histology group — code tables and
registrar notes for the NAACCR items each schema collects. That is the dimension
the data dictionary lacks, and the reason several variables reach the extractor
today with no enforceable code set at all (Grade Clinical/Pathological carry an
empty ``Code Descriptions``; Summary Stage and the nodes items carry a
site-agnostic full set). The emitted units reach the extractor through the
existing path — ``build_variable_group`` → ``scope_coding_context`` →
``reduce_valid_codes`` → ``variable.valid_codes`` — with no runtime change.

One schema compiles to one file. Like the other drivers, nothing runs without
``--run``, a schema whose output already exists is skipped so an interrupted run
resumes, and a failing schema does not abort the rest.

    python -m scripts.rule_compilation.compile_staging                      # plan, both algorithms
    python -m scripts.rule_compilation.compile_staging --algorithm eod      # plan, EOD only
    python -m scripts.rule_compilation.compile_staging --algorithm eod --run
    python -m scripts.rule_compilation.compile_staging --algorithm eod --run --schema breast
    python -m scripts.rule_compilation.compile_staging --run --item 3843 3844
    python -m scripts.rule_compilation.compile_staging --run --cache-dir .cache/staging

Every schema still writes a ``<stem>.review.txt`` that must be eyeballed before
the units are trusted; this script reports counts but does not judge them. There
is no ``<stem>.usage.json`` — no LLM, no tokens to record.

Note on TNM 2.1: it documents NAACCR items 880/890/900/910/940/950/960/970, the
2016-2017 UICC-7 TNM items. ``config/variable_groups.json`` targets 1001-1004 /
1011-1014 (AJCC 8th edition, 2018+), which SEER*RSA does not carry at all. So a
TNM run emits units only for the handful of items the two share (Behavior,
Regional Nodes Positive) until those legacy items are added to a variable group.
Remapping 940 onto 1001 would not help: ``reduce_valid_codes`` intersects
against the dictionary set for the item, and the dictionary carries only a
two-code stub for 1001, so the intersection is empty and the codes never arrive.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter

from cipoc.models import RuleUnit

from .compile_manual import upsert_manifest
from .staging_fetch import StagingRelease, fetch_algorithm
from .staging_index import ALGORITHMS, EXCLUDE_ITEMS, VARIABLE_GROUPS, schema_stem, target_items
from .staging_units import build_schema_units
from .staging_validate import validate_staging_units
from .validate import UnitValidation

RULES_DIR = Path("documents/rules")
DATA_DICTIONARY = Path("documents/manuals/naaccr_data_dictionary_v25.json")

_RULE_LIST_ADAPTER = TypeAdapter(list[RuleUnit])


def build_report(
    units: list[RuleUnit],
    results: list[UnitValidation],
    *,
    schema_id: str,
    source: str,
    sample_size: int = 4,
) -> str:
    """Review report for one ingested schema.

    ``report.build_report`` cannot be reused: it renders each unit beside the
    anchored region of a source markdown file, and this pipeline has no markdown
    and no anchor line ranges. What matters here instead is what was emitted per
    item and what the code tables actually say, so a reviewer can see at a glance
    that a site-specific set landed on the right item.
    """
    by_id = {u.rule_id: u for u in units}
    accepted = [r for r in results if r.ok]
    quarantined = [r for r in results if not r.ok]

    lines = [
        "=" * 72,
        "SEER*RSA STAGING INGEST REVIEW REPORT",
        "=" * 72,
        f"Schema: {schema_id}",
        f"Source: {source}",
        f"Units emitted: {len(units)}  |  accepted: {len(accepted)}  "
        f"|  quarantined: {len(quarantined)}",
    ]

    accepted_units = [by_id[r.rule_id] for r in accepted if r.rule_id in by_id]

    kinds = Counter(u.kind for u in accepted_units)
    lines.append("\nAccepted units by kind:")
    for kind, count in sorted(kinds.items()):
        lines.append(f"  {kind:14s} {count}")

    items = Counter(item for u in accepted_units for item in u.item_ids)
    lines.append("\nAccepted units by NAACCR item:")
    if not items:
        lines.append("  (none item-specific)")
    for item_id, count in sorted(items.items()):
        lines.append(f"  {item_id:>6} {count}")

    # One per selection-row group; a multi-row schema repeats each unit across
    # them, so the reviewer needs to see all of them, not the first.
    predicates: list[dict] = []
    for unit in accepted_units:
        dumped = unit.applies_to.model_dump(exclude_none=True) if unit.applies_to else None
        if dumped not in predicates:
            predicates.append(dumped)
    lines.append(f"\nApplicability predicates carried by this schema's units ({len(predicates)}):")
    for predicate in predicates:
        lines.append(f"  {predicate}")

    lines.append("\n" + "-" * 72)
    lines.append(f"QUARANTINED ({len(quarantined)})")
    lines.append("-" * 72)
    if not quarantined:
        lines.append("(none)")
    for result in quarantined:
        unit = by_id.get(result.rule_id)
        lines.append(f"\n[{result.rule_id}]  kind={unit.kind if unit else '?'}  "
                     f"items={unit.item_ids if unit else '?'}")
        for error in result.errors:
            lines.append(f"  - {error}")

    tables = [u for u in accepted_units if u.kind == "code_table"]
    lines.append("\n" + "-" * 72)
    lines.append(f"SAMPLED CODE TABLES (up to {sample_size} of {len(tables)})")
    lines.append("-" * 72)
    if not tables:
        lines.append("(none)")
    step = max(1, len(tables) // sample_size) if tables else 1
    for unit in tables[::step][:sample_size]:
        lines.append(f"\n[{unit.rule_id}]  items={unit.item_ids}  anchor={unit.anchor}")
        lines.append(f"  {' > '.join(unit.section_path)}")
        for code, description in list((unit.codes or {}).items())[:12]:
            summary = " ".join(description.split())
            lines.append(textwrap.shorten(f"    {code}: {summary}", width=110, placeholder=" …"))
        if len(unit.codes or {}) > 12:
            lines.append(f"    … {len(unit.codes) - 12} further code(s)")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


def compile_schema(
    release: StagingRelease,
    schema_id: str,
    *,
    items: frozenset[int],
    rules_dir: Path,
    data_dictionary_path: Path,
) -> tuple[Path, Path, list[RuleUnit], list[RuleUnit]]:
    """Build, validate, and write one schema. Returns (out, report, all, accepted)."""
    algorithm = release.algorithm
    schema = release.schema(schema_id)
    tables = release.tables_for_schema(schema)

    units = build_schema_units(schema, tables, algorithm, items=items)
    validations = validate_staging_units(units, data_dictionary_path=data_dictionary_path)
    accepted_ids = {r.rule_id for r in validations if r.ok}
    accepted = [u for u in units if u.rule_id in accepted_ids]

    stem = schema_stem(schema_id)
    out_path = rules_dir / algorithm.manual / f"{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_RULE_LIST_ADAPTER.dump_json(accepted, indent=2) + b"\n")

    # upsert_manifest types source_path as a Path only because every other
    # compiler has a markdown file; a str passes through str() unchanged, which
    # is what we want here — the provenance is a release URL, and Path() would
    # collapse its '//'.
    upsert_manifest(rules_dir, algorithm.manual, algorithm.url, defaults=algorithm.manifest_entry)

    report_path = rules_dir / algorithm.manual / f"{stem}.review.txt"
    report_path.write_text(
        build_report(units, validations, schema_id=schema_id, source=algorithm.url)
    )
    return out_path, report_path, units, accepted


def _plan(release: StagingRelease, schema_ids: list[str], items: frozenset[int], rules_dir: Path) -> None:
    algorithm = release.algorithm
    print(f"{len(schema_ids)} schema(s) from {algorithm.url}"
          f" -> {rules_dir / algorithm.manual}/\n")
    print(f"  {'stem':38s} {'items':>5s} {'tables':>6s}  in-scope NAACCR items")
    covered: Counter[int] = Counter()
    for schema_id in schema_ids:
        schema = release.schema(schema_id)
        in_scope = [
            i for i in schema.get("inputs") or []
            if i.get("naaccr_item") in items and i.get("table")
        ]
        covered.update(i["naaccr_item"] for i in in_scope)
        listed = ", ".join(str(i) for i in sorted({i["naaccr_item"] for i in in_scope}))
        print(f"  {schema_stem(schema_id)[:38]:38s} {len(in_scope):5d} "
              f"{len({i['table'] for i in in_scope}):6d}  {listed[:60]}")
    print(f"\n  Item coverage across these schemas ({len(covered)} item(s)):")
    for item_id, count in sorted(covered.items()):
        print(f"    {item_id:>6}  {count} schema(s)")
    unreached = sorted(items - set(covered))
    if unreached:
        print(f"\n  Targeted but absent from {algorithm.name}: {unreached}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algorithm", nargs="*", choices=sorted(ALGORITHMS), default=None,
                        help="Algorithms to ingest; default all.")
    parser.add_argument("--rules-dir", type=Path, default=RULES_DIR)
    parser.add_argument("--data-dictionary", type=Path, default=DATA_DICTIONARY)
    parser.add_argument("--variable-groups", type=Path, default=VARIABLE_GROUPS)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory to cache release ZIPs in, so re-runs need no network.")
    parser.add_argument("--run", action="store_true",
                        help="Execute the ingest instead of printing the plan.")
    parser.add_argument("--schema", nargs="*", default=None, help="Compile only these schema ids.")
    parser.add_argument("--item", nargs="*", type=int, default=None,
                        help="Emit code tables only for these NAACCR item IDs, overriding the "
                             "set derived from config/variable_groups.json.")
    parser.add_argument("--limit", type=int, default=None, help="Compile at most this many schemas.")
    parser.add_argument("--force", action="store_true",
                        help="Recompile schemas whose output file already exists.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort at the first failing schema.")
    args = parser.parse_args(argv)

    if args.run and not args.data_dictionary.exists():
        raise SystemExit(f"Data dictionary {args.data_dictionary} does not exist.")

    try:
        items = target_items(args.variable_groups, items=args.item)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not items:
        raise SystemExit("No NAACCR items in scope; nothing to ingest.")

    algorithms = [ALGORITHMS[name] for name in (args.algorithm or sorted(ALGORITHMS))]
    wanted_schemas = set(args.schema) if args.schema else None

    done: list[tuple[str, str]] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    started = time.monotonic()

    for algorithm in algorithms:
        print(f"{'=' * 78}\n{algorithm.manual} — {algorithm.manifest_entry['title']}\n{'=' * 78}")
        release = fetch_algorithm(algorithm, cache_dir=args.cache_dir)

        schema_ids = release.schema_ids()
        if wanted_schemas is not None:
            missing = wanted_schemas - set(schema_ids)
            schema_ids = [s for s in schema_ids if s in wanted_schemas]
            if missing and not schema_ids:
                raise SystemExit(f"No {algorithm.name} schema(s) named {sorted(missing)}.")
        if args.limit is not None:
            schema_ids = schema_ids[: args.limit]

        if not args.run:
            _plan(release, schema_ids, items, args.rules_dir)
            continue

        for n, schema_id in enumerate(schema_ids, 1):
            stem = schema_stem(schema_id)
            label = f"{algorithm.name}/{stem}"
            out_path = args.rules_dir / algorithm.manual / f"{stem}.json"
            if out_path.exists() and not args.force:
                print(f"[{n}/{len(schema_ids)}] skip {label} -- {out_path} exists (--force to recompile)")
                skipped.append(label)
                continue

            try:
                _, report_path, units, accepted = compile_schema(
                    release, schema_id,
                    items=items,
                    rules_dir=args.rules_dir,
                    data_dictionary_path=args.data_dictionary,
                )
            except Exception as error:  # one bad schema must not lose the rest of the run
                failed.append((label, f"{type(error).__name__}: {error}"))
                print(f"!! FAILED: {label} -- {type(error).__name__}: {error}")
                if args.stop_on_error:
                    break
                continue

            quarantined = len(units) - len(accepted)
            info = f"{len(accepted)} units, {quarantined} quarantined -> {report_path}"
            done.append((label, info))
            print(f"[{n}/{len(schema_ids)}] {label}: {info}")

    if not args.run:
        print("Re-run with --run to ingest. No LLM calls; one release download per algorithm.")
        if args.item is None:
            print(f"Items excluded by design: {sorted(EXCLUDE_ITEMS)} (see staging_index.EXCLUDE_ITEMS).")
        return 0

    elapsed = time.monotonic() - started
    print(f"\n{'=' * 78}\nSummary -- {len(done)} compiled, {len(skipped)} skipped, "
          f"{len(failed)} failed in {elapsed / 60:.1f} min\n{'=' * 78}")
    for label, why in failed:
        print(f"  FAIL  {label[:40]:40s} {why}")
    if skipped:
        print(f"  skip  {len(skipped)} schema(s) already compiled")
    if done:
        dirs = sorted({a.manual for a in algorithms})
        print(f"\nReview {', '.join(str(args.rules_dir / d) for d in dirs)}/<stem>.review.txt "
              "before trusting the units.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
