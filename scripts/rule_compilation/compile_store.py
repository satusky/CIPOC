"""Compile the CoC STORE 2024 manual into the rule store, item by item.

STORE fits neither existing driver. ``compile_pending`` derives its queue from
per-variable ``rule_source`` entries and selects a chapter by heading subtree;
``compile_summary_stage`` carves one item into ~100 site chapters. STORE is the
inverse of the latter: ~200 *items*, each its own heading subtree, none of them
site-partitioned. ``store_index`` does the resolution; this script drives it:

    index (pure)  →  tag (LLM)  →  validate (pure)  →  report + write

One item compiles to one file, because ``compile_sections`` forces a single
``item_ids`` list across a batch — batching two items into one call would label
each one's rules with the other's item. Each unit is forced to its item rather
than letting the model infer it: the section is the item's own chapter, so there
is nothing to infer and every inference is a way to be wrong.

Only the items listed in ``store_index.TARGETS`` compile — the ones this project
extracts that no already-compiled manual covers. See that module for the items
STORE 2024 does not document at all.

Like the other drivers, nothing runs without ``--run``, an item whose output
already exists is skipped so an interrupted run resumes, and a failing item does
not abort the rest.

    python -m scripts.rule_compilation.compile_store                    # show the plan
    python -m scripts.rule_compilation.compile_store --run              # compile everything
    python -m scripts.rule_compilation.compile_store --run --group tnm_staging
    python -m scripts.rule_compilation.compile_store --run --item 1001 1002
    python -m scripts.rule_compilation.compile_store --run --limit 1    # cost probe

Every item still writes a ``<stem>.review.txt`` that must be eyeballed before the
units are trusted; this script reports counts but does not judge them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .compile_manual import _load_llm, compile_sections, write_outputs
from .store_index import (
    ABSENT,
    MANIFEST_ENTRY,
    MANUAL,
    SOURCE,
    TARGETS,
    build_index,
)

RULES_DIR = Path("documents/rules")
DATA_DICTIONARY = Path("documents/manuals/naaccr_data_dictionary_v25.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=SOURCE, help="Source markdown path.")
    parser.add_argument("--rules-dir", type=Path, default=RULES_DIR)
    parser.add_argument("--data-dictionary", type=Path, default=DATA_DICTIONARY)
    parser.add_argument("--run", action="store_true", help="Execute the compiles instead of printing the plan.")
    parser.add_argument("--item", nargs="*", type=int, default=None, help="Compile only these NAACCR item IDs.")
    parser.add_argument("--group", nargs="*", default=None, help="Compile only these variable_groups.json group_ids, e.g. tnm_staging.")
    parser.add_argument("--limit", type=int, default=None, help="Compile at most this many items; useful as a cost probe.")
    parser.add_argument("--force", action="store_true", help="Recompile items whose output file already exists.")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort at the first failing item.")
    parser.add_argument("--agent", default="note_scanner", help="Config agent whose LLM settings to use for tagging.")
    args = parser.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"Source markdown {args.source} does not exist.")

    try:
        targets = build_index(args.source, items=args.item)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.group:
        wanted = set(args.group)
        targets = [t for t in targets if t[0].group_id in wanted]
        missing = wanted - {t.group_id for t in TARGETS}
        if missing:
            raise SystemExit(f"No STORE targets for group(s) {sorted(missing)}.")
    if args.limit is not None:
        targets = targets[: args.limit]

    if not targets:
        print("Nothing to compile after filtering.")
        return 0

    print(f"{len(targets)} item(s) from {args.source} -> {args.rules_dir / MANUAL}/\n")
    print(f"  {'item':>5}  {'stem':30s} {'group':22s} {'secs':>4s} {'dx_date_min':12s} heading")
    for item, sections in targets:
        print(
            f"  {item.item_id:>5}  {item.stem:30s} {item.group_id:22s} "
            f"{len(sections):4d} {item.dx_date_min:12s} {item.heading[:44]}"
        )
    if args.item is None and args.group is None:
        print(f"\n  Not in STORE 2024 ({len(ABSENT)} targeted item(s)):")
        for item_id, why in sorted(ABSENT.items()):
            print(f"    {item_id:>5}  {why}")
    print()

    if not args.run:
        print("Re-run with --run to compile. Each item is one or more serial LLM calls.")
        return 0

    llm = _load_llm(args.agent)
    done: list[tuple[str, str]] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    total_in = total_out = 0
    started = time.monotonic()

    for n, (item, sections) in enumerate(targets, 1):
        out_path = args.rules_dir / MANUAL / f"{item.stem}.json"
        if out_path.exists() and not args.force:
            print(f"[{n}/{len(targets)}] skip {item.stem} -- {out_path} exists (--force to recompile)")
            skipped.append(item.stem)
            continue

        print(f"[{n}/{len(targets)}] {item.stem} (item {item.item_id}, {len(sections)} section(s))")
        try:
            units, validations, usage = compile_sections(
                manual=MANUAL,
                site_group=item.stem,
                sections=sections,
                source_path=args.source,
                data_dictionary_path=args.data_dictionary,
                default_applicability=item.applicability,
                llm=llm,
                item_ids=[item.item_id],
                show_progress=True,
            )
        except Exception as error:  # one bad item must not lose the rest of the run
            failed.append((item.stem, f"{type(error).__name__}: {error}"))
            print(f"!! FAILED: {item.stem} -- {type(error).__name__}: {error}")
            if args.stop_on_error:
                break
            continue

        _, report_path, _, accepted = write_outputs(
            units, validations, usage,
            rules_dir=args.rules_dir, manual=MANUAL, site_group=item.stem,
            source_path=args.source, manifest_defaults=MANIFEST_ENTRY,
        )
        quarantined = len(units) - len(accepted)
        if usage["usage_reported"]:
            total_in += usage["input_tokens"]
            total_out += usage["output_tokens"]
        done.append((item.stem, f"{len(accepted)} units, {quarantined} quarantined -> {report_path}"))
        print(f"    {done[-1][1]}")

    elapsed = time.monotonic() - started
    print(f"\n{'=' * 78}\nSummary -- {len(done)} compiled, {len(skipped)} skipped, "
          f"{len(failed)} failed in {elapsed / 60:.1f} min\n{'=' * 78}")
    for stem, info in done:
        print(f"  ok    {stem[:40]:40s} {info}")
    for stem in skipped:
        print(f"  skip  {stem[:40]:40s} already compiled")
    for stem, why in failed:
        print(f"  FAIL  {stem[:40]:40s} {why}")

    if total_in or total_out:
        print(f"\n  Total tokens: {total_in + total_out:,} (in {total_in:,} / out {total_out:,})")
    elif done:
        print("\n  Total tokens: endpoint returned no usage metadata for these compiles.")

    if done:
        print(f"\nReview each {args.rules_dir / MANUAL}/<stem>.review.txt before trusting the units.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
