"""Compile the Summary Stage 2018 manual into the rule store.

Summary Stage does not fit the ``compile_pending`` driver: that one derives its
work queue from per-variable ``rule_source`` entries and selects each chapter by
heading subtree. This manual is one item (NAACCR 764) documented by ~100 site
chapters plus a run of site-agnostic general chapters, and its site groups are
delimited by line range rather than by heading level — Breast and Bone are ``#``
headings while every other site chapter is ``##``. ``summary_stage_index`` does
that structural work; this script drives it:

    index (pure)  →  tag (LLM)  →  validate (pure)  →  report + write

The general chapters compile together into a single site-agnostic ``general``
group; every site chapter compiles into its own file, scoped to the ICD-O-3
topography and morphology ranges parsed from its own code preamble. Every unit
is forced to item 764 rather than letting the model infer it per section.

Like ``compile_pending``, nothing runs without ``--run``, a chapter whose output
already exists is skipped so an interrupted run resumes, and a failing chapter
does not abort the rest.

    python -m scripts.rule_compilation.compile_summary_stage                 # show the plan
    python -m scripts.rule_compilation.compile_summary_stage --run           # compile everything
    python -m scripts.rule_compilation.compile_summary_stage --run --scope general
    python -m scripts.rule_compilation.compile_summary_stage --run --chapter breast lung
    python -m scripts.rule_compilation.compile_summary_stage --run --limit 3  # cost probe

Every chapter still writes a ``<site_group>.review.txt`` that must be eyeballed
before the units are trusted; this script reports counts but does not judge them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cipoc.models import RuleApplicability

from .compile_manual import _load_llm, compile_sections, write_outputs
from .segment import segment_markdown
from .summary_stage_index import (
    EFFECTIVE_DX_DATE_MIN,
    MANIFEST_ENTRY,
    SUMMARY_STAGE_ITEM_ID,
    Chapter,
    build_index,
    sections_for,
)

MANUAL = "summary_stage_2018"
SOURCE = Path("documents/markdown/Summary-Stage_v3.3.md")
RULES_DIR = Path("documents/rules")
DATA_DICTIONARY = Path("documents/manuals/naaccr_data_dictionary_v25.json")

# The site-agnostic chapters (stage-code definitions, general coding instructions,
# ambiguous terminology) compile into one group under this stem rather than 16
# single-section files.
GENERAL_GROUP = "general"

GENERAL_APPLICABILITY = RuleApplicability(dx_date_min=EFFECTIVE_DX_DATE_MIN)


def plan(chapters: list[Chapter]) -> list[tuple[str, list[Chapter], RuleApplicability]]:
    """Group the index into compile targets: one general group plus one per site chapter."""
    general = [c for c in chapters if c.scope == "general"]
    targets: list[tuple[str, list[Chapter], RuleApplicability]] = []
    if general:
        targets.append((GENERAL_GROUP, general, GENERAL_APPLICABILITY))
    for chapter in chapters:
        if chapter.scope == "site":
            targets.append((chapter.site_group, [chapter], chapter.applicability))
    return targets


def _describe(applicability: RuleApplicability) -> str:
    sites = ",".join(applicability.sites or []) or "all sites"
    histologies = ",".join(applicability.histologies or []) or "all histologies"
    return f"{sites[:44]:46s} {histologies[:44]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=SOURCE, help="Source markdown path.")
    parser.add_argument("--rules-dir", type=Path, default=RULES_DIR)
    parser.add_argument("--data-dictionary", type=Path, default=DATA_DICTIONARY)
    parser.add_argument("--run", action="store_true", help="Execute the compiles instead of printing the plan.")
    parser.add_argument("--chapter", nargs="*", default=None, help="Compile only these site-group stems, e.g. breast lung.")
    parser.add_argument("--scope", choices=("general", "site"), default=None, help="Compile only the general chapters or only the site chapters.")
    parser.add_argument("--limit", type=int, default=None, help="Compile at most this many groups; useful as a cost probe.")
    parser.add_argument("--force", action="store_true", help="Recompile groups whose output file already exists.")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort at the first failing group.")
    parser.add_argument("--include-appendices", action="store_true", help="Also compile the appendix reference tables.")
    parser.add_argument("--agent", default="note_scanner", help="Config agent whose LLM settings to use for tagging.")
    args = parser.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"Source markdown {args.source} does not exist.")

    chapters = build_index(args.source, include_appendices=args.include_appendices)
    targets = plan(chapters)

    if args.scope == "general":
        targets = [t for t in targets if t[0] == GENERAL_GROUP]
    elif args.scope == "site":
        targets = [t for t in targets if t[0] != GENERAL_GROUP]
    if args.chapter:
        wanted = set(args.chapter)
        targets = [t for t in targets if t[0] in wanted]
        missing = wanted - {t[0] for t in targets}
        if missing:
            raise SystemExit(f"No such chapter(s) in the index: {sorted(missing)}")
    if args.limit is not None:
        targets = targets[: args.limit]

    if not targets:
        print("Nothing to compile after filtering.")
        return 0

    sections = segment_markdown(args.source.read_text(), max_heading_level=2)

    print(f"{len(targets)} group(s) from {args.source} -> {args.rules_dir / MANUAL}/, item {SUMMARY_STAGE_ITEM_ID}\n")
    print(f"  {'group':38s} {'secs':>4s}  {'sites':46s} histologies")
    for site_group, group_chapters, applicability in targets:
        count = sum(len(sections_for(c, sections)) for c in group_chapters)
        print(f"  {site_group[:38]:38s} {count:4d}  {_describe(applicability)}")
    print()

    if not args.run:
        print("Re-run with --run to compile. Each group is one or more serial LLM calls.")
        return 0

    llm = _load_llm(args.agent)
    done: list[tuple[str, str]] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    total_in = total_out = 0
    started = time.monotonic()

    for n, (site_group, group_chapters, applicability) in enumerate(targets, 1):
        out_path = args.rules_dir / MANUAL / f"{site_group}.json"
        if out_path.exists() and not args.force:
            print(f"[{n}/{len(targets)}] skip {site_group} -- {out_path} exists (--force to recompile)")
            skipped.append(site_group)
            continue

        group_sections = [s for c in group_chapters for s in sections_for(c, sections)]
        print(f"[{n}/{len(targets)}] {site_group} ({len(group_sections)} section(s))")
        try:
            units, validations, usage = compile_sections(
                manual=MANUAL,
                site_group=site_group,
                sections=group_sections,
                source_path=args.source,
                data_dictionary_path=args.data_dictionary,
                default_applicability=applicability,
                llm=llm,
                item_ids=[SUMMARY_STAGE_ITEM_ID],
                show_progress=True,
            )
        except Exception as error:  # one bad chapter must not lose the rest of the run
            failed.append((site_group, f"{type(error).__name__}: {error}"))
            print(f"!! FAILED: {site_group} -- {type(error).__name__}: {error}")
            if args.stop_on_error:
                break
            continue

        _, report_path, _, accepted = write_outputs(
            units, validations, usage,
            rules_dir=args.rules_dir, manual=MANUAL, site_group=site_group,
            source_path=args.source, manifest_defaults=MANIFEST_ENTRY,
        )
        quarantined = len(units) - len(accepted)
        if usage["usage_reported"]:
            total_in += usage["input_tokens"]
            total_out += usage["output_tokens"]
        done.append((site_group, f"{len(accepted)} units, {quarantined} quarantined -> {report_path}"))
        print(f"    {done[-1][1]}")

    elapsed = time.monotonic() - started
    print(f"\n{'=' * 78}\nSummary -- {len(done)} compiled, {len(skipped)} skipped, "
          f"{len(failed)} failed in {elapsed / 60:.1f} min\n{'=' * 78}")
    for site_group, info in done:
        print(f"  ok    {site_group[:40]:40s} {info}")
    for site_group in skipped:
        print(f"  skip  {site_group[:40]:40s} already compiled")
    for site_group, why in failed:
        print(f"  FAIL  {site_group[:40]:40s} {why}")

    if total_in or total_out:
        print(f"\n  Total tokens: {total_in + total_out:,} (in {total_in:,} / out {total_out:,})")
    elif done:
        print("\n  Total tokens: endpoint returned no usage metadata for these compiles.")

    if done:
        print(f"\nReview each {args.rules_dir / MANUAL}/<site_group>.review.txt before trusting the units.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
