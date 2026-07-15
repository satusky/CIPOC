"""Offline rule-compilation driver.

Wires the pipeline end to end for one site group of one manual:

    segment (pure)  →  tag (LLM)  →  validate (pure)  →  report + write

Compilation is idempotent and re-runnable; only accepted units are written to
``documents/rules/<manual>/<site_group>.json`` and the manifest is upserted.
Quarantined units and a spot-check sample are written to a review report that
must be eyeballed before the output is trusted.

Usage:
    PYTHONPATH=src python -m scripts.rule_compilation.compile_manual \\
        --manual solid_tumor_rules --site-group breast \\
        --root-heading "Breast Equivalent Terms" \\
        --boundary-heading "Equivalent Terms and Definitions" \\
        --source documents/markdown/SolidTumorRules_Combined.md \\
        --sites C500-C509 --dx-date-min 2018-01-01

Run with --dry-run to segment and print the section plan without calling the LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from cipoc.models import RuleApplicability, RuleUnit

from .report import build_report
from .segment import segment_markdown, select_subtree
from .validate import validate_units

_RULE_LIST_ADAPTER = TypeAdapter(list[RuleUnit])

# Sections whose only content is non-normative; skipped before the LLM call to
# save tokens. Substring match on the section heading, case-insensitive.
_SKIP_HEADINGS = ("new for", "table of contents", "introduction note", "illustrations")


def _load_llm(agent: str):
    """Build the offline tagging LLM from the project config (lazy import)."""
    from cipoc.llm import agent_model_for
    from cipoc.utils.utils import load_config

    config = load_config()
    settings = config.llm_config(agent)
    settings.model = "gpt-5.6-terra"
    return agent_model_for(settings.provider)(settings)


def compile_site_group(
    *,
    manual: str,
    site_group: str,
    root_heading: str,
    boundary_heading: str | None,
    source_path: Path,
    data_dictionary_path: Path,
    default_applicability: RuleApplicability | None,
    llm,
    max_heading_level: int = 3,
    show_progress: bool = False,
) -> tuple[list[RuleUnit], list]:
    """Segment, tag, and validate one site group. Returns (units, validations).

    Set ``show_progress`` to render a per-section tqdm bar over the LLM tagging
    loop (each section is one serial model call); left off for programmatic and
    test callers.
    """
    from .tag import tag_section

    sections = segment_markdown(source_path.read_text(), max_heading_level=max_heading_level)
    subtree = select_subtree(sections, root_heading, boundary_heading_contains=boundary_heading)
    if not subtree:
        raise SystemExit(f"No section matched root heading {root_heading!r} in {source_path}.")

    taggable = [s for s in subtree if not any(skip in s.heading.casefold() for skip in _SKIP_HEADINGS)]
    progress = taggable
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(taggable, desc=f"tagging {site_group}", unit="section")

    units: list[RuleUnit] = []
    for section in progress:
        if show_progress:
            progress.set_postfix_str(section.heading[:40])
        units.extend(
            tag_section(
                section, llm,
                source_doc=manual, site_group=site_group,
                default_applicability=default_applicability,
            )
        )

    validations = validate_units(
        units, source_markdown_path=source_path, data_dictionary_path=data_dictionary_path
    )
    return units, validations


def _upsert_manifest(rules_dir: Path, manual: str, source_path: Path) -> None:
    manifest_path = rules_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entry = manifest.setdefault(manual, {})
    entry.setdefault("title", manual)
    entry.setdefault("family", "SEER")
    entry.setdefault("publication_date", "2024-01-01")
    entry["source_markdown"] = str(source_path)
    from datetime import date
    entry["compiled_at"] = date.today().isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manual", required=True, help="Manifest key / source_doc, e.g. 'solid_tumor_rules'.")
    parser.add_argument("--site-group", required=True, help="Output file stem, e.g. 'breast'.")
    parser.add_argument("--root-heading", required=True, help="Substring of the site-group root heading.")
    parser.add_argument("--boundary-heading", default=None, help="Marker that bounds the site group (e.g. 'Equivalent Terms and Definitions').")
    parser.add_argument("--source", required=True, type=Path, help="Source markdown path.")
    parser.add_argument("--data-dictionary", type=Path, default=Path("documents/naaccr_data_dictionary_v25.json"))
    parser.add_argument("--rules-dir", type=Path, default=Path("documents/rules"))
    parser.add_argument("--sites", nargs="*", default=None, help="Default applies_to sites, e.g. C500-C509.")
    parser.add_argument("--dx-date-min", default=None, help="Default applies_to dx_date_min (ISO).")
    parser.add_argument("--agent", default="note_scanner", help="Config agent whose LLM settings to use for tagging.")
    parser.add_argument("--max-heading-level", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Segment and print the section plan; no LLM call.")
    args = parser.parse_args(argv)

    default_applicability = None
    if args.sites or args.dx_date_min:
        default_applicability = RuleApplicability(sites=args.sites, dx_date_min=args.dx_date_min)

    if args.dry_run:
        sections = segment_markdown(args.source.read_text(), max_heading_level=args.max_heading_level)
        subtree = select_subtree(sections, args.root_heading, boundary_heading_contains=args.boundary_heading)
        print(f"{len(subtree)} sections in subtree {args.root_heading!r}:")
        for section in subtree:
            skipped = any(skip in section.heading.casefold() for skip in _SKIP_HEADINGS)
            flag = "  [skip]" if skipped else ""
            print(f"  {section.anchor:28s} {' > '.join(section.section_path)[-70:]}{flag}")
        return

    llm = _load_llm(args.agent)
    units, validations = compile_site_group(
        manual=args.manual, site_group=args.site_group,
        root_heading=args.root_heading, boundary_heading=args.boundary_heading,
        source_path=args.source, data_dictionary_path=args.data_dictionary,
        default_applicability=default_applicability, llm=llm,
        max_heading_level=args.max_heading_level, show_progress=True,
    )

    accepted_ids = {r.rule_id for r in validations if r.ok}
    accepted = [u for u in units if u.rule_id in accepted_ids]

    out_path = args.rules_dir / args.manual / f"{args.site_group}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_RULE_LIST_ADAPTER.dump_json(accepted, indent=2) + b"\n")
    _upsert_manifest(args.rules_dir, args.manual, args.source)

    report = build_report(units, validations, source_markdown_path=args.source)
    report_path = args.rules_dir / args.manual / f"{args.site_group}.review.txt"
    report_path.write_text(report)

    print(report)
    print(f"\nWrote {len(accepted)} accepted units -> {out_path}")
    print(f"Review report -> {report_path}")


if __name__ == "__main__":
    main()
