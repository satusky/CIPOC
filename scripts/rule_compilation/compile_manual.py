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

# Floor for the diagnosis dates this project scopes against. Wide on purpose: a
# unit tagged with its own dx_date_min overrides this, so the floor only has to be
# early enough never to contradict a tagged dx_date_max. The earlier default of
# 2018-01-01 did contradict one: merged with a model-tagged dx_date_max of
# 2017-12-31 it produced a window no date can satisfy, silently killing the rule.
DEFAULT_DX_DATE_MIN = "2000-01-01"

from .report import build_report
from .segment import segment_markdown, select_subtree
from .validate import validate_units

_RULE_LIST_ADAPTER = TypeAdapter(list[RuleUnit])


def _usage_summary(usage_metadata: dict, *, model: str, sections: int) -> dict:
    """Fold the per-model usage a callback collected into one report record.

    ``usage_metadata`` is ``{model_name: {input_tokens, output_tokens, total_tokens, ...}}``
    as accumulated by langchain's usage callback across the tagging loop. Totals
    are summed across models (normally one). ``usage_reported`` is False when the
    endpoint returned no usage block, so a zero reads as "not reported" rather
    than "free" — the distinction matters for cost tracking.
    """
    by_model = {m: dict(u) for m, u in usage_metadata.items()}
    total_in = sum(u.get("input_tokens", 0) for u in by_model.values())
    total_out = sum(u.get("output_tokens", 0) for u in by_model.values())
    return {
        "model_configured": model,
        "sections_tagged": sections,
        "usage_reported": bool(by_model) and (total_in + total_out) > 0,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "by_model": by_model,
    }

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


def compile_sections(
    *,
    manual: str,
    site_group: str,
    sections: list,
    source_path: Path,
    data_dictionary_path: Path,
    default_applicability: RuleApplicability | None,
    llm,
    item_ids: list[int] | None = None,
    show_progress: bool = False,
) -> tuple[list[RuleUnit], list, dict]:
    """Tag and validate an explicit list of already-segmented sections.

    The section-selection half of ``compile_site_group`` assumes a manual whose
    site groups are one heading subtree each. Callers that derive their own
    section spans — see ``compile_summary_stage``, which indexes chapters by line
    range — hand the sections in directly here instead.

    ``item_ids``, when given, overrides the model's per-unit item assignment for
    every unit compiled (see ``tag_section``).
    """
    from langchain_core.callbacks import get_usage_metadata_callback

    from .tag import tag_section

    taggable = [s for s in sections if not any(skip in s.heading.casefold() for skip in _SKIP_HEADINGS)]
    progress = taggable
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(taggable, desc=f"tagging {site_group}", unit="section", leave=False)

    units: list[RuleUnit] = []
    with get_usage_metadata_callback() as usage_cb:
        for section in progress:
            if show_progress:
                progress.set_postfix_str(section.heading[:40])
            units.extend(
                tag_section(
                    section, llm,
                    source_doc=manual, site_group=site_group,
                    default_applicability=default_applicability,
                    item_ids=item_ids,
                )
            )
    usage = _usage_summary(
        usage_cb.usage_metadata,
        model=getattr(getattr(llm, "_config", None), "model", "unknown"),
        sections=len(taggable),
    )

    validations = validate_units(
        units, source_markdown_path=source_path, data_dictionary_path=data_dictionary_path
    )
    return units, validations, usage


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
    root_level: int | None = None,
    max_heading_level: int = 3,
    item_ids: list[int] | None = None,
    show_progress: bool = False,
) -> tuple[list[RuleUnit], list, dict]:
    """Segment, tag, and validate one site group.

    Returns ``(units, validations, usage)``. ``usage`` is the token-count record
    from ``_usage_summary`` covering every LLM call in the tagging loop, for cost
    tracking; it reports zeros with ``usage_reported=False`` if the endpoint sent
    no usage block.

    Set ``show_progress`` to render a per-section tqdm bar over the LLM tagging
    loop (each section is one serial model call); left off for programmatic and
    test callers.
    """
    sections = segment_markdown(source_path.read_text(), max_heading_level=max_heading_level)
    subtree = select_subtree(
        sections, root_heading, boundary_heading_contains=boundary_heading, root_level=root_level
    )
    if not subtree:
        raise SystemExit(f"No section matched root heading {root_heading!r} in {source_path}.")

    return compile_sections(
        manual=manual, site_group=site_group, sections=subtree,
        source_path=source_path, data_dictionary_path=data_dictionary_path,
        default_applicability=default_applicability, llm=llm,
        item_ids=item_ids, show_progress=show_progress,
    )


def upsert_manifest(
    rules_dir: Path,
    manual: str,
    source_path: Path,
    *,
    defaults: dict | None = None,
) -> None:
    """Record this manual in the rule-store manifest, preserving existing fields.

    ``defaults`` seeds a manual's descriptive fields on first compile; they are
    set only when absent, so hand-edits to an existing entry survive a recompile.
    """
    manifest_path = rules_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entry = manifest.setdefault(manual, {})
    for key, value in {
        "title": manual, "family": "SEER", "publication_date": "2024-01-01",
        **(defaults or {}),
    }.items():
        entry.setdefault(key, value)
    entry["source_markdown"] = str(source_path)
    from datetime import date
    entry["compiled_at"] = date.today().isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def write_outputs(
    units: list[RuleUnit],
    validations: list,
    usage: dict,
    *,
    rules_dir: Path,
    manual: str,
    site_group: str,
    source_path: Path,
    manifest_defaults: dict | None = None,
) -> tuple[Path, Path, Path, list[RuleUnit]]:
    """Write the accepted units, review report, and usage record for one site group.

    Only units that passed every validation are promoted into the store; the rest
    stay quarantined in the review report. Returns the three paths written plus
    the accepted units.
    """
    accepted_ids = {r.rule_id for r in validations if r.ok}
    accepted = [u for u in units if u.rule_id in accepted_ids]

    out_path = rules_dir / manual / f"{site_group}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_RULE_LIST_ADAPTER.dump_json(accepted, indent=2) + b"\n")
    upsert_manifest(rules_dir, manual, source_path, defaults=manifest_defaults)

    report_path = rules_dir / manual / f"{site_group}.review.txt"
    report_path.write_text(build_report(units, validations, source_markdown_path=source_path))

    usage_path = rules_dir / manual / f"{site_group}.usage.json"
    usage_path.write_text(json.dumps(usage, indent=2) + "\n")

    return out_path, report_path, usage_path, accepted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manual", required=True, help="Manifest key / source_doc, e.g. 'solid_tumor_rules'.")
    parser.add_argument("--site-group", required=True, help="Output file stem, e.g. 'breast'.")
    parser.add_argument("--root-heading", required=True, help="Substring of the site-group root heading.")
    parser.add_argument("--boundary-heading", default=None, help="Marker that bounds the site group (e.g. 'Equivalent Terms and Definitions').")
    parser.add_argument("--root-level", type=int, default=None, help="Require the root heading to be at exactly this level (e.g. 2 for '##'). Use when a deeper heading quotes the chapter title and would otherwise match first.")
    parser.add_argument("--source", required=True, type=Path, help="Source markdown path.")
    parser.add_argument("--data-dictionary", type=Path, default=Path("documents/manuals/naaccr_data_dictionary_v25.json"))
    parser.add_argument("--rules-dir", type=Path, default=Path("documents/rules"))
    parser.add_argument("--sites", nargs="*", default=None, help="Default applies_to sites, e.g. C500-C509.")
    parser.add_argument(
        "--dx-date-min",
        default=DEFAULT_DX_DATE_MIN,
        help=(
            f"Default applies_to dx_date_min (ISO), default {DEFAULT_DX_DATE_MIN}. This is a floor for "
            "cases the project considers at all, not a claim about when a rule took effect: a unit's own "
            "tagged dx_date_min overrides it, and a manual's effective date lives in the manifest. Pass a "
            "later date only for a genuine sub-manual boundary (e.g. the Solid Tumor Rules breast "
            "2018 MPH/STR split)."
        ),
    )
    parser.add_argument("--histologies", nargs="*", default=None, help="Default applies_to histologies, e.g. 8000-8700 8982.")
    parser.add_argument(
        "--item-ids",
        nargs="*",
        type=int,
        default=None,
        help="Force every compiled unit to these NAACCR item IDs instead of letting the model "
             "infer them. Use for a manual that governs one known item throughout.",
    )
    parser.add_argument("--agent", default="note_scanner", help="Config agent whose LLM settings to use for tagging.")
    parser.add_argument("--max-heading-level", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Segment and print the section plan; no LLM call.")
    args = parser.parse_args(argv)

    default_applicability = None
    if args.sites or args.histologies or args.dx_date_min:
        default_applicability = RuleApplicability(
            sites=args.sites, histologies=args.histologies, dx_date_min=args.dx_date_min
        )

    if args.dry_run:
        sections = segment_markdown(args.source.read_text(), max_heading_level=args.max_heading_level)
        subtree = select_subtree(
            sections, args.root_heading,
            boundary_heading_contains=args.boundary_heading, root_level=args.root_level,
        )
        print(f"{len(subtree)} sections in subtree {args.root_heading!r}:")
        for section in subtree:
            skipped = any(skip in section.heading.casefold() for skip in _SKIP_HEADINGS)
            flag = "  [skip]" if skipped else ""
            print(f"  {section.anchor:28s} {' > '.join(section.section_path)[-70:]}{flag}")
        return

    llm = _load_llm(args.agent)
    units, validations, usage = compile_site_group(
        manual=args.manual, site_group=args.site_group,
        root_heading=args.root_heading, boundary_heading=args.boundary_heading,
        source_path=args.source, data_dictionary_path=args.data_dictionary,
        default_applicability=default_applicability, llm=llm, item_ids=args.item_ids,
        root_level=args.root_level, max_heading_level=args.max_heading_level, show_progress=True,
    )

    out_path, report_path, usage_path, accepted = write_outputs(
        units, validations, usage,
        rules_dir=args.rules_dir, manual=args.manual,
        site_group=args.site_group, source_path=args.source,
    )

    print(report_path.read_text())
    if usage["usage_reported"]:
        print(f"\nTokens: {usage['total_tokens']:,} "
              f"(in {usage['input_tokens']:,} / out {usage['output_tokens']:,}) "
              f"over {usage['sections_tagged']} sections -> {usage_path}")
    else:
        print(f"\nTokens: endpoint returned no usage metadata; nothing to track -> {usage_path}")
    print(f"Wrote {len(accepted)} accepted units -> {out_path}")
    print(f"Review report -> {report_path}")


if __name__ == "__main__":
    main()
