"""Run one case and write the canonical orchestrator result artifact.

    PYTHONPATH=src python -m scripts.run_case_state \
        --notes tests/fixtures/note_bundle.json \
        --output tests/test_outputs/case_state.json

Pass ``--item-ids`` to extract only some variables. The configured variable
groups are trimmed to those items plus any given structured data, so a known
primary site or histology still scopes the run.

Disabling LLM content capture omits only model prompts and responses. The
artifact still contains the full processed note corpus and may contain PHI.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from cipoc.agents import OrchestratorAgent
from cipoc.models import (
    ClinicalNote,
    LLMUsageSummary,
    OrchestratorRunError,
    OrchestratorRunFailure,
    OrchestratorRunResult,
)
from cipoc.tools import subset_variable_groups
from cipoc.utils import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTES = REPO_ROOT / "tests" / "fixtures" / "note_bundle.json"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "test_outputs" / "case_state.json"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_notes(path: Path) -> list[dict]:
    raw_notes = _load_json(path)
    if not isinstance(raw_notes, list):
        raise ValueError(f"{path} must contain a JSON array of clinical notes.")
    for note in raw_notes:
        ClinicalNote.model_validate(note)
    return raw_notes


def _load_structured_data(value: str | None) -> dict[int, str] | None:
    if value is None:
        return None
    raw = (
        json.loads(value)
        if value.lstrip().startswith("{")
        else _load_json(Path(value))
    )
    if not isinstance(raw, dict):
        raise ValueError("Structured data must be a JSON object keyed by item ID.")
    return {int(item_id): str(item_value) for item_id, item_value in raw.items()}


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the orchestrator and save its canonical result as JSON.",
        epilog=(
            "Disabling LLM content capture does not de-identify the artifact: "
            "the processed note corpus still contains clinical text and PHI."
        ),
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=DEFAULT_NOTES,
        help=f"JSON array of clinical notes (default: {DEFAULT_NOTES.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Canonical result JSON path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--structured-data",
        default=None,
        help=(
            "Known coded values keyed by NAACCR item ID, either an inline JSON "
            "object or a path to a JSON file."
        ),
    )
    parser.add_argument(
        "--item-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Extract only these NAACCR item IDs. Items given structured data are "
            "kept too, so they still scope the run."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="LangGraph parallel task limit.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Run without the live progress display.",
    )
    parser.add_argument(
        "--no-llm-content-capture",
        "--no-llm-capture",
        dest="no_llm_content_capture",
        action="store_true",
        help=(
            "Omit only LLM prompts and responses; usage and exchange metadata are "
            "still captured. This does not de-identify corpus PHI."
        ),
    )
    parser.add_argument(
        "--max-content-chars",
        type=_nonnegative_int,
        default=None,
        help=(
            "Maximum retained characters per LLM prompt message; unset is "
            "unbounded. This does not de-identify corpus PHI."
        ),
    )
    return parser


def _subset_agent(item_ids: list[int], directory: Path) -> OrchestratorAgent:
    """Build an orchestrator whose plan is the configured groups trimmed to
    ``item_ids``. The trimmed file must outlive the run, which fingerprints it."""
    config = load_config()
    groups = _load_json(config.documents().variable_groups_path)
    subset_path = directory / "variable_groups.json"
    subset_path.write_text(
        json.dumps(subset_variable_groups(groups, item_ids), indent=2), encoding="utf-8"
    )
    return OrchestratorAgent(config=config.with_documents(variable_groups_path=subset_path))


RunArtifact = OrchestratorRunResult | OrchestratorRunFailure


def _write_artifact(path: Path, artifact: RunArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def run_case_state(
    raw_notes: list[dict],
    output: str | Path,
    *,
    item_ids: list[int] | None = None,
    structured_data: dict[int, str] | None = None,
    max_concurrency: int | None = None,
    progress: bool = True,
    capture_llm_content: bool = True,
    max_content_chars: int | None = None,
) -> OrchestratorRunResult:
    """Run the orchestrator through its public API, optionally on a subset of
    the configured variables, and write the run artifact to ``output``.

    A failed run writes its partial failure artifact there before the
    ``OrchestratorRunError`` propagates.
    """
    output = Path(output)
    with tempfile.TemporaryDirectory() as directory:
        agent = (
            OrchestratorAgent()
            if item_ids is None
            else _subset_agent([*item_ids, *(structured_data or {})], Path(directory))
        )
        try:
            result = agent.run(
                raw_notes,
                structured_data=structured_data,
                max_concurrency=max_concurrency,
                progress=progress,
                capture_llm_content=capture_llm_content,
                max_content_chars=max_content_chars,
            )
        except OrchestratorRunError as error:
            _write_artifact(output, error.failure)
            raise
    _write_artifact(output, result)
    return result


def usage_line(summary: LLMUsageSummary | None) -> str:
    """One line of token usage. Per-call detail and timing stay in the artifact."""
    # Counts are zero-filled when no call reported usage; that is unknown, not zero.
    if summary is None or (summary.model_invocations and not summary.usage_reported_invocations):
        return "Usage: unavailable."
    line = f"Usage: {summary.total_tokens:,} tokens over {summary.model_invocations:,} model calls"
    if summary.missing_usage_invocations:
        line += f" ({summary.missing_usage_invocations:,} without reported usage)"
    return line + "."


def _print_outcome(path: Path, artifact: RunArtifact) -> None:
    print(f"Wrote {path}")
    observability = artifact.observability
    if observability.collection_status in {"partial", "unavailable"}:
        print(f"Telemetry collection: {observability.collection_status}")
        for issue in observability.collection_issues:
            print(f"  {issue.code}: {issue.message}")
    print(usage_line(observability.llm_usage_summary))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_notes = _load_notes(args.notes)
    structured_data = _load_structured_data(args.structured_data)

    try:
        result = run_case_state(
            raw_notes,
            args.output,
            item_ids=args.item_ids,
            structured_data=structured_data,
            max_concurrency=args.max_concurrency,
            progress=not args.no_progress,
            capture_llm_content=not args.no_llm_content_capture,
            max_content_chars=args.max_content_chars,
        )
    except OrchestratorRunError as error:
        _print_outcome(args.output, error.failure)
        print(f"Orchestration failed: {error}", file=sys.stderr)
        return 1

    _print_outcome(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
