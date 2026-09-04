"""Run one case and write the canonical orchestrator result artifact.

    PYTHONPATH=src python -m scripts.run_case_state \
        --notes tests/fixtures/note_bundle.json \
        --output tests/test_outputs/case_state.json

Disabling LLM content capture omits only model prompts and responses. The
artifact still contains the full processed note corpus and may contain PHI.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def run_case_state(
    raw_notes: list[dict],
    *,
    structured_data: dict[int, str] | None = None,
    max_concurrency: int | None = None,
    progress: bool = True,
    capture_llm_content: bool = True,
    max_content_chars: int | None = None,
) -> OrchestratorRunResult:
    """Run the orchestrator through its public API."""
    return OrchestratorAgent().run(
        raw_notes,
        structured_data=structured_data,
        max_concurrency=max_concurrency,
        progress=progress,
        capture_llm_content=capture_llm_content,
        max_content_chars=max_content_chars,
    )


def usage_lines(summary: LLMUsageSummary) -> list[str]:
    """Render concise provider-reported usage totals for terminal output."""
    lines = [
        (
            f"Tokens: input={summary.input_tokens:,} "
            f"output={summary.output_tokens:,} total={summary.total_tokens:,}"
        ),
        (
            f"Calls: logical={summary.logical_calls:,} "
            f"invocations={summary.model_invocations:,} "
            f"retries={summary.retry_invocations:,}"
        ),
        (
            "Usage coverage: "
            f"reported={summary.usage_reported_invocations:,} "
            f"missing={summary.missing_usage_invocations:,}"
        ),
    ]
    details = [
        f"input.{name}={count:,}"
        for name, count in sorted(summary.input_token_details.root.items())
        if count
    ]
    details.extend(
        f"output.{name}={count:,}"
        for name, count in sorted(summary.output_token_details.root.items())
        if count
    )
    if details:
        lines.append("Token details: " + ", ".join(details))
    return lines


RunArtifact = OrchestratorRunResult | OrchestratorRunFailure


def _write_artifact(path: Path, artifact: RunArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def _print_outcome(path: Path, artifact: RunArtifact) -> None:
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path} ({size_kb:.1f} KB)")
    for line in usage_lines(artifact.observability.llm_usage_summary):
        print(line)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_notes = _load_notes(args.notes)
    structured_data = _load_structured_data(args.structured_data)

    try:
        artifact = run_case_state(
            raw_notes,
            structured_data=structured_data,
            max_concurrency=args.max_concurrency,
            progress=not args.no_progress,
            capture_llm_content=not args.no_llm_content_capture,
            max_content_chars=args.max_content_chars,
        )
    except OrchestratorRunError as error:
        artifact = error.failure
        _write_artifact(args.output, artifact)
        _print_outcome(args.output, artifact)
        print(f"Orchestration failed: {error}", file=sys.stderr)
        return 1

    _write_artifact(args.output, artifact)
    _print_outcome(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
