"""Print the exact state dispatched to one orchestrator extractor branch.

The script runs note scanning, corpus characterization, eligibility checks, and
variable/rule scoping. It stops when the selected group reaches
``extract_branch``, before note retrieval or extraction executes.

Example:
    PYTHONPATH=src python -m scripts.debug_extractor_branch \
        "Initial LLM Extraction"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from cipoc.agents.orchestrator import ExtractBranchState, OrchestratorAgent
from cipoc.models import ClinicalNote, TargetGroup
from cipoc.tools import load_variable_groups


DEFAULT_GROUPS_PATH = Path("config/variable_groups.json")
DEFAULT_NOTES_PATH = Path("tests/fixtures/note_bundle.json")


def group_by_name(path: str | Path, name: str) -> TargetGroup:
    """Load exactly one flat target group by its configured display name."""
    groups = load_variable_groups(path)
    matches = [group for group in groups if group.name == name]
    if not matches:
        matches = [
            group
            for group in groups
            if group.name is not None and group.name.casefold() == name.casefold()
        ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Variable group name is not unique: {name!r}")

    available = ", ".join(repr(group.name) for group in groups)
    raise ValueError(f"Unknown variable group name {name!r}. Available names: {available}")


def run_until_extractor_branch(
    agent: OrchestratorAgent,
    group: TargetGroup,
    raw_notes: list[dict[str, Any]],
    *,
    structured_data: dict[int, str] | None = None,
) -> ExtractBranchState:
    """Run the root graph until ``group`` is dispatched to ``extract_branch``."""
    agent._target_variables = [group]
    graph_input = {
        "note_corpus": {
            note["note_id"]: ClinicalNote(**note)
            for note in raw_notes
        },
        "structured_data": structured_data or {},
    }
    events = agent._graph.stream(graph_input, stream_mode="tasks")
    try:
        for event in events:
            if event.get("name") != "extract_branch" or "input" not in event:
                continue
            branch_input = event["input"]
            if isinstance(branch_input, ExtractBranchState):
                return branch_input
            return ExtractBranchState.model_validate(branch_input)
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()

    raise RuntimeError(
        f"Variable group {group.name!r} did not reach the extractor branch. "
        "It may have no pending variables or may have failed corpus/site gating."
    )


def _load_json(path: str | Path) -> Any:
    with open(path, "r") as file:
        return json.load(file)


def _structured_data(value: str | None) -> dict[int, str] | None:
    if value is None:
        return None
    candidate = Path(value)
    raw = candidate.read_text() if candidate.is_file() else value
    return {int(item_id): str(coded_value) for item_id, coded_value in json.loads(raw).items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_name", help="Configured variable-group name, e.g. 'Initial LLM Extraction'.")
    parser.add_argument(
        "--variable-groups",
        type=Path,
        default=DEFAULT_GROUPS_PATH,
        help=f"Variable-group JSON file (default: {DEFAULT_GROUPS_PATH}).",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=DEFAULT_NOTES_PATH,
        help=f"Clinical-note JSON bundle (default: {DEFAULT_NOTES_PATH}).",
    )
    parser.add_argument(
        "--structured-data",
        default=None,
        help="Known item values as an inline JSON object or path to a JSON file.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    group = group_by_name(args.variable_groups, args.group_name)
    raw_notes = _load_json(args.notes)
    if not isinstance(raw_notes, list):
        raise ValueError(f"Clinical-note bundle must be a JSON list: {args.notes}")

    state = run_until_extractor_branch(
        OrchestratorAgent(),
        group,
        raw_notes,
        structured_data=_structured_data(args.structured_data),
    )
    print(state.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
