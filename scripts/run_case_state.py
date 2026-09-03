"""Run the orchestrator over one case and save its full output state as JSON.

``OrchestratorAgent.run()`` returns only the durable :class:`Case`, and the
compiled graph's ``output_schema`` narrows ``invoke()`` to the durable channels
``to_case()`` consumes. Both drop the scanned note corpus, the note digests, the
corpus descriptors, and the planned target groups -- exactly the material a
reviewer needs to see what the run actually did.

This script therefore drives ``agent.compiled_graph`` itself (the accessor
``BaseAgent`` exposes for external harnesses) in ``values`` and ``tasks`` stream
modes. The last root values event is the full state, while task events populate
optional execution-observability channels.

    PYTHONPATH=src python -m scripts.run_case_state \
        --notes tests/fixtures/note_bundle.json \
        --output tests/test_outputs/case_state.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from cipoc.agents import OrchestratorAgent
from cipoc.models import ClinicalNote
from cipoc.tools import load_group_hierarchy, load_variable_groups
from cipoc.utils import ObservabilityCollector, run_with_progress
from cipoc.utils.progress.events import normalize


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTES = REPO_ROOT / "tests" / "fixtures" / "note_bundle.json"
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "test_outputs" / "case_state.json"

_REJECTION_MESSAGES = {
    "note_type_mismatch": "Note type did not match the configured note filter.",
    "cancer_status_mismatch": (
        "Cancer status did not match the configured note filter."
    ),
    "missing_or_invalid_date": "Note date was missing or invalid.",
    "outside_date_window": "Note date was outside the configured date window.",
}


def to_jsonable(value: Any) -> Any:
    """Reduce a live graph state to the JSON value space.

    Each Pydantic model is dumped through *its own* class rather than the field's
    declared type. That matters for ``note_corpus``, whose declared value type is
    ``ClinicalNote | ProcessedClinicalNote``: serializing through the union can
    match the narrower member and silently drop the scan results, while dumping
    the instance keeps every field it actually carries.

    Dict keys are stringified because note IDs are ``int | str`` and JSON object
    keys are strings either way; unknown objects degrade to ``str`` so a long run
    is never lost at the final write.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        # Sorted so two runs over the same case diff cleanly; NoteCorpusDescriptors
        # holds raw sets (types, affected tissues, unique flags).
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _retriever_offered(exchanges: list[dict[str, Any]]) -> list[Any] | None:
    """Return the last successful retriever's raw proposed note IDs, if captured."""
    for exchange in reversed(exchanges):
        if exchange.get("node") != "identify_relevant_notes" or exchange.get("error"):
            continue
        response = exchange.get("response")
        if isinstance(response, Mapping):
            note_ids = response.get("note_ids")
            if note_ids is None:
                note_ids = response.get("relevant_note_ids")
        else:
            note_ids = response
        if isinstance(note_ids, (list, tuple)):
            return list(note_ids)
    return None


def _workbench_note_selection(
    durable: Any,
    llm_exchanges: Mapping[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Adapt typed durable provenance to the Workbench's display contract."""
    if not isinstance(durable, Mapping):
        return {}

    converted = {}
    for key, raw_selection in durable.items():
        selection = (
            raw_selection.model_dump(mode="json")
            if isinstance(raw_selection, BaseModel)
            else raw_selection
        )
        if not isinstance(selection, Mapping):
            continue

        filtered_out = {}
        rejected = selection.get("rejected_note_ids", {})
        if isinstance(rejected, Mapping):
            for note_id, reasons in rejected.items():
                readable = []
                for reason in reasons or []:
                    code = str(getattr(reason, "value", reason))
                    readable.append(
                        _REJECTION_MESSAGES.get(
                            code, code.replace("_", " ").capitalize() + "."
                        )
                    )
                filtered_out[str(note_id)] = " ".join(readable)

        group_key = str(key)
        workbench_selection = {
            "group_id": selection.get("group_id"),
            "candidate_note_ids": list(selection.get("candidate_note_ids", []) or []),
            "filtered_out": filtered_out,
            "selected_note_ids": list(selection.get("selected_note_ids", []) or []),
        }
        offered = _retriever_offered(llm_exchanges.get(group_key, []))
        if offered is not None:
            workbench_selection["retriever_offered"] = offered
        converted[group_key] = workbench_selection
    return converted


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_notes(path: Path) -> list[dict]:
    raw_notes = _load_json(path)
    if not isinstance(raw_notes, list):
        raise ValueError(f"{path} must contain a JSON array of clinical notes.")
    # Validate up front so a malformed bundle fails before any LLM call.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the orchestrator and save its full output state as JSON."
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
        help=f"Where to write the state JSON (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--structured-data",
        default=None,
        help=(
            "Known coded values keyed by NAACCR item ID, either an inline JSON "
            "object or a path to a JSON file. Seeded as structured-data results, "
            "skipping extraction."
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
        "--no-llm-capture",
        action="store_true",
        help="Omit model prompts and responses from the output artifact.",
    )
    return parser


def run_case_state(
    raw_notes: list[dict],
    *,
    structured_data: dict[int, str] | None = None,
    max_concurrency: int | None = None,
    progress: bool = True,
    capture_llm: bool = True,
) -> dict[str, Any]:
    """Run the orchestrator and return its last root state, JSON-ready."""
    agent = OrchestratorAgent()
    graph_input = {
        "note_corpus": {note["note_id"]: ClinicalNote(**note) for note in raw_notes},
        "structured_data": structured_data or {},
    }
    base_graph_config = (
        {"max_concurrency": max_concurrency} if max_concurrency is not None else None
    )
    collector = ObservabilityCollector(capture_llm=capture_llm)
    graph_config = collector.graph_config(base_graph_config)

    if progress:
        groups_path = agent._config.documents().variable_groups_path
        final_state = run_with_progress(
            agent.compiled_graph,
            graph_input,
            config=graph_config,
            subgraphs=True,
            description="Orchestrator",
            target_groups=load_variable_groups(groups_path),
            group_hierarchy=load_group_hierarchy(groups_path),
            pause_before_summary=True,
            event_observer=collector.observe,
        )
    else:
        final_state = None
        stream_kwargs = {
            "stream_mode": ["values", "tasks"],
            "subgraphs": True,
        }
        if graph_config is not None:
            stream_kwargs["config"] = graph_config
        for raw_item in agent.compiled_graph.stream(graph_input, **stream_kwargs):
            event = normalize(raw_item, subgraphs=True)
            if event is None:
                continue
            collector.observe(event)
            if event.kind == "values" and event.is_root:
                final_state = event.payload
        if final_state is None:
            raise RuntimeError("The orchestrator graph produced no state.")

    result = to_jsonable(final_state)
    observability = collector.snapshot()
    llm_exchanges = observability.get("llm_exchanges", {})
    result["note_selection"] = _workbench_note_selection(
        result.get("note_selection", {}), llm_exchanges
    )
    result.update(observability)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    raw_notes = _load_notes(args.notes)
    structured_data = _load_structured_data(args.structured_data)

    state = run_case_state(
        raw_notes,
        structured_data=structured_data,
        max_concurrency=args.max_concurrency,
        progress=not args.no_progress,
        capture_llm=not args.no_llm_capture,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, indent=2), encoding="utf-8")

    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.1f} KB)")
    print("  channels: " + ", ".join(sorted(state)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
