"""Phase 0 gate: verify Tap 2 (LLM callback) propagates to every subagent.

The whole demo rests on one assumption — that a single
:class:`~cipoc.demo.capture.LLMCaptureHandler` attached to the orchestrator's
stream config sees the model calls made by the scanner, retriever, and extractor
subagents (which the orchestrator invokes with a bare ``.run(progress=False)``,
no explicit ``config=``). This script runs the orchestrator once over the note
fixture and reports how many calls were captured per agent, plus whether each
capture carried the ``langgraph_node`` metadata used for map correlation.

Run it before building the rest of the demo::

    PYTHONPATH=src python -m cipoc.demo.checks
    PYTHONPATH=src python -m cipoc.demo.checks --notes tests/fixtures/note_bundle.json

Exit code 0 means scanner, retriever, and extractor calls were all captured with
node metadata — Tap 2 works and the build proceeds as planned. A non-zero exit
means propagation is incomplete for some agent and the ``llm=``-injected
instrumented-model fallback (see the plan) is needed for that path.

This needs live LLM credentials (``AZURE_OPENAI_URL`` / ``RENCI_AZURE_API_KEY``)
and network access, so it is a manual check, not a unit test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cipoc.agents.orchestrator import OrchestratorAgent
from cipoc.models import ClinicalNote

from .capture import LLMCaptureHandler
from .mapping import infer_agent, map_node_id

# The subagents whose calls MUST be captured for the demo detail panels to work.
_REQUIRED_AGENTS = ("scanner", "retriever", "extractor")

_DEFAULT_NOTES = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "note_bundle.json"
)


def run_tap2_check(notes_path: Path, *, dump: Path | None = None) -> bool:
    """Stream the orchestrator graph once and report Tap-2 capture coverage.

    Returns ``True`` when every required subagent had at least one call captured
    with a resolvable map node. When ``dump`` is set, writes the raw captures to
    that path for inspection.
    """
    raw_notes = json.loads(notes_path.read_text())
    graph_input = {
        "note_corpus": {note["note_id"]: ClinicalNote(**note) for note in raw_notes},
        "structured_data": {},
    }

    agent = OrchestratorAgent()
    handler = LLMCaptureHandler()

    print(f"Streaming orchestrator over {notes_path} (live LLM)…\n")
    for _ in agent.compiled_graph.stream(
        graph_input,
        stream_mode=["values", "tasks"],
        subgraphs=True,
        config={"callbacks": [handler]},
    ):
        pass

    calls = handler.snapshot()
    if dump is not None:
        dump.write_text(
            json.dumps([call.to_dict() for call in calls], indent=2, ensure_ascii=False)
        )
        print(f"Wrote {len(calls)} raw captures to {dump}\n")

    per_agent: dict[str, int] = {}
    with_node: dict[str, int] = {}
    mapped: dict[str, int] = {}
    for call in calls:
        agent_name = infer_agent(call.namespace)
        per_agent[agent_name] = per_agent.get(agent_name, 0) + 1
        if call.node:
            with_node[agent_name] = with_node.get(agent_name, 0) + 1
        if map_node_id(call.node, call.namespace) is not None:
            mapped[agent_name] = mapped.get(agent_name, 0) + 1

    print(f"Captured {len(calls)} LLM call(s) total.\n")
    header = f"{'agent':<14}{'calls':>7}{'w/ node':>9}{'mapped':>8}"
    print(header)
    print("-" * len(header))
    for agent_name in sorted(per_agent):
        print(
            f"{agent_name:<14}{per_agent[agent_name]:>7}"
            f"{with_node.get(agent_name, 0):>9}{mapped.get(agent_name, 0):>8}"
        )

    missing = [name for name in _REQUIRED_AGENTS if per_agent.get(name, 0) == 0]
    no_node = [
        name
        for name in _REQUIRED_AGENTS
        if per_agent.get(name, 0) and with_node.get(name, 0) == 0
    ]

    print()
    if missing:
        print(f"FAIL: no captured calls for: {', '.join(missing)}.")
        print("      Tap 2 does not propagate to these — use the llm= fallback.")
        return False
    if no_node:
        print(f"WARN: captures for {', '.join(no_node)} lack langgraph_node metadata.")
        print("      Map correlation will need the namespace-only path for these.")
    print("PASS: scanner, retriever, and extractor calls were all captured.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notes",
        type=Path,
        default=_DEFAULT_NOTES,
        help="Path to a JSON list of clinical notes (default: note_bundle fixture).",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="Optional path to write the raw captured LLM calls as JSON.",
    )
    args = parser.parse_args(argv)
    ok = run_tap2_check(args.notes, dump=args.dump)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
