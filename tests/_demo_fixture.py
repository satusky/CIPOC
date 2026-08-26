"""Generate the committed demo trace fixture, hermetically and deterministically.

The Phase-1 replay tests need a recorded trace that contains every
``DemoEventType`` — including ``llm_call`` — but tests must not touch a live LLM.
So we drive the real orchestrator topology through the LLM-free fake subagents
(``tests/fake_orchestrator``) and *inject* synthetic captures into the same
:class:`~cipoc.demo.capture.LLMCaptureHandler` the merge reads, timed to fire the
way real callbacks do: right after an LLM node's ``task_start``, so the merge
drains them into that node's ``task_start``/``task_end`` window.

Run ``python -m tests._demo_fixture`` (from the repo root, ``PYTHONPATH=src``) to
regenerate ``tests/fixtures/demo_trace.jsonl`` after an intentional schema change.
The output is deterministic (fixed clock, single note, fixed synthetic content),
so regeneration is a no-op unless the shape really changed.

``--gated <path>`` writes a *second*, larger trace that is deliberately not
committed: every note, several variable groups, and a corpus that opens some
gates and closes others. The workflow map draws one node per note, per gate and
per variable, so developing and demoing it needs a run whose shape is actually
interesting — and the committed fixture (one note, two passing groups) is not.
"""

from __future__ import annotations

import argparse
from itertools import count
from pathlib import Path
from typing import Any, Iterable, Iterator

from cipoc.demo.capture import LLMCaptureHandler
from cipoc.demo.events import LLMCall
from cipoc.demo.stream import run_demo_stream
from cipoc.utils.progress.events import ProgressEvent, normalize
from cipoc.utils.progress.model import DEFAULT_NODE_KINDS

from tests.fake_orchestrator import (
    Outcome,
    Script,
    build_fake_orchestrator,
    graph_input,
    load_notes,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "demo_trace.jsonl"

# One note plus two small variable groups keeps the fixture compact while still
# exercising scanner -> retriever -> extractor and every event type. Item 674 is
# scripted to fail validation once, so the trace also covers the repair loop.
_FIXTURE_NOTE_ID = 51
_FIXTURE_GROUP_IDS = ("initial_llm_extraction", "lymph_node_removal")
_REPAIR_ITEM_ID = 674

# The gated variant. Two gates open and one stays shut, so the map has a group
# whose whole subtree must never light up; one open group retrieves no notes, so
# it reaches the relevant-notes decision and is skipped for a different reason.
_GATED_GROUP_IDS = (
    "initial_llm_extraction",   # ungated
    "first_course_treatment",   # gate:treatment -> open
    "metastases",               # gate:mets      -> shut
    "lymph_node_removal",       # gate:nodes     -> open
    "site_specific_codes",      # ungated, but retrieves nothing
)
_GATED_CONCEPTS = {
    "cancer": True,
    "metastasis": False,        # shuts gate:mets
    "surgery": True,
    "chemotherapy": True,
    "radiation": True,
    "lymph_nodes_removed": True,
}


def _step_clock(step: float = 0.25):
    """A deterministic monotonic clock so the committed fixture never churns."""
    ticks = count()
    return lambda: next(ticks) * step


def _synthetic_call(event: ProgressEvent, run_seq: int) -> LLMCall:
    """A stand-in capture for the LLM node ``event`` just started.

    ``namespace`` is the node's own scope (``parent + "{node}:{task_id}"``) — the
    same ``langgraph_checkpoint_ns`` a real capture from inside this node carries,
    so agent/map resolution matches a live trace.
    """
    node = event.node
    return LLMCall(
        node=node,
        namespace=event.scope,
        run_id=f"run-{run_seq:03d}",
        parent_run_id=None,
        model="fake-model",
        prompt_messages=[
            {"role": "system", "content": f"You are the {node} step."},
            {"role": "human", "content": f"Synthetic prompt for {node}."},
        ],
        reasoning=None,
        response=f'{{"node": "{node}", "result": "synthetic"}}',
        usage={"input_tokens": 128, "output_tokens": 32, "total_tokens": 160},
    )


def _inject_captures(
    raw_items: Any, handler: LLMCaptureHandler, *, subgraphs: bool = True
) -> Iterator[Any]:
    """Pass raw stream items through, appending a synthetic call after each LLM
    node's ``task_start`` so the merge sees it on the following item."""
    run_seq = count()
    for item in raw_items:
        yield item
        event = normalize(item, subgraphs=subgraphs)
        if (
            event is not None
            and event.kind == "task_start"
            and DEFAULT_NODE_KINDS.get(event.node) == "llm"
        ):
            handler.calls.append(_synthetic_call(event, next(run_seq)))


def record_trace(
    *,
    notes: Iterable[Any],
    group_ids: Iterable[str],
    script: Script,
    path: Path | None = None,
) -> list[Any]:
    """Drive the fake orchestrator and return (optionally record) its events."""
    agent = build_fake_orchestrator(script)
    # Trim the plan to the requested groups so the extractor fans out over a
    # handful of variables rather than the full data-dictionary set — the graph
    # plans from this list.
    wanted = set(group_ids)
    agent._target_variables = [
        group for group in agent._target_variables if group.group_id in wanted
    ]
    handler = LLMCaptureHandler()

    # Wrap the real stream so synthetic captures land in ``handler`` mid-run,
    # exactly where the merge expects live callbacks. ``run_demo_stream`` is given
    # this same handler, so it never attaches a second (no-op) one.
    original_stream = agent.compiled_graph.stream

    def wrapped_stream(*args: Any, **kwargs: Any) -> Iterator[Any]:
        return _inject_captures(original_stream(*args, **kwargs), handler)

    agent.compiled_graph.stream = wrapped_stream  # type: ignore[method-assign]
    try:
        return list(
            run_demo_stream(
                agent.compiled_graph,
                graph_input(list(notes)),
                record_path=path,
                handler=handler,
                clock=_step_clock(),
            )
        )
    finally:
        agent.compiled_graph.stream = original_stream  # type: ignore[method-assign]


def generate_fixture(path: Path = FIXTURE_PATH) -> int:
    """Record the committed fixture trace to ``path``; return the event count."""
    note = next(n for n in load_notes() if n.note_id == _FIXTURE_NOTE_ID)
    return len(
        record_trace(
            notes=[note],
            group_ids=_FIXTURE_GROUP_IDS,
            script=Script(outcomes={_REPAIR_ITEM_ID: Outcome(repairs=1)}),
            path=path,
        )
    )


def build_gated_trace(path: Path | None = None) -> list[Any]:
    """A run whose gates disagree: two open, one shut, one group retrieving none.

    Not committed — it is large, and it exists so the workflow map can be built
    and demoed against a run with more than one of everything.
    """
    return record_trace(
        notes=load_notes(),
        group_ids=_GATED_GROUP_IDS,
        script=Script(
            concepts=dict(_GATED_CONCEPTS),
            outcomes={_REPAIR_ITEM_ID: Outcome(repairs=1)},
            retrieved={"site_specific_codes": 0},
        ),
        path=path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gated",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the gated demo trace here instead of the committed fixture.",
    )
    args = parser.parse_args()

    if args.gated is not None:
        written = len(build_gated_trace(args.gated))
        target = args.gated
    else:
        written = generate_fixture()
        target = FIXTURE_PATH
    print(f"Wrote {written} events to {target} ({target.stat().st_size / 1024:.1f} KiB).")
