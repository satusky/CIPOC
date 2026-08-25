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
"""

from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Any, Iterator

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


def generate_fixture(path: Path = FIXTURE_PATH) -> int:
    """Record the fixture trace to ``path``; return the number of events."""
    note = next(n for n in load_notes() if n.note_id == _FIXTURE_NOTE_ID)
    script = Script(outcomes={_REPAIR_ITEM_ID: Outcome(repairs=1)})
    agent = build_fake_orchestrator(script)
    # Trim to two groups so the extractor fans out over a handful of variables
    # rather than the full data-dictionary set — the graph plans from this list.
    agent._target_variables = [
        group
        for group in agent._target_variables
        if group.group_id in _FIXTURE_GROUP_IDS
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
        events = list(
            run_demo_stream(
                agent.compiled_graph,
                graph_input([note]),
                record_path=path,
                handler=handler,
                clock=_step_clock(),
            )
        )
    finally:
        agent.compiled_graph.stream = original_stream  # type: ignore[method-assign]
    return len(events)


if __name__ == "__main__":
    count_written = generate_fixture()
    size_kb = FIXTURE_PATH.stat().st_size / 1024
    print(f"Wrote {count_written} events to {FIXTURE_PATH} ({size_kb:.1f} KiB).")
