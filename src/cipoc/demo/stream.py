"""Phase 1 — drive the graph, merge the two taps, yield one ordered event stream.

:func:`run_demo_stream` streams the compiled orchestrator graph exactly as the
progress runner does (``stream_mode=["values","tasks"]``, ``subgraphs=True``) but
with an :class:`~cipoc.demo.capture.LLMCaptureHandler` attached to the stream
config, then folds the two taps into a single ordered list of
:class:`~cipoc.demo.events.DemoEvent`\\ s and optionally records them to a JSONL
trace as they are produced.

**Ordering.** LangGraph's stream is pull-based: by the time an item is yielded,
the node work that produced it — including every LLM callback it fired — has
already run. So after each Tap-1 item we drain whatever Tap-2 captures have
accumulated since the last item and emit them first. A model call made inside
node *X* therefore lands between *X*'s ``task_start`` and its ``task_end``, which
is where the presenter expects to see it.

The merge core (:func:`merge_events`) is deliberately split out from the graph
driver so it can be tested against a scripted item sequence with no live LLM.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from itertools import count
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from langgraph.graph.state import CompiledStateGraph

from cipoc.utils.progress.events import ProgressEvent, normalize

from .capture import LLMCaptureHandler
from .events import DemoEvent, LLMCall
from .mapping import infer_agent, map_node_id
from .serialize import to_jsonable
from .trace import TraceWriter


def _with_callback(
    config: Mapping[str, Any] | None, handler: LLMCaptureHandler
) -> dict[str, Any]:
    """Return a copy of ``config`` with ``handler`` added to its callbacks.

    Preserves any caller config (e.g. the orchestrator's ``max_concurrency``) and
    any callbacks already present, so instrumentation is purely additive.
    """
    merged = dict(config or {})
    callbacks = merged.get("callbacks")
    if callbacks is None:
        merged["callbacks"] = [handler]
    elif isinstance(callbacks, list):
        merged["callbacks"] = [*callbacks, handler]
    else:
        # A single handler or a CallbackManager — keep it alongside ours.
        merged["callbacks"] = [callbacks, handler]
    return merged


def _llm_event(seq: int, t: float, call: LLMCall) -> DemoEvent:
    return DemoEvent(
        seq=seq,
        t=t,
        type="llm_call",
        node=call.node,
        namespace=call.namespace,
        map_node_id=map_node_id(call.node, call.namespace),
        agent=infer_agent(call.namespace),
        payload=call.to_dict(),
        error=call.error,
    )


def _tap1_event(seq: int, t: float, event: ProgressEvent) -> DemoEvent:
    if event.kind == "values":
        return DemoEvent(
            seq=seq,
            t=t,
            type="values",
            namespace=event.namespace,
            agent=infer_agent(event.namespace),
            payload=to_jsonable(event.payload),
        )
    return DemoEvent(
        seq=seq,
        t=t,
        type=event.kind,
        node=event.node,
        task_id=event.task_id,
        namespace=event.namespace,
        map_node_id=map_node_id(event.node, event.namespace),
        agent=infer_agent(event.namespace),
        payload=to_jsonable(event.payload),
        error=to_jsonable(event.error),
    )


def merge_events(
    raw_items: Iterable[Any],
    handler: LLMCaptureHandler,
    *,
    subgraphs: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[DemoEvent]:
    """Fold raw stream items + captured LLM calls into ordered ``DemoEvent``\\ s.

    ``raw_items`` is the live ``graph.stream(...)`` iterator (or a scripted
    stand-in in tests); ``handler`` is the same capture handler attached to that
    stream's config, whose ``calls`` grow as nodes execute. Brackets the run with
    ``run_start`` / ``run_end`` and assigns each event a monotonic ``seq`` and an
    elapsed ``t`` from ``clock``.
    """
    start = clock()
    seq = count()
    emitted_calls = 0

    yield DemoEvent(seq=next(seq), t=0.0, type="run_start")

    for raw_item in raw_items:
        event = normalize(raw_item, subgraphs=subgraphs)
        if event is None:
            continue
        # Captures made during the just-finished step belong *before* this item.
        calls = handler.snapshot()
        for call in calls[emitted_calls:]:
            yield _llm_event(next(seq), clock() - start, call)
        emitted_calls = len(calls)
        yield _tap1_event(next(seq), clock() - start, event)

    # Trailing captures from the final node, if any, before we close the run.
    calls = handler.snapshot()
    for call in calls[emitted_calls:]:
        yield _llm_event(next(seq), clock() - start, call)

    yield DemoEvent(seq=next(seq), t=clock() - start, type="run_end")


def run_demo_stream(
    graph: CompiledStateGraph,
    graph_input: Any,
    *,
    config: Mapping[str, Any] | None = None,
    record_path: str | Path | None = None,
    subgraphs: bool = True,
    handler: LLMCaptureHandler | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[DemoEvent]:
    """Stream ``graph`` and yield merged ``DemoEvent``\\ s (Tap 1 + Tap 2).

    When ``record_path`` is set, each event is flushed to a JSONL trace as it is
    yielded, so an interrupted run still leaves a replayable partial trace. Pass a
    ``handler`` to reuse an existing capture handler (tests do this); otherwise a
    fresh one is created and attached to the stream config.
    """
    handler = handler or LLMCaptureHandler()
    stream_config = _with_callback(config, handler)
    raw_items = graph.stream(
        graph_input,
        stream_mode=["values", "tasks"],
        subgraphs=subgraphs,
        config=stream_config,
    )

    writer_cm = TraceWriter(record_path) if record_path is not None else nullcontext()
    with writer_cm as writer:
        for event in merge_events(
            raw_items, handler, subgraphs=subgraphs, clock=clock
        ):
            if writer is not None:
                writer.write(event)
            yield event


def record_demo_trace(
    graph: CompiledStateGraph,
    graph_input: Any,
    record_path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    subgraphs: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run ``graph`` to completion, recording a trace; return the event count.

    A thin driver for the headless ``record`` path: it consumes
    :func:`run_demo_stream` for its side effect (writing ``record_path``) without
    the caller having to iterate the generator.
    """
    return sum(
        1
        for _ in run_demo_stream(
            graph,
            graph_input,
            config=config,
            record_path=record_path,
            subgraphs=subgraphs,
            clock=clock,
        )
    )


__all__ = ["merge_events", "run_demo_stream", "record_demo_trace"]
