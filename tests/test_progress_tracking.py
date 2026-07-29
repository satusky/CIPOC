import asyncio
import io
import os
import threading
import unittest
from unittest.mock import patch

from cipoc.agents.extractor import ExtractorAgent
from cipoc.agents.note_retriever import NoteRetrieverAgent
from cipoc.agents.note_scanner import NoteScannerAgent
from cipoc.models import ClinicalNote
from cipoc.utils.progress.renderers import AnsiAltScreen, NotebookDisplay, PlainLog
from cipoc.utils.progress.runner import (
    _select_renderer,
    arun_with_progress,
    astream_events,
    astream_with_progress,
    run_with_progress,
)


class _TTY(io.StringIO):
    def isatty(self):
        return True


class _Graph:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error
        self.calls = []

    def stream(self, graph_input, **kwargs):
        self.calls.append((graph_input, kwargs))
        yield from self.events
        if self.error is not None:
            raise self.error


class _AsyncGraph(_Graph):
    """The same stub over ``astream``, so the async runner is driven by an event
    sequence identical to the sync one and the two can be compared directly."""

    async def astream(self, graph_input, **kwargs):
        self.calls.append((graph_input, kwargs))
        for event in self.events:
            # Hand control back between items: a runner that quietly blocked the
            # loop would still pass without this.
            await asyncio.sleep(0)
            yield event
        if self.error is not None:
            raise self.error


class _RecordingRenderer:
    min_interval = 0.01

    def __init__(self):
        self.paints = []
        self.closed = False

    def paint(self, snapshot, **kwargs):
        self.paints.append((snapshot, kwargs))
        return True

    def close(self):
        self.closed = True


class _BrokenRenderer(_RecordingRenderer):
    def paint(self, snapshot, **kwargs):
        raise OSError("display failed")

    def close(self):
        raise OSError("close failed")


class _BlockedRenderer(_RecordingRenderer):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.close_finished = threading.Event()

    def paint(self, snapshot, **kwargs):
        self.release.wait()
        return super().paint(snapshot, **kwargs)

    def close(self):
        super().close()
        self.close_finished.set()


class _InterruptedCloseRenderer(_RecordingRenderer):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.close_calls == 1:
            raise KeyboardInterrupt("during close")
        super().close()


class RendererSelectionTests(unittest.TestCase):
    def test_notebook_wins_over_terminal(self):
        renderer = _select_renderer(
            _TTY(),
            notebook=True,
            size_provider=lambda: os.terminal_size((80, 24)),
        )
        self.assertIsInstance(renderer, NotebookDisplay)

    def test_usable_terminal_gets_alternate_screen(self):
        renderer = _select_renderer(
            _TTY(),
            notebook=False,
            size_provider=lambda: os.terminal_size((80, 12)),
        )
        self.assertIsInstance(renderer, AnsiAltScreen)

    def test_redirected_or_short_terminal_gets_plain_log(self):
        self.assertIsInstance(
            _select_renderer(io.StringIO(), notebook=False),
            PlainLog,
        )
        self.assertIsInstance(
            _select_renderer(
                _TTY(),
                notebook=False,
                size_provider=lambda: os.terminal_size((80, 11)),
            ),
            PlainLog,
        )


class ProgressRunnerTests(unittest.TestCase):
    def test_streams_tasks_and_values_and_returns_last_root_state(self):
        result = {"answer": 42}
        graph = _Graph(
            [
                ("tasks", {"id": "1", "name": "initialize", "input": {}}),
                (
                    "tasks",
                    {
                        "id": "1",
                        "name": "initialize",
                        "result": {},
                        "error": None,
                    },
                ),
                ("values", result),
            ]
        )
        renderer = _RecordingRenderer()

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ):
            actual = run_with_progress(graph, {"input": True}, description="Test")

        self.assertIs(actual, result)
        self.assertEqual(
            graph.calls,
            [
                (
                    {"input": True},
                    {"stream_mode": ["values", "tasks"], "subgraphs": False},
                )
            ],
        )
        final_snapshot, final_kwargs = renderer.paints[-1]
        self.assertTrue(final_snapshot.finished)
        self.assertEqual(final_snapshot.nodes[0].state, "ok")
        self.assertTrue(final_kwargs["final"])
        self.assertTrue(renderer.closed)

    def test_keyboard_interrupt_marks_final_snapshot_and_closes_renderer(self):
        graph = _Graph([], KeyboardInterrupt("cancelled"))
        renderer = _RecordingRenderer()

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_with_progress(graph, {})

        snapshot, kwargs = renderer.paints[-1]
        self.assertEqual(snapshot.fatal, "cancelled")
        self.assertTrue(snapshot.finished)
        self.assertTrue(kwargs["final"])
        self.assertTrue(renderer.closed)

    def test_renderer_failure_does_not_replace_result_or_graph_error(self):
        successful = _Graph([("values", {"answer": 1})])
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=_BrokenRenderer(),
        ):
            self.assertEqual(run_with_progress(successful, {}), {"answer": 1})

        failing = _Graph([], ValueError("graph failed"))
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=_BrokenRenderer(),
        ):
            with self.assertRaisesRegex(ValueError, "graph failed"):
                run_with_progress(failing, {})

    def test_thread_start_failure_does_not_prevent_graph_run(self):
        graph = _Graph([("values", {"answer": 1})])
        renderer = _RecordingRenderer()

        with (
            patch(
                "cipoc.utils.progress.runner._select_renderer",
                return_value=renderer,
            ),
            patch(
                "cipoc.utils.progress.runner._RepaintLoop.start",
                side_effect=RuntimeError("no threads"),
            ),
        ):
            self.assertEqual(run_with_progress(graph, {}), {"answer": 1})

        self.assertTrue(renderer.closed)
        self.assertTrue(renderer.paints[-1][1]["final"])

    def test_blocked_renderer_does_not_block_graph_return(self):
        graph = _Graph([("values", {"answer": 1})])
        renderer = _BlockedRenderer()
        try:
            with (
                patch(
                    "cipoc.utils.progress.runner._select_renderer",
                    return_value=renderer,
                ),
                patch("cipoc.utils.progress.runner._REPAINT_JOIN_TIMEOUT", 0.02),
            ):
                self.assertEqual(run_with_progress(graph, {}), {"answer": 1})
            self.assertFalse(renderer.closed)
        finally:
            renderer.release.set()

        self.assertTrue(renderer.close_finished.wait(1.0))

    def test_close_interruption_is_retried_after_terminal_restore(self):
        graph = _Graph([("values", {"answer": 1})])
        renderer = _InterruptedCloseRenderer()

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ):
            with self.assertRaisesRegex(KeyboardInterrupt, "during close"):
                run_with_progress(graph, {})

        self.assertEqual(renderer.close_calls, 2)
        self.assertTrue(renderer.closed)

    def test_show_branches_controls_synthetic_variable_table(self):
        graph_input = {
            "requested_variables": {
                "group_id": "group",
                "name": "Group",
                "variables": [{"item_id": 390, "name": "Date of Diagnosis"}],
            }
        }
        graph = _Graph([("values", {"relevant_note_ids": [1]})])
        compact = _RecordingRenderer()
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=compact,
        ):
            run_with_progress(graph, graph_input, show_branches=False)
        self.assertEqual(compact.paints[-1][0].mode, "compact")

        table = _RecordingRenderer()
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=table,
        ):
            run_with_progress(graph, graph_input, show_branches=True)
        self.assertEqual(table.paints[-1][0].mode, "standalone")
        self.assertEqual(table.paints[-1][0].total_variables, 1)

    def test_plain_log_keeps_fast_intermediate_transitions(self):
        stream = io.StringIO()
        graph = _Graph(
            [
                ("tasks", {"id": "1", "name": "initialize", "input": {}}),
                (
                    "tasks",
                    {"id": "1", "name": "initialize", "result": {}, "error": None},
                ),
                ("values", {"answer": 1}),
            ]
        )
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=PlainLog(stream),
        ):
            run_with_progress(graph, {}, description="Fast")

        lines = stream.getvalue().splitlines()
        self.assertIn("node initialize [deterministic]: running (0/1)", lines)
        self.assertIn("node initialize [deterministic]: complete (1/1)", lines)

    def test_alternate_screen_restores_before_expanded_summary(self):
        stream = _TTY()
        renderer = AnsiAltScreen(
            stream,
            color=False,
            size_provider=lambda: os.terminal_size((80, 24)),
        )
        graph = _Graph(
            [
                (
                    "values",
                    {
                        "variable_results": {
                            390: {
                                "status": "extracted",
                                "value": "20260101",
                            }
                        }
                    },
                )
            ]
        )
        groups = [
            {
                "group_id": "initial",
                "name": "Initial",
                "variables": [{"item_id": 390, "name": "Date of Diagnosis"}],
            }
        ]

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ):
            run_with_progress(
                graph,
                {},
                description="Test",
                target_groups=groups,
            )

        output = stream.getvalue()
        self.assertIn("\x1b[?1049l", output)
        summary = output.rsplit("\x1b[?1049l", 1)[1]
        self.assertIn("Date of Diagnosis", summary)
        self.assertIn("202601", summary)
        self.assertNotIn("\x1b", summary)


TASK_EVENTS = [
    ("tasks", {"id": "1", "name": "initialize", "input": {}}),
    ("tasks", {"id": "1", "name": "initialize", "result": {}, "error": None}),
    ("values", {"answer": 42}),
]


class AsyncProgressRunnerTests(unittest.TestCase):
    """The async runner shares ``_ProgressSession`` with the sync one, so these
    pin the half that is not shared: pulling items off ``astream`` and handing
    each normalized event to the caller."""

    @staticmethod
    def _drain(coro):
        return asyncio.run(coro)

    def test_arun_returns_the_last_root_state_and_paints_the_same_frames(self):
        graph = _AsyncGraph(TASK_EVENTS)
        renderer = _RecordingRenderer()

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ):
            actual = self._drain(arun_with_progress(graph, {"input": True}, description="Test"))

        self.assertEqual(actual, {"answer": 42})
        self.assertEqual(
            graph.calls,
            [({"input": True}, {"stream_mode": ["values", "tasks"], "subgraphs": False})],
        )
        final_snapshot, final_kwargs = renderer.paints[-1]
        self.assertTrue(final_snapshot.finished)
        self.assertEqual(final_snapshot.nodes[0].state, "ok")
        self.assertTrue(final_kwargs["final"])
        self.assertTrue(renderer.closed)

    def test_the_sync_and_async_runners_agree_on_the_final_snapshot(self):
        sync_renderer, async_renderer = _RecordingRenderer(), _RecordingRenderer()
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=sync_renderer,
        ):
            run_with_progress(_Graph(TASK_EVENTS), {}, description="Test")
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=async_renderer,
        ):
            self._drain(arun_with_progress(_AsyncGraph(TASK_EVENTS), {}, description="Test"))

        sync_final, async_final = sync_renderer.paints[-1][0], async_renderer.paints[-1][0]
        self.assertEqual(sync_final.counts, async_final.counts)
        self.assertEqual(
            [(node.name, node.started, node.done) for node in sync_final.nodes],
            [(node.name, node.started, node.done) for node in async_final.nodes],
        )

    def test_astream_yields_every_normalized_event_as_it_arrives(self):
        graph = _AsyncGraph(TASK_EVENTS)

        async def collect():
            seen = []
            with patch(
                "cipoc.utils.progress.runner._select_renderer",
                return_value=_RecordingRenderer(),
            ):
                async for event in astream_with_progress(graph, {}):
                    seen.append(event.kind)
            return seen

        self.assertEqual(self._drain(collect()), ["task_start", "task_end", "values"])

    def test_updates_reach_the_caller_without_disturbing_the_dashboard(self):
        """The one reason ``astream_results`` may widen ``stream_mode``: the model
        ignores the kind, so the painted frames are unchanged."""
        graph = _AsyncGraph(
            [("updates", {"extract": {"variable_results": {}}}), ("values", {"answer": 1})]
        )
        renderer = _RecordingRenderer()

        async def collect():
            seen = []
            with patch(
                "cipoc.utils.progress.runner._select_renderer",
                return_value=renderer,
            ):
                async for event in astream_with_progress(
                    graph, {}, stream_mode=["values", "tasks", "updates"]
                ):
                    seen.append(event)
            return seen

        seen = self._drain(collect())
        self.assertEqual([event.kind for event in seen], ["updates", "values"])
        self.assertEqual(seen[0].writes, (("extract", {"variable_results": {}}),))
        self.assertEqual(
            graph.calls[0][1]["stream_mode"], ["values", "tasks", "updates"]
        )
        self.assertEqual(renderer.paints[-1][0].nodes, ())

    def test_a_graph_failure_propagates_and_still_closes_the_renderer(self):
        graph = _AsyncGraph([], ValueError("graph failed"))
        renderer = _RecordingRenderer()

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ):
            with self.assertRaisesRegex(ValueError, "graph failed"):
                self._drain(arun_with_progress(graph, {}))

        snapshot, kwargs = renderer.paints[-1]
        self.assertEqual(snapshot.fatal, "graph failed")
        self.assertTrue(snapshot.finished)
        self.assertTrue(kwargs["final"])
        self.assertTrue(renderer.closed)

    def test_a_run_with_no_root_state_is_an_error_in_both_modes(self):
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=_RecordingRenderer(),
        ):
            with self.assertRaisesRegex(RuntimeError, "no final state"):
                run_with_progress(_Graph([]), {})
            with self.assertRaisesRegex(RuntimeError, "no final state"):
                self._drain(arun_with_progress(_AsyncGraph([]), {}))

    def test_astream_events_streams_without_building_a_display(self):
        graph = _AsyncGraph(TASK_EVENTS)

        async def collect():
            return [
                event.kind
                async for event in astream_events(graph, {}, subgraphs=False)
            ]

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            side_effect=AssertionError("no renderer should be built"),
        ):
            self.assertEqual(
                self._drain(collect()), ["task_start", "task_end", "values"]
            )


class _FakeGraph:
    def __init__(self, result):
        self.result = result
        self.inputs = []

    def invoke(self, graph_input):
        self.inputs.append(graph_input)
        return self.result

    async def ainvoke(self, graph_input):
        return self.invoke(graph_input)


class ProgressDisabledTests(unittest.TestCase):
    NOTE = ClinicalNote(note_id=1, date="2026-01-01", type="test", content="")

    def test_subagents_invoke_graph_without_creating_progress_displays(self):
        scanner = object.__new__(NoteScannerAgent)
        scanner._graph = _FakeGraph({})
        self.assertEqual(scanner.run(self.NOTE, progress=False).note_id, 1)

        retriever = object.__new__(NoteRetrieverAgent)
        retriever._graph = _FakeGraph({"relevant_note_ids": [1]})
        self.assertEqual(retriever.run({}, progress=False), [1])

        extractor = object.__new__(ExtractorAgent)
        extractor._graph = _FakeGraph({"extracted_values": None})
        self.assertIsNone(extractor.run({}, progress=False).extracted_values)

    def test_subagents_await_the_graph_without_creating_progress_displays(self):
        """``arun(progress=False)`` must reach ``ainvoke`` directly. The display
        settings each ``arun`` now passes through are inert in that case, so a
        renderer built here would mean they had leaked into the no-progress path."""
        scanner = object.__new__(NoteScannerAgent)
        scanner._async = True
        scanner._graph = _FakeGraph({})

        retriever = object.__new__(NoteRetrieverAgent)
        retriever._async = True
        retriever._graph = _FakeGraph({"relevant_note_ids": [1]})

        extractor = object.__new__(ExtractorAgent)
        extractor._async = True
        extractor._graph = _FakeGraph({"extracted_values": None})

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            side_effect=AssertionError("no renderer should be built"),
        ):
            self.assertEqual(
                asyncio.run(scanner.arun(self.NOTE, progress=False)).note_id, 1
            )
            self.assertEqual(asyncio.run(retriever.arun({}, progress=False)), [1])
            self.assertIsNone(
                asyncio.run(extractor.arun({}, progress=False)).extracted_values
            )


if __name__ == "__main__":
    unittest.main()
