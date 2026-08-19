import io
import os
import threading
import unittest
from unittest.mock import Mock, patch

from cipoc.agents.extractor import ExtractorAgent
from cipoc.agents.note_retriever import NoteRetrieverAgent
from cipoc.agents.note_scanner import NoteScannerAgent
from cipoc.models import ClinicalNote
from cipoc.utils.progress.renderers import AnsiAltScreen, NotebookDisplay, PlainLog
from cipoc.utils.progress.runner import _select_renderer, run_with_progress


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
    def test_forwards_graph_config(self):
        graph = _Graph([("values", {"answer": 42})])

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=_RecordingRenderer(),
        ):
            run_with_progress(graph, {}, config={"max_concurrency": 32})

        self.assertEqual(
            graph.calls,
            [
                (
                    {},
                    {
                        "stream_mode": ["values", "tasks"],
                        "subgraphs": False,
                        "config": {"max_concurrency": 32},
                    },
                )
            ],
        )

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
        ), patch("builtins.input", return_value="") as read_input:
            run_with_progress(
                graph,
                {},
                description="Test",
                target_groups=groups,
                pause_before_summary=True,
            )

        read_input.assert_called_once_with()
        output = stream.getvalue()
        self.assertIn("\x1b[?1049l", output)
        summary = output.rsplit("\x1b[?1049l", 1)[1]
        self.assertIn("Date of Diagnosis", summary)
        self.assertIn("202601", summary)
        self.assertNotIn("\x1b", summary)

    def test_final_dashboard_stays_open_while_waiting_for_enter(self):
        stream = _TTY()
        renderer = AnsiAltScreen(
            stream,
            color=False,
            size_provider=lambda: os.terminal_size((80, 24)),
        )
        graph = _Graph([("values", {"answer": 1})])

        def press_enter():
            waiting_output = stream.getvalue()
            self.assertIn("Press Enter to view report", waiting_output)
            self.assertIn("\x1b[?1049h", waiting_output)
            self.assertNotIn("\x1b[?1049l", waiting_output)
            return ""

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ), patch("builtins.input", side_effect=press_enter):
            run_with_progress(graph, {}, pause_before_summary=True)

        output = stream.getvalue()
        exit_position = output.index("\x1b[?1049l")
        self.assertLess(exit_position, output.rindex("CIPOC"))

    def test_pause_is_skipped_for_non_tty_notebook_failure_and_default_runs(self):
        renderers_and_graphs = (
            (PlainLog(io.StringIO()), _Graph([("values", {"answer": 1})]), True),
            (
                NotebookDisplay(
                    display_fn=lambda value, **kwargs: None,
                    html_factory=lambda value: value,
                ),
                _Graph([("values", {"answer": 1})]),
                True,
            ),
            (
                AnsiAltScreen(
                    _TTY(),
                    color=False,
                    size_provider=lambda: os.terminal_size((80, 24)),
                ),
                _Graph([], ValueError("failed")),
                True,
            ),
            (
                AnsiAltScreen(
                    _TTY(),
                    color=False,
                    size_provider=lambda: os.terminal_size((80, 24)),
                ),
                _Graph([("values", {"answer": 1})]),
                False,
            ),
        )
        for renderer, graph, pause in renderers_and_graphs:
            with self.subTest(renderer=type(renderer).__name__, pause=pause):
                read_input = Mock()
                with patch(
                    "cipoc.utils.progress.runner._select_renderer",
                    return_value=renderer,
                ), patch("builtins.input", read_input):
                    if graph.error is None:
                        run_with_progress(graph, {}, pause_before_summary=pause)
                    else:
                        with self.assertRaisesRegex(ValueError, "failed"):
                            run_with_progress(graph, {}, pause_before_summary=pause)
                read_input.assert_not_called()

    def test_eof_restores_terminal_and_prints_report(self):
        stream = _TTY()
        renderer = AnsiAltScreen(
            stream,
            color=False,
            size_provider=lambda: os.terminal_size((80, 24)),
        )
        graph = _Graph([("values", {"answer": 1})])

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ), patch("builtins.input", side_effect=EOFError):
            run_with_progress(graph, {}, pause_before_summary=True)

        output = stream.getvalue()
        self.assertIn("\x1b[?25h\x1b[?1049l", output)
        self.assertIn("CIPOC", output.rsplit("\x1b[?1049l", 1)[1])

    def test_pause_interruption_restores_terminal(self):
        stream = _TTY()
        renderer = AnsiAltScreen(
            stream,
            color=False,
            size_provider=lambda: os.terminal_size((80, 24)),
        )
        graph = _Graph([("values", {"answer": 1})])

        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=renderer,
        ), patch("builtins.input", side_effect=KeyboardInterrupt("cancelled")):
            with self.assertRaisesRegex(KeyboardInterrupt, "cancelled"):
                run_with_progress(graph, {}, pause_before_summary=True)

        self.assertIn("\x1b[?25h\x1b[?1049l", stream.getvalue())


class ProgressDisabledTests(unittest.TestCase):
    def test_subagents_invoke_graph_without_creating_progress_displays(self):
        class FakeGraph:
            def __init__(self, result):
                self.result = result
                self.inputs = []

            def invoke(self, graph_input):
                self.inputs.append(graph_input)
                return self.result

        note = ClinicalNote(note_id=1, date="2026-01-01", note_type="test", content="")

        scanner = object.__new__(NoteScannerAgent)
        scanner._graph = FakeGraph({})
        self.assertEqual(scanner.run(note, progress=False).note_id, 1)

        retriever = object.__new__(NoteRetrieverAgent)
        retriever._graph = FakeGraph({"relevant_note_ids": [1]})
        self.assertEqual(retriever.run({}, progress=False), [1])

        extractor = object.__new__(ExtractorAgent)
        extractor._graph = FakeGraph({"extracted_values": None})
        self.assertIsNone(extractor.run({}, progress=False).extracted_values)


if __name__ == "__main__":
    unittest.main()
