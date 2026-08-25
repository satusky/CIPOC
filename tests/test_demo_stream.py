"""Merge (Tap 1 + Tap 2) ordering, config plumbing, and end-to-end record/replay."""

import tempfile
import unittest
from itertools import count
from pathlib import Path

from cipoc.demo.capture import LLMCaptureHandler
from cipoc.demo.events import LLMCall
from cipoc.demo.stream import _with_callback, merge_events, run_demo_stream
from cipoc.demo.trace import read_trace
from cipoc.utils.progress.events import normalize

from tests._demo_fixture import (
    _FIXTURE_GROUP_IDS,
    _FIXTURE_NOTE_ID,
    _REPAIR_ITEM_ID,
    _step_clock,
)
from tests.fake_orchestrator import (
    Outcome,
    Script,
    build_fake_orchestrator,
    graph_input,
    load_notes,
)


def _task_start(ns, node, task_id="t"):
    return (ns, "tasks", {"id": task_id, "name": node, "input": {"note": "n"}})


def _task_end(ns, node, task_id="t"):
    return (ns, "tasks", {"id": task_id, "name": node, "result": {"ok": True}})


def _values(ns, payload):
    return (ns, "values", payload)


class WithCallbackTests(unittest.TestCase):
    def test_adds_handler_when_no_config(self):
        handler = LLMCaptureHandler()
        self.assertEqual(_with_callback(None, handler)["callbacks"], [handler])

    def test_preserves_existing_config_keys_and_callbacks(self):
        handler = LLMCaptureHandler()
        existing = object()
        merged = _with_callback({"max_concurrency": 4, "callbacks": [existing]}, handler)
        self.assertEqual(merged["max_concurrency"], 4)
        self.assertEqual(merged["callbacks"], [existing, handler])

    def test_wraps_a_non_list_callbacks_value(self):
        handler = LLMCaptureHandler()
        manager = object()
        merged = _with_callback({"callbacks": manager}, handler)
        self.assertEqual(merged["callbacks"], [manager, handler])


class MergeOrderingTests(unittest.TestCase):
    def _clock(self):
        return _step_clock(1.0)

    def test_brackets_with_run_start_and_run_end(self):
        events = list(merge_events(iter(()), LLMCaptureHandler(), clock=self._clock()))
        self.assertEqual([e.type for e in events], ["run_start", "run_end"])
        self.assertEqual([e.seq for e in events], [0, 1])

    def test_llm_call_lands_between_its_nodes_start_and_end(self):
        handler = LLMCaptureHandler()
        ns = ("note_branch:1",)

        def raw_items():
            # summarize_note starts; its model call is captured during execution,
            # i.e. before the next stream item (its task_end) is pulled.
            yield _task_start(ns, "summarize_note")
            handler.calls.append(
                LLMCall(node="summarize_note", namespace=(*ns, "summarize_note:t"), run_id="r")
            )
            yield _task_end(ns, "summarize_note")

        events = list(merge_events(raw_items(), handler, clock=self._clock()))
        types = [e.type for e in events]
        self.assertEqual(types, ["run_start", "task_start", "llm_call", "task_end", "run_end"])
        # The llm_call resolved to the scanner's summarize map node.
        llm = events[types.index("llm_call")]
        self.assertEqual(llm.map_node_id, "scanner_summarize_note")
        self.assertEqual(llm.agent, "scanner")

    def test_trailing_captures_are_emitted_before_run_end(self):
        handler = LLMCaptureHandler()

        def raw_items():
            yield _task_start(("extract:1",), "extract_individual_value")
            # Capture appended after the LAST stream item is consumed.
            handler.calls.append(
                LLMCall(node="extract_individual_value", namespace=("extract:1",), run_id="r")
            )

        events = list(merge_events(raw_items(), handler, clock=self._clock()))
        self.assertEqual(
            [e.type for e in events],
            ["run_start", "task_start", "llm_call", "run_end"],
        )

    def test_seq_is_dense_and_time_is_monotonic(self):
        handler = LLMCaptureHandler()
        raw = [
            _task_start((), "scan_notes"),
            _values((), {"note_corpus": {}}),
            _task_end((), "scan_notes"),
        ]
        events = list(merge_events(iter(raw), handler, clock=self._clock()))
        self.assertEqual([e.seq for e in events], list(range(len(events))))
        times = [e.t for e in events]
        self.assertEqual(times, sorted(times))

    def test_values_event_has_no_node_but_keeps_payload(self):
        handler = LLMCaptureHandler()
        raw = [_values((), {"variable_results": []})]
        events = list(merge_events(iter(raw), handler, clock=self._clock()))
        values = next(e for e in events if e.type == "values")
        self.assertEqual(values.node, "")
        self.assertIsNone(values.map_node_id)
        self.assertEqual(values.payload, {"variable_results": []})


class RecordReplayEndToEndTests(unittest.TestCase):
    """A hermetic full run (no LLM): streaming, recording, and replay all agree."""

    @classmethod
    def setUpClass(cls):
        note = next(n for n in load_notes() if n.note_id == _FIXTURE_NOTE_ID)
        script = Script(outcomes={_REPAIR_ITEM_ID: Outcome(repairs=1)})
        agent = build_fake_orchestrator(script)
        agent._target_variables = [
            g for g in agent._target_variables if g.group_id in _FIXTURE_GROUP_IDS
        ]
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "trace.jsonl"
        cls.streamed = list(
            run_demo_stream(
                agent.compiled_graph,
                graph_input([note]),
                record_path=path,
                clock=_step_clock(),
            )
        )
        cls.replayed = read_trace(path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_recorded_trace_matches_the_streamed_events(self):
        self.assertEqual(
            [e.to_dict() for e in self.replayed],
            [e.to_dict() for e in self.streamed],
        )

    def test_run_is_bracketed_and_seq_is_dense(self):
        self.assertEqual(self.streamed[0].type, "run_start")
        self.assertEqual(self.streamed[-1].type, "run_end")
        self.assertEqual([e.seq for e in self.streamed], list(range(len(self.streamed))))

    def test_every_task_event_maps_to_a_known_node(self):
        # No LLM fired (fakes), so no llm_call events — but every task event must
        # still resolve to a real map node id (the mapping covers the topology).
        task_events = [e for e in self.streamed if e.type in ("task_start", "task_end")]
        self.assertTrue(task_events)
        self.assertTrue(all(e.map_node_id is not None for e in task_events))

    def test_payloads_are_json_safe(self):
        import json

        for event in self.streamed:
            json.dumps(event.to_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
