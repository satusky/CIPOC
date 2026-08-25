"""Trace format: DemoEvent/LLMCall serialization and JSONL read/write/append."""

import json
import tempfile
import unittest
from pathlib import Path

from cipoc.demo.events import DemoEvent, LLMCall
from cipoc.demo.trace import TraceWriter, iter_trace, read_trace, write_trace

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "demo_trace.jsonl"


def _event(seq: int, **kw) -> DemoEvent:
    kw.setdefault("t", seq * 0.5)
    kw.setdefault("type", "task_start")
    return DemoEvent(seq=seq, **kw)


class DemoEventSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_fields_and_namespace_tuple(self):
        event = DemoEvent(
            seq=3,
            t=1.25,
            type="task_end",
            node="summarize_note",
            namespace=("note_branch:abc", "summarize_note:def"),
            map_node_id="scanner_summarize_note",
            agent="scanner",
            payload={"summary": "x", "note_ids": {"1": "a"}},
            error=None,
        )
        restored = DemoEvent.from_dict(event.to_dict())
        self.assertEqual(restored, event)
        # namespace survives as a tuple, not the list it is stored as.
        self.assertIsInstance(restored.namespace, tuple)

    def test_to_dict_stores_namespace_as_list_and_rounds_time(self):
        event = DemoEvent(seq=1, t=1.2345678, type="values", namespace=("a:b",))
        data = event.to_dict()
        self.assertEqual(data["namespace"], ["a:b"])
        self.assertEqual(data["t"], 1.234568)

    def test_from_dict_tolerates_missing_optional_fields(self):
        restored = DemoEvent.from_dict({"seq": 0, "type": "run_start"})
        self.assertEqual(restored.node, "")
        self.assertEqual(restored.namespace, ())
        self.assertIsNone(restored.map_node_id)


class LLMCallSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_namespace_tuple(self):
        call = LLMCall(
            node="extract_individual_value",
            namespace=("extract:1", "variable_branch:2"),
            run_id="r1",
            parent_run_id="r0",
            model="m",
            prompt_messages=[{"role": "human", "content": "hi"}],
            reasoning=None,
            response='{"value": "1"}',
            usage={"total_tokens": 10},
        )
        restored = LLMCall.from_dict(call.to_dict())
        self.assertEqual(restored, call)
        self.assertIsInstance(restored.namespace, tuple)

    def test_to_dict_is_json_serializable(self):
        call = LLMCall(node="n", namespace=("a:b",), run_id="r")
        json.dumps(call.to_dict())  # must not raise


class TraceIOTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            DemoEvent(seq=0, t=0.0, type="run_start"),
            _event(1, node="scan_notes", namespace=("scan_notes:1",)),
            DemoEvent(seq=2, t=1.0, type="values", payload={"note_corpus": {"1": {}}}),
            DemoEvent(seq=3, t=1.5, type="run_end"),
        ]

    def test_write_then_read_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            count = write_trace(path, self.events)
            self.assertEqual(count, len(self.events))
            restored = read_trace(path)
            self.assertEqual(
                [e.to_dict() for e in restored],
                [e.to_dict() for e in self.events],
            )

    def test_write_trace_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "dir" / "trace.jsonl"
            write_trace(path, self.events)
            self.assertTrue(path.exists())

    def test_iter_trace_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            write_trace(path, self.events)
            with path.open("a") as handle:
                handle.write("\n   \n")
            self.assertEqual(len(list(iter_trace(path))), len(self.events))

    def test_trace_writer_appends_and_flushes_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            writer = TraceWriter(path)
            writer.write(self.events[0])
            writer.write(self.events[1])
            # A partial (still-open) trace is already readable up to the last line.
            partial = read_trace(path)
            self.assertEqual(len(partial), 2)
            writer.close()

    def test_trace_writer_context_manager_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with TraceWriter(path) as writer:
                for event in self.events:
                    writer.write(event)
                self.assertEqual(writer.count, len(self.events))
            self.assertEqual(len(read_trace(path)), len(self.events))


class FixtureTraceTests(unittest.TestCase):
    """Invariants the committed fixture must satisfy for replay to be well-formed."""

    @classmethod
    def setUpClass(cls):
        cls.events = read_trace(FIXTURE)

    def test_fixture_is_nonempty(self):
        self.assertGreater(len(self.events), 0)

    def test_seq_is_dense_and_monotonic(self):
        self.assertEqual([e.seq for e in self.events], list(range(len(self.events))))

    def test_brackets_are_run_start_and_run_end(self):
        self.assertEqual(self.events[0].type, "run_start")
        self.assertEqual(self.events[-1].type, "run_end")

    def test_every_event_type_is_present(self):
        seen = {e.type for e in self.events}
        self.assertEqual(
            seen,
            {"run_start", "task_start", "task_end", "values", "llm_call", "run_end"},
        )

    def test_llm_calls_carry_a_serializable_payload(self):
        llm_events = [e for e in self.events if e.type == "llm_call"]
        self.assertTrue(llm_events)
        for event in llm_events:
            self.assertIsNotNone(event.payload)
            self.assertIn("prompt_messages", event.payload)
            self.assertIsNotNone(event.map_node_id)

    def test_fixture_round_trips_through_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "copy.jsonl"
            write_trace(path, self.events)
            self.assertEqual(
                [e.to_dict() for e in read_trace(path)],
                [e.to_dict() for e in self.events],
            )


if __name__ == "__main__":
    unittest.main()
