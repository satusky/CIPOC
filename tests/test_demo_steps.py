"""Phase 2 — grouping the event stream into presenter steps (pause boundaries)."""

import unittest
from pathlib import Path

from cipoc.demo.events import DemoEvent
from cipoc.demo.state import replay
from cipoc.demo.steps import build_steps
from cipoc.demo.trace import read_trace

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "demo_trace.jsonl"


def _root_start(seq, node, payload=None, map_id=None, agent="orchestrator"):
    return DemoEvent(seq=seq, t=seq * 0.1, type="task_start", node=node, task_id="t",
                     namespace=(), map_node_id=map_id, agent=agent, payload=payload)


def _nested(seq, node, ns):
    return DemoEvent(seq=seq, t=seq * 0.1, type="task_start", node=node, task_id="x",
                     namespace=ns, map_node_id="scanner_summarize_note", agent="scanner")


class BoundaryTests(unittest.TestCase):
    def test_empty_stream_has_no_steps(self):
        self.assertEqual(build_steps([]), [])

    def test_intro_step_precedes_first_root_task(self):
        events = [
            DemoEvent(seq=0, t=0.0, type="run_start"),
            DemoEvent(seq=1, t=0.1, type="values", namespace=(), payload={"note_corpus": {}}),
            _root_start(2, "initialize", map_id="initialize_case"),
            DemoEvent(seq=3, t=0.3, type="task_end", node="initialize", task_id="t",
                      namespace=(), map_node_id="initialize_case", agent="orchestrator"),
        ]
        steps = build_steps(events)
        self.assertEqual(steps[0].title, "Run start")
        self.assertEqual(steps[0].start_seq, 0)
        self.assertEqual(steps[0].end_seq, 1)
        self.assertEqual(steps[1].title, "Initialize case")

    def test_nested_events_belong_to_their_root_task_step(self):
        events = [
            _root_start(0, "note_branch", payload={"note_id": 51, "note_type": "Path"},
                        map_id="scanner_initialize"),
            _nested(1, "summarize_note", ("note_branch:1",)),
            _nested(2, "detect_concepts", ("note_branch:1",)),
            _root_start(3, "characterize_corpus", map_id="characterize_corpus"),
        ]
        steps = build_steps(events)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].start_seq, 0)
        self.assertEqual(steps[0].end_seq, 2)  # absorbs the nested events
        self.assertEqual(steps[1].node, "characterize_corpus")

    def test_repeated_nodes_get_an_instance_counter(self):
        events = [
            _root_start(0, "note_branch", payload={"note_id": 1, "note_type": "A"}),
            _root_start(1, "note_branch", payload={"note_id": 2, "note_type": "B"}),
        ]
        steps = build_steps(events)
        self.assertEqual(steps[0].title, "Characterize note 1")
        self.assertEqual(steps[1].title, "Characterize note 2")

    def test_subtitle_from_payload(self):
        events = [
            _root_start(0, "note_branch", payload={"note_id": 51, "note_type": "Pathology"}),
            _root_start(1, "extract_branch",
                        payload={"requested_variables": {"name": "Lymph Nodes", "group_id": "ln"}}),
        ]
        steps = build_steps(events)
        self.assertEqual(steps[0].subtitle, "Pathology #51")
        self.assertEqual(steps[1].subtitle, "Lymph Nodes")


class FixtureTilingTests(unittest.TestCase):
    """The steps must partition the whole trace with no gaps or overlaps."""

    @classmethod
    def setUpClass(cls):
        cls.events = read_trace(FIXTURE)
        cls.steps = build_steps(cls.events)

    def test_steps_are_nonempty_and_indexed_densely(self):
        self.assertTrue(self.steps)
        self.assertEqual([s.index for s in self.steps], list(range(len(self.steps))))

    def test_steps_tile_the_stream_without_gaps(self):
        self.assertEqual(self.steps[0].start_seq, self.events[0].seq)
        self.assertEqual(self.steps[-1].end_seq, self.events[-1].seq)
        for earlier, later in zip(self.steps, self.steps[1:]):
            self.assertEqual(later.start_seq, earlier.end_seq + 1)

    def test_replay_to_each_step_end_produces_a_valid_snapshot(self):
        for step in self.steps:
            state = replay(self.events[: step.end_seq + 1])
            state.snapshot()  # must not raise

    def test_expected_component_steps_are_present(self):
        titles = [s.title for s in self.steps]
        self.assertIn("Initialize case", titles)
        self.assertIn("Characterize corpus", titles)
        self.assertIn("Plan extraction", titles)
        self.assertIn("Finalize case", titles)
        # Two variable groups -> two extract steps.
        self.assertEqual(sum(t.startswith("Extract group") for t in titles), 2)

    def test_to_dict_round_trips_fields(self):
        step = self.steps[1]
        data = step.to_dict()
        self.assertEqual(data["index"], step.index)
        self.assertEqual(data["title"], step.title)
        self.assertEqual(data["end_seq"], step.end_seq)


if __name__ == "__main__":
    unittest.main()
