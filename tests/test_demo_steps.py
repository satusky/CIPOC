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
            _root_start(0, "extract_branch",
                        payload={"requested_variables": {"name": "A"}}),
            _root_start(1, "extract_branch",
                        payload={"requested_variables": {"name": "B"}}),
        ]
        steps = build_steps(events)
        self.assertEqual(steps[0].title, "Extract group 1")
        self.assertEqual(steps[1].title, "Extract group 2")

    def test_parallel_note_fan_out_collapses_into_one_step(self):
        # The interleaved note_branch instances (all one map node) collapse into
        # a single "Characterize notes" step marked as a fan-out, rather than one
        # empty "active" step per note plus a final step holding everyone's work.
        events = [
            _root_start(0, "note_branch", payload={"note_id": 1, "note_type": "A"},
                        map_id="scanner_initialize"),
            _root_start(1, "note_branch", payload={"note_id": 2, "note_type": "B"},
                        map_id="scanner_initialize"),
            _nested(2, "summarize_note", ("note_branch:t",)),
            _root_start(3, "characterize_corpus", map_id="characterize_corpus"),
        ]
        steps = build_steps(events)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].title, "Characterize notes")
        self.assertTrue(steps[0].fanout)
        self.assertEqual(steps[0].start_seq, 0)
        self.assertEqual(steps[0].end_seq, 2)  # spans both notes + nested work
        self.assertFalse(steps[1].fanout)

    def test_scan_notes_folds_into_the_characterization_step(self):
        # scan_notes is just the fan-out that hands each note to a note_branch,
        # so it shares the step with the instances it spawned — and the extended
        # step must still be a fan-out step (per-note cards, later instances
        # collapsing into it) rather than reverting to a plain one.
        events = [
            _root_start(0, "scan_notes", map_id="fan_out_notes"),
            DemoEvent(seq=1, t=0.1, type="task_end", node="scan_notes", task_id="t",
                      namespace=(), map_node_id="fan_out_notes", agent="orchestrator"),
            _root_start(2, "note_branch", payload={"note_id": 1, "note_type": "A"},
                        map_id="scanner_initialize"),
            _root_start(3, "note_branch", payload={"note_id": 2, "note_type": "B"},
                        map_id="scanner_initialize"),
            _nested(4, "summarize_note", ("note_branch:t",)),
            _root_start(5, "characterize_corpus", map_id="characterize_corpus"),
        ]
        steps = build_steps(events)
        self.assertEqual([s.title for s in steps],
                         ["Scan & characterize notes", "Characterize corpus"])
        self.assertTrue(steps[0].fanout)
        self.assertEqual(steps[0].node, "note_branch")
        self.assertEqual((steps[0].start_seq, steps[0].end_seq), (0, 4))

    def test_check_state_and_plan_extraction_share_one_step(self):
        # They are two root tasks describing one decision (are there groups left,
        # and which) — the presenter should not have to advance between them.
        events = [
            _root_start(0, "check_state", map_id="check_state"),
            DemoEvent(seq=1, t=0.1, type="task_end", node="check_state", task_id="t",
                      namespace=(), map_node_id="check_state", agent="orchestrator"),
            _root_start(2, "plan_extraction", map_id="plan_extraction"),
            DemoEvent(seq=3, t=0.3, type="task_end", node="plan_extraction", task_id="t",
                      namespace=(), map_node_id="plan_extraction", agent="orchestrator"),
            _root_start(4, "finalize_case", map_id="finalize_case"),
        ]
        steps = build_steps(events)
        self.assertEqual([s.title for s in steps],
                         ["Check state & plan extraction", "Finalize case"])
        self.assertEqual(steps[0].start_seq, 0)
        self.assertEqual(steps[0].end_seq, 3)
        # The merged step re-centers on the incoming node, which is what Panel 2
        # keys the extraction-plan view off.
        self.assertEqual(steps[0].node, "plan_extraction")
        self.assertEqual(steps[0].map_node_id, "plan_extraction")

    def test_check_state_alone_still_gets_its_own_step(self):
        # The final gate ("no groups remain") has no plan after it.
        events = [
            _root_start(0, "check_state", map_id="check_state"),
            _root_start(1, "finalize_case", map_id="finalize_case"),
        ]
        self.assertEqual([s.title for s in build_steps(events)],
                         ["Check state", "Finalize case"])

    def test_step_carries_the_opening_task_id(self):
        # Panel 2 selects a group's variable instances by this task_id prefix.
        events = [_root_start(0, "extract_branch", payload={"requested_variables": {}})]
        self.assertEqual(build_steps(events)[0].task_id, "t")

    def test_subtitle_from_payload(self):
        events = [
            _root_start(0, "extract_branch",
                        payload={"requested_variables": {"name": "Lymph Nodes", "group_id": "ln"}}),
        ]
        steps = build_steps(events)
        self.assertEqual(steps[0].subtitle, "Lymph Nodes")


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
        self.assertIn("Check state & plan extraction", titles)
        self.assertIn("Finalize case", titles)
        # The gate and the plan it produces are never two separate steps.
        self.assertNotIn("Plan extraction", titles)
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
