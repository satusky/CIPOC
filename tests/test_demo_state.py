"""Phase 2 — DemoState folds the merged event stream into presentable state."""

import json
import unittest
from pathlib import Path

from cipoc.demo.events import DemoEvent, LLMCall
from cipoc.demo.state import DemoState, NodeDetail, replay
from cipoc.demo.trace import read_trace

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "demo_trace.jsonl"


def _task_start(seq, node, task_id, namespace, map_id, agent, payload=None, t=None):
    return DemoEvent(
        seq=seq, t=seq * 0.1 if t is None else t, type="task_start",
        node=node, task_id=task_id, namespace=namespace,
        map_node_id=map_id, agent=agent, payload=payload,
    )


def _task_end(seq, node, task_id, namespace, map_id, agent, payload=None, error=None):
    return DemoEvent(
        seq=seq, t=seq * 0.1, type="task_end",
        node=node, task_id=task_id, namespace=namespace,
        map_node_id=map_id, agent=agent, payload=payload, error=error,
    )


class MapActivityTests(unittest.TestCase):
    """Active/visited map nodes and fan-out multiplicity from task events."""

    def test_task_start_marks_node_active_and_visited(self):
        state = DemoState()
        state.ingest(DemoEvent(seq=0, t=0.0, type="run_start"))
        state.ingest(_task_start(1, "summarize_note", "a", ("note_branch:1",),
                                 "scanner_summarize_note", "scanner"))
        snap = state.snapshot()
        self.assertEqual(snap.active_map_nodes, ("scanner_summarize_note",))
        self.assertEqual(snap.visited_map_nodes, ("scanner_summarize_note",))
        self.assertEqual(snap.current_map_node, "scanner_summarize_note")
        self.assertEqual(snap.current_agent, "scanner")

    def test_task_end_clears_active_but_keeps_visited(self):
        state = DemoState()
        state.ingest(_task_start(1, "summarize_note", "a", ("note_branch:1",),
                                 "scanner_summarize_note", "scanner"))
        state.ingest(_task_end(2, "summarize_note", "a", ("note_branch:1",),
                               "scanner_summarize_note", "scanner"))
        snap = state.snapshot()
        self.assertEqual(snap.active_map_nodes, ())
        self.assertEqual(snap.visited_map_nodes, ("scanner_summarize_note",))

    def test_fanout_multiplicity_counts_concurrent_tasks(self):
        state = DemoState()
        for i in range(3):
            state.ingest(_task_start(i, "extract_individual_value", str(i),
                                     ("extract:1", f"variable_branch:{i}"),
                                     "extractor_extract_individual_value", "extractor"))
        snap = state.snapshot()
        self.assertEqual(snap.node_multiplicity["extractor_extract_individual_value"], 3)
        # One finishes; multiplicity drops but the node stays active.
        state.ingest(_task_end(9, "extract_individual_value", "0",
                               ("extract:1", "variable_branch:0"),
                               "extractor_extract_individual_value", "extractor"))
        snap = state.snapshot()
        self.assertEqual(snap.node_multiplicity["extractor_extract_individual_value"], 2)

    def test_unmapped_node_is_ignored_for_activity(self):
        state = DemoState()
        state.ingest(_task_start(1, "mystery_node", "a", (), None, "orchestrator"))
        snap = state.snapshot()
        self.assertEqual(snap.active_map_nodes, ())
        self.assertEqual(snap.visited_map_nodes, ())

    def test_run_end_clears_current_node(self):
        state = DemoState()
        state.ingest(_task_start(1, "finalize_case", "a", (), "finalize_case", "orchestrator"))
        state.ingest(DemoEvent(seq=2, t=2.0, type="run_end"))
        snap = state.snapshot()
        self.assertTrue(snap.finished)
        self.assertIsNone(snap.current_map_node)


class DetailTests(unittest.TestCase):
    """Per-node detail accumulation (task input/result + correlated LLM calls)."""

    def test_llm_call_attaches_to_its_map_node(self):
        state = DemoState()
        state.ingest(_task_start(1, "summarize_note", "a", ("note_branch:1",),
                                 "scanner_summarize_note", "scanner"))
        call = LLMCall(node="summarize_note", namespace=("note_branch:1", "summarize_note:a"),
                       run_id="r", response='{"summary": "x"}')
        state.ingest(DemoEvent(seq=2, t=0.2, type="llm_call", node="summarize_note",
                               namespace=("note_branch:1", "summarize_note:a"),
                               map_node_id="scanner_summarize_note", agent="scanner",
                               payload=call.to_dict()))
        detail = state.snapshot().details["scanner_summarize_note"]
        self.assertEqual(len(detail.llm_calls), 1)
        self.assertEqual(detail.llm_calls[0]["response"], '{"summary": "x"}')

    def test_detail_captures_input_then_result_and_status(self):
        state = DemoState()
        state.ingest(_task_start(1, "retrieve_notes", "a", ("extract_branch:1",),
                                 "hard_filter_notes", "retriever", payload={"in": 1}))
        active = state.snapshot().details["hard_filter_notes"]
        self.assertEqual(active.status, "active")
        self.assertEqual(active.input, {"in": 1})
        state.ingest(_task_end(2, "retrieve_notes", "a", ("extract_branch:1",),
                               "hard_filter_notes", "retriever", payload={"retrieved_note_ids": [1]}))
        done = state.snapshot().details["hard_filter_notes"]
        self.assertEqual(done.status, "done")
        self.assertEqual(done.result, {"retrieved_note_ids": [1]})

    def test_task_error_marks_detail_status_error(self):
        state = DemoState()
        state.ingest(_task_start(1, "extract_individual_value", "a", ("extract:1",),
                                 "extractor_extract_individual_value", "extractor"))
        state.ingest(_task_end(2, "extract_individual_value", "a", ("extract:1",),
                               "extractor_extract_individual_value", "extractor",
                               error="boom"))
        self.assertEqual(
            state.snapshot().details["extractor_extract_individual_value"].status, "error"
        )


class LazyModelTests(unittest.TestCase):
    """The variable table's ProgressModel is built lazily from the streamed plan."""

    def test_no_progress_before_a_plan_is_seen(self):
        state = DemoState()
        state.ingest(DemoEvent(seq=0, t=0.0, type="run_start"))
        state.ingest(DemoEvent(seq=1, t=0.1, type="values", namespace=(),
                               payload={"note_corpus": {"1": {"note_id": 1}}}))
        self.assertIsNone(state.snapshot().progress)

    def test_plan_values_build_the_model_and_buffered_events_replay(self):
        # A task starts before the plan lands; once target_variables arrives the
        # model must be built and the earlier event folded in.
        state = DemoState()
        state.ingest(DemoEvent(seq=0, t=0.0, type="run_start"))
        state.ingest(DemoEvent(
            seq=1, t=0.1, type="values", namespace=(),
            payload={
                "target_variables": [
                    {"group_id": "g", "name": "Group", "stage": "initial",
                     "variables": [{"item_id": 1, "name": "V1"}]}
                ],
                "note_corpus": {},
            },
        ))
        progress = state.snapshot().progress
        self.assertIsNotNone(progress)
        self.assertEqual(progress.total_variables, 1)


class FixtureReplayTests(unittest.TestCase):
    """Replaying the committed trace yields a coherent, JSON-safe final state."""

    @classmethod
    def setUpClass(cls):
        cls.events = read_trace(FIXTURE)
        cls.state = replay(cls.events)
        cls.snap = cls.state.snapshot()

    def test_final_snapshot_is_finished_with_no_active_nodes(self):
        self.assertTrue(self.snap.finished)
        self.assertEqual(self.snap.active_map_nodes, ())
        self.assertIsNone(self.snap.current_map_node)

    def test_all_variables_reach_a_terminal_status(self):
        progress = self.snap.progress
        self.assertEqual(progress.total_variables, 7)
        self.assertEqual(progress.terminal_variables, 7)
        self.assertEqual(progress.done_groups, progress.total_groups)

    def test_gated_group_eligibility_is_resolved_from_hydrated_values(self):
        # The lymph-node group is gate-annotated; hydrating values lets the model
        # run the real gate predicate and stamp the verdict.
        progress = self.snap.progress
        gated = next(g for g in progress.groups if g.group_id == "lymph_node_removal")
        self.assertIn("✓", gated.annotation)

    def test_llm_details_are_present_for_llm_nodes(self):
        details = self.snap.details
        self.assertTrue(details["scanner_summarize_note"].llm_calls)
        self.assertTrue(details["extractor_repair_invalid_extraction"].llm_calls)

    def test_snapshot_to_dict_is_json_serializable(self):
        json.dumps(self.snap.to_dict())  # must not raise

    def test_latest_case_exposes_hydrated_variable_results(self):
        case = self.state.latest_case
        self.assertIsNotNone(case)
        self.assertEqual(set(case.variable_results), {390, 400, 410, 672, 674, 676, 682})

    def test_mid_run_snapshot_shows_in_flight_activity(self):
        mid = replay(self.events[:60])
        snap = mid.snapshot()
        self.assertFalse(snap.finished)
        self.assertTrue(snap.active_map_nodes)
        self.assertIsNotNone(snap.current_map_node)


if __name__ == "__main__":
    unittest.main()
