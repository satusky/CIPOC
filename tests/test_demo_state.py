"""Phase 2 — DemoState folds the merged event stream into presentable state."""

import json
import tempfile
import unittest
from pathlib import Path

from cipoc.demo.events import DemoEvent, LLMCall
from cipoc.demo.server import DemoSession, LiveDemoSession, load_replay_session
from cipoc.demo.state import DemoSnapshot, DemoState, NodeDetail, replay
from cipoc.demo.stream import DemoRun
from cipoc.demo.trace import read_trace
from cipoc.models import CaseFacts
from tests.fake_orchestrator import Outcome, Script, build_fake_orchestrator, load_notes

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


class InstanceDetailTests(unittest.TestCase):
    """Fan-out instances (e.g. per-note characterization) stay grouped by scope."""

    def _values(self, seq, ns, payload):
        return DemoEvent(seq=seq, t=seq * 0.1, type="values", namespace=ns, payload=payload)

    def _llm(self, seq, ns, response):
        call = LLMCall(node="summarize_note", namespace=ns, run_id="r", response=response)
        return DemoEvent(seq=seq, t=seq * 0.1, type="llm_call", node="summarize_note",
                         namespace=ns, map_node_id="scanner_summarize_note", agent="scanner",
                         payload=call.to_dict())

    def test_two_parallel_notes_keep_their_own_material(self):
        # Two note_branch instances run interleaved; each note's summary and its
        # own LLM call must land on its own instance, not pile onto a shared node.
        a, b = ("note_branch:a",), ("note_branch:b",)
        events = [
            _task_start(1, "note_branch", "a", (), "scanner_initialize", "scanner",
                        payload={"note_id": 50, "note_type": "Path"}),
            _task_start(2, "note_branch", "b", (), "scanner_initialize", "scanner",
                        payload={"note_id": 51, "note_type": "Rad"}),
            self._llm(3, a + ("summarize_note:x",), '{"summary": "A"}'),
            self._llm(4, b + ("summarize_note:y",), '{"summary": "B"}'),
            self._values(5, a, {"summary": "note A summary"}),
            self._values(6, b, {"summary": "note B summary", "concepts": {"c": {"presence": True}}}),
            _task_end(7, "note_branch", "a", (), "scanner_initialize", "scanner"),
        ]
        snap = replay(events).snapshot()
        insts = snap.instances
        self.assertEqual(list(insts), ["note_branch:a", "note_branch:b"])

        inst_a = insts["note_branch:a"]
        self.assertEqual(inst_a.index, 1)
        self.assertEqual(inst_a.label, "Path #50")
        self.assertEqual(inst_a.status, "done")  # its task_end arrived
        self.assertEqual(inst_a.result["summary"], "note A summary")
        self.assertEqual(len(inst_a.llm_calls), 1)
        self.assertEqual(inst_a.llm_calls[0]["response"], '{"summary": "A"}')

        inst_b = insts["note_branch:b"]
        self.assertEqual(inst_b.index, 2)
        self.assertEqual(inst_b.label, "Rad #51")
        self.assertEqual(inst_b.status, "active")  # no task_end yet
        self.assertEqual(len(inst_b.llm_calls), 1)
        self.assertEqual(inst_b.llm_calls[0]["response"], '{"summary": "B"}')
        self.assertIn("concepts", inst_b.result)

    def test_nested_values_accumulate_into_one_result(self):
        ns = ("note_branch:a",)
        events = [
            _task_start(1, "note_branch", "a", (), "scanner_initialize", "scanner",
                        payload={"note_id": 1, "note_type": "T"}),
            self._values(2, ns, {"summary": "s"}),
            self._values(3, ns, {"summary": "s", "concepts": {"x": {}}}),
            self._values(4, ns, {"summary": "s", "concepts": {"x": {}}, "cancer_mentions": [{"status": "m"}]}),
        ]
        inst = replay(events).snapshot().instances["note_branch:a"]
        self.assertEqual(set(inst.result), {"summary", "concepts", "cancer_mentions"})

    def test_extract_branch_is_tracked_as_a_fan_out_instance(self):
        # A pass fans out over every eligible group at once, so each group needs
        # its own instance — folded onto one NodeDetail the last group's
        # retriever verdict would overwrite the others'.
        events = [
            _task_start(1, "extract_branch", "g", (), "fan_out_groups", "orchestrator",
                        payload={"requested_variables": {"name": "Metastases", "group_id": "mets"}}),
            _task_start(2, "extract_branch", "h", (), "fan_out_groups", "orchestrator",
                        payload={"requested_variables": {"name": "Staging", "group_id": "tnm"}}),
            DemoEvent(seq=3, t=0.3, type="values", namespace=("extract_branch:g",),
                      agent="orchestrator", payload={"relevant_note_ids": [50, 51]}),
            DemoEvent(seq=4, t=0.4, type="values", namespace=("extract_branch:h",),
                      agent="orchestrator", payload={"relevant_note_ids": []}),
        ]
        insts = replay(events).snapshot().instances
        self.assertEqual(
            [i.label for i in insts.values()], ["Metastases", "Staging"]
        )
        self.assertEqual(insts["extract_branch:g"].result["relevant_note_ids"], [50, 51])
        self.assertEqual(insts["extract_branch:h"].result["relevant_note_ids"], [])


class VariableInstanceTests(unittest.TestCase):
    """Per-variable fan-out, two namespace levels down inside the extractor."""

    GROUP = ("extract_branch:g", "extract:e")

    def _variable_events(self, seq, task_id, name, item_id, passes):
        """One variable_branch: start, each validate/repair pass, then finish.

        ``passes`` is a list of ``(node, attempt, is_valid, errors)``.
        """
        scope = self.GROUP + (f"variable_branch:{task_id}",)
        events = [
            _task_start(seq, "variable_branch", task_id, self.GROUP, "fan_out_variables",
                        "extractor",
                        payload={"task": {"variable": {"item_id": item_id, "name": name}}}),
        ]
        for node, attempt, is_valid, errors in passes:
            seq += 1
            events.append(_task_end(
                seq, node, f"{task_id}-{attempt}-{node}", scope,
                f"extractor_{node}", "extractor",
                payload={"task": {"candidate": {"value": "V"}, "is_valid": is_valid,
                                  "validation_errors": errors, "extraction_attempts": attempt}},
            ))
        last = passes[-1]
        events.append(DemoEvent(
            seq=seq + 1, t=(seq + 1) * 0.1, type="values", namespace=scope, agent="extractor",
            payload={"variable_results": [{"item_id": item_id, "value": "V",
                                           "is_valid": last[2], "validation_errors": last[3],
                                           "extraction_attempts": last[1]}]},
        ))
        events.append(_task_end(seq + 2, "variable_branch", task_id, self.GROUP,
                                "fan_out_variables", "extractor"))
        return events

    def test_parallel_variables_keep_their_own_material(self):
        events = [
            *self._variable_events(1, "v1", "Primary Site", 400,
                                   [("validate_extraction", 1, True, [])]),
            *self._variable_events(10, "v2", "Laterality", 410,
                                   [("validate_extraction", 1, True, [])]),
        ]
        insts = replay(events).snapshot().instances
        self.assertEqual(
            list(insts),
            ["extract_branch:g/extract:e/variable_branch:v1",
             "extract_branch:g/extract:e/variable_branch:v2"],
        )
        first = insts["extract_branch:g/extract:e/variable_branch:v1"]
        self.assertEqual(first.node, "variable_branch")
        self.assertEqual(first.label, "Primary Site")
        self.assertEqual(first.status, "done")
        self.assertEqual(first.result["variable_results"][0]["item_id"], 400)

    def test_repair_loop_records_condensed_attempts(self):
        events = self._variable_events(1, "v1", "Nodes Positive", 820, [
            ("validate_extraction", 1, False, ["not an allowable code"]),
            ("repair_invalid_extraction", 2, False, []),
            ("validate_extraction", 2, True, []),
        ])
        inst = replay(events).snapshot().instances[
            "extract_branch:g/extract:e/variable_branch:v1"
        ]
        self.assertEqual([a["node"] for a in inst.attempts],
                         ["validate_extraction", "repair_invalid_extraction",
                          "validate_extraction"])
        self.assertEqual(inst.attempts[0]["validation_errors"], ["not an allowable code"])
        self.assertEqual(inst.attempts[0]["value"], "V")
        # Repaired successfully: the settled verdict is the *validation*, not the
        # repair pass that cleared the flag on its way to being re-checked.
        self.assertEqual(inst.status, "done")
        self.assertIs(inst.final_is_valid, True)

    def test_exhausted_repair_reports_invalid_not_error(self):
        events = self._variable_events(1, "v1", "Nodes Positive", 820, [
            ("validate_extraction", 1, False, ["bad"]),
            ("repair_invalid_extraction", 2, False, []),
            ("validate_extraction", 2, False, ["still bad"]),
        ])
        inst = replay(events).snapshot().instances[
            "extract_branch:g/extract:e/variable_branch:v1"
        ]
        self.assertEqual(inst.status, "invalid")
        self.assertEqual(inst.errors, 0)  # nothing *raised*

    def test_nested_llm_call_lands_on_its_variable(self):
        # A repair call's namespace nests one level deeper than the variable's;
        # it must resolve to the innermost enclosing fan-out.
        scope = self.GROUP + ("variable_branch:v1",)
        call = LLMCall(node="repair_invalid_extraction",
                       namespace=scope + ("repair_invalid_extraction:r",),
                       run_id="r", response="{}")
        events = [
            _task_start(1, "variable_branch", "v1", self.GROUP, "fan_out_variables",
                        "extractor", payload={"task": {"variable": {"name": "X", "item_id": 1}}}),
            DemoEvent(seq=2, t=0.2, type="llm_call", node="repair_invalid_extraction",
                      namespace=scope + ("repair_invalid_extraction:r",),
                      map_node_id="extractor_repair_invalid_extraction", agent="extractor",
                      payload=call.to_dict()),
        ]
        inst = replay(events).snapshot().instances[
            "extract_branch:g/extract:e/variable_branch:v1"
        ]
        self.assertEqual(len(inst.llm_calls), 1)
        self.assertEqual(inst.status, "active")

    def test_fixture_groups_every_variable_under_its_own_group(self):
        snap = replay(read_trace(FIXTURE)).snapshot()
        variables = [i for i in snap.instances.values() if i.node == "variable_branch"]
        self.assertTrue(variables)
        self.assertTrue(all(v.label for v in variables), "every variable needs a name")
        # Keys are scoped by extract_branch, so a group's cards can be selected.
        groups = {v.key.split("/", 1)[0] for v in variables}
        self.assertEqual(len(groups), 2)
        # The fixture scripts item 674 to fail validation once, then repair.
        repaired = [v for v in variables if len(v.attempts) > 1]
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0].status, "done")

    def test_variable_instances_serialize_their_attempts(self):
        events = self._variable_events(1, "v1", "Grade", 440,
                                       [("validate_extraction", 1, True, [])])
        data = replay(events).snapshot().to_dict()
        inst = data["instances"]["extract_branch:g/extract:e/variable_branch:v1"]
        self.assertEqual(inst["label"], "Grade")
        self.assertEqual(len(inst["attempts"]), 1)
        json.dumps(data)  # the whole snapshot must stay JSON-serializable


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


class CaseFactsSnapshotTests(unittest.TestCase):
    """Facts are a detached read of root values, independent of task progress."""

    UNKNOWN = {
        "primary_site": None,
        "gross_primary_site": None,
        "histology": None,
        "behavior": None,
        "sex": None,
        "date_of_diagnosis": None,
    }

    @staticmethod
    def values(seq, payload, namespace=()):
        return DemoEvent(seq=seq, t=seq * 0.1, type="values",
                         namespace=namespace, payload=payload)

    def test_direct_constructor_keeps_previous_positional_signature(self):
        snap = DemoSnapshot(0, 0.0, False, None, None, (), (), {}, {}, {}, None)
        self.assertIsNone(snap.case_facts)
        self.assertIn("case_facts", snap.to_dict())
        self.assertIsNone(snap.to_dict()["case_facts"])

    def test_startup_is_unavailable_even_with_preconfigured_progress(self):
        for state in (DemoState(), DemoState(target_groups=[], graph_input={
            "case_facts": {"primary_site": "C509"},
        })):
            with self.subTest(preconfigured=state.snapshot().progress is not None):
                self.assertIsNone(state.snapshot().case_facts)
                state.ingest(DemoEvent(seq=0, t=0.0, type="run_start"))
                data = json.loads(json.dumps(state.snapshot().to_dict(), allow_nan=False))
                self.assertIn("case_facts", data)
                self.assertIsNone(data["case_facts"])

    def test_missing_null_or_unhydratable_root_is_unavailable(self):
        for payload in (
            {}, {"case_facts": None}, None, [],
            {"case_facts": {"primary_site": ["invalid"]}},
            {"case_facts": {"primary_site": "C509"}, "note_corpus": {"1": {"note_id": 1}}},
        ):
            with self.subTest(payload=payload):
                state = replay([self.values(1, {"case_facts": {"primary_site": "C509"}})])
                state.ingest(self.values(2, payload))
                self.assertIsNone(state.snapshot().case_facts)
                self.assertIsNone(state.snapshot().to_dict()["case_facts"])

    def test_explicit_unknown_object_has_all_six_nullable_fields(self):
        self.assertEqual(set(CaseFacts.model_fields), set(self.UNKNOWN))
        for facts in ({}, self.UNKNOWN):
            with self.subTest(facts=facts):
                snap = replay([self.values(1, {"case_facts": facts})]).snapshot()
                self.assertEqual(snap.case_facts, self.UNKNOWN)
                self.assertEqual(snap.to_dict()["case_facts"], self.UNKNOWN)

    def test_new_root_facts_replace_prior_fields_including_explicit_nulls(self):
        state = replay([self.values(1, {"case_facts": {
            "primary_site": "C509", "gross_primary_site": "breast", "histology": "8500",
        }})])
        earlier = state.snapshot()
        state.ingest(self.values(2, {"case_facts": {"primary_site": None, "sex": "2"}}))
        self.assertEqual(state.snapshot().case_facts, {**self.UNKNOWN, "sex": "2"})
        state.ingest(self.values(3, {"case_facts": None}))
        self.assertIsNone(state.snapshot().case_facts)
        self.assertEqual(earlier.case_facts["primary_site"], "C509")

    def test_results_and_subagent_or_task_facts_cannot_override_root(self):
        for root_facts in (None, {}, {"primary_site": "C509"}):
            with self.subTest(root_facts=root_facts):
                state = replay([self.values(1, {
                    "case_facts": root_facts,
                    "structured_data": {400: "C349"},
                    "variable_results": {
                        400: {"item_id": 400, "status": "structured_data", "value": "C349"},
                    },
                })])
                self.assertIsNotNone(state.latest_case)
                expected = None if root_facts is None else {**self.UNKNOWN, **root_facts}
                self.assertEqual(state.snapshot().case_facts, expected)
                events = [
                    _task_start(2, "extract_branch", "g", (), "fan_out_groups", "orchestrator",
                                payload={"case_facts": {"primary_site": "C349"}}),
                    self.values(3, {"case_facts": {"primary_site": "C349"}}, ("extract_branch:g",)),
                    _task_end(4, "merge_and_update", "m", (), "merge_and_update", "orchestrator",
                              payload={"case_facts": {"primary_site": "C349"}}),
                ]
                for event in events:
                    state.ingest(event)
                    self.assertEqual(state.snapshot().case_facts, expected)

    def test_snapshot_copies_model_and_returns_fresh_json_safe_facts(self):
        original = {**self.UNKNOWN, "gross_primary_site": "left bréast", "behavior": "3"}
        state = replay([self.values(1, {"case_facts": original})])
        snap = state.snapshot()
        with self.assertRaises(TypeError):
            snap.case_facts["behavior"] = "2"
        data = json.loads(json.dumps(snap.to_dict(), allow_nan=False))
        self.assertEqual(data["case_facts"], original)
        detached = snap.to_dict()
        detached["case_facts"]["behavior"] = "2"
        state.latest_case.case_facts.behavior = "0"
        self.assertEqual(snap.to_dict()["case_facts"], original)
        self.assertEqual(state.snapshot().case_facts["behavior"], "0")
        state.ingest(self.values(2, {"case_facts": {"primary_site": "C509"}}))
        self.assertEqual(snap.case_facts, original)


class CurrentCaseFactsReplayTests(unittest.TestCase):
    """Real root updates survive live capture, trace round trips, and rewinds."""

    def test_structured_characterized_and_extracted_facts_are_cursor_local(self):
        agent = build_fake_orchestrator(Script(outcomes={400: Outcome(value="C509")}))
        agent._target_variables = agent._target_variables[:1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            job = DemoRun(
                [load_notes()[0].model_dump()], structured_data={390: "20250224"},
                agent_factory=lambda: agent, max_concurrency=2,
                output_dir=path, record_path=path / "trace.jsonl",
            )
            live = LiveDemoSession(job)
            captured = []

            def consume(event):
                live.append(event)
                if event.type == "values" and not event.namespace:
                    captured.append((event, live._latest.snapshot()))

            job.execute(consume)
            self.assertEqual(job.status, "completed", job.error)
            self.assertFalse(job.issues)
            events = read_trace(path / "trace.jsonl")
            session = load_replay_session(path / "trace.jsonl", artifact_path=job.output_path)
            self.assertEqual(session.artifact["case"]["case_facts"]["primary_site"], "C509")

            structured = {**CaseFactsSnapshotTests.UNKNOWN, "date_of_diagnosis": "20250224"}
            characterized = {**structured, "gross_primary_site": "breast"}
            extracted = {**characterized, "primary_site": "C509"}
            self.assertIsNone(captured[0][1].case_facts)
            observed = [snap.to_dict()["case_facts"] for _, snap in captured]
            self.assertIn(structured, observed)
            self.assertIn(characterized, observed)
            self.assertEqual(observed[-1], extracted)
            # Extraction results arrive before merge_and_update writes scoping facts.
            self.assertTrue(any(
                event.payload.get("variable_results", {}).get("400", {}).get("value") == "C509"
                and snap.case_facts == characterized
                for event, snap in captured
            ))

            for event, snap in reversed(captured):
                with self.subTest(seq=event.seq):
                    expected = event.payload.get("case_facts")
                    self.assertEqual(snap.case_facts, expected)  # retained live snapshot
                    prefix = [item for item in events if item.seq <= event.seq]
                    self.assertEqual(replay(prefix).snapshot().case_facts, expected)
                    self.assertEqual(live.snapshot_at_seq(event.seq)["case_facts"], expected)
                    self.assertEqual(session.snapshot_at_seq(event.seq)["case_facts"], expected)

            # Step caching and final artifacts must not leak facts into earlier steps.
            session.goto(len(session.steps) - 1)
            for step in reversed(session.steps):
                prefix = [event for event in events if event.seq <= step.end_seq]
                expected = replay(prefix).snapshot().to_dict()["case_facts"]
                self.assertEqual(session.goto(step.index)["snapshot"]["case_facts"], expected)


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

    def test_legacy_facts_follow_exact_root_values_at_every_step(self):
        state = DemoState()
        expected = None
        for event in self.events:
            state.ingest(event)
            if event.type == "values" and not event.namespace:
                expected = event.payload.get("case_facts")
            with self.subTest(seq=event.seq):
                self.assertEqual(state.snapshot().to_dict()["case_facts"], expected)
        self.assertIsNotNone(expected)
        session = DemoSession(self.events)
        session.goto(len(session.steps) - 1)
        for step in reversed(session.steps):
            prefix = [event for event in self.events if event.seq <= step.end_seq]
            self.assertEqual(session.goto(step.index)["snapshot"]["case_facts"],
                             replay(prefix).snapshot().to_dict()["case_facts"])


if __name__ == "__main__":
    unittest.main()
