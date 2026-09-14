import unittest

from cipoc.tools import GroupNode, load_group_hierarchy, load_variable_groups
from cipoc.utils.progress.events import ProgressEvent, normalize
from cipoc.utils.progress.model import ProgressModel, Stage
from tests.fake_orchestrator import (
    Outcome,
    Script,
    VARIABLE_GROUPS,
    graph_input,
    record_events,
)


GROUP = {
    "group_id": "initial",
    "name": "Initial LLM Extraction",
    "stage": "initial",
    "variables": [{"item_id": 390, "name": "Date of Diagnosis"}],
}
HIERARCHY = [
    GroupNode("case", "Case", None, ()),
    GroupNode("initial", "Initial LLM Extraction", "case", (390,)),
]
GROUP_SCOPE = ("extract_branch:group-task",)
EXTRACT_SCOPE = (*GROUP_SCOPE, "extract:extract-task")
VARIABLE_SCOPE = (*EXTRACT_SCOPE, "variable_branch:variable-task")


def task_start(namespace, task_id, name, graph_input):
    return (
        namespace,
        "tasks",
        {"id": task_id, "name": name, "input": graph_input, "triggers": []},
    )


def task_end(namespace, task_id, name, *, result=None, error=None):
    return (
        namespace,
        "tasks",
        {
            "id": task_id,
            "name": name,
            "result": result,
            "error": error,
            "interrupts": [],
        },
    )


def values(namespace, payload):
    return namespace, "values", payload


class ProgressEventTests(unittest.TestCase):
    def test_normalizes_subgraph_task_start_and_scope(self):
        event = normalize(
            task_start(GROUP_SCOPE, "variable-task", "variable_branch", {"task": {}}),
            subgraphs=True,
        )

        self.assertEqual(
            event,
            ProgressEvent(
                kind="task_start",
                namespace=GROUP_SCOPE,
                node="variable_branch",
                task_id="variable-task",
                payload={"task": {}},
            ),
        )
        self.assertEqual(event.scope, (*GROUP_SCOPE, "variable_branch:variable-task"))

    def test_normalizes_task_end_and_non_subgraph_values(self):
        event = normalize(
            ("tasks", task_end((), "task-1", "initialize", result={})[2]),
            subgraphs=False,
        )
        root_values = normalize(("values", {"answer": 1}), subgraphs=False)

        self.assertEqual(event.kind, "task_end")
        self.assertEqual(event.node, "initialize")
        self.assertEqual(event.payload, {})
        self.assertTrue(event.is_root)
        self.assertEqual(root_values, ProgressEvent(kind="values", namespace=(), payload={"answer": 1}))

    def test_ignores_unrequested_stream_modes(self):
        self.assertIsNone(normalize(((), "updates", {}), subgraphs=True))


class ProgressModelTests(unittest.TestCase):
    def setUp(self):
        self.model = ProgressModel(
            "Orchestrator",
            10.0,
            target_groups=[GROUP],
            group_hierarchy=HIERARCHY,
            graph_input={"note_corpus": {1: {}, 2: {}}},
            show_note_counts=True,
        )
        self.now = 10.0

    def ingest(self, raw_event):
        self.now += 0.1
        event = normalize(raw_event, subgraphs=True)
        self.assertIsNotNone(event)
        self.model.ingest(event, self.now)
        return self.model.snapshot()

    @staticmethod
    def group(snapshot, group_id):
        return next(group for group in snapshot.groups if group.group_id == group_id)

    def test_recorded_tuples_drive_namespace_bound_stage_transition(self):
        initial = self.model.snapshot()
        self.assertEqual(initial.variables[390].stage, Stage.IDLE)
        self.assertEqual([group.depth for group in initial.groups], [0, 1])

        self.ingest(task_start((), "note-task", "note_branch", {"note": {}}))
        scanned = self.ingest(task_end((), "note-task", "note_branch", result={}))
        self.assertEqual(scanned.notes_done, 1)

        retrieving = self.ingest(
            task_start(
                (),
                "group-task",
                "extract_branch",
                {"requested_variables": GROUP},
            )
        )
        self.assertEqual(retrieving.variables[390].stage, Stage.RETRIEVE)
        self.assertTrue(self.group(retrieving, "initial").active)
        self.assertEqual(len(retrieving.branches), 1)

        self.ingest(task_start(GROUP_SCOPE, "retrieve-task", "retrieve_notes", {}))
        retrieved = self.ingest(
            task_end(
                GROUP_SCOPE,
                "retrieve-task",
                "retrieve_notes",
                result={"retrieved_note_ids": [1, 2]},
            )
        )
        self.assertEqual(self.group(retrieved, "initial").note_count, 2)
        self.assertEqual(retrieved.branches[0].note_count, 2)

        extracting = self.ingest(
            task_start(GROUP_SCOPE, "extract-task", "extract", {})
        )
        self.assertEqual(extracting.variables[390].stage, Stage.EXTRACT)

        self.ingest(
            task_start(
                EXTRACT_SCOPE,
                "variable-task",
                "variable_branch",
                {"task": {"variable": {"item_id": 390}, "extraction_attempts": 1}},
            )
        )
        validating = self.ingest(
            task_start(
                VARIABLE_SCOPE,
                "validate-task",
                "validate_extraction",
                {"task": {"extraction_attempts": 2}},
            )
        )
        self.assertEqual(validating.variables[390].stage, Stage.VALIDATE)
        self.assertEqual(validating.variables[390].attempt, 2)

        nested_values = self.ingest(
            values(
                VARIABLE_SCOPE,
                {"variable_results": {390: {"status": "error", "reason": "ignore"}}},
            )
        )
        self.assertEqual(nested_values.variables[390].status, "pending")

        terminal = self.ingest(
            values(
                (),
                {
                    "variable_results": {
                        390: {
                            "status": "extracted",
                            "value": "20260101",
                            "extraction": {
                                "presence_confidence": "low",
                                "is_valid": True,
                            },
                        }
                    },
                    "report": {"flags": [{"item_id": 390}]},
                },
            )
        )
        variable = terminal.variables[390]
        self.assertEqual(variable.stage, Stage.DONE)
        self.assertEqual(variable.status, "extracted")
        self.assertEqual(variable.value, "20260101")
        self.assertEqual(variable.confidence, "low")
        self.assertEqual(variable.flag, "?")
        self.assertEqual(terminal.review_flags, 1)

        completed = self.ingest(
            task_end((), "group-task", "extract_branch", result={})
        )
        self.assertFalse(self.group(completed, "initial").active)
        self.assertEqual(completed.branches, ())

    def test_root_values_project_every_terminal_result_fact(self):
        item_ids = range(1, 7)
        group = {
            "group_id": "statuses",
            "name": "Statuses",
            "variables": [
                {"item_id": item_id, "name": f"Variable {item_id}"}
                for item_id in item_ids
            ],
        }
        model = ProgressModel("Orchestrator", 0.0, target_groups=[group])
        model.ingest(
            ProgressEvent(
                kind="values",
                namespace=(),
                payload={
                    "variable_results": {
                        1: {"status": "structured_data", "value": "A"},
                        2: {
                            "status": "extracted",
                            "value": "1",
                            "extraction": {
                                "presence_confidence": "low",
                                "is_valid": True,
                            },
                        },
                        3: {
                            "status": "not_found",
                            "extraction": {
                                "presence_confidence": "low",
                                "is_valid": True,
                            },
                        },
                        4: {"status": "not_applicable", "reason": "Wrong site"},
                        5: {"status": "blocked", "blocking_item_ids": [390, 400]},
                        6: {
                            "status": "error",
                            "reason": "Invalid code",
                            "extraction": {
                                "presence_confidence": "medium",
                                "is_valid": False,
                            },
                        },
                    }
                },
            ),
            1.0,
        )

        snapshot = model.snapshot()
        self.assertEqual(snapshot.terminal_variables, 6)
        self.assertEqual(snapshot.variables[1].value, "A")
        self.assertEqual(snapshot.variables[2].flag, "?")
        self.assertIsNone(snapshot.variables[3].flag)
        self.assertEqual(snapshot.variables[4].detail, "Wrong site")
        self.assertEqual(snapshot.variables[5].detail, "←390,400")
        self.assertEqual(snapshot.variables[6].detail, "Invalid code")
        self.assertEqual(snapshot.variables[6].flag, "!")

    def test_standalone_extractor_projects_both_output_shapes(self):
        graph_input = {
            "requested_variables": {
                "group_id": "standalone",
                "name": "Standalone",
                "variables": [
                    {"item_id": 390, "name": "Date of Diagnosis"},
                    {"item_id": 400, "name": "Primary Site"},
                ],
            }
        }
        model = ProgressModel("Extractor", 0.0, graph_input=graph_input)
        model.ingest(
            ProgressEvent(
                kind="values",
                namespace=(),
                payload={
                    "variable_results": [
                        {
                            "item_id": 390,
                            "value": "20260101",
                            "is_valid": True,
                            "presence_confidence": "high",
                        }
                    ],
                    "extracted_values": {
                        "variables": [
                            {
                                "item_id": 400,
                                "value": "C50.9",
                                "is_valid": True,
                                "presence_confidence": "low",
                            }
                        ]
                    },
                },
            ),
            1.0,
        )

        snapshot = model.snapshot()
        self.assertEqual(snapshot.mode, "standalone")
        self.assertEqual(snapshot.variables[390].status, "extracted")
        self.assertEqual(snapshot.variables[390].value, "20260101")
        self.assertEqual(snapshot.variables[400].status, "extracted")
        self.assertEqual(snapshot.variables[400].value, "C50.9")
        self.assertEqual(snapshot.variables[400].flag, "?")
        self.assertEqual(snapshot.review_flags, 1)

    def test_compact_run_tracks_root_tasks_without_variable_results(self):
        model = ProgressModel("Note Retriever", 0.0, graph_input={})
        model.ingest(
            ProgressEvent(
                kind="task_start",
                namespace=(),
                node="identify_relevant_notes",
                task_id="task-1",
                payload={},
            ),
            1.0,
        )
        active = model.snapshot()
        self.assertEqual(active.mode, "compact")
        self.assertEqual((active.completed_tasks, active.total_tasks), (0, 1))

        model.ingest(
            ProgressEvent(
                kind="task_end",
                namespace=(),
                node="identify_relevant_notes",
                task_id="task-1",
                payload={},
            ),
            2.0,
        )
        model.finish()
        completed = model.snapshot()
        self.assertEqual((completed.completed_tasks, completed.total_tasks), (1, 1))
        self.assertEqual(completed.nodes[0].state, "ok")


class FakeOrchestratorStreamTests(unittest.TestCase):
    def test_spike_stream_reaches_validate_attempt_two_and_terminal_snapshot(self):
        input_state = graph_input()
        target_groups = load_variable_groups(VARIABLE_GROUPS)
        model = ProgressModel(
            "Orchestrator",
            0.0,
            target_groups=target_groups,
            group_hierarchy=load_group_hierarchy(VARIABLE_GROUPS),
            graph_input=input_state,
        )
        raw_events = record_events(Script(outcomes={390: Outcome(repairs=1)}))
        task_ids = {
            str(payload["id"])
            for _, mode, payload in raw_events
            if mode == "tasks"
        }
        namespace_task_ids = {
            segment.rsplit(":", 1)[1]
            for namespace, _, _ in raw_events
            for segment in namespace
        }
        self.assertTrue(namespace_task_ids <= task_ids)

        transitions = []
        previous = None
        for index, raw_event in enumerate(raw_events, 1):
            event = normalize(raw_event, subgraphs=True)
            self.assertIsNotNone(event)
            model.ingest(event, index / 10)
            variable = model.snapshot().variables[390]
            current = (variable.stage, variable.attempt, variable.status)
            if current != previous:
                transitions.append(current)
                previous = current

        expected = [
            (Stage.IDLE, 0, "pending"),
            (Stage.RETRIEVE, 0, "pending"),
            (Stage.EXTRACT, 1, "pending"),
            (Stage.VALIDATE, 2, "pending"),
            (Stage.DONE, 0, "extracted"),
        ]
        offset = 0
        for transition in expected:
            offset = transitions.index(transition, offset) + 1

        model.finish()
        snapshot = model.snapshot()
        expected_variables = sum(len(group.variables) for group in target_groups)
        self.assertEqual(snapshot.total_variables, expected_variables)
        self.assertEqual(snapshot.terminal_variables, expected_variables)
        self.assertEqual(
            (snapshot.done_groups, snapshot.total_groups),
            (len(target_groups), len(target_groups)),
        )
        self.assertEqual(snapshot.notes_done, len(input_state["note_corpus"]))
        self.assertEqual(snapshot.branches, ())
        self.assertEqual(snapshot.counts.get("pending", 0), 0)
        self.assertTrue(snapshot.finished)


if __name__ == "__main__":
    unittest.main()
