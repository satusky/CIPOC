import unittest
from typing import get_args, get_type_hints

from pydantic import ValidationError

from cipoc.agents.orchestrator import (
    CaseState,
    ExtractBranchState,
    OrchestratorAgent,
    OrchestratorOutput,
    dict_merge_reducer,
)
from cipoc.agents.note_retriever import NoteRetrieverAgent, RelevantNoteIDs, RetrieverState
from cipoc.models import (
    Case,
    NoteDigest,
    NoteFilter,
    NoteSelectionProvenance,
    NoteSelectionRejectionCode,
    NoteSelectionUnevaluatedCode,
    ProcessedClinicalNote,
    TargetGroup,
    VariableInfo,
)
from tests.fake_orchestrator import Script, build_fake_orchestrator, graph_input


class FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def run(self, request, *, progress=True):
        self.requests.append(request)
        return self.result


class FakeStructuredRetrieverModel:
    def structured(self, schema, messages):
        return RelevantNoteIDs(note_ids=["path-1", "invented"])


class NoteSelectionProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.selection = NoteSelectionProvenance(
            group_id="treatment",
            requested_item_ids=[1280, 1290],
            candidate_note_ids=["path-1"],
            rejected_note_ids={
                2: [
                    NoteSelectionRejectionCode.NOTE_TYPE_MISMATCH,
                    NoteSelectionRejectionCode.CANCER_STATUS_MISMATCH,
                ],
                "dated-note": [
                    NoteSelectionRejectionCode.MISSING_OR_INVALID_DATE,
                    NoteSelectionRejectionCode.OUTSIDE_DATE_WINDOW,
                ],
            },
            selected_note_ids=["path-1"],
            discarded_note_ids=["invented-A"],
            unevaluated_checks=[
                NoteSelectionUnevaluatedCode.KEYWORD_FILTER_DISABLED,
                NoteSelectionUnevaluatedCode.TEMPORAL_ANCHOR_UNAVAILABLE,
            ],
        )

    def test_case_state_copies_typed_selection_into_durable_case(self):
        key = "group:treatment"
        case = CaseState(note_selection={key: self.selection}).to_case()

        self.assertEqual(case.note_selection[key], self.selection)
        serialized = case.model_dump(mode="json")["note_selection"][key]
        self.assertEqual(serialized["requested_item_ids"], [1280, 1290])
        self.assertEqual(
            serialized["rejected_note_ids"]["2"],
            ["note_type_mismatch", "cancer_status_mismatch"],
        )
        self.assertEqual(
            serialized["unevaluated_checks"],
            ["keyword_filter_disabled", "temporal_anchor_unavailable"],
        )
        self.assertEqual(serialized["discarded_note_ids"], ["invented-A"])

    def test_case_rejects_selection_key_that_does_not_match_group(self):
        with self.assertRaises(ValidationError):
            Case(note_selection={"treatment": self.selection})

    def test_root_and_branch_channels_use_dict_merge_reducer(self):
        for state_type in (CaseState, ExtractBranchState):
            annotation = get_type_hints(state_type, include_extras=True)[
                "note_selection"
            ]
            self.assertIn(dict_merge_reducer, get_args(annotation))

        other = self.selection.model_copy(update={"group_id": "other"})
        merged = dict_merge_reducer(
            {"group:treatment": self.selection},
            {"group:other": other},
        )
        self.assertEqual(set(merged), {"group:treatment", "group:other"})

    def test_output_contract_exposes_typed_selection(self):
        output = OrchestratorOutput(
            note_selection={"group:treatment": self.selection}
        )

        self.assertEqual(output.note_selection["group:treatment"], self.selection)


class RetrievalFunnelTests(unittest.TestCase):
    def setUp(self):
        self.pathology = ProcessedClinicalNote(
            note_id="path-1",
            date="2025-02-20",
            note_type="Pathology",
            content="Breast biopsy.",
            cancer_status={"current"},
        )
        self.second_pathology = self.pathology.model_copy(
            update={"note_id": "path-2"}
        )
        self.rejected = self.pathology.model_copy(
            update={
                "note_id": "old-rad",
                "note_type": "Radiology",
                "cancer_status": {"historical"},
            }
        )
        self.group = TargetGroup(
            group_id="diagnosis",
            name="Diagnosis",
            variables=[VariableInfo(item_id=400), VariableInfo(item_id=522)],
            note_filter=NoteFilter(
                note_types=["Pathology"],
                keywords=["biopsy"],
                cancer_status=["current"],
                within_days=30,
            ),
        )

    @staticmethod
    def digest(note):
        return NoteDigest(
            note_id=note.note_id,
            note_type=note.note_type,
            summary=note.summary,
            flags=note.flags,
        )

    def state(self, notes, digests=None):
        return ExtractBranchState(
            requested_variables=self.group,
            branch_note_corpus={note.note_id: note for note in notes},
            branch_note_digests=(
                {note.note_id: self.digest(note) for note in notes}
                if digests is None
                else digests
            ),
        )

    @staticmethod
    def agent(result):
        agent = object.__new__(OrchestratorAgent)
        agent._retriever = FakeRetriever(result)
        return agent

    def test_nonempty_funnel_records_candidates_rejections_and_valid_selection(self):
        agent = self.agent(["path-1", "path-2", "old-rad", "invented"])
        state = self.state(
            [self.pathology, self.second_pathology, self.rejected],
            {"path-1": self.digest(self.pathology)},
        )

        update = agent.retrieve_notes(state)
        selection = update["note_selection"]["group:diagnosis"]

        self.assertEqual(update["retrieved_note_ids"], ["path-1"])
        self.assertEqual(selection.requested_item_ids, [400, 522])
        self.assertEqual(selection.candidate_note_ids, ["path-1", "path-2"])
        self.assertEqual(
            selection.rejected_note_ids["old-rad"],
            [
                NoteSelectionRejectionCode.NOTE_TYPE_MISMATCH,
                NoteSelectionRejectionCode.CANCER_STATUS_MISMATCH,
            ],
        )
        self.assertEqual(
            selection.unevaluated_checks,
            [
                NoteSelectionUnevaluatedCode.KEYWORD_FILTER_DISABLED,
                NoteSelectionUnevaluatedCode.TEMPORAL_ANCHOR_UNAVAILABLE,
            ],
        )
        self.assertEqual(selection.selected_note_ids, ["path-1"])
        self.assertEqual(
            selection.discarded_note_ids,
            ["path-2", "old-rad", "invented"],
        )
        self.assertEqual(
            list(agent._retriever.requests[0].available_digests), ["path-1"]
        )
        serialized = Case(note_selection=update["note_selection"]).model_dump(mode="json")
        self.assertEqual(
            serialized["note_selection"]["group:diagnosis"]["discarded_note_ids"],
            ["path-2", "old-rad", "invented"],
        )
        self.assertNotIn("old-rad", serialized["note_selection"]["group:diagnosis"]["selected_note_ids"])

    def test_retriever_preserves_unoffered_ids_for_orchestrator_validation(self):
        retriever = object.__new__(NoteRetrieverAgent)
        retriever.agent = FakeStructuredRetrieverModel()
        state = RetrieverState(
            requested_variables=self.group.to_variable_group(),
            available_digests={"path-1": self.digest(self.pathology)},
            messages=[],
        )

        update = retriever.identify_relevant_notes(state)

        self.assertEqual(update["relevant_note_ids"], ["path-1", "invented"])

    def test_empty_candidates_record_the_funnel_without_calling_retriever(self):
        agent = self.agent(["invented"])

        update = agent.retrieve_notes(self.state([self.rejected]))
        selection = update["note_selection"]["group:diagnosis"]

        self.assertEqual(update["retrieved_note_ids"], [])
        self.assertEqual(selection.candidate_note_ids, [])
        self.assertEqual(selection.selected_note_ids, [])
        self.assertEqual(selection.discarded_note_ids, [])
        self.assertEqual(list(selection.rejected_note_ids), ["old-rad"])
        self.assertEqual(agent._retriever.requests, [])

    def test_empty_retriever_selection_still_records_candidates(self):
        agent = self.agent(None)

        update = agent.retrieve_notes(self.state([self.pathology]))
        selection = update["note_selection"]["group:diagnosis"]

        self.assertEqual(update["retrieved_note_ids"], [])
        self.assertEqual(selection.candidate_note_ids, ["path-1"])
        self.assertEqual(selection.selected_note_ids, [])
        self.assertEqual(selection.discarded_note_ids, [])
        self.assertEqual(len(agent._retriever.requests), 1)

    def test_empty_corpus_retains_configured_unevaluated_checks(self):
        agent = self.agent(None)

        update = agent.retrieve_notes(self.state([]))
        selection = update["note_selection"]["group:diagnosis"]

        self.assertEqual(
            selection.unevaluated_checks,
            [
                NoteSelectionUnevaluatedCode.KEYWORD_FILTER_DISABLED,
                NoteSelectionUnevaluatedCode.TEMPORAL_ANCHOR_UNAVAILABLE,
            ],
        )
        self.assertEqual(agent._retriever.requests, [])

    def test_concurrent_group_records_merge_through_graph_and_reach_case(self):
        agent = build_fake_orchestrator()

        final_state = CaseState(**agent._graph.invoke(graph_input()))
        case = final_state.to_case()

        self.assertGreater(len(case.note_selection), 1)
        self.assertEqual(case.note_selection, final_state.note_selection)
        self.assertEqual(
            set(case.note_selection),
            {
                f"group:{selection.group_id}"
                for selection in case.note_selection.values()
            },
        )
        round_trip = Case.model_validate_json(case.model_dump_json())
        self.assertEqual(round_trip.note_selection, case.note_selection)

    def test_groups_excluded_before_retrieval_have_no_record(self):
        script = Script(
            concepts={
                "cancer": True,
                "metastasis": False,
                "surgery": False,
                "chemotherapy": False,
                "radiation": False,
                "lymph_nodes_removed": False,
            }
        )
        agent = build_fake_orchestrator(script)

        final_state = CaseState(**agent._graph.invoke(graph_input()))

        self.assertNotIn("group:first_course_treatment", final_state.note_selection)
        self.assertNotIn("group:metastases", final_state.note_selection)
        self.assertNotIn("group:lymph_node_removal", final_state.note_selection)
        self.assertIn("group:initial_llm_extraction", final_state.note_selection)


if __name__ == "__main__":
    unittest.main()
