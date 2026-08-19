import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cipoc.agents.orchestrator import CaseState, OrchestratorAgent
from cipoc.models import CancerMention, CaseFacts, ClinicalNote, ConfidenceLevel, ProcessedClinicalNote, TextSpan
from cipoc.tools import build_corpus_descriptors, load_variable_groups, site_applies


VARIABLE_GROUPS = Path(__file__).resolve().parents[1] / "config" / "variable_groups.json"
BASE_DICTIONARY = (
    Path(__file__).resolve().parents[1]
    / "documents"
    / "manuals"
    / "naaccr_data_dictionary_v25.json"
)
SITE_DICTIONARY = (
    Path(__file__).resolve().parents[1] / "documents" / "cipoc_data_dictionary.json"
)


class SiteApplicabilityTests(unittest.TestCase):
    def test_item_832_accepts_coded_breast_primary_site(self):
        group = next(
            group
            for group in load_variable_groups(VARIABLE_GROUPS)
            if any(variable.item_id == 832 for variable in group.variables)
        )

        self.assertTrue(site_applies(group.applies_to, CaseFacts(primary_site="C50.4")))
        self.assertFalse(site_applies(group.applies_to, CaseFacts(primary_site="C34.9")))


class CorpusCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.note = ProcessedClinicalNote(
            note_id=1,
            date="2025-02-20",
            note_type="Pathology",
            content="Left breast core biopsy.",
            cancer_mentions=[
                CancerMention(
                    presence=True,
                    confidence=ConfidenceLevel.HIGH,
                    evidence=[TextSpan(note_id=1, text="Left breast core biopsy.")],
                    status="current",
                    affected_tissue="left breast",
                    metastasis=False,
                )
            ],
            cancer_status={"current"},
        )
        self.breast_note = self.note.model_copy(
            update={
                "note_id": 2,
                "cancer_mentions": [
                    self.note.cancer_mentions[0].model_copy(
                        update={"affected_tissue": "breast"}
                    )
                ],
            }
        )

    def agent(self):
        agent = object.__new__(OrchestratorAgent)
        agent._data_dictionary_path = BASE_DICTIONARY
        agent._site_data_dictionary_path = SITE_DICTIONARY
        return agent

    def test_affected_tissue_is_kept_as_a_complete_name(self):
        descriptors = build_corpus_descriptors({1: self.note, 2: self.breast_note})

        self.assertEqual(
            descriptors.affected_tissues, {"current": {"left breast", "breast"}}
        )

    def test_characterization_sets_gross_primary_site_before_planning(self):
        state = CaseState(note_corpus={1: self.note, 2: self.breast_note})

        update = self.agent().characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    def test_single_cancer_mention_sets_gross_primary_site(self):
        state = CaseState(note_corpus={1: self.note})

        update = self.agent().characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    @patch("cipoc.agents.orchestrator.build_corpus_descriptors")
    def test_scalar_affected_tissue_is_not_split_into_characters(self, descriptors):
        descriptors.return_value = SimpleNamespace(
            affected_tissues={"current": "breast"}
        )
        state = CaseState(note_corpus={1: self.note})

        update = self.agent().characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    def test_initial_group_uses_breast_codes_after_characterization(self):
        state = CaseState(note_corpus={1: self.note, 2: self.breast_note})
        agent = self.agent()
        facts = agent.characterize_corpus(state)["case_facts"]
        initial_group = load_variable_groups(VARIABLE_GROUPS)[0]

        scoped = agent._scope_group(initial_group, facts)
        primary_site = next(
            variable for variable in scoped.variables if variable.item_id == 400
        )

        self.assertEqual(len(primary_site.valid_codes), 9)
        self.assertIn("C504", primary_site.valid_codes)
        self.assertNotIn("C341", primary_site.valid_codes)


class OrchestratorRunTests(unittest.TestCase):
    @patch("cipoc.agents.orchestrator.run_with_progress")
    def test_progress_can_be_disabled(self, run_with_progress):
        agent = object.__new__(OrchestratorAgent)
        agent._graph = MagicMock()
        agent._graph.invoke.return_value = {}

        result = agent.run(
            [
                {
                    "note_id": 1,
                    "date": "2025-02-20",
                    "note_type": "Pathology",
                    "content": "Left breast core biopsy.",
                }
            ],
            progress=False,
        )

        run_with_progress.assert_not_called()
        agent._graph.invoke.assert_called_once()
        graph_input = agent._graph.invoke.call_args.args[0]
        self.assertIsInstance(graph_input["note_corpus"][1], ClinicalNote)
        self.assertEqual(result.variable_results, {})


if __name__ == "__main__":
    unittest.main()
