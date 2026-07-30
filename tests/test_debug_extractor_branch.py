import unittest

from scripts.debug_extractor_branch import group_by_name, run_until_extractor_branch
from tests.fake_orchestrator import (
    VARIABLE_GROUPS,
    build_fake_orchestrator,
    load_notes,
)


class GroupSelectionTests(unittest.TestCase):
    def test_selects_nested_group_by_display_name(self):
        group = group_by_name(VARIABLE_GROUPS, "Histologic Type and Behavior Code")

        self.assertEqual(group.group_id, "histologic_type_and_behavior")
        self.assertEqual([variable.item_id for variable in group.variables], [522, 523])

    def test_unknown_name_lists_available_names(self):
        with self.assertRaisesRegex(ValueError, "Available names.*Initial LLM Extraction"):
            group_by_name(VARIABLE_GROUPS, "missing")


class ExtractorBranchCaptureTests(unittest.TestCase):
    def test_returns_dispatch_state_without_running_extract_branch(self):
        agent = build_fake_orchestrator()
        group = group_by_name(VARIABLE_GROUPS, "Initial LLM Extraction")
        agent._retriever.run = lambda *args, **kwargs: self.fail(  # type: ignore[method-assign]
            "extract branch should not execute"
        )
        notes = load_notes()

        state = run_until_extractor_branch(
            agent,
            group,
            [note.model_dump() for note in notes],
        )

        self.assertEqual(state.requested_variables.group_id, "initial_llm_extraction")
        self.assertEqual(
            [variable.item_id for variable in state.requested_variables.variables],
            [390, 400, 410],
        )
        self.assertEqual(set(state.branch_note_corpus), {note.note_id for note in notes})
        self.assertEqual(set(state.branch_note_digests), {note.note_id for note in notes})
        self.assertEqual(state.retrieved_note_ids, [])


if __name__ == "__main__":
    unittest.main()
