import unittest

from cipoc.agents.orchestrator import CaseState, OrchestratorInput
from cipoc.models import ClinicalNote


class NoteIDTests(unittest.TestCase):
    def test_orchestrator_state_accepts_alphanumeric_note_ids(self):
        note = ClinicalNote(
            note_id="439464515c",
            date="2026-08-17",
            note_type="Progress Note",
            content="Example note",
        )

        orchestrator_input = OrchestratorInput(note_corpus={note.note_id: note})
        state = CaseState(note_corpus=orchestrator_input.note_corpus)

        self.assertEqual(state.note_corpus["439464515c"].note_id, "439464515c")


if __name__ == "__main__":
    unittest.main()
