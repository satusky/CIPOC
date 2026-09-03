import unittest
from datetime import date

from cipoc.models import (
    NoteFilter,
    NoteFilterEvaluation,
    NoteSelectionRejectionCode,
    NoteSelectionUnevaluatedCode,
    ProcessedClinicalNote,
)
from cipoc.tools import evaluate_note_filter, note_matches_filter, prefilter_notes


class NoteFilterEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.note = ProcessedClinicalNote(
            note_id="path-1",
            date="2025-02-20",
            note_type="Pathology",
            content="Breast biopsy.",
            summary="Current breast malignancy.",
            cancer_status={"current"},
            flags=["breast", "biopsy"],
        )
        self.anchor = date(2025, 2, 25)

    def test_none_filter_passes_without_reasons(self):
        result = evaluate_note_filter(self.note, None)

        self.assertIsInstance(result, NoteFilterEvaluation)
        self.assertTrue(result.passes)
        self.assertEqual(result.rejection_reasons, [])
        self.assertEqual(result.unevaluated_checks, [])

    def test_note_type_is_case_insensitive_but_requires_exact_match(self):
        matching = evaluate_note_filter(
            self.note, NoteFilter(note_types=[" pathology "])
        )
        rejected = evaluate_note_filter(
            self.note, NoteFilter(note_types=["path"])
        )

        self.assertTrue(matching.passes)
        self.assertEqual(
            rejected.rejection_reasons,
            [NoteSelectionRejectionCode.NOTE_TYPE_MISMATCH],
        )

    def test_cancer_status_requires_set_intersection(self):
        matching = evaluate_note_filter(
            self.note, NoteFilter(cancer_status=["historical", "current"])
        )
        rejected = evaluate_note_filter(
            self.note, NoteFilter(cancer_status=["historical"])
        )

        self.assertTrue(matching.passes)
        self.assertEqual(
            rejected.rejection_reasons,
            [NoteSelectionRejectionCode.CANCER_STATUS_MISMATCH],
        )

    def test_missing_or_invalid_date_is_reported(self):
        for invalid_date in ("", "02/20/2025", "not-a-date"):
            with self.subTest(date=invalid_date):
                result = evaluate_note_filter(
                    self.note.model_copy(update={"date": invalid_date}),
                    NoteFilter(within_days=30),
                    anchor=self.anchor,
                )
                self.assertEqual(
                    result.rejection_reasons,
                    [NoteSelectionRejectionCode.MISSING_OR_INVALID_DATE],
                )

    def test_note_outside_date_window_is_reported(self):
        result = evaluate_note_filter(
            self.note,
            NoteFilter(within_days=3),
            anchor=self.anchor,
        )

        self.assertEqual(
            result.rejection_reasons,
            [NoteSelectionRejectionCode.OUTSIDE_DATE_WINDOW],
        )

    def test_configured_keyword_filter_is_reported_and_skipped(self):
        result = evaluate_note_filter(
            self.note,
            NoteFilter(keywords=["absent-keyword"]),
        )

        self.assertTrue(result.passes)
        self.assertEqual(
            result.unevaluated_checks,
            [NoteSelectionUnevaluatedCode.KEYWORD_FILTER_DISABLED],
        )

    def test_date_window_without_anchor_is_reported_and_skipped(self):
        result = evaluate_note_filter(
            self.note.model_copy(update={"date": "not-a-date"}),
            NoteFilter(within_days=1),
        )

        self.assertTrue(result.passes)
        self.assertEqual(result.rejection_reasons, [])
        self.assertEqual(
            result.unevaluated_checks,
            [NoteSelectionUnevaluatedCode.TEMPORAL_ANCHOR_UNAVAILABLE],
        )

    def test_all_applicable_reasons_and_skipped_checks_are_reported(self):
        result = evaluate_note_filter(
            self.note.model_copy(update={"date": "invalid"}),
            NoteFilter(
                note_types=["Radiology"],
                keywords=["breast"],
                cancer_status=["historical"],
                within_days=30,
            ),
            anchor=self.anchor,
        )

        self.assertFalse(result.passes)
        self.assertEqual(
            result.rejection_reasons,
            [
                NoteSelectionRejectionCode.NOTE_TYPE_MISMATCH,
                NoteSelectionRejectionCode.CANCER_STATUS_MISMATCH,
                NoteSelectionRejectionCode.MISSING_OR_INVALID_DATE,
            ],
        )
        self.assertEqual(
            result.unevaluated_checks,
            [NoteSelectionUnevaluatedCode.KEYWORD_FILTER_DISABLED],
        )

    def test_boolean_api_has_parity_for_prior_filter_behaviors(self):
        scenarios = [
            (None, None, True),
            (NoteFilter(), None, True),
            (NoteFilter(note_types=["PATHOLOGY"]), None, True),
            (NoteFilter(note_types=["Path"]), None, False),
            (NoteFilter(keywords=["missing"]), None, True),
            (NoteFilter(cancer_status=["current"]), None, True),
            (NoteFilter(cancer_status=["recent"]), None, False),
            (NoteFilter(within_days=1), None, True),
            (NoteFilter(within_days=5), self.anchor, True),
            (NoteFilter(within_days=4), self.anchor, False),
        ]

        for note_filter, anchor, expected in scenarios:
            with self.subTest(note_filter=note_filter, anchor=anchor):
                evaluation = evaluate_note_filter(
                    self.note, note_filter, anchor=anchor
                )
                self.assertEqual(evaluation.passes, expected)
                self.assertEqual(
                    note_matches_filter(self.note, note_filter, anchor=anchor),
                    expected,
                )

    def test_prefilter_preserves_note_order(self):
        notes = [
            self.note.model_copy(update={"note_id": 3}),
            self.note.model_copy(update={"note_id": 1, "note_type": "Radiology"}),
            self.note.model_copy(update={"note_id": 2}),
        ]

        filtered = prefilter_notes(notes, NoteFilter(note_types=["Pathology"]))

        self.assertEqual([note.note_id for note in filtered], [3, 2])


if __name__ == "__main__":
    unittest.main()
