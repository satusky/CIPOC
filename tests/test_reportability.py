import unittest

from cipoc.models import (
    CancerMention,
    CaseFacts,
    CaseVariableResult,
    ClinicalNote,
    ConfidenceLevel,
    ExtractionStop,
    ProcessedClinicalNote,
    TextSpan,
    VariableStatus,
)
from cipoc.tools import (
    CNS_TISSUES,
    checkpoint_one_stop_reason,
    is_reportable,
    mention_continues,
    stop_pending_results,
)


def _mention(tissue="lung", behavior="malignant", certainty="definite", text="Adenocarcinoma of the lung."):
    return CancerMention(
        presence=True,
        confidence=ConfidenceLevel.HIGH,
        evidence=[TextSpan(note_id=1, text=text)],
        status="current",
        affected_tissue=tissue,
        metastasis=False,
        behavior=behavior,
        diagnostic_certainty=certainty,
    )


def _corpus(*mentions_by_note):
    return {
        note_id: ProcessedClinicalNote(
            note_id=note_id, date="2026-09-01", note_type="Pathology",
            content="Synthetic note.", cancer_mentions=mentions,
        )
        for note_id, mentions in enumerate(mentions_by_note, start=1)
    }


class IsReportableTests(unittest.TestCase):
    def test_behavior_and_site_table(self):
        cases = [
            ("3", "C500", True),
            ("3", None, True),
            ("2", "C449", True),
            ("2", "C809", True),
            ("0", "C700", True),
            ("0", "C710", True),
            ("1", "C729", True),
            ("0", "C751", True),
            ("1", "C753", True),
            ("0", "C71.9", True),
            ("0", "c710", True),
            ("0", "C730", False),
            ("0", "C750", False),
            ("1", "C754", False),
            ("0", "C500", False),
            ("1", "C569", False),
            ("0", None, None),
            ("0", "C809", None),
            ("0", "not-a-site", None),
            (None, "C500", None),
            ("9", "C500", None),
        ]
        for behavior, site, expected in cases:
            with self.subTest(behavior=behavior, site=site):
                self.assertIs(is_reportable(CaseFacts(behavior=behavior, primary_site=site)), expected)

    def test_no_facts_are_unknown(self):
        self.assertIsNone(is_reportable(None))
        self.assertIsNone(is_reportable(CaseFacts()))


class MentionContinuesTests(unittest.TestCase):
    def test_definite_or_probable_with_a_known_behavior_continues(self):
        for behavior in ("malignant", "benign"):
            for certainty in ("definite", "probable"):
                with self.subTest(behavior=behavior, certainty=certainty):
                    self.assertTrue(mention_continues(_mention(behavior=behavior, certainty=certainty)))

    def test_possible_wording_never_continues(self):
        for behavior in ("malignant", "benign", "unknown"):
            with self.subTest(behavior=behavior):
                self.assertFalse(mention_continues(_mention("brain", behavior, "possible")))

    def test_unknown_behavior_continues_only_in_cns_tissues(self):
        self.assertEqual(
            CNS_TISSUES,
            {"meninges", "brain", "spinal_cord_and_cranial_nerves", "other_endocrine_gland"},
        )
        for tissue in sorted(CNS_TISSUES):
            for certainty in ("definite", "probable"):
                with self.subTest(tissue=tissue, certainty=certainty):
                    self.assertTrue(mention_continues(_mention(tissue, "unknown", certainty)))
        for tissue in ("lung", "unspecified", "thyroid_gland"):
            with self.subTest(tissue=tissue):
                self.assertFalse(mention_continues(_mention(tissue, "unknown")))


class CheckpointOneStopReasonTests(unittest.TestCase):
    def test_no_mentions_stops(self):
        reason = "Not reportable: no note mentions a neoplasm."
        self.assertEqual(checkpoint_one_stop_reason(None, {}), reason)
        self.assertEqual(checkpoint_one_stop_reason(None, _corpus(None, [])), reason)
        unscanned = {1: ClinicalNote(note_id=1, date="2026-09-01", note_type="Pathology", content="x")}
        self.assertEqual(checkpoint_one_stop_reason(None, unscanned), reason)

    def test_any_continuing_mention_lets_the_case_through(self):
        corpus = _corpus([_mention(certainty="possible")], [_mention()])
        self.assertIsNone(checkpoint_one_stop_reason(None, corpus))

    def test_stop_reason_cites_every_mention(self):
        corpus = _corpus(
            [_mention("lung", "unknown", text="4 mm nodule in the right upper lobe.")],
            [_mention("pancreas", "malignant", "possible", text="Possible pancreatic cancer.")],
        )
        reason = checkpoint_one_stop_reason(None, corpus)
        self.assertTrue(reason.startswith("Not reportable: no mention is a definite or probable neoplasm"))
        self.assertIn('note 1 (lung, behavior unknown, definite): "4 mm nodule in the right upper lobe."', reason)
        self.assertIn('note 2 (pancreas, behavior malignant, possible): "Possible pancreatic cancer."', reason)

    def test_supplied_facts_decide_before_the_mentions(self):
        malignant = _corpus([_mention("breast")])
        reason = checkpoint_one_stop_reason(CaseFacts(primary_site="C500", behavior="0"), malignant)
        self.assertEqual(
            reason,
            "Not reportable: supplied primary site C500, behavior 0. Benign and borderline "
            "tumors are reportable only at CNS sites (C700-C729, C751-C753).",
        )
        for facts in (
            CaseFacts(primary_site="C341"),
            CaseFacts(histology="8140"),
            CaseFacts(behavior="0"),
            CaseFacts(primary_site="C710", behavior="0"),
        ):
            with self.subTest(facts=facts):
                self.assertIsNone(checkpoint_one_stop_reason(facts, {}))

    def test_other_facts_leave_it_to_the_mentions(self):
        facts = CaseFacts(sex="2", date_of_diagnosis="2026-08-01", gross_primary_site="breast")
        self.assertIsNotNone(checkpoint_one_stop_reason(facts, {}))
        self.assertIsNone(checkpoint_one_stop_reason(facts, _corpus([_mention()])))


class StopPendingResultsTests(unittest.TestCase):
    def test_only_pending_variables_close(self):
        results = {
            400: CaseVariableResult(item_id=400, status=VariableStatus.STRUCTURED_DATA, value="C500"),
            410: CaseVariableResult(item_id=410),
            420: CaseVariableResult(item_id=420, status=VariableStatus.NOT_APPLICABLE, reason="Gate unmet."),
            430: CaseVariableResult(item_id=430),
        }
        updates = stop_pending_results(results, ExtractionStop.NOT_REPORTABLE)
        self.assertEqual(set(updates), {410, 430})
        for item_id, result in updates.items():
            self.assertEqual(result.item_id, item_id)
            self.assertEqual(result.status, VariableStatus.NOT_APPLICABLE)
            self.assertEqual(result.stopped_by, ExtractionStop.NOT_REPORTABLE)
            self.assertIsNone(result.reason)
        self.assertEqual(stop_pending_results({}, ExtractionStop.NOT_REPORTABLE), {})


if __name__ == "__main__":
    unittest.main()
