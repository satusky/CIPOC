import unittest

from pydantic import ValidationError

from cipoc.agents.note_scanner import (
    ConceptFinding,
    NoteScannerAgent,
    ScannerState,
    concept_findings_model,
)
from cipoc.models import (
    CancerMention,
    ClinicalNote,
    CONCEPT_DESCRIPTIONS,
    TextSpan,
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def structured(self, schema, messages, **kwargs):
        self.messages = messages
        return self.response


def _finding(present=False, *, evidence="carcinoma"):
    return ConceptFinding(
        presence=present,
        confidence="max",
        evidence=[TextSpan(note_id="wrong", text=evidence)] if present else [],
    )


def _findings(**present):
    schema = concept_findings_model(CONCEPT_DESCRIPTIONS)
    return schema(
        **{
            name: _finding(present.get(name, False), evidence=name)
            for name in CONCEPT_DESCRIPTIONS
        }
    )


def _scanner_with(response):
    scanner = object.__new__(NoteScannerAgent)
    scanner.agent = FakeLLM(response)
    return scanner


def _state():
    return ScannerState(
        note=ClinicalNote(
            note_id=1,
            date="2025-01-01",
            note_type="Pathology Report",
            content="Final diagnosis: invasive carcinoma.",
        ),
        messages=[],
    )


class ConceptDetectionTests(unittest.TestCase):
    def test_schema_requires_every_configured_concept(self):
        descriptions = {**CONCEPT_DESCRIPTIONS, "immunotherapy": "Cancer immunotherapy."}
        schema = concept_findings_model(descriptions).model_json_schema()

        self.assertEqual(set(schema["required"]), set(descriptions))

    def test_finding_requires_model_populated_fields(self):
        with self.assertRaises(ValidationError):
            ConceptFinding()

    def test_complete_schema_rejects_an_incomplete_response(self):
        schema = concept_findings_model(CONCEPT_DESCRIPTIONS)

        with self.assertRaises(ValidationError):
            schema(cancer=_finding(True))

    def test_detect_concepts_returns_all_findings_and_normalizes_evidence_ids(self):
        scanner = _scanner_with(_findings(cancer=True, surgery=True))

        result = scanner.detect_concepts(_state())
        concepts = result["concepts"]

        self.assertEqual(set(concepts), set(CONCEPT_DESCRIPTIONS))
        self.assertTrue(concepts["surgery"].presence)
        self.assertTrue(concepts["cancer"].presence)
        self.assertEqual(concepts["surgery"].evidence[0].note_id, 1)

    def test_cancer_is_implied_by_cancer_directed_treatment(self):
        scanner = _scanner_with(_findings(chemotherapy=True))

        result = scanner.detect_concepts(_state())

        self.assertTrue(result["concepts"]["cancer"].presence)
        self.assertEqual(result["concepts"]["cancer"].evidence[0].text, "chemotherapy")

    def test_detect_concepts_passes_template_and_returns_findings(self):
        scanner = _scanner_with(_findings(cancer=True))

        result = scanner.detect_concepts(_state())

        self.assertTrue(result["concepts"]["cancer"].presence)
        self.assertEqual(set(result["concepts"]), set(CONCEPT_DESCRIPTIONS))
        prompt = scanner.agent.messages[-1].content
        for name, description in CONCEPT_DESCRIPTIONS.items():
            self.assertIn(f'"{name}":', prompt)
            self.assertIn(description, prompt)


class CancerMentionSchemaTests(unittest.TestCase):
    def mention(self, **updates):
        values = {
            "presence": True,
            "confidence": "max",
            "evidence": [TextSpan(note_id=1, text="invasive breast carcinoma")],
            "status": "current",
            "affected_tissue": "breast",
            "metastasis": False,
        }
        values.update(updates)
        return CancerMention(**values)

    def test_assertion_fields_are_required(self):
        schema = CancerMention.model_json_schema()

        self.assertTrue(
            {"presence", "confidence", "evidence"}.issubset(schema["required"])
        )
        self.assertTrue(schema["properties"]["presence"]["const"])

    def test_presence_must_be_true(self):
        with self.assertRaises(ValidationError):
            self.mention(presence=False)

    def test_evidence_must_not_be_empty(self):
        with self.assertRaises(ValidationError):
            self.mention(evidence=[])

    def test_complete_cancer_mention_is_valid(self):
        mention = self.mention()

        self.assertTrue(mention.presence)
        self.assertEqual(mention.affected_tissue, "breast")


if __name__ == "__main__":
    unittest.main()
