import unittest
from typing import get_args

from pydantic import ValidationError

from cipoc.agents.note_scanner import (
    CancerMentionFinding,
    CancerMentions,
    ConceptFinding,
    NoteScannerAgent,
    ScannerState,
    concept_findings_model,
)
from cipoc.models import (
    CancerMention,
    ClinicalNote,
    CONCEPT_DESCRIPTIONS,
    CorpusGate,
    GrossSite,
    ProcessedClinicalNote,
    TextSpan,
)
from cipoc.prompts.note_scanner import CANCER_MENTIONS_PROMPT
from cipoc.tools.orchestration import build_corpus_descriptors, corpus_gate_passes, is_diagnosed_malignant


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
        descriptions = {**CONCEPT_DESCRIPTIONS, "targeted_therapy": "Cancer targeted therapy."}
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
        for concept in ("chemotherapy", "hormonal_therapy", "immunotherapy"):
            with self.subTest(concept=concept):
                scanner = _scanner_with(_findings(**{concept: True}))
                result = scanner.detect_concepts(_state())
                self.assertTrue(result["concepts"]["cancer"].presence)
                self.assertEqual(result["concepts"]["cancer"].evidence[0].text, concept)

    def test_detect_concepts_passes_template_and_returns_findings(self):
        scanner = _scanner_with(_findings(cancer=True))

        result = scanner.detect_concepts(_state())

        self.assertTrue(result["concepts"]["cancer"].presence)
        self.assertEqual(set(result["concepts"]), set(CONCEPT_DESCRIPTIONS))
        prompt = scanner.agent.messages[-1].content
        for name, description in CONCEPT_DESCRIPTIONS.items():
            self.assertIn(f'"{name}":', prompt)
            self.assertIn(description, prompt)

    def test_treatment_definitions_are_cancer_directed_and_not_cytotoxic_chemo(self):
        self.assertEqual(
            CONCEPT_DESCRIPTIONS["chemotherapy"],
            "Any systemic cytotoxic chemotherapy that was administered or planned.",
        )
        for concept in ("hormonal_therapy", "immunotherapy"):
            with self.subTest(concept=concept):
                self.assertIn("cancer-directed", CONCEPT_DESCRIPTIONS[concept])
                self.assertIn("administered or planned", CONCEPT_DESCRIPTIONS[concept])
                self.assertIn("non-cancer", CONCEPT_DESCRIPTIONS[concept])

    def test_real_scanner_graph_preserves_treatment_provenance_and_negative_findings(self):
        for concept, note_id, content, present in (
            ("hormonal_therapy", "001", "Anastrozole planned for breast cancer.", True),
            ("immunotherapy", "note-A", "Pembrolizumab administered for melanoma.", True),
            ("hormonal_therapy", "001", "Hormone replacement for menopause; no cancer.", False),
            ("immunotherapy", "note-A", "Allergen immunotherapy for rhinitis; no cancer.", False),
        ):
            with self.subTest(concept=concept, present=present):
                class ScriptedLLM:
                    def structured(self, schema, messages, **kwargs):
                        if schema.__name__ == "NoteSummary":
                            return schema(summary=content, keywords=[concept])
                        if schema.__name__ == "CancerMentions":
                            return schema(mentions=[])
                        return schema(**{
                            name: _finding(present and name == concept, evidence=content)
                            for name in CONCEPT_DESCRIPTIONS
                        })

                scanner = _scanner_with(None)
                scanner.agent = ScriptedLLM()
                scanner._retry_policy = None
                scanner._graph = scanner._build_graph()
                note = scanner.run(ClinicalNote(
                    note_id=note_id, date="2025-01-01", note_type="Oncology", content=content
                ), progress=False)
                restored = ProcessedClinicalNote.model_validate_json(note.model_dump_json())
                self.assertEqual(restored.concepts[concept].presence, present)
                self.assertEqual(restored.concepts["cancer"].presence, present)
                self.assertFalse(restored.concepts["chemotherapy"].presence)
                if present:
                    self.assertEqual(restored.concepts[concept].evidence, [TextSpan(note_id=note_id, text=content)])
                    self.assertEqual(restored.concepts[concept].confidence, "max")
                else:
                    self.assertEqual(restored.concepts[concept].evidence, [])
                corpus = build_corpus_descriptors({note_id: restored})
                self.assertEqual(corpus_gate_passes([CorpusGate.TREATMENT_PRESENT], corpus), present)


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

    def test_scanner_output_answers_behavior_and_certainty_and_saves_the_base_mention(self):
        values = self.mention().model_dump(exclude={"behavior", "diagnostic_certainty"})
        with self.assertRaises(ValidationError):
            CancerMentionFinding(**values)
        finding = CancerMentionFinding(**values, behavior="benign", diagnostic_certainty="possible")
        scanner = _scanner_with(CancerMentions(mentions=[finding]))

        result = scanner.get_cancer_mentions(_state())

        [mention] = result["cancer_mentions"]
        self.assertIs(type(mention), CancerMention)
        self.assertEqual((mention.behavior, mention.diagnostic_certainty), ("benign", "possible"))
        prompt = scanner.agent.messages[-1].content
        self.assertIn("- behavior:", prompt)
        self.assertIn("- diagnostic_certainty:", prompt)


    def test_scanner_output_chooses_the_tissue_from_the_vocabulary(self):
        sent = CancerMentions.model_json_schema()["$defs"]["CancerMentionFinding"]
        tissue = sent["properties"]["affected_tissue"]
        self.assertEqual(tissue["enum"], list(get_args(GrossSite)))
        self.assertIn("affected_tissue", sent["required"])
        self.assertIn("'unspecified'", tissue["description"])
        self.assertNotIn("enum", CancerMention.model_json_schema()["properties"]["affected_tissue"])

        values = self.mention().model_dump()
        self.assertEqual(CancerMentionFinding(**values).affected_tissue, "breast")
        with self.assertRaises(ValidationError):
            CancerMentionFinding(**{**values, "affected_tissue": "left breast"})
        self.assertEqual(self.mention(affected_tissue="left breast").affected_tissue, "left breast")


class DiagnosedMalignantMentionTests(unittest.TestCase):
    """Only definite or probable malignant mentions feed note status and affected tissues."""

    def finding(self, status, tissue, behavior="malignant", diagnostic_certainty="definite"):
        return CancerMentionFinding(
            presence=True,
            confidence="max",
            evidence=[TextSpan(note_id=1, text="Final diagnosis: invasive carcinoma.")],
            status=status,
            affected_tissue=tissue,
            metastasis=False,
            behavior=behavior,
            diagnostic_certainty=diagnostic_certainty,
        )

    def scan(self, *findings):
        """Run the mention step on one note, then roll the corpus up from it."""
        state = _state()
        result = _scanner_with(CancerMentions(mentions=list(findings))).get_cancer_mentions(state)
        note = ProcessedClinicalNote(**state.note.model_dump(), **result)
        return note, build_corpus_descriptors({note.note_id: note})

    def test_benign_unknown_and_possible_mentions_are_excluded(self):
        for behavior, certainty in (
            ("benign", "definite"),
            ("benign", "probable"),
            ("unknown", "definite"),
            ("malignant", "possible"),
        ):
            with self.subTest(behavior=behavior, diagnostic_certainty=certainty):
                note, descriptors = self.scan(self.finding("current", "lung", behavior, certainty))

                self.assertEqual(len(note.cancer_mentions), 1)
                self.assertIsNone(note.cancer_status)
                self.assertEqual(descriptors.affected_tissues, {})

    def test_definite_and_probable_malignant_mentions_are_included(self):
        for certainty in ("definite", "probable"):
            with self.subTest(diagnostic_certainty=certainty):
                note, descriptors = self.scan(
                    self.finding("current", "breast", diagnostic_certainty=certainty)
                )

                self.assertEqual(note.cancer_status, {"current"})
                self.assertEqual(descriptors.affected_tissues, {"current": {"breast"}})

    def test_mixed_note_keeps_only_definite_and_probable_malignant_mentions(self):
        note, descriptors = self.scan(
            self.finding("current", "breast"),
            self.finding("current", "lung", behavior="unknown"),
            self.finding("recent", "colon", diagnostic_certainty="probable"),
            self.finding("recent", "pancreas", diagnostic_certainty="possible"),
            self.finding("historical", "ovary", behavior="benign"),
        )

        self.assertEqual(len(note.cancer_mentions), 5)
        self.assertEqual(note.cancer_status, {"current", "recent"})
        self.assertEqual(descriptors.affected_tissues, {"current": {"breast"}, "recent": {"colon"}})

    def test_legacy_mentions_without_the_fields_still_count(self):
        legacy = self.finding("historical", "prostate").model_dump(
            mode="json", exclude={"behavior", "diagnostic_certainty"}
        )
        note = ProcessedClinicalNote.model_validate(
            {**_state().note.model_dump(), "cancer_mentions": [legacy], "cancer_status": ["historical"]}
        )

        [mention] = note.cancer_mentions
        self.assertTrue(is_diagnosed_malignant(mention))
        self.assertEqual(
            build_corpus_descriptors({note.note_id: note}).affected_tissues,
            {"historical": {"prostate"}},
        )


class NeoplasmGateTests(unittest.TestCase):
    """The neoplasm concept opens the mention step without implying cancer."""

    def mention(self, tissue, behavior, content):
        return CancerMentionFinding(
            presence=True,
            confidence="max",
            evidence=[TextSpan(note_id=1, text=content)],
            status="current",
            affected_tissue=tissue,
            metastasis=False,
            behavior=behavior,
            diagnostic_certainty="definite",
        )

    def run_scanner(self, content, present, mentions):
        """Run the real scanner graph with scripted concept and mention answers."""
        class ScriptedLLM:
            def structured(self, schema, messages, **kwargs):
                if schema.__name__ == "NoteSummary":
                    return schema(summary=content, keywords=["imaging"])
                if schema.__name__ == "CancerMentions":
                    return schema(mentions=mentions)
                return schema(**{
                    name: _finding(name in present, evidence=content)
                    for name in CONCEPT_DESCRIPTIONS
                })

        scanner = _scanner_with(None)
        scanner.agent = ScriptedLLM()
        scanner._retry_policy = None
        scanner._graph = scanner._build_graph()
        note = scanner.run(ClinicalNote(
            note_id=1, date="2025-01-01", note_type="Radiology", content=content
        ), progress=False)
        return ProcessedClinicalNote.model_validate_json(note.model_dump_json())

    def test_neoplasm_does_not_imply_cancer(self):
        result = _scanner_with(_findings(neoplasm=True)).detect_concepts(_state())
        self.assertTrue(result["concepts"]["neoplasm"].presence)
        self.assertFalse(result["concepts"]["cancer"].presence)

        result = _scanner_with(_findings(neoplasm=True, chemotherapy=True)).detect_concepts(_state())
        self.assertEqual(result["concepts"]["cancer"].evidence[0].text, "chemotherapy")

    def test_benign_only_note_reaches_the_mention_step(self):
        content = "MRI brain: 2 cm dural-based meningioma, WHO grade 1."
        meningioma = self.mention("meninges", "benign", content)

        note = self.run_scanner(content, {"neoplasm"}, [meningioma])

        self.assertTrue(note.concepts["neoplasm"].presence)
        self.assertFalse(note.concepts["cancer"].presence)
        [mention] = note.cancer_mentions
        self.assertEqual(mention.behavior, "benign")
        self.assertIsNone(note.cancer_status)

        # With neither concept present, the scripted mention is never requested.
        self.assertIsNone(self.run_scanner(content, set(), [meningioma]).cancer_mentions)

    def test_lone_lung_nodule_is_saved_with_unknown_behavior(self):
        behavior_line = next(
            line for line in CANCER_MENTIONS_PROMPT.splitlines() if line.startswith("- behavior:")
        )
        self.assertIn("nodule", behavior_line.split('"unknown"', 1)[1])
        sent_schema = CancerMentions.model_json_schema()
        behavior = sent_schema["$defs"]["CancerMentionFinding"]["properties"]["behavior"]
        self.assertIn("nodule", behavior["description"].split("'unknown'", 1)[1])
        scope = CANCER_MENTIONS_PROMPT.split("For each mention", 1)[0]
        for finding in ("benign", "borderline", "mass", "nodule", "lesion"):
            with self.subTest(finding=finding):
                self.assertIn(finding, scope)

        content = "CT chest: 8 mm indeterminate right upper lobe nodule. Recommend follow-up CT in 3 months."
        note = self.run_scanner(content, {"neoplasm"}, [self.mention("lung", "unknown", content)])

        [mention] = note.cancer_mentions
        self.assertEqual(mention.behavior, "unknown")
        self.assertIsNone(note.cancer_status)
        self.assertEqual(build_corpus_descriptors({note.note_id: note}).affected_tissues, {})


if __name__ == "__main__":
    unittest.main()
