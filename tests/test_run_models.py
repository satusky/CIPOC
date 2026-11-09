import unittest
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from cipoc.models import (
    CancerMention,
    Case,
    CaseVariableResult,
    ClinicalNote,
    ConfidenceLevel,
    ExtractionStop,
    NoteCorpusDescriptors,
    NoteDigest,
    OrchestratorConfigFingerprint,
    OrchestratorRunCorpus,
    OrchestratorRunError,
    OrchestratorRunFailure,
    OrchestratorRunInfo,
    OrchestratorRunInputs,
    OrchestratorRunResult,
    ProcessedClinicalNote,
    ReportabilityStop,
    RunObservability,
    TargetGroup,
    TextSpan,
    ValidatedVariableOutput,
    VariableInfo,
    VariableStatus,
)


def fingerprint():
    return OrchestratorConfigFingerprint(
        agent_llm_config={
            "note_scanner": {
                "model": "test-model",
                "reasoning": {"effort": "medium"},
            }
        },
        retry={"note_scanner": {"max_attempts": 3, "jitter": True}},
        max_extraction_attempts=3,
        variable_groups_digest="sha256:variables",
        data_dictionary_digest="sha256:data",
        site_data_dictionary_digest="sha256:site",
        prompt_digests={"note_scanner.py": "sha256:prompt"},
        cipoc_version="1.0.0",
    )


def run_info(status="completed"):
    return OrchestratorRunInfo(
        run_id=uuid4(),
        started_at=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 4, 12, 0, 2, tzinfo=timezone.utc),
        duration_seconds=2.0,
        status=status,
        config_fingerprint=fingerprint(),
    )


def processed_note():
    evidence = TextSpan(note_id="note-A", text="Invasive breast carcinoma.")
    return ProcessedClinicalNote(
        note_id="note-A",
        date="2026-09-03",
        note_type="Pathology",
        content="Invasive breast carcinoma.",
        summary="Current breast malignancy.",
        concepts={
            "cancer": {
                "presence": True,
                "confidence": ConfidenceLevel.HIGH,
                "evidence": [evidence],
            }
        },
        cancer_status={"current"},
        cancer_mentions=[
            CancerMention(
                presence=True,
                confidence=ConfidenceLevel.HIGH,
                evidence=[evidence],
                status="current",
                affected_tissue="breast",
                metastasis=False,
            )
        ],
        flags=["pathology", "breast"],
    )


def inputs():
    return OrchestratorRunInputs(
        target_variables=[
            TargetGroup(
                group_id="initial",
                name="Initial",
                variables=[VariableInfo(item_id=400, name="Primary Site")],
                stage="initial",
            )
        ],
        structured_data={390: "8500"},
    )


def corpus():
    note = processed_note()
    return OrchestratorRunCorpus(
        note_corpus={note.note_id: note},
        note_digests={
            note.note_id: NoteDigest(
                note_id=note.note_id,
                note_type=note.note_type,
                summary=note.summary,
                flags=note.flags,
            )
        },
        note_corpus_descriptors=NoteCorpusDescriptors(
            note_count=1,
            date_range=(note.date, note.date),
            types={note.note_type},
            affected_tissues={"current": {"breast"}},
            concepts=note.concepts,
            unique_flags=set(note.flags),
        ),
    )


def observability():
    return RunObservability(llm_content_captured=False, collection_status="complete")


class OrchestratorRunModelTests(unittest.TestCase):
    def test_completed_result_round_trips_through_json(self):
        result = OrchestratorRunResult(
            run=run_info(),
            case=Case(),
            inputs=inputs(),
            corpus=corpus(),
            observability=observability(),
        )

        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())

        self.assertEqual(restored, result)
        self.assertEqual(restored.schema_version, "1.3")
        self.assertEqual(restored.run.started_at.tzinfo, timezone.utc)
        self.assertNotIn(
            "api_key",
            restored.run.config_fingerprint.agent_llm_config["note_scanner"],
        )

    def test_1_3_artifact_round_trips_mention_behavior_and_certainty(self):
        payload = OrchestratorRunResult(
            run=run_info(), case=Case(), inputs=inputs(), corpus=corpus(),
            observability=observability(),
        ).model_dump(mode="json")
        payload["corpus"]["note_corpus"]["note-A"]["cancer_mentions"][0].update(
            behavior="benign", diagnostic_certainty="probable",
        )

        result = OrchestratorRunResult.model_validate(payload)
        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())

        self.assertEqual(restored, result)
        self.assertEqual(restored.schema_version, "1.3")
        mention = restored.corpus.note_corpus["note-A"].cancer_mentions[0]
        self.assertEqual((mention.behavior, mention.diagnostic_certainty), ("benign", "probable"))

    def test_stopped_1_3_artifact_round_trips_the_flag_and_the_case_stop(self):
        stop = ReportabilityStop(checkpoint=1, reason="Not reportable: no note mentions a neoplasm.")
        case = Case(
            variable_results={
                390: CaseVariableResult(item_id=390, status=VariableStatus.STRUCTURED_DATA, value="8500"),
                400: CaseVariableResult(
                    item_id=400, status=VariableStatus.NOT_APPLICABLE,
                    stopped_by=ExtractionStop.NOT_REPORTABLE,
                ),
            },
            reportability_stop=stop,
        )
        result = OrchestratorRunResult(
            run=run_info(), case=case, inputs=inputs(), corpus=corpus(),
            observability=observability(),
        )

        payload = result.model_dump(mode="json")
        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())

        self.assertEqual(restored, result)
        self.assertEqual(restored.schema_version, "1.3")
        self.assertEqual(payload["case"]["reportability_stop"], {"checkpoint": 1, "reason": stop.reason})
        self.assertEqual(payload["case"]["variable_results"]["400"]["stopped_by"], "not_reportable")
        self.assertIsNone(payload["case"]["variable_results"]["400"]["reason"])
        self.assertIsNone(payload["case"]["variable_results"]["390"]["stopped_by"])

    def test_legacy_artifacts_load_without_a_stop(self):
        payload = OrchestratorRunResult(
            run=run_info(), case=Case(), inputs=inputs(), corpus=corpus(),
            observability=observability(),
        ).model_dump(mode="json")
        payload["schema_version"] = "1.2"
        del payload["case"]["reportability_stop"]
        payload["case"]["variable_results"] = {
            "400": {"item_id": 400, "status": "not_applicable", "reason": "Corpus gate not met."},
        }

        restored = OrchestratorRunResult.model_validate(payload)

        self.assertIsNone(restored.case.reportability_stop)
        self.assertIsNone(restored.case.variable_results[400].stopped_by)
        self.assertEqual(restored.case.variable_results[400].reason, "Corpus gate not met.")

    def test_legacy_mentions_load_as_definite_malignant(self):
        payload = OrchestratorRunResult(
            run=run_info(), case=Case(), inputs=inputs(), corpus=corpus(),
            observability=observability(),
        ).model_dump(mode="json")
        payload["schema_version"] = "1.2"
        legacy = payload["corpus"]["note_corpus"]["note-A"]["cancer_mentions"][0]
        del legacy["behavior"], legacy["diagnostic_certainty"]

        restored = OrchestratorRunResult.model_validate(payload)

        self.assertEqual(restored.schema_version, "1.2")
        mention = restored.corpus.note_corpus["note-A"].cancer_mentions[0]
        self.assertEqual((mention.behavior, mention.diagnostic_certainty), ("malignant", "definite"))

    def test_legacy_free_text_tissues_load_as_written(self):
        payload = OrchestratorRunResult(
            run=run_info(), case=Case(), inputs=inputs(), corpus=corpus(),
            observability=observability(),
        ).model_dump(mode="json")
        payload["schema_version"] = "1.2"
        payload["corpus"]["note_corpus"]["note-A"]["cancer_mentions"][0]["affected_tissue"] = "left breast"
        payload["inputs"]["target_variables"][0]["applies_to"] = {"gross_primary_sites": ["Left Breast"]}

        restored = OrchestratorRunResult.model_validate(payload)

        mention = restored.corpus.note_corpus["note-A"].cancer_mentions[0]
        self.assertEqual(mention.affected_tissue, "left breast")
        self.assertEqual(
            restored.inputs.target_variables[0].applies_to.gross_primary_sites, ["Left Breast"]
        )

    def test_case_facts_without_candidates_load_and_new_ones_round_trip(self):
        payload = OrchestratorRunResult(
            run=run_info(), case=Case(), inputs=inputs(), corpus=corpus(),
            observability=observability(),
        ).model_dump(mode="json")
        payload["schema_version"] = "1.2"
        payload["case"]["case_facts"] = {"gross_primary_site": "left breast"}

        restored = OrchestratorRunResult.model_validate(payload)

        self.assertEqual(restored.case.case_facts.gross_primary_site, "left breast")
        self.assertEqual(restored.case.case_facts.candidate_gross_sites, [])

        payload["schema_version"] = "1.3"
        payload["case"]["case_facts"] = {"candidate_gross_sites": ["breast", "ovary"]}
        restored = OrchestratorRunResult.model_validate_json(
            OrchestratorRunResult.model_validate(payload).model_dump_json()
        )
        self.assertEqual(restored.case.case_facts.candidate_gross_sites, ["breast", "ovary"])

    def test_unknown_schema_version_is_rejected(self):
        result = OrchestratorRunResult(
            run=run_info(),
            case=Case(),
            inputs=inputs(),
            corpus=corpus(),
            observability=observability(),
        ).model_dump(mode="json")
        result["schema_version"] = "2.0"

        with self.assertRaises(ValidationError):
            OrchestratorRunResult.model_validate(result)

    def test_legacy_observability_is_readable_but_new_artifacts_require_status(self):
        result = OrchestratorRunResult(
            run=run_info(), case=Case(), inputs=inputs(), corpus=corpus(),
            observability=observability(),
        ).model_dump(mode="json")
        result["observability"].pop("collection_status")
        with self.assertRaisesRegex(ValidationError, "explicit observability collection status"):
            OrchestratorRunResult.model_validate(result)
        result["schema_version"] = "1.0"
        # Historical targeting labels remain readable without executing today's
        # configuration checks or recomputing past clinical decisions.
        result["inputs"]["target_variables"][0]["applies_to"] = {
            "histology_families": ["legacy-family"],
        }
        restored = OrchestratorRunResult.model_validate(result)
        self.assertIsNone(restored.observability.collection_status)
        self.assertEqual(restored.schema_version, "1.0")
        self.assertEqual(restored.inputs.target_variables[0].applies_to.histology_families, ["legacy-family"])

    def test_fingerprint_rejects_secrets_and_non_json_values(self):
        values = fingerprint().model_dump()
        values["agent_llm_config"]["extractor"] = {"api_key": "secret"}
        with self.assertRaises(ValidationError):
            OrchestratorConfigFingerprint.model_validate(values)

        values = fingerprint().model_dump()
        values["agent_llm_config"]["extractor"] = {"custom": object()}
        with self.assertRaises(ValidationError):
            OrchestratorConfigFingerprint.model_validate(values)

    def test_completed_and_failed_versions_preserve_legacy_and_require_modern_status(self):
        for model, extra in (
            (OrchestratorRunResult, {"run": run_info(), "case": Case(), "corpus": corpus()}),
            (OrchestratorRunFailure, {"run": run_info("failed"), "error": "failure"}),
        ):
            artifact = model(inputs=inputs(), observability=observability(), **extra)
            self.assertEqual(artifact.schema_version, "1.3")
            for version in ("1.0", "1.1", "1.2", "1.3"):
                with self.subTest(model=model.__name__, version=version):
                    payload = {**artifact.model_dump(mode="json"), "schema_version": version}
                    restored = model.model_validate(payload)
                    self.assertEqual(restored.schema_version, version)
                    self.assertEqual(model.model_validate_json(restored.model_dump_json()), restored)
                    payload["observability"].pop("collection_status")
                    if version == "1.0":
                        self.assertIsNone(model.model_validate(payload).observability.collection_status)
                    else:
                        with self.assertRaisesRegex(ValidationError, "explicit observability collection status"):
                            model.model_validate(payload)

    def test_processed_note_scan_fields_survive_json_round_trip(self):
        restored = OrchestratorRunCorpus.model_validate_json(corpus().model_dump_json())
        note = restored.note_corpus["note-A"]

        self.assertIsInstance(note, ProcessedClinicalNote)
        self.assertEqual(note.summary, "Current breast malignancy.")
        self.assertEqual(note.cancer_status, {"current"})
        self.assertEqual(note.cancer_mentions[0].affected_tissue, "breast")
        self.assertEqual(note.flags, ["pathology", "breast"])
        self.assertTrue(note.concepts["cancer"].presence)

        raw_note = ClinicalNote(
            note_id="note-B",
            date="2026-09-03",
            note_type="Pathology",
            content="Unscanned note.",
        )
        with self.assertRaises(ValidationError):
            OrchestratorRunCorpus(note_corpus={raw_note.note_id: raw_note})

    def test_timestamps_must_be_timezone_aware(self):
        values = run_info().model_dump()
        values["started_at"] = datetime(2026, 9, 4, 12, 0)

        with self.assertRaises(ValidationError):
            OrchestratorRunInfo.model_validate(values)

    def test_failure_and_exception_retain_partial_payload_without_case(self):
        failure = OrchestratorRunFailure(
            run=run_info(status="failed"),
            inputs=inputs(),
            corpus=None,
            observability=observability(),
            error="scanner endpoint unavailable",
        )
        restored = OrchestratorRunFailure.model_validate_json(
            failure.model_dump_json()
        )
        error = OrchestratorRunError(restored)

        self.assertIs(error.failure, restored)
        self.assertEqual(str(error), "scanner endpoint unavailable")
        self.assertIsNone(error.failure.corpus)
        self.assertNotIn("case", error.failure.model_dump())

        with self.assertRaises(ValidationError):
            OrchestratorRunFailure.model_validate(
                {**failure.model_dump(), "case": Case()}
            )

    def test_result_and_failure_require_their_matching_status(self):
        common = {
            "inputs": inputs(),
            "corpus": corpus(),
            "observability": observability(),
        }
        with self.assertRaises(ValidationError):
            OrchestratorRunResult(
                run=run_info(status="failed"), case=Case(), **common
            )
        with self.assertRaises(ValidationError):
            OrchestratorRunFailure(
                run=run_info(), error="failed", **common
            )



class CaseVariableStopTests(unittest.TestCase):
    """``stopped_by`` marks a variable closed by a case-level determination."""

    def test_only_not_applicable_variables_may_carry_the_flag(self):
        extraction = dict(
            explanation="Scripted.", most_important_note=None, spans=[],
            presence_confidence=ConfidenceLevel.LOW, is_valid=True,
        )
        others = {
            VariableStatus.PENDING: {},
            VariableStatus.STRUCTURED_DATA: {"value": "C500"},
            VariableStatus.EXTRACTED: {
                "value": "C500",
                "extraction": ValidatedVariableOutput(item_id=400, value="C500", **extraction),
            },
            VariableStatus.NOT_FOUND: {
                "extraction": ValidatedVariableOutput(item_id=400, value=None, **extraction),
            },
            VariableStatus.ERROR: {"reason": "Extraction failed."},
            VariableStatus.BLOCKED: {"reason": "Waiting.", "blocking_item_ids": [390]},
        }
        self.assertEqual(set(others) | {VariableStatus.NOT_APPLICABLE}, set(VariableStatus))
        for status, fields in others.items():
            with self.subTest(status=status):
                CaseVariableResult(item_id=400, status=status, **fields)
                with self.assertRaisesRegex(ValidationError, "case-level stop"):
                    CaseVariableResult(
                        item_id=400, status=status, stopped_by=ExtractionStop.NOT_REPORTABLE, **fields
                    )

    def test_not_applicable_needs_a_reason_or_the_flag(self):
        with self.assertRaisesRegex(ValidationError, "require a reason or a case-level stop"):
            CaseVariableResult(item_id=400, status=VariableStatus.NOT_APPLICABLE)
        stopped = CaseVariableResult(
            item_id=400, status=VariableStatus.NOT_APPLICABLE, stopped_by=ExtractionStop.NOT_REPORTABLE
        )
        self.assertIsNone(stopped.reason)
        gated = CaseVariableResult(item_id=400, status=VariableStatus.NOT_APPLICABLE, reason="Gate unmet.")
        self.assertIsNone(gated.stopped_by)


if __name__ == "__main__":
    unittest.main()
