import unittest
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from cipoc.models import (
    CancerMention,
    Case,
    ClinicalNote,
    ConfidenceLevel,
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
    RunObservability,
    TargetGroup,
    TextSpan,
    VariableInfo,
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
    return RunObservability(llm_content_captured=False)


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
        self.assertEqual(restored.schema_version, "1.0")
        self.assertEqual(restored.run.started_at.tzinfo, timezone.utc)
        self.assertNotIn(
            "api_key",
            restored.run.config_fingerprint.agent_llm_config["note_scanner"],
        )

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

    def test_fingerprint_rejects_secrets_and_non_json_values(self):
        values = fingerprint().model_dump()
        values["agent_llm_config"]["extractor"] = {"api_key": "secret"}
        with self.assertRaises(ValidationError):
            OrchestratorConfigFingerprint.model_validate(values)

        values = fingerprint().model_dump()
        values["agent_llm_config"]["extractor"] = {"custom": object()}
        with self.assertRaises(ValidationError):
            OrchestratorConfigFingerprint.model_validate(values)

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


if __name__ == "__main__":
    unittest.main()
