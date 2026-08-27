import csv
import json
import tempfile
import unittest
from pathlib import Path

from cipoc.export import NOTE_NLP_FIELDS, OmopExporter
from cipoc.models import (
    Case,
    CaseVariableResult,
    ClinicalNote,
    ConfidenceLevel,
    TextSpan,
    ValidatedVariableOutput,
    VariableStatus,
)


def extraction(
    item_id: int,
    value: str | None,
    spans: list[TextSpan],
    *,
    is_valid: bool = True,
) -> ValidatedVariableOutput:
    return ValidatedVariableOutput(
        item_id=item_id,
        value=value,
        explanation="Test extraction",
        most_important_note=1 if value is not None else None,
        spans=spans,
        presence_confidence=ConfidenceLevel.MAX,
        is_valid=is_valid,
        validation_errors=[] if is_valid else ["Invalid test extraction"],
        extraction_attempts=1,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class OmopExporterTests(unittest.TestCase):
    def setUp(self):
        self.note = ClinicalNote(
            note_id="note-A",
            date="2025-02-24",
            note_type="Pathology Report",
            content="Final Diagnosis: Invasive ductal carcinoma of the left breast.",
        )

    def test_exports_notes_and_valid_non_null_extraction_spans(self):
        first_span = TextSpan(
            note_id="note-A",
            text="Invasive ductal carcinoma",
        )
        second_span = TextSpan(note_id="note-A", text="left breast")
        case = Case(
            variable_results={
                400: CaseVariableResult(
                    item_id=400,
                    status=VariableStatus.EXTRACTED,
                    value="C504",
                    extraction=extraction(400, "C504", [first_span, second_span]),
                ),
                410: CaseVariableResult(
                    item_id=410,
                    status=VariableStatus.STRUCTURED_DATA,
                    value="2",
                ),
                522: CaseVariableResult(
                    item_id=522,
                    status=VariableStatus.NOT_FOUND,
                    extraction=extraction(522, None, []),
                ),
                820: CaseVariableResult(
                    item_id=820,
                    status=VariableStatus.ERROR,
                    extraction=extraction(
                        820,
                        "1",
                        [first_span],
                        is_valid=False,
                    ),
                ),
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            result = OmopExporter(
                person_id="source-person-7",
                nlp_system="CIPOC:test-model",
                nlp_date="2025-04-12",
            ).export(notes=[self.note], case=case, output_directory=directory)

            note_rows = read_csv(result.note_path)
            nlp_rows = read_csv(result.note_nlp_path)
            error_report = read_json(result.error_path)

        self.assertEqual(result.note_count, 1)
        self.assertEqual(result.note_nlp_count, 2)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(error_report, {"errors": []})

        self.assertEqual(note_rows[0]["note_id"], "note-A")
        self.assertEqual(note_rows[0]["person_id"], "source-person-7")
        self.assertEqual(note_rows[0]["note_date"], "2025-02-24")
        self.assertEqual(note_rows[0]["note_type_concept_id"], "EHR")
        self.assertEqual(
            note_rows[0]["note_class_concept_id"], "Pathology Report"
        )
        self.assertEqual(note_rows[0]["encoding_concept_id"], "UTF-8")

        self.assertEqual(
            [row["note_nlp_id"] for row in nlp_rows],
            ["note-A:400:1", "note-A:400:2"],
        )
        self.assertEqual(nlp_rows[0]["note_nlp_concept_id"], "C504")
        self.assertEqual(nlp_rows[0]["note_nlp_source_concept_id"], "400")
        self.assertEqual(nlp_rows[0]["nlp_date"], "2025-04-12")
        self.assertEqual(
            nlp_rows[0]["offset"],
            str(self.note.content.index(first_span.text)),
        )
        self.assertEqual(
            json.loads(nlp_rows[0]["term_modifiers"]),
            {"item_id": 400, "value": "C504", "confidence": "max"},
        )

    def test_incomplete_note_nlp_rows_are_written_to_error_file(self):
        case = Case(
            variable_results={
                400: CaseVariableResult(
                    item_id=400,
                    status=VariableStatus.EXTRACTED,
                    value="C504",
                    extraction=extraction(400, "C504", []),
                ),
                410: CaseVariableResult(
                    item_id=410,
                    status=VariableStatus.EXTRACTED,
                    value="2",
                    extraction=extraction(
                        410,
                        "2",
                        [TextSpan(note_id="missing-note", text="left breast")],
                    ),
                ),
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            result = OmopExporter(
                person_id=7,
                nlp_date="2025-04-12",
            ).export(notes=[self.note], case=case, output_directory=directory)

            nlp_rows = read_csv(result.note_nlp_path)
            error_rows = read_json(result.error_path)["errors"]

        self.assertEqual(nlp_rows, [])
        self.assertEqual(result.error_count, 2)
        self.assertEqual({row["table_name"] for row in error_rows}, {"note_nlp"})

        empty_span_error = next(
            row for row in error_rows if row["source_id"] == "400:1"
        )
        self.assertEqual(
            {issue["field"] for issue in empty_span_error["issues"]},
            {"note_id", "lexical_variant"},
        )
        partial_row = empty_span_error["row_data"]
        self.assertEqual(partial_row["note_nlp_source_concept_id"], 400)
        self.assertEqual(partial_row["note_nlp_concept_id"], "C504")

        missing_note_error = next(
            row for row in error_rows if row["source_id"] == "410:1"
        )
        self.assertEqual(
            [issue["field"] for issue in missing_note_error["issues"]],
            ["note_id"],
        )
        self.assertEqual(
            missing_note_error["issues"][0]["type"],
            "invalid_reference",
        )

    def test_incomplete_note_rows_are_preserved_as_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            result = OmopExporter(
                person_id=None,
                nlp_date="2025-04-12",
            ).export(notes=[self.note], case=Case(), output_directory=directory)

            note_rows = read_csv(result.note_path)
            error_rows = read_json(result.error_path)["errors"]

        self.assertEqual(note_rows, [])
        self.assertEqual(result.note_count, 0)
        self.assertEqual(result.error_count, 1)
        self.assertEqual(error_rows[0]["table_name"], "note")
        self.assertEqual(
            [issue["field"] for issue in error_rows[0]["issues"]],
            ["person_id"],
        )
        self.assertEqual(
            error_rows[0]["row_data"]["note_id"],
            "note-A",
        )

    def test_ambiguous_evidence_does_not_invent_an_offset(self):
        note = self.note.model_copy(
            update={"content": "left breast and left breast"}
        )
        case = Case(
            variable_results={
                410: CaseVariableResult(
                    item_id=410,
                    status=VariableStatus.EXTRACTED,
                    value="2",
                    extraction=extraction(
                        410,
                        "2",
                        [TextSpan(note_id="note-A", text="left breast")],
                    ),
                )
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            result = OmopExporter(
                person_id=7,
                nlp_date="2025-04-12",
            ).export(notes=[note], case=case, output_directory=directory)
            nlp_rows = read_csv(result.note_nlp_path)

        self.assertEqual(result.note_nlp_count, 1)
        self.assertEqual(nlp_rows[0]["offset"], "")


if __name__ == "__main__":
    unittest.main()


class OmopBuildTests(unittest.TestCase):
    """``build`` is what ``export`` writes, minus the writing.

    The demo renders these rows in the browser, so the two have to be the same
    computation rather than two descriptions of one — a second reconstruction of
    the row shapes is exactly what would drift from the CSVs.
    """

    def setUp(self):
        self.notes = [
            ClinicalNote(
                note_id="note-A",
                date="2025-02-24",
                note_type="Pathology Report",
                content="Final Diagnosis: Invasive ductal carcinoma of the left breast.",
            ),
            ClinicalNote(
                note_id="note-B",
                date="2025-03-02",
                note_type="Progress Note",
                content="Left breast primary confirmed on review.",
            ),
        ]
        span_a = TextSpan(note_id="note-A", text="Invasive ductal carcinoma")
        span_b = TextSpan(note_id="note-B", text="Left breast primary")
        self.case = Case(
            variable_results={
                400: CaseVariableResult(
                    item_id=400,
                    status=VariableStatus.EXTRACTED,
                    value="C504",
                    extraction=extraction(400, "C504", [span_a]),
                ),
                522: CaseVariableResult(
                    item_id=522,
                    status=VariableStatus.EXTRACTED,
                    value="8500",
                    extraction=extraction(522, "8500", [span_b]),
                ),
            }
        )

    def build(self, **kwargs):
        return OmopExporter(person_id=7).build(
            notes=self.notes, case=self.case, **kwargs
        )

    def test_build_matches_what_export_writes(self):
        exporter = OmopExporter(person_id=7)
        tables = exporter.build(notes=self.notes, case=self.case)
        with tempfile.TemporaryDirectory() as directory:
            result = exporter.export(
                notes=self.notes, case=self.case, output_directory=directory
            )
            note_csv = read_csv(result.note_path)
            note_nlp_csv = read_csv(result.note_nlp_path)

        self.assertEqual(len(tables.note_rows), result.note_count)
        self.assertEqual(len(tables.note_nlp_rows), result.note_nlp_count)
        self.assertEqual(len(tables.errors), result.error_count)

        # ``DictWriter`` writes ``None`` as an empty field, so the in-memory rows
        # have to be coerced the same way before they can be compared.
        def as_csv(row):
            return {
                key: "" if value is None else str(value)
                for key, value in row.model_dump().items()
            }

        self.assertEqual([as_csv(row) for row in tables.note_rows], note_csv)
        self.assertEqual([as_csv(row) for row in tables.note_nlp_rows], note_nlp_csv)
        # The column order the demo publishes to the browser is the CSV header.
        self.assertEqual(list(note_nlp_csv[0]), list(NOTE_NLP_FIELDS))

    def test_item_ids_selects_one_variable(self):
        tables = self.build(item_ids=[400])
        self.assertEqual(
            [row.note_nlp_source_concept_id for row in tables.note_nlp_rows], [400]
        )

    def test_item_ids_does_not_narrow_the_note_table(self):
        """NOTE stays whole-corpus even when one item is asked for.

        ``valid_note_ids`` and the duplicate-id check are corpus-wide properties;
        a caller that wants only the notes an item cites narrows the result
        itself, because that is a view rather than a validation.
        """
        self.assertEqual(len(self.build(item_ids=[400]).note_rows), 2)

    def test_item_ids_none_means_every_variable(self):
        self.assertEqual(
            [row.note_nlp_id for row in self.build().note_nlp_rows],
            [
                row.note_nlp_id
                for row in self.build(item_ids=list(self.case.variable_results)).note_nlp_rows
            ],
        )

    def test_an_empty_filter_selects_nothing(self):
        tables = self.build(item_ids=[])
        self.assertEqual(tables.note_nlp_rows, [])
        self.assertEqual(len(tables.note_rows), 2)
