import csv
import json
import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import export_omop
from cipoc.export import OmopExporter
from cipoc.export import omop as omop_module
from cipoc.export.models import OmopNoteNlpRow, OmopNoteRow
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

    def evidence_case(self):
        return Case(
            variable_results={
                400: CaseVariableResult(
                    item_id=400,
                    status=VariableStatus.EXTRACTED,
                    value="C504",
                    extraction=extraction(
                        400, "C504", [TextSpan(note_id="note-A", text="left breast")]
                    ),
                )
            }
        )

    def test_invalid_note_dates_are_row_errors_and_invalidate_references(self):
        for value in (
            "2025-02-29", "2024-02-30", "2025-13-01", "20250224", "2025-2-24",
            "", "private-date",
        ):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                note = self.note.model_copy(update={"date": value})
                result = OmopExporter(person_id=7).export(
                    notes=[note], case=self.evidence_case(), output_directory=directory
                )
                self.assertEqual(
                    (result.note_count, result.note_nlp_count, result.error_count),
                    (0, 0, 2),
                )
                errors = read_json(result.error_path)["errors"]
                self.assertEqual(errors[0]["row_data"]["note_date"], value)
                self.assertEqual(errors[0]["issues"][0]["field"], "note_date")
                if value:
                    self.assertNotIn(value, errors[0]["issues"][0]["message"])
                self.assertEqual(errors[1]["issues"][0]["type"], "invalid_reference")

    def test_invalid_explicit_nlp_dates_and_timestamps_are_row_errors(self):
        for options, field, invalid in (
            ({"nlp_date": ""}, "nlp_date", ""),
            ({"nlp_date": "2025-02-29"}, "nlp_date", "2025-02-29"),
            ({"nlp_date": "20250412"}, "nlp_date", "20250412"),
            ({"nlp_date": "", "nlp_datetime": "2024-02-29T12:00:00"}, "nlp_date", ""),
            ({"nlp_datetime": "private-timestamp"}, "nlp_datetime", "private-timestamp"),
            ({"nlp_datetime": "2025-02-29T12:00:00"}, "nlp_datetime", "2025-02-29T12:00:00"),
            ({"nlp_datetime": "2024-02-29T25:00:00"}, "nlp_datetime", "2024-02-29T25:00:00"),
            (
                {"nlp_date": "2024-02-29", "nlp_datetime": "2024-02-29T12:60:00"},
                "nlp_datetime", "2024-02-29T12:60:00",
            ),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                exporter = OmopExporter(person_id=7, **options)
                result = exporter.export(
                    notes=[self.note], case=self.evidence_case(), output_directory=directory
                )
                self.assertEqual(
                    (result.note_count, result.note_nlp_count, result.error_count),
                    (1, 0, 1),
                )
                error = read_json(result.error_path)["errors"][0]
                self.assertEqual(error["row_data"][field], invalid)
                self.assertIn(field, {issue["field"] for issue in error["issues"]})
                self.assertEqual(read_csv(result.note_nlp_path), [])
                if "nlp_date" not in options:
                    self.assertEqual(error["row_data"]["nlp_date"], "")

    def test_dates_derive_from_parsed_iso_timestamps_and_omitted_defaults(self):
        for options, expected_date in (
            ({"nlp_datetime": "2024-02-29T23:59:59Z"}, "2024-02-29"),
            ({"nlp_datetime": "2024-02-29 23:59:59.123456+05:30"}, "2024-02-29"),
            ({"nlp_datetime": "20240229T235959"}, "2024-02-29"),
            ({"nlp_datetime": "2024-W09-4T23:59:59"}, "2024-02-29"),
            ({"nlp_datetime": datetime(2024, 2, 29, 23, 59, 59)}, "2024-02-29"),
            ({"nlp_date": date(2024, 2, 29)}, "2024-02-29"),
            ({"nlp_date": "2024-02-29", "nlp_datetime": "2025-01-01T12:00:00"}, "2024-02-29"),
            ({"nlp_datetime": ""}, date.today().isoformat()),
            ({}, date.today().isoformat()),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                note = self.note.model_copy(update={"date": "2024-02-29"})
                result = OmopExporter(person_id=7, **options).export(
                    notes=[note], case=self.evidence_case(), output_directory=directory
                )
                self.assertEqual(result.error_count, 0)
                self.assertEqual(read_csv(result.note_path)[0]["note_date"], "2024-02-29")
                self.assertEqual(read_csv(result.note_nlp_path)[0]["nlp_date"], expected_date)

    def test_duplicate_notes_are_rejected_before_publishing_referenced_rows(self):
        notes = [self.note, self.note.model_copy()]
        with tempfile.TemporaryDirectory() as directory:
            result = OmopExporter(person_id=7).export(
                notes=notes, case=self.evidence_case(), output_directory=directory
            )
            self.assertEqual(
                (result.note_count, result.note_nlp_count, result.error_count), (0, 0, 3)
            )
            errors = read_json(result.error_path)["errors"]
            self.assertEqual(
                [error["issues"][0]["type"] for error in errors],
                ["duplicate", "duplicate", "invalid_reference"],
            )

    def test_staging_failures_preserve_all_three_existing_files(self):
        for failed_file in ("note.csv", "note_nlp.csv", "omop_errors.json"):
            with self.subTest(failed_file=failed_file), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                exporter = OmopExporter(person_id=7)
                exporter.export(notes=[], case=Case(), output_directory=output)
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                write_csv_file, write_text = omop_module._write_csv, Path.write_text
                invalid_note = self.note.model_copy(
                    update={"note_id": "invalid", "date": "2025-02-29"}
                )

                def fail_csv(path, fields, rows):
                    write_csv_file(path, fields, rows)
                    if path.name == failed_file:
                        raise OSError("injected staging failure")

                def fail_text(path, *args, **kwargs):
                    count = write_text(path, *args, **kwargs)
                    if path.name == failed_file:
                        raise OSError("injected staging failure")
                    return count

                with (
                    patch.object(omop_module, "_write_csv", side_effect=fail_csv),
                    patch.object(Path, "write_text", new=fail_text),
                ):
                    with self.assertRaisesRegex(OSError, "injected"):
                        exporter.export(
                            notes=[self.note, invalid_note],
                            case=self.evidence_case(),
                            output_directory=output,
                        )
                self.assertEqual(
                    {path.name: path.read_bytes() for path in output.iterdir()}, before
                )

    def test_error_serialization_failure_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            exporter = OmopExporter(person_id=7)
            exporter.export(notes=[], case=Case(), output_directory=output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with patch.object(
                omop_module.OmopErrorReport, "model_dump_json",
                side_effect=TypeError("injected serialization failure"),
            ):
                with self.assertRaisesRegex(TypeError, "injected"):
                    exporter.export(
                        notes=[self.note], case=self.evidence_case(), output_directory=output
                    )
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.iterdir()}, before
            )

    def test_all_three_files_close_before_replace_and_replacements_are_not_a_transaction(self):
        for fail_at in (1, 2, 3):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                exporter = OmopExporter(person_id=7)
                exporter.export(notes=[], case=Case(), output_directory=output)
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                streams, published = [], []
                open_path, replace = Path.open, os.replace

                def track_open(path, *args, **kwargs):
                    stream = open_path(path, *args, **kwargs)
                    if (args[0] if args else kwargs.get("mode")) == "w":
                        streams.append(stream)
                    return stream

                def fail_replace(src, dst):
                    self.assertEqual(len(streams), 3)
                    self.assertTrue(all(stream.closed for stream in streams))
                    self.assertEqual(Path(src).parent.parent, output)
                    self.assertTrue(Path(src).is_file())
                    if len(published) + 1 == fail_at:
                        raise OSError("injected replacement failure")
                    replace(src, dst)
                    published.append(Path(dst).name)

                with (
                    patch.object(Path, "open", new=track_open),
                    patch("os.replace", side_effect=fail_replace),
                ):
                    with self.assertRaisesRegex(OSError, "injected"):
                        exporter.export(
                            notes=[self.note], case=self.evidence_case(),
                            output_directory=output,
                        )
                self.assertEqual(len(published), fail_at - 1)
                self.assertEqual({path.name for path in output.iterdir()}, set(before))
                for name, data in before.items():
                    if name in published:
                        self.assertNotEqual((output / name).read_bytes(), data)
                    else:
                        self.assertEqual((output / name).read_bytes(), data)


class OmopDateSchemaTests(unittest.TestCase):
    def test_both_row_schemas_validate_dates_and_optional_datetimes(self):
        rows = (
            (OmopNoteRow, "note_date", "note_datetime", {
                "note_id": "1", "person_id": "1", "note_type_concept_id": "EHR",
                "note_class_concept_id": "Test", "note_text": "Synthetic",
                "encoding_concept_id": "UTF-8", "language_concept_id": "English",
            }),
            (OmopNoteNlpRow, "nlp_date", "nlp_datetime", {
                "note_nlp_id": "1", "note_id": "1", "lexical_variant": "Synthetic",
            }),
        )
        for model, date_field, datetime_field, row in rows:
            for invalid in (
                "2025-02-29", "2024-04-31", "20240229", "2024-2-29",
                "2024-02-29T12:00:00", "",
            ):
                with self.subTest(model=model.__name__, invalid=invalid):
                    with self.assertRaises(ValueError):
                        model.model_validate({**row, date_field: invalid})
            for timestamp in (None, "", "2024-02-29T12:00:00Z", "20240229T120000"):
                with self.subTest(model=model.__name__, timestamp=timestamp):
                    result = model.model_validate(
                        {**row, date_field: "2024-02-29", datetime_field: timestamp}
                    )
                    self.assertEqual(getattr(result, datetime_field), timestamp)
                    self.assertIsInstance(getattr(result, date_field), str)
            for invalid in ("2024-02-30T12:00:00", "2024-02-29T24:00:00", "not-a-time", " "):
                with self.subTest(model=model.__name__, invalid=invalid):
                    with self.assertRaises(ValueError):
                        model.model_validate(
                            {**row, date_field: "2024-02-29", datetime_field: invalid}
                        )


class OmopExportScriptTests(unittest.TestCase):
    def test_main_passes_run_result_case_to_exporter(self):
        case = Case()
        run_result = SimpleNamespace(case=case)
        exported = SimpleNamespace(
            note_count=1,
            note_path=Path("note.csv"),
            note_nlp_count=0,
            note_nlp_path=Path("note_nlp.csv"),
            error_count=0,
            error_path=Path("omop_errors.json"),
        )
        args = SimpleNamespace(
            notes=Path("notes.json"),
            person_id="person-1",
            output_directory=Path("output"),
            nlp_date=None,
            nlp_system="CIPOC",
            structured_data=None,
            max_concurrency=None,
            no_progress=True,
        )
        note = ClinicalNote(
            note_id=1,
            date="2025-02-24",
            note_type="Pathology Report",
            content="No reportable finding.",
        )
        parser = SimpleNamespace(parse_args=lambda: args)
        agent = SimpleNamespace(run=lambda *args, **kwargs: run_result)
        exporter = SimpleNamespace(export=lambda **kwargs: exported)

        with (
            patch.object(export_omop, "build_parser", return_value=parser),
            patch.object(export_omop, "_load_notes", return_value=[note]),
            patch.object(export_omop, "_load_structured_data", return_value=None),
            patch.object(export_omop, "OrchestratorAgent", return_value=agent),
            patch.object(export_omop, "OmopExporter", return_value=exporter),
            patch.object(exporter, "export", wraps=exporter.export) as export,
        ):
            export_omop.main()

        self.assertIs(export.call_args.kwargs["case"], case)


if __name__ == "__main__":
    unittest.main()
