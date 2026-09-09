import csv
import io
import os
import tempfile
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch

from cipoc.export import OmopExporter, merge_omop_csvs
from cipoc.export import merge as merge_module
from cipoc.export.models import (
    NOTE_FIELDS,
    NOTE_NLP_FIELDS,
    OmopNoteNlpRow,
    OmopNoteRow,
)
from cipoc.models import (
    Case,
    CaseVariableResult,
    ClinicalNote,
    ConfidenceLevel,
    TextSpan,
    ValidatedVariableOutput,
    VariableStatus,
)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def note_row(person_id: str) -> dict:
    return OmopNoteRow(
        note_id="note/1",
        person_id=person_id,
        note_date="2025-02-24",
        note_type_concept_id="EHR",
        note_class_concept_id="Pathology",
        note_text="Invasive carcinoma.",
        encoding_concept_id="UTF-8",
        language_concept_id="English",
    ).model_dump()


def note_nlp_row(*, note_id: str = "note/1") -> dict:
    return OmopNoteNlpRow(
        note_nlp_id="finding/1",
        note_id=note_id,
        lexical_variant="Invasive carcinoma",
        note_nlp_concept_id="C504",
        note_nlp_source_concept_id=400,
        nlp_date="2025-04-12",
    ).model_dump()


def write_export(directory: Path, person_id: str) -> None:
    write_csv(directory / "note.csv", NOTE_FIELDS, [note_row(person_id)])
    write_csv(
        directory / "note_nlp.csv",
        NOTE_NLP_FIELDS,
        [note_nlp_row()],
    )


class OmopMergeTests(unittest.TestCase):
    def test_merges_exports_and_namespaces_ids_by_person(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            write_export(first, "patient:1")
            write_export(second, "patient/2")

            result = merge_omop_csvs([first, second], root / "merged")
            notes = read_csv(result.note_path)
            note_nlp = read_csv(result.note_nlp_path)

        self.assertEqual(result.source_count, 2)
        self.assertEqual(result.note_count, 2)
        self.assertEqual(result.note_nlp_count, 2)
        self.assertEqual(
            [row["note_id"] for row in notes],
            ["patient%3A1:note%2F1", "patient%2F2:note%2F1"],
        )
        self.assertEqual(
            [row["note_nlp_id"] for row in note_nlp],
            ["patient%3A1:finding%2F1", "patient%2F2:finding%2F1"],
        )
        self.assertEqual(
            [row["note_id"] for row in note_nlp],
            [row["note_id"] for row in notes],
        )
        self.assertEqual(
            [row["person_id"] for row in notes],
            ["patient:1", "patient/2"],
        )

    def test_rejects_note_nlp_with_a_dangling_note_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            write_csv(source / "note.csv", NOTE_FIELDS, [note_row("patient-1")])
            write_csv(
                source / "note_nlp.csv",
                NOTE_NLP_FIELDS,
                [note_nlp_row(note_id="missing-note")],
            )

            with self.assertRaisesRegex(ValueError, "does not exist"):
                merge_omop_csvs([source], root / "merged")

            self.assertFalse((root / "merged").exists())

    def test_rejects_duplicate_namespaced_note_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            write_export(first, "patient-1")
            write_export(second, "patient-1")

            with self.assertRaisesRegex(ValueError, "duplicate note_id"):
                merge_omop_csvs([first, second], root / "merged")

    def test_rejects_incompatible_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            write_csv(source / "note.csv", ("note_id",), [{"note_id": "1"}])

            with self.assertRaisesRegex(ValueError, "incompatible schema"):
                merge_omop_csvs([source], root / "merged")

    def test_invalid_sources_preserve_destinations_and_redact_errors(self):
        marker = "SYNTHETIC_PRIVATE_VALUE"
        for table, fields, row in (
            ("note", NOTE_FIELDS, note_row("patient-1")),
            ("note_nlp", NOTE_NLP_FIELDS, note_nlp_row()),
        ):
            buffer = io.StringIO(newline="")
            csv.DictWriter(buffer, fieldnames=fields).writerow(row)
            valid_row = buffer.getvalue()
            invalid_rows = {
                "short": valid_row.rsplit(",", 1)[0] + "\n",
                "wide": valid_row.rstrip("\r\n") + f",{marker}\n",
                "unterminated": f'"{marker}\ncontinued',
                "bad_quote": f'"{marker}"junk,rest\n',
                "blank_record": "\n",
            }
            for kind, invalid_row in invalid_rows.items():
                with (
                    self.subTest(table=table, kind=kind),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    source = root / marker
                    first = root / "first"
                    write_export(first, "patient-2")
                    write_export(source, "patient-1")
                    (source / f"{table}.csv").write_text(
                        ",".join(fields) + "\n" + valid_row + invalid_row,
                        encoding="utf-8",
                    )
                    output = root / "merged"
                    write_export(output, "old-person")
                    before = {path.name: path.read_bytes() for path in output.iterdir()}
                    try:
                        merge_omop_csvs([first, source], output)
                    except ValueError as error:
                        self.assertNotIn(marker, traceback.format_exc())
                        self.assertIn("CSV", str(error))
                        self.assertNotIn("duplicate", str(error))
                    else:
                        self.fail("Malformed CSV was accepted")
                    self.assertEqual(
                        {path.name: path.read_bytes() for path in output.iterdir()}, before
                    )

    def test_validates_every_row_with_the_omop_schema(self):
        marker = "SYNTHETIC_PRIVATE_VALUE"
        for table, fields, base, invalid_fields in (
            ("note", NOTE_FIELDS, note_row("patient-1"), {
                "note_text": " ", "note_type_concept_id": "",
                "note_date": "2025-02-29", "note_datetime": "2025-01-01T25:00:00",
            }),
            ("note_nlp", NOTE_NLP_FIELDS, note_nlp_row(), {
                "lexical_variant": "", "nlp_date": "20250412",
                "nlp_datetime": marker,
            }),
        ):
            for field, value in invalid_fields.items():
                with (
                    self.subTest(table=table, field=field),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    source = root / "source"
                    write_export(source, "patient-1")
                    write_csv(source / f"{table}.csv", fields, [{**base, field: value}])
                    output = root / "merged"
                    write_export(output, "old-person")
                    before = {path.name: path.read_bytes() for path in output.iterdir()}
                    previous_limit = csv.field_size_limit(777)
                    try:
                        try:
                            merge_omop_csvs([source], output)
                        except ValueError as error:
                            self.assertIn(field, str(error))
                            self.assertNotIn(marker, traceback.format_exc())
                        else:
                            self.fail("Invalid OMOP row was accepted")
                        self.assertEqual(csv.field_size_limit(), 777)
                        self.assertEqual(
                            {path.name: path.read_bytes() for path in output.iterdir()},
                            before,
                        )
                    finally:
                        csv.field_size_limit(previous_limit)

    def test_header_and_encoding_errors_do_not_expose_input(self):
        marker = "SYNTHETIC_PRIVATE_VALUE"
        for content in (
            b"",
            (marker + "," + ",".join(NOTE_FIELDS[1:]) + "\n").encode(),
            (",".join((NOTE_FIELDS[1], *NOTE_FIELDS[1:])) + "\n").encode(),
            (",".join(NOTE_FIELDS) + "\n" + marker).encode() + b"\xff",
        ):
            with self.subTest(content=content[:10]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / marker
                write_export(source, "patient-1")
                (source / "note.csv").write_bytes(content)
                try:
                    merge_omop_csvs([source], root / "merged")
                except ValueError:
                    self.assertNotIn(marker, traceback.format_exc())
                else:
                    self.fail("Invalid header or encoding was accepted")
                self.assertFalse((root / "merged").exists())

    def test_reordered_headers_multiline_and_optional_empty_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            note = {**note_row("patient-1"), "note_text": 'First, "quoted" line\r\nSecond line'}
            write_csv(source / "note.csv", tuple(reversed(NOTE_FIELDS)), [note])
            write_csv(source / "note_nlp.csv", tuple(reversed(NOTE_NLP_FIELDS)), [note_nlp_row()])
            result = merge_omop_csvs([source], root / "merged")
            self.assertEqual(read_csv(result.note_path)[0]["note_text"], note["note_text"])
            self.assertEqual(read_csv(result.note_path)[0]["note_datetime"], "")
            self.assertEqual(read_csv(result.note_nlp_path)[0]["nlp_datetime"], "")

    def test_header_only_sources_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            write_csv(source / "note.csv", NOTE_FIELDS, [])
            write_csv(source / "note_nlp.csv", NOTE_NLP_FIELDS, [])
            result = merge_omop_csvs([source], root / "merged")
            self.assertEqual((result.note_count, result.note_nlp_count), (0, 0))
            self.assertEqual(read_csv(result.note_path), [])
            self.assertEqual(read_csv(result.note_nlp_path), [])

    def test_duplicate_ids_and_references_preserve_existing_outputs(self):
        for kind in ("local_note", "local_nlp", "cross_source_nlp", "reference"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                write_export(source, "patient-1")
                sources = [source]
                if kind == "local_note":
                    write_csv(source / "note.csv", NOTE_FIELDS, [note_row("patient-1")] * 2)
                elif kind == "local_nlp":
                    other_note = {**note_row("patient-2"), "note_id": "other"}
                    write_csv(
                        source / "note.csv", NOTE_FIELDS,
                        [note_row("patient-1"), other_note],
                    )
                    write_csv(
                        source / "note_nlp.csv", NOTE_NLP_FIELDS,
                        [note_nlp_row(), note_nlp_row(note_id="other")],
                    )
                elif kind == "cross_source_nlp":
                    other = root / "other"
                    write_csv(
                        other / "note.csv", NOTE_FIELDS,
                        [{**note_row("patient-1"), "note_id": "other"}],
                    )
                    write_csv(
                        other / "note_nlp.csv", NOTE_NLP_FIELDS,
                        [note_nlp_row(note_id="other")],
                    )
                    sources.append(other)
                else:
                    write_csv(
                        source / "note_nlp.csv", NOTE_NLP_FIELDS,
                        [note_nlp_row(note_id="SYNTHETIC_PRIVATE_VALUE")],
                    )
                output = root / "merged"
                write_export(output, "old-person")
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                with self.assertRaises(ValueError) as caught:
                    merge_omop_csvs(sources, output)
                self.assertNotIn("SYNTHETIC_PRIVATE_VALUE", str(caught.exception))
                self.assertEqual(
                    {path.name: path.read_bytes() for path in output.iterdir()}, before
                )

    def test_staging_write_failures_preserve_existing_outputs(self):
        for failed_file in ("note.csv", "note_nlp.csv"):
            with self.subTest(failed_file=failed_file), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / "source", root / "merged"
                write_export(source, "new-person")
                write_export(output, "old-person")
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                write = merge_module._write_csv

                def fail_write(path, fields, rows):
                    write(path, fields, rows)
                    if path.name == failed_file:
                        raise OSError("injected staging failure")

                with patch.object(merge_module, "_write_csv", side_effect=fail_write):
                    with self.assertRaisesRegex(OSError, "injected"):
                        merge_omop_csvs([source], output)
                self.assertEqual(
                    {path.name: path.read_bytes() for path in output.iterdir()}, before
                )

    def test_all_staged_files_close_before_replace_and_partial_publication_is_possible(self):
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, output = root / "source", root / "merged"
                write_export(source, "new-person")
                write_export(output, "old-person")
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                streams, published = [], []
                open_path, replace = Path.open, os.replace

                def track_open(path, *args, **kwargs):
                    stream = open_path(path, *args, **kwargs)
                    if (args[0] if args else kwargs.get("mode")) == "w":
                        streams.append(stream)
                    return stream

                def fail_replace(src, dst):
                    self.assertEqual(len(streams), 2)
                    self.assertTrue(all(stream.closed for stream in streams))
                    self.assertEqual(Path(src).parent.parent, output)
                    self.assertNotEqual(Path(src).parent, output)
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
                        merge_omop_csvs([source], output)
                self.assertEqual(len(published), fail_at - 1)
                self.assertEqual({path.name for path in output.iterdir()}, set(before))
                for name, data in before.items():
                    if name in published:
                        self.assertNotEqual((output / name).read_bytes(), data)
                    else:
                        self.assertEqual((output / name).read_bytes(), data)

    def test_large_unicode_multiline_export_merge_roundtrip_restores_limit(self):
        content = ('Synthetic \u03bb, "quoted" line\r\n' * 7000) + "end"
        self.assertGreater(len(content), 131072)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            note = ClinicalNote(
                note_id="large", date="2024-02-29", note_type="Test", content=content
            )
            extracted = ValidatedVariableOutput(
                item_id=400,
                value="C504",
                explanation="Synthetic long evidence",
                most_important_note=None,
                spans=[TextSpan(note_id="large", text=content)],
                presence_confidence=ConfidenceLevel.MAX,
                is_valid=True,
                validation_errors=[],
                extraction_attempts=1,
            )
            case = Case(variable_results={
                400: CaseVariableResult(
                    item_id=400, status=VariableStatus.EXTRACTED,
                    value="C504", extraction=extracted,
                )
            })
            exported = OmopExporter(person_id="person").export(
                notes=[note], case=case, output_directory=root / "source"
            )
            self.assertEqual(exported.error_count, 0)
            previous_limit = csv.field_size_limit(131072)
            try:
                result = merge_omop_csvs([root / "source"], root / "merged")
                self.assertEqual(csv.field_size_limit(), 131072)
                csv.field_size_limit(len(content) + 1)
                self.assertEqual(read_csv(exported.note_path)[0]["note_text"], content)
                self.assertEqual(read_csv(exported.note_nlp_path)[0]["lexical_variant"], content)
                self.assertEqual(read_csv(result.note_path)[0]["note_text"], content)
                self.assertEqual(read_csv(result.note_nlp_path)[0]["lexical_variant"], content)
            finally:
                csv.field_size_limit(previous_limit)

    def test_platform_max_field_limit_handles_narrow_c_integers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            write_export(source, "patient-1")
            field_size_limit, reader = csv.field_size_limit, csv.reader
            previous_limit = field_size_limit()
            platform_max = (1 << 31) - 1

            def narrow_limit(value=None):
                if value is None:
                    return field_size_limit()
                if value > platform_max:
                    raise OverflowError("simulated narrow C integer")
                return field_size_limit(value)

            def check_reader(*args, **kwargs):
                self.assertEqual(field_size_limit(), platform_max)
                return reader(*args, **kwargs)

            try:
                with (
                    patch.object(csv, "field_size_limit", side_effect=narrow_limit),
                    patch.object(csv, "reader", side_effect=check_reader) as reads,
                ):
                    merge_omop_csvs([source], root / "merged")
                self.assertEqual(reads.call_count, 2)
                self.assertEqual(field_size_limit(), previous_limit)
            finally:
                field_size_limit(previous_limit)

    def test_cooperating_reads_serialize_limit_changes_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            write_export(source, "patient-1")
            first_read, release_first, second_acquire = Event(), Event(), Event()
            acquisition_lock, reader_lock = Lock(), Lock()
            acquisitions, reads = [], []
            reader = csv.reader

            class ObservedLock:
                def __enter__(self):
                    with acquisition_lock:
                        acquisitions.append(True)
                        if len(acquisitions) == 2:
                            second_acquire.set()
                    reader_lock.acquire()

                def __exit__(self, *args):
                    reader_lock.release()

            def blocking_reader(*args, **kwargs):
                reads.append(csv.field_size_limit())
                if len(reads) == 1:
                    first_read.set()
                    if not release_first.wait(5):
                        raise AssertionError("reader was not released")
                return reader(*args, **kwargs)

            previous_limit = csv.field_size_limit(777)
            try:
                with (
                    patch.object(merge_module, "_CSV_READER_LOCK", ObservedLock()),
                    patch.object(csv, "reader", side_effect=blocking_reader),
                    ThreadPoolExecutor(max_workers=2) as pool,
                ):
                    first = pool.submit(merge_omop_csvs, [source], root / "first")
                    try:
                        self.assertTrue(first_read.wait(5))
                        second = pool.submit(merge_omop_csvs, [source], root / "second")
                        self.assertTrue(second_acquire.wait(5))
                        self.assertEqual(len(reads), 1)
                    finally:
                        release_first.set()
                    self.assertEqual(first.result(timeout=5).note_count, 1)
                    self.assertEqual(second.result(timeout=5).note_count, 1)
                self.assertEqual(len(reads), 4)
                self.assertTrue(all(limit > 131072 for limit in reads))
                self.assertEqual(csv.field_size_limit(), 777)
            finally:
                csv.field_size_limit(previous_limit)

    def test_parser_failure_restores_custom_field_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            write_export(source, "patient-1")
            (source / "note.csv").write_text(
                ",".join(NOTE_FIELDS) + '\n"' + "x" * 140000, encoding="utf-8"
            )
            previous_limit = csv.field_size_limit(777)
            try:
                with self.assertRaisesRegex(ValueError, "CSV"):
                    merge_omop_csvs([source], root / "merged")
                self.assertEqual(csv.field_size_limit(), 777)
                self.assertFalse((root / "merged").exists())
            finally:
                csv.field_size_limit(previous_limit)


if __name__ == "__main__":
    unittest.main()
