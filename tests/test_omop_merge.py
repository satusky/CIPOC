import csv
import tempfile
import unittest
from pathlib import Path

from cipoc.export import merge_omop_csvs
from cipoc.export.models import (
    NOTE_FIELDS,
    NOTE_NLP_FIELDS,
    OmopNoteNlpRow,
    OmopNoteRow,
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


if __name__ == "__main__":
    unittest.main()
