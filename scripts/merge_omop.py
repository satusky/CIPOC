"""Merge per-patient OMOP staging CSVs into one export."""

from __future__ import annotations

import argparse
from pathlib import Path

from cipoc.export import merge_omop_csvs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge per-patient NOTE and NOTE_NLP staging CSVs."
    )
    parser.add_argument(
        "input_directories",
        type=Path,
        nargs="+",
        help="Directories containing per-patient note.csv and note_nlp.csv files.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for the merged note.csv and note_nlp.csv files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = merge_omop_csvs(
        args.input_directories,
        args.output_directory,
    )

    print(f"Sources: {result.source_count}")
    print(f"NOTE rows: {result.note_count} -> {result.note_path}")
    print(f"NOTE_NLP rows: {result.note_nlp_count} -> {result.note_nlp_path}")


if __name__ == "__main__":
    main()
