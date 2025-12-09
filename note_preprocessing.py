import re
from datetime import datetime, timedelta
import pandas as pd


def split_notes_at_separator(
    notes: str,
    separator: str = r"<NOTE:([\s\S]+?)(\d{4}-\d{2}-\d{2})>",
    capture_count: int = 2,
    regex_flags=0
) -> list[tuple]:
    parts = re.split(re.compile(separator, flags=regex_flags), notes)
    split_list = []
    for i in range(1, len(parts) -1, capture_count + 1):
        split_list.append(tuple(parts[i+j].strip() for j in range(capture_count + 1)))

    return split_list

def split_notes_to_dict(split_notes: tuple | list[tuple], ordered_keys: list[str] = ["note_type", "date", "note"]) -> dict | list[dict]:
    def tuple_to_dict(tup, keys):
        assert len(tup) == len(keys), f"Note tuple length ({len(tup)}) does not match number of keys ({len(keys)})"
        return dict(zip(keys, tup))
    
    if isinstance(split_notes, tuple):
        split_dicts = tuple_to_dict(split_notes, ordered_keys)
    else:
        split_dicts = [tuple_to_dict(note, ordered_keys) for note in split_notes]

    return split_dicts

def select_notes_within_date_of_diagnosis(row: pd.Series, note_column: str, days_before: int, days_after: int, dod_column: str = "DATE_OF_DIAGNOSIS_N390") -> str:
    date_of_diagnosis = str(row[dod_column])
    date_of_diagnosis = datetime.strptime(date_of_diagnosis, "%Y%m%d")
    date_of_diagnosis = date_of_diagnosis.date()

    split_list = split_notes_at_separator(row[note_column]) # type: ignore
    split_dicts = split_notes_to_dict(split_list)

    notes_list = []
    for note_group in split_dicts:
        note_date = datetime.strptime(note_group["date"], "%Y%m%d")
        note_date = note_date.date()

        start_date = date_of_diagnosis - timedelta(days=days_before)
        end_date = date_of_diagnosis + timedelta(days=days_after)

        if start_date <= note_date <= end_date:
            start_string = "<NOTE: " + note_group["note_type"] + " " + note_group["date"] + ">"
            notes_list.append(start_string + note_group["note"])

    kept_notes = " ".join(notes_list)
    return kept_notes

def select_notes_by_note_type(row: pd.Series, note_column: str, note_types_to_keep: list[str]) -> str:
    split_list = split_notes_at_separator(row[note_column]) # type: ignore
    split_dicts = split_notes_to_dict(split_list)

    notes_list = []
    for note_group in split_dicts:
        note_type = note_group["note_type"]
        if note_type in note_types_to_keep:
            start_string = "<NOTE: " + note_group["note_type"] + " " + note_group["date"] + ">"
            notes_list.append(start_string + note_group["note"])

    kept_notes = " ".join(notes_list)
    return kept_notes

# def filter_notes(notes_df: pd.DataFrame, note_types: list[str] | pd.Series, days_before: int, days_after: int, note_column: str = "ALL_FILES") -> list[str]:
#     notes_df["kept_notes"] = notes_df.apply(select_notes_within_date_of_diagnosis, axis=1, args=(note_column, days_before, days_after))
#     notes_df = notes_df.drop(columns=[note_column])
#     notes_df["kept_notes"] = notes_df.apply(select_notes_by_note_type, axis=1, args=(note_column, note_types))
#     return notes_df["kept_notes"].to_list()

def filter_notes(
        row: pd.Series,
        note_column: str,
        days_before_after: tuple[int, int] | None = None,
        date_of_diagnosis_column: str = "DATE_OF_DIAGNOSIS_N390",
        note_types: list[str] | None = None
    ) -> str:
    if days_before_after is None and note_types is None:
        raise Exception("No filter criteria given. Please provide date constraints or note types")

    split_list = split_notes_at_separator(row[note_column]) # type: ignore
    split_dicts = split_notes_to_dict(split_list)

    if days_before_after is not None:
        date_of_diagnosis = datetime.strptime(str(row[date_of_diagnosis_column]), "%Y%m%d").date()
        start_date = date_of_diagnosis - timedelta(days=days_before_after[0])
        end_date = date_of_diagnosis + timedelta(days=days_before_after[1])

    notes_list = []
    for note_group in split_dicts:
        if days_before_after and not start_date <= datetime.strptime(note_group["date"], "%Y%m%d").date() <= end_date:
            continue

        if note_types and note_group["note_type"] not in note_types:
            continue

        start_string = "<NOTE: " + note_group["note_type"] + " " + note_group["date"] + ">"
        notes_list.append(start_string + note_group["note"])

    kept_notes = " ".join(notes_list)
    return kept_notes
