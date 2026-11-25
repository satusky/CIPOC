import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from transforms.api import LightweightInput, LightweightOutput, transform, Input, Output

from cipoc.export import OmopErrorReport, OmopExporter, merge_omop_csvs
from cipoc.models import OrchestratorRunResult

EXTRACT_DATASET_RID = "<cipoc_extract output rid>"  # one row per patient
NOTE_DATASET_RID = "<note output rid>"
NOTE_NLP_DATASET_RID = "<note_nlp output rid>"
ERRORS_DATASET_RID = "<errors output rid>"
NLP_SYSTEM = "CIPOC"

ERROR_COLUMNS = {"table_name": "string", "source_id": "string", "issues": "string", "row_data": "string"}


def note_date(value):
    """A date, timestamp, or date string as 'YYYY-MM-DD'. Unparseable values come back unchanged, so they reach the errors table."""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return value


def export_patient(mspi, result_json: str, directory: Path) -> list[tuple]:
    """Write one completed run's note.csv and note_nlp.csv to directory; return its error rows."""
    result = OrchestratorRunResult.model_validate_json(result_json)
    notes = [note.model_copy(update={"date": note_date(note.date)}) for note in result.corpus.note_corpus.values()]
    exported = OmopExporter(
        # str(): a numpy integer isn't an int, so the row models would reject it.
        person_id=None if pd.isna(mspi) else str(mspi),
        nlp_system=NLP_SYSTEM,
        # The run's own date, so rebuilding this transform doesn't change it.
        nlp_date=result.run.finished_at.date(),
    ).export(
        notes=notes,
        case=result.case,
        output_directory=directory,
    )
    report = OmopErrorReport.model_validate_json(exported.error_path.read_text(encoding="utf-8"))
    return [
        (
            mspi,
            error.table_name,
            error.source_id,
            json.dumps([issue.model_dump() for issue in error.issues]),
            json.dumps(error.row_data),
        )
        for error in report.errors
    ]


def read_staging_csv(path: Path) -> pd.DataFrame:
    """A merged staging CSV as string columns. Only empty cells become null, so note text like 'NA' stays text."""
    return pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""])


@transform.using(
    extract=Input(EXTRACT_DATASET_RID),
    note=Output(NOTE_DATASET_RID),
    note_nlp=Output(NOTE_NLP_DATASET_RID),
    errors=Output(ERRORS_DATASET_RID),
)
def export_omop(
    extract: LightweightInput,
    note: LightweightOutput,
    note_nlp: LightweightOutput,
    errors: LightweightOutput,
):
    extract_df = extract.dataframe()
    # failed and invalid_input rows have no case to export; the extract output already records them.
    completed = extract_df[extract_df["status"] == "completed"]

    error_rows = []
    with TemporaryDirectory() as directory:
        # Per-patient folders, then the same merge as scripts/merge_omop.py, which
        # namespaces note_id and note_nlp_id by person_id.
        patient_directories = []
        for index, (mspi, result_json) in enumerate(zip(completed["mspi"], completed["result_json"])):
            patient_directory = Path(directory) / "patients" / str(index)
            error_rows += export_patient(mspi, result_json, patient_directory)
            patient_directories.append(patient_directory)
        merged = merge_omop_csvs(patient_directories, Path(directory) / "merged")
        note.write_table(read_staging_csv(merged.note_path))
        note_nlp.write_table(read_staging_csv(merged.note_nlp_path))

    errors_df = pd.DataFrame(error_rows, columns=["mspi", *ERROR_COLUMNS])
    # mspi keeps the input's type, which an empty frame couldn't infer. pandas holds strings as object.
    mspi_type = "string" if extract_df["mspi"].dtype == object else extract_df["mspi"].dtype
    errors.write_table(errors_df.astype({"mspi": mspi_type, **ERROR_COLUMNS}))
