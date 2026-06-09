import os
import json
import textwrap
from tqdm import tqdm

import pandas as pd
import pyspark.sql.dataframe as spark
import mlflow
mlflow.openai.autolog(disable=True)

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from pydantic import BaseModel, Field
from typing import Any
from dataclasses import dataclass

from llm_client import ChatClient
from note_filter import NoteFilter
from utils import get_model_run_state, append_temp_file_to_output


class NAACCRVariable(BaseModel):
    """ Structured output for an extracted NAACCR variable """
    item_id: int = Field(description="NAACCR item ID number")
    item_name: str = Field(description="NAACCR variable name")
    explanation: str = Field(description="Reasoning for assigning selected value")
    value: str = Field(description="Valid coding value assigned based on the context in the appropriate format")


@dataclass
class ExtractionConfig:
    model: str
    target_vars: list | pd.Series
    target_df: pd.DataFrame
    note_column: str
    id_column: str
    date_of_diagnosis_column: str
    date_of_diagnosis_format: str
    note_date_format: str
    note_types: list[str]
    note_days_before: int
    note_days_after: int
    llm_client: ChatClient
    output_file: str
    error_file: str
    output_dir: str
    start_index: int
    end_index: int
    filter_batch_size: int | None


def get_targets_from_file(target_file: str, target_codes_file: str) -> pd.DataFrame:
    with open(target_codes_file, "r") as f:
        target_codes = json.load(f)

    targets = pd.read_csv(target_file)
    targets["Codes"] = [json.dumps(target_codes[str(item_id)]) for item_id in targets["Item_Number"]]

    return targets


def get_variable_info_from_id(variable_id: str | int, df: pd.DataFrame) -> dict:
    col_map = {
        "Item_Number": "variable_id",
        "Item_Name": "variable_name",
        "Item_Description": "variable_description",
        "Codes": "variable_codes",
        "Instructions": "variable_instructions"
    }

    # Remove invalid keys
    col_map = {key: val for key, val in col_map.items() if key in df.columns}
    return df.loc[df["Item_Number"] == int(variable_id), list(col_map.keys())].rename(columns=col_map).to_dict("records")[0]  # type: ignore


def build_prompt(
    variable_id: str | int,
    variable_name: str,
    variable_description: str,
    variable_codes: Any,
    variable_instructions: str | None = None,
) -> str:
    # Doing it this way to preserve indentation
    instruction_string = f"""
    The coding instructions for the variable are:
    {variable_instructions}
    """

    user_prompt = f"""
    Your task is to read a set of clinical notes and extract a NAACCR variable for entry into a cancer registry.
    Here is the name, ID number, and a description of the variable:

    Variable name: {variable_name}
    NAACCR ID: {variable_id}
    Description:
    {variable_description}
    {instruction_string if variable_instructions is not None else ""}
    The possible values for the variable are shown in the following dictionary, where the keys are the output codes and the values are a description of that code:
    {variable_codes}

    Provide the most appropriate output code for th variable based on the following notes:
    """
    return textwrap.dedent(user_prompt)


def extract_naaccr_variable(variable_info: dict, notes: str, llm_client: ChatClient, **kwargs) -> NAACCRVariable:
    user_prompt = build_prompt(**variable_info)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant to a cancer registrar"
        },
        {
            "role": "user",
            "content": user_prompt + notes
        }
    ]

    completion = llm_client.chat(
        messages=messages,
        **kwargs
    )

    tool_call = completion.choices[0].message.tool_calls[0]
    tool_input = tool_call.function.arguments
    parsed = NAACCRVariable.model_validate_json(tool_input)

    return parsed


def submit_naaccr_chat_request(variable_info: dict, notes: str, llm_client: ChatClient, **kwargs) -> ChatCompletion:
    user_prompt = build_prompt(**variable_info)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant to a cancer registrar"
        },
        {
            "role": "user",
            "content": user_prompt + notes
        }
    ]

    completion = llm_client.chat(messages=messages, **kwargs)
    return completion


def validate_tool_call(tool_call: ChatCompletionMessageToolCall, data_model: type[BaseModel] = NAACCRVariable):
    tool_input = tool_call.function.arguments
    parsed = data_model.model_validate_json(tool_input)
    return parsed


def run_extraction(
    notes_df: pd.DataFrame,
    config: ExtractionConfig,
    current_index: int = 0,
    max_tool_retries: int = 1,
    pbar = None
):
    if pbar is None:
        pbar = tqdm(total=config.end_index - config.start_index, position=0, desc="Patients")

    os.makedirs(config.output_dir, exist_ok=True)

    temp_output_file = f"TEMP_{config.output_file}"
    temp_error_file = f"TEMP_{config.error_file}"

    temp_output_path = os.path.join(config.output_dir, temp_output_file)
    output_path = os.path.join(config.output_dir, config.output_file)
    temp_error_path = os.path.join(config.output_dir, temp_error_file)
    error_path = os.path.join(config.output_dir, config.error_file)

    with open(output_path.replace(".json", f"_{current_index}.json"), "w") as f, open(error_path.replace(".json", f"_{current_index}.json"), "w") as error_log:
        for i, (patient_id, notes) in enumerate(zip(notes_df[config.id_column], notes_df["KEPT_NOTES"]), start=current_index):
            if i >= config.end_index:
                break

            if i < config.start_index:
                continue

            for target_id in tqdm(config.target_vars, position=1, desc="Variables"):
                target_info = get_variable_info_from_id(target_id, config.target_df)

                for attempt in range(max_tool_retries):
                    try:
                        output_code = extract_naaccr_variable(target_info, notes, config.llm_client)
                        break
                    except TypeError:
                        if attempt < max_tool_retries - 1:
                            continue
                        output_code = NAACCRVariable(item_id=target_info["variable_id"], item_name=target_info["variable_name"], explanation="Error", value="tool_call_error")
                        error = {"patient_id": patient_id, "item_id": target_info["variable_id"], "error": "Tool not called"}
                        error_log.write(json.dumps(error) + "\n")
                        break
                    except Exception as e:
                        output_code = NAACCRVariable(item_id=target_info["variable_id"], item_name=target_info["variable_name"], explanation="Error", value="api_error")
                        error = {"patient_id": patient_id, "item_id": target_info["variable_id"], "error": getattr(e, "message", str(e))}
                        error_log.write(json.dumps(error) + "\n")
                        break

                write_dict = output_code.model_dump()
                write_dict.update({"patient_id": patient_id})
                f.write(json.dumps(write_dict) + "\n")

            pbar.update(1)


def extract_batches(
    notes_df: pd.DataFrame,
    config: ExtractionConfig,
    extract_batch_size: int,
    max_tool_retries: int = 1,
):
    pbar = tqdm(total=config.end_index - config.start_index, position=0, desc="Patients")
    current_index = 0
    while current_index < config.end_index:
        # Skip if whole batch is before start
        if current_index + extract_batch_size < config.start_index:
            current_index += extract_batch_size
            continue

        extract_batch_end = min(current_index + extract_batch_size, len(notes_df))
        run_extraction(notes_df=notes_df.iloc[current_index: extract_batch_end], config=config, current_index=current_index, pbar=pbar, max_tool_retries=max_tool_retries)

        current_index += extract_batch_size


def run_extraction_batch_filter(
    notes_db: spark.DataFrame,
    config: ExtractionConfig,
    patient_ids: list | None = None,
    extract_batch_size: int | None = None,
    resume_run: bool = False,
    max_tool_retries: int = 1,
):
    if config.filter_batch_size is None:
        raise ValueError("Batch size must be set in ExtractionConfig to run in batch mode.")

    note_filter = NoteFilter(
        note_types_to_keep=config.note_types,
        days_before=config.note_days_before,
        days_after=config.note_days_after,
        reference_date_format=config.date_of_diagnosis_format,
        note_date_format=config.note_date_format,
    )

    if extract_batch_size is None:
        extract_batch_size = config.end_index - config.start_index

    if patient_ids is None:
        patient_ids = get_patient_ids(db=notes_db, config=config, resume_run=resume_run, output_directory=config.output_dir)
    else:
        config.start_index = 0
        config.end_index = len(patient_ids)

    notes_df = batch_apply_filter(note_filter=note_filter, patient_ids=patient_ids, notes_db=notes_db, config=config)
    if resume_run:
        config.start_index = 0
        config.end_index = len(patient_ids)

    extract_batches(notes_df=notes_df, config=config, extract_batch_size=extract_batch_size, max_tool_retries=max_tool_retries)


def get_patient_ids(db: spark.DataFrame, config: ExtractionConfig | None = None, resume_run: bool = False, output_directory: str = ".") -> list:
    patient_ids = db.rdd.map(lambda x: x.PERSON_ID).collect()
    if resume_run:
        if config is None:
            raise TypeError("ExtractionConfig must be provided to resume a run")

        run_state = get_model_run_state(model_name=config.model, output_directory=output_directory)
        if run_state is not None:
            completed = [patient_id for patient_id, result_count in run_state.items() if result_count >= len(config.target_vars)]
            patient_ids = list(set(patient_ids) - set(completed))

    return patient_ids


def batch_apply_filter(note_filter, patient_ids, notes_db, config):
    notes_df = pd.DataFrame(columns=["PERSON_ID", "MRN", "KEPT_NOTES"])
    for patient_batch in batch_patients(patient_ids, notes_db, batch_size=config.filter_batch_size):
        filter_batch_df = notes_db.select(["PERSON_ID", "MRN", config.note_column, config.date_of_diagnosis_column]).where(notes_db.PERSON_ID.isin(patient_batch)).toPandas()
        filter_batch_df = filter_batch_df.astype(str)
        filter_batch_df["KEPT_NOTES"] = [note_filter.apply_filters(notes, date) for notes, date in zip(filter_batch_df.pop(config.note_column), filter_batch_df.pop(config.date_of_diagnosis_column))]
        notes_df = pd.concat((notes_df, filter_batch_df))

    return notes_df


def batch_patients(patient_ids: list, db: spark.DataFrame, batch_size=100):
    batch_start = 0
    while batch_start < len(patient_ids):
        batch_end = batch_start + batch_size if batch_start + batch_size < len(patient_ids) else len(patient_ids)
        batch = patient_ids[batch_start:batch_end]
        batch_start += batch_size

        yield batch