import json
import textwrap
from tqdm import tqdm

import pandas as pd
import pyspark.sql.dataframe as spark

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from pydantic import BaseModel, Field
from typing import Any
from dataclasses import dataclass

from llm_client import ChatClient
from note_filter import NoteFilter


class NAACCRVariable(BaseModel):
    """ Structured output for an extracted NAACCR variable """
    item_id: int = Field(description="NAACCR item ID number")
    item_name: str = Field(description="NAACCR variable name")
    explanation: str = Field(description="Reasoning for assigning selected value")
    value: str = Field(description="Valid coding value assigned based on the context in the appropriate format")


@dataclass
class ExtractionConfig:
    target_vars: list | pd.Series
    target_df: pd.DataFrame
    note_column: str
    id_column: str
    note_types: list[str]
    note_days_before: int
    note_days_after: int
    llm_client: ChatClient
    output_file: str
    error_file: str
    start_index: int
    end_index: int
    batch_size: int | None


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
    return df.loc[df["Item_Number"] == int(variable_id), list(col_map.keys())].rename(columns=col_map).to_dict("records")[0] # type: ignore

def build_prompt(
    variable_id: str | int,
    variable_name: str,
    variable_description: str,
    variable_codes: Any,
    variable_instructions: str | None = None
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
    pbar = None
):
    if pbar is None:
        pbar = tqdm(total=config.end_index - config.start_index, position=0, desc="Patients")

    with open(config.output_file, "a") as f, open(config.error_file, "a") as error_log:
        for i, (patient_id, notes) in enumerate(zip(notes_df[config.id_column], notes_df["KEPT_NOTES"]), start=current_index):
            if i >= config.end_index:
                break
            
            if i < config.start_index:
                continue

            for target_id in tqdm(config.target_vars, position=1, desc="Variables"):
                target_info = get_variable_info_from_id(target_id, config.target_df)
                try:
                    output_code = extract_naaccr_variable(target_info, notes, config.llm_client)
                except Exception as e:
                    output_code = None
                    error = {"patient_id": patient_id, "item_id": target_info["variable_id"], "error": getattr(e, "message", str(e))}
                    error_log.write(json.dumps(error) + "\n")

                if output_code is None:
                    output_code = NAACCRVariable(item_id=target_info["variable_id"], item_name=target_info["variable_name"], explanation="Error", value="")

                write_dict = output_code.model_dump()
                write_dict.update({"patient_id": patient_id})
                f.write(json.dumps(write_dict) + "\n")

            pbar.update(1)

def run_extraction_batches(
    notes_db: spark.DataFrame,
    config: ExtractionConfig
):
    if config.batch_size is None:
        raise ValueError("Batch size must be set in ExtractionConfig to run in batch mode.")
    
    pbar = tqdm(total=config.end_index - config.start_index, position=0, desc="Patients")
    current_index = 0
    for patient_batch in batch_patients(notes_db, batch_size=config.batch_size):
        # Check if whole batch is out of bounds to reduce calls to db
        if current_index >= config.end_index:
            break

        if current_index + config.batch_size < config.start_index:
            current_index += config.batch_size
            continue

        notes_df = notes_db.select(["PERSON_ID", "MRN", config.note_column]).where(notes_db.PERSON_ID.isin(patient_batch)).toPandas()
        note_filter = NoteFilter(note_types_to_keep=config.note_types, days_before=config.note_days_before, days_after=config.note_days_after)
        notes_df["KEPT_NOTES"] = note_filter.apply_filters(notes_df.pop(config.note_column).to_list())
        run_extraction(notes_df=notes_df, config=config, current_index=current_index, pbar=pbar)

        current_index += config.batch_size

def run_extraction_batch_filter(
    notes_db: spark.DataFrame,
    config: ExtractionConfig
):
    if config.batch_size is None:
        raise ValueError("Batch size must be set in ExtractionConfig to run in batch mode.")
    
    notes_df = pd.DataFrame(columns=["PERSON_ID", "MRN", "KEPT_NOTES"])
    for patient_batch in batch_patients(notes_db, batch_size=config.batch_size):
        batch_df = notes_db.select(["PERSON_ID", "MRN", config.note_column]).where(notes_db.PERSON_ID.isin(patient_batch)).toPandas()
        note_filter = NoteFilter(note_types_to_keep=config.note_types, days_before=config.note_days_before, days_after=config.note_days_after)
        batch_df["KEPT_NOTES"] = note_filter.apply_filters(batch_df.pop(config.note_column).to_list())
        notes_df = pd.concat((notes_df, batch_df))
        
    run_extraction(notes_df=notes_df, config=config)

def batch_patients(db: spark.DataFrame, batch_size=100):
    person_ids = db.rdd.map(lambda x: x.PERSON_ID).collect()
    batch_start = 0
    while batch_start < len(person_ids):
        batch_end = batch_start + batch_size if batch_start + batch_size < len(person_ids) else len(person_ids)
        batch = person_ids[batch_start:batch_end]
        batch_start += batch_size

        yield batch