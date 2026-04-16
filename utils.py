import os
import shutil
import json
import glob
from collections import defaultdict
from typing import Literal


def find_output_files(output_type: Literal["results", "error"], directory: str = ".") -> dict | None:
    output_files = glob.glob(os.path.join(directory, f"*{output_type}*.json"))
    if not output_files:
        return None

    output_file_dict = defaultdict(list)
    for file in output_files:
        model_name = os.path.basename(file).split("_")[0]
        output_file_dict[model_name].append(file)

    return dict(output_file_dict)


def get_model_run_state(model_name: str, state_type: Literal["results", "error"] = "results", output_directory: str = ".") -> dict | None:
    """ Get number of results for each patient for a given model """
    results = find_output_files(state_type, output_directory)
    if results is None:
        return None

    model_state = defaultdict(int)
    for file in results[model_name]:
        with open(file, "r") as f:
            for line in f.readlines():
                line_json = json.loads(line)
                if state_type == "results" or (state_type == "error" and line_json["explanation"] == "Error"):
                    model_state[line_json["patient_id"]] += 1

    return dict(model_state)


def find_temp_files(model_name: str, state_type: Literal["results", "error"] = "results", output_directory: str = ".") -> dict:
    output_type_string = "error_log" if state_type == "error" else state_type
    file_prefix = f"{model_name}_{output_type_string}_"
    model_outputs = glob.glob(os.path.join(output_directory, f"{file_prefix}*.json"))

    output_groups = defaultdict(list)
    for file in model_outputs:
        file_split = os.path.basename(file).replace(file_prefix, "").replace(".json", "").split("_")
        if len(file_split) > 1:
            output_groups[file_split[0]].append(file)

    return dict(output_groups)


def append_temp_file_to_output(temp_file: str, output_file: str) -> None:
    with open(temp_file, "rb") as tempf, open(output_file, "ab") as outf:
        shutil.copyfileobj(tempf, outf)


def cleanup_temp_files(model_name: str, output_directory: str = ".") -> None:
    print(model_name)
    for state_type in ["results", "error"]:
        temp_groups = find_temp_files(model_name, state_type, output_directory)
        if len(temp_groups) == 0:
            print(f"No {state_type} temp files found.")
            continue

        temp_file_count = sum([len(group) for group in temp_groups.values()])
        print(f"Found {temp_file_count} temp {state_type} files across {len(temp_groups)} dates.")

        output_type_string = "error_log" if state_type == "error" else state_type
        for date, group in temp_groups.items():
            output_file = f"{model_name}_{output_type_string}_{date}.json"
            output_path = os.path.join(output_directory, output_file)
            for temp_file in group:
                append_temp_file_to_output(temp_file, output_path)
                os.remove(temp_file)