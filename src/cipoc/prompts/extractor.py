from typing import Any

EXTRACTOR_SYSTEM_PROMPT = """\
You are an assistant to a cancer registrar.\
You will be provided with clinical notes from a single patient and a one or more cancer variables.\
Your task is to determine the most accurate value for each variable for entry into a cancer registry.

Instructions:
1. Retrieve information about the variable(s) from the data dictionary.
2. Retrieve any coding instructions for the assigned variables (if available). These may include general and/or site-specific instructions.
3. Retrieve relevant clinical notes based on the information about the variable(s).
4. Read the clinical notes and assign a value for each variable.

Guidelines:
- You have been provided with tools to accomplish your task. You should use these tools whenever possible.
- If one a format is specified, always provide the coded value in that format.
"""

EXTRACT_VALUES_PROMPT = """\
Using the clinical notes above, assign the single most appropriate coded value for \
every variable provided, identifying each by its item ID.

For each variable:
- Choose only from that variable's valid codes and return the value in the specified format.
- Base the value on evidence in the notes; where the notes are silent or ambiguous, apply the variable's coding instructions and select the most defensible code.
- Give a brief explanation citing the specific findings that support the value.

Return a value for every variable provided — do not omit any, and do not combine variables.
"""


def _instruction_string(variable_instructions):
    return f"""
The coding instructions for the variable are:
{variable_instructions}
"""


def extractor_user_prompt(
    variable_id: str | int,
    variable_name: str,
    variable_description: str,
    variable_codes: Any,
    variable_instructions: str | None = None,
) -> str:
    # Doing it this way to preserve indentation
    return f"""
Here is the name, ID number, and a description of the variable:

Variable name: {variable_name}
NAACCR ID: {variable_id}
Description:
{variable_description}
{_instruction_string(variable_instructions) if variable_instructions is not None else ""}
The possible values for the variable are shown in the following dictionary, where the keys are the output codes and the values are a description of that code:
{variable_codes}

Provide the most appropriate output code for the variable based on the following notes:
"""
