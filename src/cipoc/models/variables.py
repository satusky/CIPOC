from pydantic import BaseModel, Field, ConfigDict

from .base import TextSpan, ConfidenceLevel, confidence_field


class VariableInfo(BaseModel):
    """ Information about a variable """
    item_id: int = Field(description="Item ID number.")
    name: str | None = Field(default=None, description="Variable name.")
    description: str | None = Field(default=None, description="Variable description.")
    data_type: str | None = Field(default=None, description="Data type defined by the data dictionary.")
    length: int | None = Field(default=None, description="Maximum field length defined by the data dictionary.")
    allowable_values: str | None = Field(default=None, description="Allowable values defined by the data dictionary.")
    format: str | None = Field(default=None, description="Format for coded value as defined by the data dictionary and/or instructions.")
    valid_codes: str | dict | None = Field(default=None, description="Valid codes from the data dictionary. When applicable, the scope is reduced when based on case specifics (e.g., primary site).")
    model_config = ConfigDict(protected_namespaces=())


class VariableGroupInfo(BaseModel):
    variables: list[VariableInfo] = Field(description="List of variables in group with variable-level information.")
    extract_group: bool = Field(default=False, description="Extract the entire group together (True) or individually (False).")


class VariableOutput(BaseModel):
    """ Structured output for an extracted variable """
    item_id: int = Field(description="Item ID number.")
    value: str | None = Field(description="Coded value for the variable. Must be selected from the valid codes and in the appropriate format. Return `None` if no value can be determined.")
    explanation: str = Field(description="Reasoning used for assigning the selected value.")
    spans: list[TextSpan] = Field(description="List of text span(s) in the clinical note that provide evidence for this claim. A span containing newline characters should be split into multiple spans.")
    presence_confidence: ConfidenceLevel = confidence_field()


class VariableGroupOutput(BaseModel):
    variables: list[VariableOutput] = Field(description="List of coded values for each variable in group with explanations and confidence level.")
