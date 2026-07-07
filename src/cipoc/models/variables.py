from pydantic import BaseModel, Field, ConfigDict

from .base import ConfidenceLevel, confidence_field


class VariableInfo(BaseModel):
    """ Information about a variable """
    item_id: int = Field(description="Item ID number.")
    name: str | None = Field(default=None, description="Variable name.")
    description: str | None = Field(default=None, description="Variable description.")
    format: str | None = Field(default=None, description="Format for coded value as defined by the data dictionary and/or instructions.")
    valid_codes: str | dict | None = Field(default=None, description="Valid codes from the data dictionary.")
    model_config = ConfigDict(protected_namespaces=())


class VariableGroupInfo(BaseModel):
    variables: list[VariableInfo] = Field(description="List of variables in group with variable-level information.")


class VariableOutput(BaseModel):
    """ Structured output for an extracted variable """
    item_id: int = Field(description="Item ID number.")
    explanation: str = Field(description="Reasoning used for assigning the selected value.")
    value: str | None = Field(description="Coded value for the variable. Must be selected from the valid codes and in the appropriate format. Return `None` if no value can be determined.")
    presence_confidence: ConfidenceLevel = confidence_field()


class VariableGroupOutput(BaseModel):
    variables: list[VariableOutput] = Field(description="List of coded values for each variable in group with explanations and confidence level.")


# DEPRECATED
class NAACCRVariable(BaseModel):
    """ Structured output for an extracted NAACCR variable """
    item_id: int = Field(description="NAACCR item ID number")
    item_name: str = Field(description="NAACCR variable name")
    explanation: str = Field(description="Reasoning for assigning selected value")
    value: str = Field(description="Valid coding value assigned based on the context in the appropriate format")
