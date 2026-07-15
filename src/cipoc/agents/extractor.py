import json
from collections import Counter
from pathlib import Path

from operator import add
from typing_extensions import Annotated, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage, SystemMessage

from cipoc.llm import BaseAgentModel
from cipoc.models import (
    CaseFacts,
    ClinicalNote,
    ConfidenceLevel,
    VariableGroupInfo,
    VariableGroupOutput,
    VariableInfo,
    VariableOutput,
)
from cipoc.prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    EXTRACT_GROUP_VALUES_PROMPT,
    EXTRACT_VARIABLE_VALUE_PROMPT,
    REPAIR_VARIABLE_VALUE_PROMPT,
)
from cipoc.tools import VariableValueValidator, build_variable_group, load_rule_store
from cipoc.utils import CipocConfig, run_with_progress

from .base import BaseAgent


# Graph state
class ExtractorInput(BaseModel):
    requested_variables: VariableGroupInfo = Field(description="The target variable(s) to extract from the clinical notes.")


class ValidatedVariableOutput(VariableOutput):
    """A VariableOutput carrying the pipeline's validation verdict.

    Built by the extractor after validation; never use as an LLM structured-output
    schema, or the model will be asked to fill in the verdict fields itself.
    """
    is_valid: bool = Field(description="Whether the emitted result passed validation; False means it exhausted its repair attempts still failing.")
    validation_errors: list[str] = Field(default_factory=list)
    extraction_attempts: int = 0


class ValidatedVariableGroupOutput(VariableGroupOutput):
    variables: list[ValidatedVariableOutput] = Field(description="List of coded values for each variable in group, with validation verdicts.")


class ExtractorOutput(BaseModel):
    extracted_values: ValidatedVariableGroupOutput | None = Field(default=None, description="The values for variable(s) extracted from the clinical notes, with per-variable validation verdicts.")


class ExtractorState(ExtractorInput, ExtractorOutput):
    messages: Annotated[list[AnyMessage], add_messages]
    notes: list[ClinicalNote] | None = Field(default=None, description="Corpus of relevant clinical notes for deriving coded value(s).")
    max_extraction_attempts: int = Field(default=2, description="Maximum number of attempts to repair an invalid extraction.")
    variable_results: Annotated[list[ValidatedVariableOutput], add] = Field(default_factory=list)


# Subgraphs
class VariableExtractionTask(BaseModel):
    variable: VariableInfo
    extraction_mode: Literal["group", "individual"]
    coding_instructions: str | None = None
    candidate: VariableOutput | None = None
    validation_errors: list[str] = Field(default_factory=list)
    extraction_attempts: int = 0
    is_valid: bool = False


class VariableBranchState(BaseModel):
    task: VariableExtractionTask
    notes: list[ClinicalNote]
    messages: list[AnyMessage]
    group_context: VariableGroupOutput | None = None
    max_extraction_attempts: int = Field(default=2, description="Maximum number of attempts to repair an invalid extraction.")


class VariableBranchOutput(BaseModel):
    variable_results: list[ValidatedVariableOutput]


class ExtractorAgent(BaseAgent):
    """Extracts coded variables from clinical notes."""
    _state = ExtractorState
    _input_schema = ExtractorInput
    _output_schema = ExtractorOutput

    def __init__(self, llm: BaseAgentModel | None = None, *, config: CipocConfig | None = None, **kwargs):
        self._value_validator = VariableValueValidator()
        super().__init__(agent_type="extractor", llm=llm, config=config, **kwargs)

    # --- Nodes (bound methods: (state) -> dict) ---
    # Initial nodes
    def initialize(self, state: ExtractorState) -> dict:
        """Seed the conversation with the shared persona and the variables to extract."""
        return {"messages": [SystemMessage(EXTRACTOR_SYSTEM_PROMPT)]}

    # TODO: need to create the database and tooling for this step. For now, just retrieve all the synthetic notes from `tests/fixtures/`
    def retrieve_clinical_notes(self, state: ExtractorState) -> dict:
        """Retrieve clinical notes containing information relevant to the desired variables"""
        note_path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "note_bundle.json"
        with open(note_path, "r") as f:
            notes = [ClinicalNote(**note) for note in json.load(f)]

        note_string = "\n".join(note.model_dump_json() for note in notes)
        return {"notes": notes, "messages": [HumanMessage("Clinical notes:\n" + note_string)]}

    def variables_to_extract(self, state: ExtractorState) -> Literal["extract_group_values"] | list[Send]:
        variables = state.requested_variables.variables

        if len(variables) > 1 and state.requested_variables.extract_group:
            return "extract_group_values"

        return [
            Send(
                "variable_branch",
                VariableBranchState(
                    task=VariableExtractionTask(variable=variable, extraction_mode="individual"),
                    notes=state.notes or [],
                    messages=list(state.messages),
                    max_extraction_attempts=state.max_extraction_attempts,
                ),
            )
            for variable in variables
        ]

    def extract_group_values(self, state: ExtractorState) -> Command[Literal["variable_branch"]]:
        group_output = self.agent.model.with_structured_output(
            VariableGroupOutput
        ).invoke(
            state.messages
            + [
                HumanMessage(
                    "Variables to extract:\n"
                    + state.requested_variables.model_dump_json()
                ),
                HumanMessage(EXTRACT_GROUP_VALUES_PROMPT),
            ]
        )

        output_counts = Counter(output.item_id for output in group_output.variables)
        outputs_by_id = {
            output.item_id: output
            for output in group_output.variables
        }

        sends = [
            Send(
                "variable_branch",
                VariableBranchState(
                    task=VariableExtractionTask(
                        variable=variable,
                        extraction_mode="group",
                        candidate=outputs_by_id.get(variable.item_id),
                        validation_errors=(
                            ["Group extraction did not return this requested variable."]
                            if output_counts[variable.item_id] == 0
                            else ["Group extraction returned this variable more than once."]
                            if output_counts[variable.item_id] > 1
                            else []
                        ),
                        extraction_attempts=1,
                    ),
                    notes=state.notes or [],
                    messages=list(state.messages),
                    group_context=group_output,
                    max_extraction_attempts=state.max_extraction_attempts,
                ),
            )
            for variable in state.requested_variables.variables
        ]

        return Command(goto=sends)

    def merge_variable_results(self, state: ExtractorState) -> dict:
        results_by_id = {result.item_id: result for result in state.variable_results}

        ordered_results = [
            results_by_id.get(variable.item_id)
            or ValidatedVariableOutput(
                item_id=variable.item_id,
                value=None,
                explanation="No extraction result was produced for this variable.",
                spans=[],
                presence_confidence=ConfidenceLevel.LOW,
                is_valid=False,
                validation_errors=["No branch result was produced for this variable."],
            )
            for variable in state.requested_variables.variables
        ]

        return {"extracted_values": ValidatedVariableGroupOutput(variables=ordered_results)}


    # Variable branch
    def route_variable_entry(
        self, state: VariableBranchState
    ) -> Literal["extract_individual_value", "validate_extraction"]:
        if state.task.extraction_mode == "individual":
            return "extract_individual_value"
        return "validate_extraction"

    def extract_individual_value(self, state: VariableBranchState) -> dict:
        """Extract one variable from the clinical notes."""
        extracted_value = self.agent.model.with_structured_output(VariableOutput).invoke(
            state.messages
            + [
                HumanMessage("Variable to extract:\n" + state.task.variable.model_dump_json()),
                HumanMessage(EXTRACT_VARIABLE_VALUE_PROMPT),
            ]
        )
        return {
            "task": state.task.model_copy(
                update={
                    "candidate": extracted_value,
                    "validation_errors": [],
                    "extraction_attempts": state.task.extraction_attempts + 1,
                    "is_valid": False,
                }
            )
        }

    def validate_extraction(self, state: VariableBranchState) -> dict:
        errors = list(state.task.validation_errors)
        if state.task.candidate is None:
            errors.append("No extraction candidate was returned.")
        else:
            errors.extend(
                self._value_validator.validate(
                    state.task.variable,
                    state.task.candidate,
                )
            )
            if state.task.candidate.value is None and state.task.candidate.spans:
                errors.append("Supporting text spans must be empty when no value is returned.")
            elif state.task.candidate.value is not None and not state.task.candidate.spans:
                errors.append("No supporting text spans were returned.")
            for index, span in enumerate(state.task.candidate.spans, start=1):
                if not span.text.strip():
                    errors.append(f"Supporting text span {index} is empty.")
                    continue
                if "\n" in span.text or "\r" in span.text:
                    errors.append(
                        f"Supporting text span {index} contains newline characters."
                    )
                if not any(span.text in note.content for note in state.notes):
                    errors.append(
                        f"Supporting text span {index} is not verbatim text from a clinical note."
                    )

        return {
            "task": state.task.model_copy(
                update={"validation_errors": errors, "is_valid": not errors}
            )
        }

    def route_after_validation(
        self, state: VariableBranchState
    ) -> Literal["repair_invalid_extraction", "complete_variable"]:
        if state.task.is_valid:
            return "complete_variable"

        if state.task.extraction_attempts >= state.max_extraction_attempts:
            return "complete_variable"

        return "repair_invalid_extraction"

    def repair_invalid_extraction(self, state: VariableBranchState) -> dict:
        repair_context = {
            "variable": state.task.variable.model_dump(),
            "invalid_candidate": (
                state.task.candidate.model_dump()
                if state.task.candidate is not None
                else None
            ),
            "validation_errors": state.task.validation_errors,
            "group_context": (
                state.group_context.model_dump()
                if state.group_context is not None
                else None
            ),
        }
        repaired_value = self.agent.model.with_structured_output(VariableOutput).invoke(
            state.messages + [HumanMessage(REPAIR_VARIABLE_VALUE_PROMPT + "\nRepair context:\n" + json.dumps(repair_context))]
        )

        return {
            "task": state.task.model_copy(
                update={
                    "candidate": repaired_value,
                    "validation_errors": [],
                    "extraction_attempts": state.task.extraction_attempts + 1,
                    "is_valid": False,
                }
            )
        }

    def complete_variable(self, state: VariableBranchState) -> dict:
        candidate = state.task.candidate or VariableOutput(
            item_id=state.task.variable.item_id,
            explanation="No extraction candidate was produced after the allowed attempts.",
            value=None,
            spans=[],
            presence_confidence=ConfidenceLevel.LOW,
        )
        validated = ValidatedVariableOutput(
            **candidate.model_dump(),
            is_valid=state.task.is_valid,
            validation_errors=list(state.task.validation_errors),
            extraction_attempts=state.task.extraction_attempts,
        )
        return {"variable_results": [validated]}

    def _build_variable_branch(self):
        branch = StateGraph(VariableBranchState, output_schema=VariableBranchOutput)
        branch.add_node("extract_individual_value", self.extract_individual_value)
        branch.add_node("validate_extraction", self.validate_extraction)
        branch.add_node("repair_invalid_extraction", self.repair_invalid_extraction)
        branch.add_node("complete_variable", self.complete_variable)

        branch.add_conditional_edges(
            START,
            self.route_variable_entry,
            ["extract_individual_value", "validate_extraction"],
        )
        branch.add_edge("extract_individual_value", "validate_extraction")
        branch.add_conditional_edges("validate_extraction", self.route_after_validation)
        branch.add_edge("repair_invalid_extraction", "validate_extraction")
        branch.add_edge("complete_variable", END)

        return branch.compile()

    # --- Graph wiring (compiled once per instance) ---
    def _wire_graph(self, workflow: StateGraph) -> None:
        variable_branch = self._build_variable_branch()

        workflow.add_node("initialize", self.initialize)
        workflow.add_node("retrieve_clinical_notes", self.retrieve_clinical_notes)
        workflow.add_node("extract_group_values", self.extract_group_values, destinations=("variable_branch",))
        workflow.add_node("variable_branch", variable_branch)
        workflow.add_node("merge_variable_results", self.merge_variable_results)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "retrieve_clinical_notes")
        workflow.add_conditional_edges(
            "retrieve_clinical_notes",
            self.variables_to_extract,
            ["extract_group_values", "variable_branch"],
        )
        workflow.add_edge("variable_branch", "merge_variable_results")
        workflow.add_edge("merge_variable_results", END)

    # --- Public API ---
    def run(
        self,
        item_ids: int | list[int],
        *,
        extract_group: bool = False,
        case_facts: CaseFacts | None = None,
        rules_dir: str | Path | None = None,
    ) -> ExtractorOutput:
        """Extract coded values for the given NAACCR item ID(s) from the clinical notes.

        Pass ``case_facts`` to scope manual coding instructions and valid codes to
        the case deterministically. The rule store is read from ``rules_dir`` when
        given, else from the config's ``documents.rules_path``; without either,
        variables carry unscoped data-dictionary metadata only.
        """
        rule_store = None
        if case_facts is not None:
            rules_path = rules_dir or getattr(self._config.documents(), "rules_path", None)
            if rules_path is not None:
                rule_store = load_rule_store(rules_path)

        variable_group = build_variable_group(
            item_ids,
            data_dictionary_path=self._config.documents().data_dictionary_path,
            case_facts=case_facts,
            rule_store=rule_store,
        )
        if not variable_group.variables:
            raise ValueError("None of the requested item IDs exist in the data dictionary.")
        variable_group = variable_group.model_copy(update={"extract_group": extract_group})
        # result = self._graph.invoke({"requested_variables": variable_group})
        result = run_with_progress(
            self._graph,
            {"requested_variables": variable_group},
            subgraphs=True,
            description="Extractor",
        )
        return ExtractorOutput(**result)


if __name__ == "__main__":
    agent = ExtractorAgent()
    agent.draw(path="src/cipoc/agents/visualization/extractor.png")
    # Case facts matching tests/fixtures/note_bundle.json (left breast, dx 2025);
    # scopes coding instructions and valid codes from documents/rules.
    facts = CaseFacts(primary_site="C504", date_of_diagnosis="2025-02-24", sex="female")
    result = agent.run([400, 410, 522], case_facts=facts)  # Primary Site, Laterality, Histology
    print(json.dumps(result.model_dump(), indent=2))
