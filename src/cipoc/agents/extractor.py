import json
from pathlib import Path

from typing_extensions import Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage

from cipoc.llm import BaseAgentModel
from cipoc.models import VariableGroupInfo, VariableGroupOutput, ClinicalNote
from cipoc.prompts import EXTRACTOR_SYSTEM_PROMPT, EXTRACT_VALUES_PROMPT
from cipoc.tools import build_variable_group
from cipoc.utils import CipocConfig

from .base import BaseAgent


# Graph state
class ExtractorInput(BaseModel):
    variables: VariableGroupInfo = Field(description="The variable(s) to extract from the clinical notes.")


class ExtractorOutput(BaseModel):
    extracted_values: VariableGroupOutput | None = Field(default=None, description="The values for variable(s) extracted from the clinical notes.")


class ExtractorState(ExtractorInput, ExtractorOutput):
    messages: Annotated[list[AnyMessage], add_messages]
    notes: list[ClinicalNote] | None = Field(default=None, description="Corpus of relevant clinical notes for deriving coded value(s).")


class ExtractorAgent(BaseAgent):
    """Extracts coded variables from clinical notes."""
    _state = ExtractorState
    _input_schema = ExtractorInput
    _output_schema = ExtractorOutput

    def __init__(self, llm: BaseAgentModel | None = None, *, config: CipocConfig | None = None, **kwargs):
        super().__init__(agent_type="extractor", llm=llm, config=config, **kwargs)

    # --- Nodes (bound methods: (state) -> dict) ---
    # Initial nodes
    def initialize(self, state: ExtractorState) -> dict:
        """Seed the conversation with the shared persona and the variables to extract."""
        return {"messages": [
            SystemMessage(EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(f"Variables to extract:\n{state.variables.model_dump_json(indent=2)}"),
        ]}

    # TODO: need to create the database and tooling for this step. For now, just retrieve all the synthetic notes from `tests/fixtures/`
    def retrieve_clinical_notes(self, state: ExtractorState) -> dict:
        """Retrieve clinical notes containing information relevant to the desired variables"""
        note_path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "note_bundle.json"
        with open(note_path, "r") as f:
            notes = [ClinicalNote(**note) for note in json.load(f)]

        note_string = "\n".join(note.model_dump_json() for note in notes)
        return {"notes": notes, "messages": [AIMessage("Clinical notes:\n" + note_string)]}

    # TODO: create this tool. Skip for now
    # For each variable
    def retrieve_coding_instructions(self, state: ExtractorState):
        """Look up variable-level coding instructions from documents"""
        pass

    def extract_values(self, state: ExtractorState) -> dict:
        """Extract values for all variables from the clinical notes."""
        extracted_values = self.agent.model.with_structured_output(VariableGroupOutput).invoke(
            state.messages + [HumanMessage(EXTRACT_VALUES_PROMPT)]
        )
        return {"extracted_values": extracted_values}

    # --- Graph wiring (compiled once per instance) ---
    def _wire_graph(self, workflow: StateGraph) -> None:
        workflow.add_node("initialize", self.initialize)
        workflow.add_node("retrieve_clinical_notes", self.retrieve_clinical_notes)
        workflow.add_node("extract_values", self.extract_values)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "retrieve_clinical_notes")
        workflow.add_edge("retrieve_clinical_notes", "extract_values")
        workflow.add_edge("extract_values", END)

    # --- Public API ---
    def run(self, item_ids: int | list[int]) -> ExtractorOutput:
        """Extract coded values for the given NAACCR item ID(s) from the clinical notes."""
        variable_group = build_variable_group(
            item_ids, data_dictionary_path=self._config.documents().data_dictionary_path
        )
        result = self._graph.invoke({"variables": variable_group})
        return ExtractorOutput(**result)


if __name__ == "__main__":
    agent = ExtractorAgent()
    agent.draw(path="src/cipoc/agents/visualization/extractor.png")
    result = agent.run([400, 410, 440])  # Primary Site, Laterality, Grade
    print(json.dumps(result.model_dump(), indent=2))
