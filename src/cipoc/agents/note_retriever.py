from typing_extensions import Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage, SystemMessage

from cipoc.llm import BaseAgentModel
from cipoc.models import NoteDigest, VariableGroupInfo
from cipoc.utils import CipocConfig, run_with_progress
from cipoc.prompts.note_retriever import NOTE_RETRIEVER_SYSTEM_PROMPT, SELECT_NOTES_PROMPT

from .base import BaseAgent


class RelevantNoteIDs(BaseModel):
    note_ids: list[int | str] = Field(default_factory=list, description="ID values for notes identified as relevant based on the filter criteria.")


# Graph state
class RetrieverInput(BaseModel):
    requested_variables: VariableGroupInfo = Field(description="The target variable(s) to extract from the clinical notes.")
    available_digests: dict[int | str, NoteDigest] = Field(description="Dictionary of clinical note metadata keyed by note ID for identification of relevance of notes for an extraction task. Includes note type, summary, and keywords.")


class RetrieverOutput(BaseModel):
    relevant_note_ids: list[int | str] | None = Field(default=None, description="ID values for notes identified as relevant based on the filter criteria. `None` if no note could plausibly be relevant.")


class RetrieverState(RetrieverInput, RetrieverOutput):
    messages: Annotated[list[AnyMessage], add_messages]
    

class NoteRetrieverAgent(BaseAgent):
    """Selects which notes a downstream extractor should read, judging relevance
    from note digests against the requested variables."""
    _state = RetrieverState
    _input_schema = RetrieverInput
    _output_schema = RetrieverOutput

    def __init__(self, llm: BaseAgentModel | None = None, *, config: CipocConfig | None = None, **kwargs):
        super().__init__(agent_type="note_retriever", llm=llm, config=config, **kwargs)

    # --- Nodes (bound methods: (state) -> dict) ---
    def initialize(self, state: RetrieverState) -> dict:
        """Seed the conversation with the shared persona + the requested variables (the cacheable prefix)."""
        return {"messages": [
            SystemMessage(NOTE_RETRIEVER_SYSTEM_PROMPT),
            HumanMessage("Variables to extract:\n" + state.requested_variables.model_dump_json()),
        ]}

    def identify_relevant_notes(self, state: RetrieverState) -> dict:
        """LLM call to identify relevant notes based on task and note digests."""
        digest_string = "\n".join(digest.model_dump_json() for digest in state.available_digests.values())
        response = self.agent.structured(
            RelevantNoteIDs, state.messages + [HumanMessage(SELECT_NOTES_PROMPT), HumanMessage("Note digests:\n" + digest_string)]
        )

        input_note_ids = set(state.available_digests)
        valid_ids = [note_id for note_id in response.note_ids if note_id in input_note_ids]

        if not valid_ids:
            return {"relevant_note_ids": None}
        return {"relevant_note_ids": valid_ids}


    # --- Graph wiring (compiled once per instance) ---
    def _wire_graph(self, workflow: StateGraph) -> None:
        workflow.add_node("initialize", self.initialize)
        workflow.add_node(
            "identify_relevant_notes", self.identify_relevant_notes, retry_policy=self.retry_policy
        )

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "identify_relevant_notes")
        workflow.add_edge("identify_relevant_notes", END)

    # --- Public API ---
    def run(
        self,
        retriever_input: RetrieverInput | dict,
        *,
        progress: bool = True,
    ) -> list[int | str] | None:
        """Select relevant notes for the requested variables; returns their IDs (or None)."""
        result = (
            run_with_progress(
                self._graph,
                retriever_input,
                description="Note Retriever",
            )
            if progress
            else self._graph.invoke(retriever_input)
        )
        return result["relevant_note_ids"]


if __name__ == "__main__":
    import json
    from pathlib import Path

    from cipoc.tools import build_variable_group

    agent = NoteRetrieverAgent()
    agent.draw(path="src/cipoc/agents/visualization/note_retriever.png")

    scanned_path = Path(__file__).resolve().parents[3] / "tests" / "test_outputs" / "scanner_test.json"
    with open(scanned_path, "r") as f:
        digests = {note["note_id"]: NoteDigest(**note) for note in json.load(f)}

    # Same variables the extractor demo extracts (Primary Site, Laterality, Histology).
    variable_group = build_variable_group(
        [400, 410, 522],
        data_dictionary_path=agent._config.documents().data_dictionary_path,
    )

    relevant_ids = agent.run(
        RetrieverInput(requested_variables=variable_group, available_digests=digests)
    )
    print(json.dumps({"relevant_note_ids": relevant_ids}, indent=2))
