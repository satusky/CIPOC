from typing_extensions import Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage, SystemMessage

from cipoc.llm import BaseAgentModel
from cipoc.models import ClinicalNote, ProcessedClinicalNote, CancerMentionList, CancerStatus, ConfidenceLevel, confidence_field
from cipoc.utils.utils import load_config, CipocConfig
from cipoc.prompts.note_scanner import NOTE_SCANNER_SYSTEM_PROMPT, CANCER_IN_NOTE_PROMPT, NOTE_SUMMARY_PROMPT, CANCER_MENTIONS_PROMPT

from .base import BaseAgent


# Graph state
class ScannerInput(BaseModel):
    note: ClinicalNote = Field(description="A clinical note object for a single patient visit.")


class CancerPresent(BaseModel):
    cancer_present: bool = Field(description="Cancer is mentioned in the note. If uncertain, default to `True`.")
    presence_confidence: ConfidenceLevel = confidence_field()


class ScannerOutput(BaseModel):
    cancer_present: bool | None = Field(default=None, description="Cancer is mentioned in the note. If uncertain, default to `True`.")
    cancer_status: set[CancerStatus] | None = Field(default=None, description="Distinct temporality statuses across all cancer mentions in the note. `None` when no cancer is present.")
    summary: str | None = Field(default=None, description="Summary of clinical note.")
    cancer_mentions: CancerMentionList | None = Field(default=None, description="List of cancer mentions.")
    presence_confidence: ConfidenceLevel | None = confidence_field(default=None)


class ScannerState(ScannerInput, ScannerOutput):
    messages: Annotated[list[AnyMessage], add_messages]


class NoteScannerAgent(BaseAgent):
    """Scans a single clinical note: gates on cancer presence, then fans out to
    summarization and cancer-mention extraction."""
    _state = ScannerState
    _input_schema = ScannerInput
    _output_schema = ScannerOutput

    def __init__(self, llm: BaseAgentModel | None = None, *, config: CipocConfig | None = None, **kwargs):
        super().__init__(agent_type="note_scanner", llm=llm, config=config, **kwargs)

    # --- Nodes (bound methods: (state) -> dict) ---
    def initialize(self, state: ScannerState) -> dict:
        """Seed the conversation once with the shared persona + the note (the cacheable prefix)."""
        return {"messages": [
            SystemMessage(NOTE_SCANNER_SYSTEM_PROMPT),
            HumanMessage(f"Clinical note:\n{state.note.model_dump_json(indent=2)}"),
        ]}

    def check_note_for_cancer(self, state: ScannerState) -> dict:
        """First LLM call to check for cancer mention(s) in a clinical note."""
        response = self.agent.model.with_structured_output(CancerPresent).invoke(
            state.messages + [HumanMessage(CANCER_IN_NOTE_PROMPT)]
        )
        return {"cancer_present": response.cancer_present, "presence_confidence": response.presence_confidence}

    def summarize_note(self, state: ScannerState) -> dict:
        """Summarize a clinical note."""
        response = self.agent.model.invoke(state.messages + [HumanMessage(NOTE_SUMMARY_PROMPT)])
        return {"summary": response.text}

    def get_cancer_mentions(self, state: ScannerState) -> dict:
        """Detail any mentions of cancer in a clinical note."""
        mentions = self.agent.model.with_structured_output(CancerMentionList).invoke(
            state.messages + [HumanMessage(CANCER_MENTIONS_PROMPT)]
        )
        # Roll the per-mention temporality up to a note-level set (empty -> None).
        return {
            "cancer_mentions": mentions,
            "cancer_status": {m.status for m in mentions.mentions} or None,
        }

    @staticmethod
    def cancer_gate(state: ScannerState) -> list[str] | str:
        """Gate function: fan out to both summary + mentions nodes if cancer mentioned, else stop."""
        return ["summarize_note", "get_cancer_mentions"] if state.cancer_present else END

    # --- Graph wiring (compiled once per instance) ---
    def _wire_graph(self, workflow: StateGraph) -> None:
        workflow.add_node("initialize", self.initialize)
        workflow.add_node("check_note_for_cancer", self.check_note_for_cancer)
        workflow.add_node("summarize_note", self.summarize_note)
        workflow.add_node("get_cancer_mentions", self.get_cancer_mentions)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "check_note_for_cancer")
        workflow.add_conditional_edges(
            "check_note_for_cancer", self.cancer_gate, ["summarize_note", "get_cancer_mentions", END]
        )
        workflow.add_edge("summarize_note", END)
        workflow.add_edge("get_cancer_mentions", END)

    # --- Public API ---
    def run(self, note: ClinicalNote | dict) -> ProcessedClinicalNote:
        """Run the scanner over a single note and return the enriched note."""
        if isinstance(note, dict):
            note = ClinicalNote(**note)
        result = self._graph.invoke({"note": note})
        return ProcessedClinicalNote(**note.model_dump(), **result)


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Synthetic clinical note used to exercise the scanner end-to-end.
    note_path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "synthetic_note.json"
    with open(note_path, "r") as f:
        note_data = json.load(f)

    agent = NoteScannerAgent()
    agent.draw(path="src/cipoc/agents/visualization/note_scanner.png")
    result = agent.run(ClinicalNote(**note_data))
    print(json.dumps(result.model_dump(), indent=2))
