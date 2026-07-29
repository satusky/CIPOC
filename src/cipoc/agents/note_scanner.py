from typing_extensions import Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage, SystemMessage

from cipoc.llm import BaseAgentModel
from cipoc.models import (
    ClinicalNote,
    ProcessedClinicalNote,
    CancerMention,
    CancerStatus,
    ConceptWithEvidence,
    CONCEPT_DESCRIPTIONS,
    build_concept_presence_dict,
)
from cipoc.utils import CipocConfig, run_with_progress
from cipoc.prompts.note_scanner import (
    NOTE_SCANNER_SYSTEM_PROMPT,
    CONCEPT_DETECTION_PROMPT,
    NOTE_SUMMARY_PROMPT,
    CANCER_MENTIONS_PROMPT,
)

from .base import BaseAgent


# Rendered once: the concept vocabulary the detection call must judge, with descriptions.
_CONCEPT_LIST = "\n".join(f"- {name}: {desc}" for name, desc in CONCEPT_DESCRIPTIONS.items())
_CONCEPT_DETECTION_PROMPT = CONCEPT_DETECTION_PROMPT.format(concept_list=_CONCEPT_LIST)


# Graph state
class ScannerInput(BaseModel):
    note: ClinicalNote = Field(description="A clinical note object for a single patient visit.")


class ConceptFinding(ConceptWithEvidence):
    """One detected concept: presence/confidence/evidence plus which concept it is."""
    concept: str = Field(description="Concept key, exactly as given in the list of concepts to evaluate.")


class ConceptFindingList(BaseModel):
    findings: list[ConceptFinding] = Field(description="One finding per concept evaluated.")


class CancerMentions(BaseModel):
    mentions: list[CancerMention] = Field(description="List of cancer mentions; empty if no cancer is mentioned.")


class NoteSummary(BaseModel):
    summary: str = Field(description="Summary of clinical note.")
    keywords: list[str] = Field(description="Three to eight keywords associated with the note contents for search. Never empty.")


class ScannerOutput(BaseModel):
    concepts: dict[str, ConceptWithEvidence] | None = Field(default=None, description="Per-concept presence/evidence findings keyed by concept name.")
    cancer_status: set[CancerStatus] | None = Field(default=None, description="Distinct temporality statuses across all cancer mentions in the note. `None` when no cancer is present.")
    summary: str | None = Field(default=None, description="Summary of clinical note.")
    flags: list[str] | None = Field(default=None, description="Keywords associated with the note contents for search.")
    cancer_mentions: list[CancerMention] | None = Field(default=None, description="List of cancer mentions.")


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
    # Each LLM node is a prompt builder plus a result folder, so the sync node and
    # its async twin differ only in the call verb and share every line of logic.
    def initialize(self, state: ScannerState) -> dict:
        """Seed the conversation once with the shared persona + the note (the cacheable prefix)."""
        return {"messages": [
            SystemMessage(NOTE_SCANNER_SYSTEM_PROMPT),
            HumanMessage(f"Clinical note:\n{state.note.model_dump_json(indent=2)}"),
        ]}

    @staticmethod
    def _concept_messages(state: ScannerState) -> list[AnyMessage]:
        return state.messages + [HumanMessage(_CONCEPT_DETECTION_PROMPT)]

    def _concept_result(self, response: ConceptFindingList) -> dict:
        return {"concepts": self._findings_to_concepts(response.findings)}

    def detect_concepts(self, state: ScannerState) -> dict:
        """Single LLM call detecting presence/evidence for every tracked concept."""
        return self._concept_result(
            self.agent.structured(ConceptFindingList, self._concept_messages(state))
        )

    async def adetect_concepts(self, state: ScannerState) -> dict:
        return self._concept_result(
            await self.agent.astructured(ConceptFindingList, self._concept_messages(state))
        )

    @staticmethod
    def _findings_to_concepts(findings: list[ConceptFinding]) -> dict[str, ConceptWithEvidence]:
        """Fold detected findings into the full concept dict.

        Starts from the default all-absent dict so every tracked concept is always
        present as a key; unknown concept names returned by the model are ignored.
        """
        concepts = build_concept_presence_dict(with_evidence=True)
        for finding in findings:
            if finding.concept in concepts:
                concepts[finding.concept] = ConceptWithEvidence(
                    presence=finding.presence,
                    confidence=finding.confidence,
                    evidence=finding.evidence,
                )
        return concepts

    @staticmethod
    def _summary_messages(state: ScannerState) -> list[AnyMessage]:
        return state.messages + [HumanMessage(NOTE_SUMMARY_PROMPT)]

    @staticmethod
    def _summary_result(response: NoteSummary) -> dict:
        return {"summary": response.summary, "flags": response.keywords}

    def summarize_note(self, state: ScannerState) -> dict:
        """Summarize a clinical note and tag it with search keywords."""
        return self._summary_result(
            self.agent.structured(NoteSummary, self._summary_messages(state))
        )

    async def asummarize_note(self, state: ScannerState) -> dict:
        return self._summary_result(
            await self.agent.astructured(NoteSummary, self._summary_messages(state))
        )

    @staticmethod
    def _mentions_messages(state: ScannerState) -> list[AnyMessage]:
        return state.messages + [HumanMessage(CANCER_MENTIONS_PROMPT)]

    @staticmethod
    def _mentions_result(response: CancerMentions) -> dict:
        # Roll the per-mention temporality up to a note-level set (empty -> None).
        return {
            "cancer_mentions": response.mentions,
            "cancer_status": {m.status for m in response.mentions} or None,
        }

    def get_cancer_mentions(self, state: ScannerState) -> dict:
        """Detail any mentions of cancer in a clinical note."""
        return self._mentions_result(
            self.agent.structured(CancerMentions, self._mentions_messages(state))
        )

    async def aget_cancer_mentions(self, state: ScannerState) -> dict:
        return self._mentions_result(
            await self.agent.astructured(CancerMentions, self._mentions_messages(state))
        )

    @staticmethod
    def cancer_gate(state: ScannerState) -> str:
        """Gate function: detail cancer mentions if the cancer concept is present, else stop."""
        cancer = (state.concepts or {}).get("cancer")
        return "get_cancer_mentions" if cancer and cancer.presence else END

    # --- Graph wiring (compiled once per instance) ---
    def _wire_graph(self, workflow: StateGraph) -> None:
        workflow.add_node("initialize", self.initialize)
        # All three scan nodes call the model; each retries its own request.
        # _node picks the sync method or its async twin; the policy is unchanged
        # either way, so the same nodes retry in both modes.
        workflow.add_node("detect_concepts", self._node("detect_concepts"), retry_policy=self.retry_policy)
        workflow.add_node("summarize_note", self._node("summarize_note"), retry_policy=self.retry_policy)
        workflow.add_node("get_cancer_mentions", self._node("get_cancer_mentions"), retry_policy=self.retry_policy)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "summarize_note")
        workflow.add_edge("summarize_note", "detect_concepts")
        workflow.add_conditional_edges(
            "detect_concepts", self.cancer_gate, ["get_cancer_mentions", END]
        )
        workflow.add_edge("get_cancer_mentions", END)

    # --- Public API ---
    @staticmethod
    def _as_note(notes: ClinicalNote | dict) -> ClinicalNote:
        return ClinicalNote(**notes) if isinstance(notes, dict) else notes

    def run(
        self,
        notes: ClinicalNote | dict,
        *,
        progress: bool = True,
    ) -> ProcessedClinicalNote:
        """Run the scanner over a single note and return the enriched note."""
        self._require_mode(False)
        note = self._as_note(notes)
        graph_input = {"note": note}
        result = (
            run_with_progress(
                self._graph,
                graph_input,
                description="Note Scanner",
            )
            if progress
            else self._graph.invoke(graph_input)
        )
        return ProcessedClinicalNote(**note.model_dump(), **result)

    async def arun(
        self,
        notes: ClinicalNote | dict,
        *,
        progress: bool = False,
    ) -> ProcessedClinicalNote:
        """Async twin of :meth:`run`; requires an agent built with ``use_async=True``."""
        note = self._as_note(notes)
        result = await self._arun_graph(
            {"note": note}, progress=progress, description="Note Scanner"
        )
        return ProcessedClinicalNote(**note.model_dump(), **result)


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Synthetic clinical note used to exercise the scanner end-to-end.
    note_path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "note_bundle.json"
    with open(note_path, "r") as f:
        note_data = json.load(f)

    agent = NoteScannerAgent()
    agent.draw(path="src/cipoc/agents/visualization/note_scanner.png")
    if isinstance(note_data, dict):
        note_data = [note_data]

    result = [agent.run(note).model_dump() for note in note_data]
    output_path = Path(__file__).resolve().parents[3] / "tests" / "test_outputs" / "scanner_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
