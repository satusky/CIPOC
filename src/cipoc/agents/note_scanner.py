import json

from typing_extensions import Annotated
from pydantic import BaseModel, Field, create_model

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage, SystemMessage

from cipoc.llm import BaseAgentModel
from cipoc.models import (
    ClinicalNote,
    ProcessedClinicalNote,
    CancerMention,
    CancerStatus,
    ConfidenceLevel,
    ConceptWithEvidence,
    CONCEPT_DESCRIPTIONS,
    DiagnosticCertainty,
    GrossSite,
    MentionBehavior,
    TextSpan,
    behavior_field,
    confidence_field,
    diagnostic_certainty_field,
)
from cipoc.tools import is_diagnosed_malignant
from cipoc.utils import CipocConfig, run_with_progress
from cipoc.prompts.note_scanner import (
    NOTE_SCANNER_SYSTEM_PROMPT,
    CONCEPT_DETECTION_PROMPT,
    NOTE_SUMMARY_PROMPT,
    CANCER_MENTIONS_PROMPT,
)

from .base import BaseAgent


# Graph state
class ScannerInput(BaseModel):
    note: ClinicalNote = Field(description="A clinical note object for a single patient visit.")


class ConceptFinding(BaseModel):
    """Required presence, confidence, and evidence for one tracked concept."""
    presence: bool = Field(description="Whether the concept is present in the note.")
    confidence: ConfidenceLevel = confidence_field()
    evidence: list[TextSpan] = Field(
        description="Verbatim supporting spans; empty when the concept is absent."
    )


def concept_findings_model(
    concept_descriptions: dict[str, str],
) -> type[BaseModel]:
    """Build a structured-output schema requiring every configured concept."""
    return create_model(
        "ConceptFindings",
        **{
            name: (ConceptFinding, Field(description=description))
            for name, description in concept_descriptions.items()
        },
    )


class CancerMentionFinding(CancerMention):
    """Scanner output for one mention, requiring the fields the saved model defaults for old artifacts
    and choosing the tissue from the vocabulary the saved model keeps as a string."""
    affected_tissue: GrossSite = Field(description="Primary organ or tissue affected, as its ICD-O-3 topography group. {'unspecified': the mention names no tissue; 'unknown_primary_site': the notes call the primary site unknown; 'colon': colorectal with no segment named; 'uterus': endometrium; 'female_genital_organ': fallopian tube, not 'ovary'; 'liver': intrahepatic bile duct, not 'biliary_tract'; 'parotid_gland': parotid, not 'major_salivary_gland'; 'other_endocrine_gland': pituitary; 'spinal_cord_and_cranial_nerves': acoustic neuroma; 'heart_mediastinum_and_pleura': pleura; 'hematopoietic_system': spleen; 'urinary_system': urethra}")
    behavior: MentionBehavior = behavior_field()
    diagnostic_certainty: DiagnosticCertainty = diagnostic_certainty_field()


class CancerMentions(BaseModel):
    mentions: list[CancerMentionFinding] = Field(description="List of neoplasm mentions, including masses, nodules, and lesions that could be neoplasms; empty if the note mentions none.")


class NoteSummary(BaseModel):
    summary: str = Field(description="Summary of clinical note.")
    keywords: list[str] = Field(description="Three to eight keywords associated with the note contents for search. Never empty.")


class ScannerOutput(BaseModel):
    concepts: dict[str, ConceptWithEvidence] | None = Field(default=None, description="Per-concept presence/evidence findings keyed by concept name.")
    cancer_status: set[CancerStatus] | None = Field(default=None, description="Distinct temporality statuses across the note's definite or probable malignant mentions. `None` when the note has none.")
    summary: str | None = Field(default=None, description="Summary of clinical note.")
    flags: list[str] | None = Field(default=None, description="Keywords associated with the note contents for search.")
    cancer_mentions: list[CancerMention] | None = Field(default=None, description="List of cancer mentions.")


class ScannerState(ScannerInput, ScannerOutput):
    messages: Annotated[list[AnyMessage], add_messages]


class NoteScannerAgent(BaseAgent):
    """Scans a single clinical note: summarizes it, detects concepts, then gates
    mention extraction on cancer or neoplasm presence."""
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

    def detect_concepts(self, state: ScannerState) -> dict:
        """Single LLM call detecting presence/evidence for every tracked concept."""
        prompt = CONCEPT_DETECTION_PROMPT.format(
            concept_list=json.dumps(CONCEPT_DESCRIPTIONS, indent=2, ensure_ascii=False)
        )
        findings_schema = concept_findings_model(CONCEPT_DESCRIPTIONS)
        response = self.agent.structured(
            findings_schema, state.messages + [HumanMessage(prompt)]
        )

        concepts: dict[str, ConceptWithEvidence] = {}
        for name in CONCEPT_DESCRIPTIONS:
            finding = getattr(response, name)
            concepts[name] = ConceptWithEvidence(
                presence=finding.presence,
                confidence=finding.confidence,
                evidence=[
                    TextSpan(note_id=state.note.note_id, text=span.text)
                    for span in finding.evidence
                ] if finding.presence else [],
            )

        # Every concept but neoplasm is cancer-specific by definition. Preserve
        # the broad cancer gate when treatment is the note's only cancer signal.
        # A neoplasm can be benign, so it never implies cancer.
        cancer = concepts["cancer"]
        if not cancer.presence:
            implied_by = next(
                (
                    finding for name, finding in concepts.items()
                    if name not in ("cancer", "neoplasm") and finding.presence
                ),
                None,
            )
            if implied_by is not None:
                concepts["cancer"] = implied_by.model_copy()

        return {"concepts": concepts}

    def summarize_note(self, state: ScannerState) -> dict:
        """Summarize a clinical note and tag it with search keywords."""
        response = self.agent.structured(
            NoteSummary, state.messages + [HumanMessage(NOTE_SUMMARY_PROMPT)]
        )
        return {"summary": response.summary, "flags": response.keywords}

    def get_cancer_mentions(self, state: ScannerState) -> dict:
        """Detail any mentions of cancer in a clinical note."""
        response = self.agent.structured(
            CancerMentions, state.messages + [HumanMessage(CANCER_MENTIONS_PROMPT)]
        )
        mentions = [CancerMention.model_validate(m.model_dump()) for m in response.mentions]
        # Roll the definite or probable malignant mentions' temporality up to a
        # note-level set (empty -> None). Every mention is still saved.
        return {
            "cancer_mentions": mentions,
            "cancer_status": {m.status for m in mentions if is_diagnosed_malignant(m)} or None,
        }

    @staticmethod
    def cancer_gate(state: ScannerState) -> str:
        """Gate function: detail mentions if the cancer or neoplasm concept is present, else stop."""
        concepts = state.concepts or {}
        present = any(
            concept is not None and concept.presence
            for concept in (concepts.get("cancer"), concepts.get("neoplasm"))
        )
        return "get_cancer_mentions" if present else END

    # --- Graph wiring (compiled once per instance) ---
    def _wire_graph(self, workflow: StateGraph) -> None:
        workflow.add_node("initialize", self.initialize)
        # All three scan nodes call the model; each retries its own request.
        workflow.add_node("detect_concepts", self.detect_concepts, retry_policy=self.retry_policy)
        workflow.add_node("summarize_note", self.summarize_note, retry_policy=self.retry_policy)
        workflow.add_node("get_cancer_mentions", self.get_cancer_mentions, retry_policy=self.retry_policy)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "summarize_note")
        workflow.add_edge("summarize_note", "detect_concepts")
        workflow.add_conditional_edges(
            "detect_concepts", self.cancer_gate, ["get_cancer_mentions", END]
        )
        workflow.add_edge("get_cancer_mentions", END)

    # --- Public API ---
    def run(
        self,
        notes: ClinicalNote | dict,
        *,
        progress: bool = True,
    ) -> ProcessedClinicalNote:
        """Run the scanner over a single note and return the enriched note."""
        if isinstance(notes, dict):
            notes = ClinicalNote(**notes)
        graph_input = {"note": notes}
        result = (
            run_with_progress(
                self._graph,
                graph_input,
                description="Note Scanner",
            )
            if progress
            else self._graph.invoke(graph_input)
        )
        return ProcessedClinicalNote(**notes.model_dump(), **result)


if __name__ == "__main__":
    from pathlib import Path

    # Synthetic clinical note used to exercise the scanner end-to-end.
    note_path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "note_bundle.json"
    with open(note_path, "r") as f:
        note_data = json.load(f)

    agent = NoteScannerAgent()
    # agent.draw(path="src/cipoc/agents/visualization/note_scanner.png")
    if isinstance(note_data, dict):
        note_data = [note_data]

    result = agent.run(note_data[2]).model_dump()
    # result = [agent.run(note).model_dump() for note in note_data]
    output_path = Path(__file__).resolve().parents[3] / "tests" / "test_outputs" / "scanner_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(json.dumps(result, indent=2))
