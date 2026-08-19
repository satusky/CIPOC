"""Deterministic stand-in for the orchestrator graph: same shape, no LLM calls.

The progress dashboard reads LangGraph task/values events, so its tests need a
graph whose *event stream* is indistinguishable from a live run. This module
builds one by keeping the real ``OrchestratorAgent`` — its topology, its
deterministic nodes, and the real ``variable_results`` roll-up — and swapping in
fake scanner/retriever/extractor subagents that are themselves compiled graphs
with the real node names and state models. Every task payload is therefore a
real Pydantic model of the same type a live run would emit.

``record_events`` streams such a graph and returns the raw
``(namespace, mode, payload)`` tuples for replay in tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

from cipoc.agents.extractor import (
    ExtractorInput,
    ExtractorOutput,
    ExtractorState,
    VariableBranchState,
    VariableBranchOutput,
    VariableExtractionTask,
)
from cipoc.agents.note_retriever import RetrieverInput, RetrieverOutput, RetrieverState
from cipoc.agents.note_scanner import ScannerInput, ScannerOutput, ScannerState
from cipoc.agents.orchestrator import OrchestratorAgent
from cipoc.models import (
    CancerMention,
    ClinicalNote,
    ConceptWithEvidence,
    ConfidenceLevel,
    ProcessedClinicalNote,
    TextSpan,
    ValidatedVariableGroupOutput,
    ValidatedVariableOutput,
    VariableOutput,
    build_concept_presence_dict,
)
from cipoc.tools import load_group_hierarchy, load_variable_groups
from cipoc.utils.utils import CipocConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTE_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "note_bundle.json"
VARIABLE_GROUPS = REPO_ROOT / "config" / "variable_groups.json"


def load_notes() -> list[ClinicalNote]:
    with open(NOTE_BUNDLE, "r") as f:
        return [ClinicalNote(**note) for note in json.load(f)]


@dataclass
class Outcome:
    """What the fake extractor should produce for one variable."""

    value: str | None = "1"
    confidence: ConfidenceLevel = ConfidenceLevel.MAX
    # Number of validation failures before the value is accepted. Each failure
    # costs one repair round, so ``repairs=2`` walks validate -> repair ->
    # validate -> repair -> validate exactly like a real bounded repair loop.
    repairs: int = 0
    # True when the repair budget is exhausted while still invalid (-> ERROR).
    exhausted: bool = False


@dataclass
class Script:
    """Everything the fakes need to behave deterministically.

    ``concepts`` decides which corpus gates open, ``retrieved`` how many notes
    each group's retriever keeps, and ``outcomes`` the per-variable extraction
    result. Anything unlisted falls back to the defaults, which produce a clean
    fully-coded case.
    """

    concepts: dict[str, bool] = field(
        default_factory=lambda: {
            "cancer": True,
            "metastasis": True,
            "surgery": True,
            "chemotherapy": True,
            "radiation": True,
            "lymph_nodes_removed": True,
        }
    )
    outcomes: dict[int, Outcome] = field(default_factory=dict)
    retrieved: dict[str, int] = field(default_factory=dict)
    default_outcome: Outcome = field(default_factory=Outcome)
    delay: float = 0.0

    def outcome(self, item_id: int) -> Outcome:
        return self.outcomes.get(item_id, self.default_outcome)

    def pause(self, scale: float = 1.0) -> None:
        if self.delay:
            time.sleep(self.delay * scale)


# --- Fake subagents (compiled graphs mirroring the real node names) ---


class FakeNoteScanner:
    """Scanner shape: initialize -> summarize_note -> detect_concepts -> gate."""

    def __init__(self, script: Script):
        self._script = script
        graph = StateGraph(ScannerState, input_schema=ScannerInput, output_schema=ScannerOutput)
        graph.add_node("initialize", self.initialize)
        graph.add_node("summarize_note", self.summarize_note)
        graph.add_node("detect_concepts", self.detect_concepts)
        graph.add_node("get_cancer_mentions", self.get_cancer_mentions)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "summarize_note")
        graph.add_edge("summarize_note", "detect_concepts")
        graph.add_conditional_edges(
            "detect_concepts", self.cancer_gate, ["get_cancer_mentions", END]
        )
        graph.add_edge("get_cancer_mentions", END)
        self._graph = graph.compile()

    def initialize(self, state: ScannerState) -> dict:
        return {}

    def summarize_note(self, state: ScannerState) -> dict:
        self._script.pause()
        return {
            "summary": f"{state.note.note_type} recorded {state.note.date}.",
            "flags": [
                "metasta",
                "therapy",
                "lymph node",
                state.note.note_type.split()[0].lower(),
            ],
        }

    def detect_concepts(self, state: ScannerState) -> dict:
        self._script.pause()
        concepts = build_concept_presence_dict()
        for name, present in self._script.concepts.items():
            if name in concepts:
                concepts[name] = ConceptWithEvidence(
                    presence=present, confidence=ConfidenceLevel.HIGH, evidence=None
                )
        return {"concepts": concepts}

    @staticmethod
    def cancer_gate(state: ScannerState) -> str:
        cancer = (state.concepts or {}).get("cancer")
        return "get_cancer_mentions" if cancer and cancer.presence else END

    def get_cancer_mentions(self, state: ScannerState) -> dict:
        self._script.pause()
        mention = CancerMention(
            presence=True,
            confidence=ConfidenceLevel.HIGH,
            evidence=[TextSpan(note_id=state.note.note_id, text=state.note.content)],
            status="current",
            affected_tissue="breast",
            metastasis=self._script.concepts.get("metastasis", False),
        )
        return {"cancer_mentions": [mention], "cancer_status": {"current"}}

    def run(self, note: ClinicalNote, *, progress: bool = True) -> ProcessedClinicalNote:
        result = self._graph.invoke({"note": note})
        return ProcessedClinicalNote(**note.model_dump(), **result)


class FakeNoteRetriever:
    """Retriever shape: initialize -> identify_relevant_notes."""

    def __init__(self, script: Script):
        self._script = script
        graph = StateGraph(
            RetrieverState, input_schema=RetrieverInput, output_schema=RetrieverOutput
        )
        graph.add_node("initialize", self.initialize)
        graph.add_node("identify_relevant_notes", self.identify_relevant_notes)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "identify_relevant_notes")
        graph.add_edge("identify_relevant_notes", END)
        self._graph = graph.compile()

    def initialize(self, state: RetrieverState) -> dict:
        return {}

    def identify_relevant_notes(self, state: RetrieverState) -> dict:
        self._script.pause()
        note_ids = sorted(state.available_digests)
        keep = self._script.retrieved.get(state.requested_variables.group_id or "", len(note_ids))
        selected = note_ids[:keep]
        return {"relevant_note_ids": selected or None}

    def run(self, retriever_input: Any, *, progress: bool = True) -> list[int] | None:
        return self._graph.invoke(retriever_input)["relevant_note_ids"]


class FakeExtractor:
    """Extractor shape, including the per-variable branch and its repair loop."""

    def __init__(self, script: Script):
        self._script = script
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

        graph = StateGraph(
            ExtractorState, input_schema=ExtractorInput, output_schema=ExtractorOutput
        )
        graph.add_node("initialize", self.initialize)
        graph.add_node("load_notes", self.load_notes)
        graph.add_node(
            "extract_group_values", self.extract_group_values, destinations=("variable_branch",)
        )
        graph.add_node("variable_branch", branch.compile())
        graph.add_node("merge_variable_results", self.merge_variable_results)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "load_notes")
        graph.add_conditional_edges(
            "load_notes",
            self.variables_to_extract,
            ["extract_group_values", "variable_branch"],
        )
        graph.add_edge("variable_branch", "merge_variable_results")
        graph.add_edge("merge_variable_results", END)
        self._graph = graph.compile()

    # Shared nodes
    def initialize(self, state: ExtractorState) -> dict:
        return {}

    def load_notes(self, state: ExtractorState) -> dict:
        return {}

    def variables_to_extract(self, state: ExtractorState):
        variables = state.requested_variables.variables
        if len(variables) > 1 and state.requested_variables.extract_as_group:
            return "extract_group_values"
        return [
            Send(
                "variable_branch",
                VariableBranchState(
                    task=VariableExtractionTask(variable=variable, extraction_mode="individual"),
                    notes=state.notes or [],
                    messages=[],
                    max_extraction_attempts=state.max_extraction_attempts,
                ),
            )
            for variable in variables
        ]

    def extract_group_values(self, state: ExtractorState) -> Command:
        self._script.pause(2)
        return Command(
            goto=[
                Send(
                    "variable_branch",
                    VariableBranchState(
                        task=VariableExtractionTask(
                            variable=variable,
                            extraction_mode="group",
                            candidate=self._candidate(variable.item_id, state.notes),
                            extraction_attempts=1,
                        ),
                        notes=state.notes or [],
                        messages=[],
                        max_extraction_attempts=state.max_extraction_attempts,
                    ),
                )
                for variable in state.requested_variables.variables
            ]
        )

    def merge_variable_results(self, state: ExtractorState) -> dict:
        by_id = {result.item_id: result for result in state.variable_results}
        ordered = [
            by_id[variable.item_id]
            for variable in state.requested_variables.variables
            if variable.item_id in by_id
        ]
        return {"extracted_values": ValidatedVariableGroupOutput(variables=ordered)}

    # Variable branch
    def _candidate(self, item_id: int, notes: list[ClinicalNote]) -> VariableOutput:
        outcome = self._script.outcome(item_id)
        note = notes[0] if notes else None
        spans = (
            [TextSpan(note_id=note.note_id, text=note.content.splitlines()[0])]
            if outcome.value is not None and note is not None
            else []
        )
        return VariableOutput(
            item_id=item_id,
            value=outcome.value,
            explanation=f"Scripted outcome for item {item_id}.",
            most_important_note=note.note_id if note is not None else None,
            spans=spans,
            presence_confidence=outcome.confidence,
        )

    def route_variable_entry(self, state: VariableBranchState) -> str:
        if state.task.extraction_mode == "individual":
            return "extract_individual_value"
        return "validate_extraction"

    def extract_individual_value(self, state: VariableBranchState) -> dict:
        self._script.pause()
        return {
            "task": state.task.model_copy(
                update={
                    "candidate": self._candidate(state.task.variable.item_id, state.notes),
                    "validation_errors": [],
                    "extraction_attempts": state.task.extraction_attempts + 1,
                    "is_valid": False,
                }
            )
        }

    def validate_extraction(self, state: VariableBranchState) -> dict:
        outcome = self._script.outcome(state.task.variable.item_id)
        failing = outcome.exhausted or state.task.extraction_attempts <= outcome.repairs
        errors = [f"Value {state.task.candidate.value!r} is not an allowable code."] if failing else []
        return {
            "task": state.task.model_copy(
                update={"validation_errors": errors, "is_valid": not errors}
            )
        }

    def route_after_validation(self, state: VariableBranchState) -> str:
        if state.task.is_valid or state.task.extraction_attempts >= state.max_extraction_attempts:
            return "complete_variable"
        return "repair_invalid_extraction"

    def repair_invalid_extraction(self, state: VariableBranchState) -> dict:
        self._script.pause()
        return {
            "task": state.task.model_copy(
                update={
                    "candidate": self._candidate(state.task.variable.item_id, state.notes),
                    "validation_errors": [],
                    "extraction_attempts": state.task.extraction_attempts + 1,
                    "is_valid": False,
                }
            )
        }

    def complete_variable(self, state: VariableBranchState) -> dict:
        candidate = state.task.candidate
        return {
            "variable_results": [
                ValidatedVariableOutput(
                    **candidate.model_dump(),
                    is_valid=state.task.is_valid,
                    validation_errors=list(state.task.validation_errors),
                    extraction_attempts=state.task.extraction_attempts,
                )
            ]
        }

    def run(self, extractor_input: Any, *, progress: bool = True) -> ExtractorOutput:
        return ExtractorOutput(**self._graph.invoke(extractor_input))


# --- The orchestrator itself ---


def build_fake_orchestrator(script: Script | None = None) -> OrchestratorAgent:
    """The real orchestrator with fake subagents and no LLM/config dependency.

    ``_scope_group`` is stubbed to a pass-through so the 2 MB data dictionary is
    never read: it only fills variable metadata the dashboard does not display,
    and skipping it keeps the fixture fast and hermetic.
    """
    script = script or Script()
    agent = object.__new__(OrchestratorAgent)
    agent._config = CipocConfig({"documents": {"variable_groups_path": str(VARIABLE_GROUPS)}})
    agent._value_validator = None
    agent._scanner = FakeNoteScanner(script)
    agent._retriever = FakeNoteRetriever(script)
    agent._extractor = FakeExtractor(script)
    agent._target_variables = load_variable_groups(VARIABLE_GROUPS)
    agent._target_group_hierarchy = load_group_hierarchy(VARIABLE_GROUPS)
    agent._data_dictionary_path = None
    agent._rule_store = None
    agent._scope_group = lambda group, case_facts: group  # type: ignore[method-assign]
    agent._graph = agent._build_graph()
    return agent


def graph_input(notes: list[ClinicalNote] | None = None, **extra) -> dict:
    notes = notes if notes is not None else load_notes()
    return {"note_corpus": {note.note_id: note for note in notes}, **extra}


def record_events(
    script: Script | None = None,
    notes: list[ClinicalNote] | None = None,
    **extra,
) -> list[tuple[tuple[str, ...], str, Any]]:
    """Stream a fake run and return its raw ``(namespace, mode, payload)`` tuples."""
    agent = build_fake_orchestrator(script)
    return list(
        agent._graph.stream(
            graph_input(notes, **extra),
            stream_mode=["values", "tasks"],
            subgraphs=True,
        )
    )


if __name__ == "__main__":
    events = record_events()
    modes = {}
    for namespace, mode, payload in events:
        modes[mode] = modes.get(mode, 0) + 1
    print(f"{len(events)} events: {modes}")
    depths = sorted({len(ns) for ns, _, _ in events})
    print("namespace depths:", depths)
    seen: set[str] = set()
    for namespace, mode, payload in events:
        if mode == "tasks":
            key = f"{'/'.join(part.split(':')[0] for part in namespace)}|{payload['name']}"
            if key not in seen:
                seen.add(key)
                print(" ", key)
