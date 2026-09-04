import json
import hashlib
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from operator import add
from typing import Any, Callable, Mapping
from typing_extensions import Annotated, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage, SystemMessage

from cipoc.llm import BaseAgentModel
from cipoc.tools import build_corpus_descriptors, build_corpus_digests, VariableValueValidator, build_variable_group, load_group_hierarchy, load_variable_groups, evaluate_note_filter, eligible_groups, pending_group, resolve_leftovers, derive_case_facts, not_found_results, to_case_results, build_report, resolve_site_key
from cipoc.utils import CipocConfig, ObservabilityCollector, run_graph_stream
from cipoc.utils.progress.events import ProgressEvent
from cipoc.models import (
    Case,
    CaseFacts,
    CaseReport,
    CaseVariableResult,
    ClinicalNote,
    ConfidenceLevel,
    NoteDigest,
    NoteCorpusDescriptors,
    NoteSelectionProvenance,
    NoteSelectionUnevaluatedCode,
    OrchestratorConfigFingerprint,
    OrchestratorRunCorpus,
    OrchestratorRunError,
    OrchestratorRunFailure,
    OrchestratorRunInfo,
    OrchestratorRunInputs,
    OrchestratorRunResult,
    RunObservability,
    TargetGroup,
    VariableGroupOutput,
    VariableInfo,
    VariableGroupInfo,
    VariableOutput,
    VariableStatus,
    ProcessedClinicalNote,
    CancerStatus,
    confidence_field,
)

from .base import BaseAgent
from .extractor import ExtractorAgent, ExtractorInput, ExtractorState
from .note_scanner import NoteScannerAgent, ScannerState
from .note_retriever import NoteRetrieverAgent, RetrieverInput

ORCHESTRATOR_SYSTEM_PROMPT = ""

def index_notes(note_corpus: dict[int | str, ProcessedClinicalNote]) -> dict[int | str, str | None]:
    return {note_id: processed_note.summary for note_id, processed_note in note_corpus.items()}

def dict_merge_reducer(left: dict, right: dict) -> dict:
    return {**left, **right}


class CaseState(BaseModel):
    case_facts: CaseFacts | None = Field(
        default=None,
        description="Data-dictionary scoping facts; derived during the run, absent until then.",
    )
    target_variables: list[TargetGroup] = Field(
        default_factory=list,
        description="Variable groups planned for extraction; set once at initialization.",
    )
    structured_data: dict[int, str] = Field(
        default_factory=dict,
        description="Caller-supplied coded values keyed by NAACCR item ID; seeded at init and never extracted.",
    )
    variable_results: Annotated[dict[int, CaseVariableResult], dict_merge_reducer] = Field(
        default_factory=dict,
        description="Per-variable orchestration results keyed by item ID; written concurrently by extraction branches.",
    )
    note_selection: Annotated[dict[str, NoteSelectionProvenance], dict_merge_reducer] = Field(
        default_factory=dict,
        description="Per-group note-selection provenance merged from extraction branches.",
    )
    fatal_blocker: str | None = Field(
        default=None,
        description="Reason no further extraction can be attempted for this case.",
    )
    report: CaseReport | None = Field(
        default=None,
        description="Review roll-up assembled at finalization; absent until then.",
    )
    note_corpus: Annotated[dict[int | str, ClinicalNote | ProcessedClinicalNote], dict_merge_reducer] = Field(
        default_factory=dict,
        description="Dictionary of processed clinical notes keyed by note ID."
    )
    note_digests: dict[int | str, NoteDigest] = Field(
        default_factory=dict,
        description="Dictionary of note digests keyed by note ID. Written once by "
        "characterize_corpus; the extract branch reads a copy under its own "
        "branch_note_digests channel, so nothing writes this concurrently.",
    )
    note_corpus_descriptors: NoteCorpusDescriptors | None = Field(
        default=None,
        description="Corpus-level characterization produced by the note scan.",
    )
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

    @property
    def outstanding_item_ids(self) -> set[int]:
        """Item IDs still awaiting a result — the graph's continue/stop signal."""
        return {
            item_id
            for item_id, result in self.variable_results.items()
            if result.status == VariableStatus.PENDING
        }

    def to_case(self) -> Case:
        """Assemble the validated durable snapshot from live graph state."""
        return Case(
            case_facts=self.case_facts,
            variable_results=dict(self.variable_results),
            note_selection=dict(self.note_selection),
            fatal_blocker=self.fatal_blocker,
            report=self.report,
        )


class ExtractorBranchInput(BaseModel):
    requested_variables: TargetGroup
    # Named distinctly from the parent CaseState channels: these are read-only
    # inputs seeded via Send, so keeping the names disjoint stops the subgraph
    # from echoing them back on fan-in (only variable_results should flow back).
    branch_note_corpus: dict[int | str, ProcessedClinicalNote]
    branch_note_digests: dict[int | str, NoteDigest]


class ExtractBranchState(ExtractorBranchInput):
    """Per-group state for the extract subgraph (retrieve_notes -> extract).

    ``variable_results`` and ``note_selection`` share parent ``CaseState`` keys
    and reducers, so each branch's outputs merge back into the case as the fan-out
    joins. The note inputs deliberately do not share parent channel names.
    """
    retrieved_note_ids: list[int | str] = Field(default_factory=list)
    variable_results: Annotated[dict[int, CaseVariableResult], dict_merge_reducer] = Field(
        default_factory=dict,
    )
    note_selection: Annotated[dict[str, NoteSelectionProvenance], dict_merge_reducer] = Field(
        default_factory=dict,
    )


class OrchestratorInput(BaseModel):
    """The only channel a caller seeds: raw notes keyed by note ID.

    Notes enter as ``ClinicalNote`` and are upgraded to ``ProcessedClinicalNote``
    in place during the scan. Everything else (targets, case facts, results) is
    derived during the run, so it is deliberately absent from the input contract.
    """
    note_corpus: dict[int | str, ClinicalNote] = Field(default_factory=dict)
    structured_data: dict[int, str] = Field(
        default_factory=dict,
        description="Optional known coded values keyed by NAACCR item ID; seeded as structured-data results, skipping extraction.",
    )


class OrchestratorOutput(BaseModel):
    """Exactly the channels ``CaseState.to_case()`` consumes to build the snapshot."""
    case_facts: CaseFacts | None = None
    target_variables: list[TargetGroup] = Field(default_factory=list)
    variable_results: dict[int, CaseVariableResult] = Field(default_factory=dict)
    note_selection: dict[str, NoteSelectionProvenance] = Field(default_factory=dict)
    fatal_blocker: str | None = None
    report: CaseReport | None = None


class OrchestratorAgent(BaseAgent):
    """Extracts coded variables from clinical notes."""
    _state = CaseState
    _input_schema = OrchestratorInput
    _output_schema = OrchestratorOutput

    def __init__(self, llm: BaseAgentModel | None = None, *, config: CipocConfig | None = None, **kwargs):
        self._value_validator = VariableValueValidator()
        super().__init__(agent_type="orchestrator", llm=llm, config=config, **kwargs)
        self._scanner = NoteScannerAgent(config=self._config)
        self._retriever = NoteRetrieverAgent(config=self._config)
        self._extractor = ExtractorAgent(config=self._config)
        variable_groups_path = self._config.documents().variable_groups_path
        self._variable_groups_path = variable_groups_path
        self._target_variables = load_variable_groups(variable_groups_path)
        self._target_group_hierarchy = load_group_hierarchy(variable_groups_path)
        # Config groups carry only item_id/name; the NAACCR dictionary supplies
        # metadata and the tissue-keyed dictionary supplies case-scoped codes.
        self._data_dictionary_path = self._config.documents().data_dictionary_path
        self._site_data_dictionary_path = self._config.documents().site_data_dictionary_path

    # --- Graph wiring (compiled once per instance) ---
    def _wire_graph(self, workflow: StateGraph) -> None:
        extract_branch = self._build_extract_branch()

        workflow.add_node("initialize", self.initialize)
        workflow.add_node("scan_notes", self.scan_notes, destinations=("note_branch",))
        workflow.add_node("note_branch", self.note_branch)
        workflow.add_node("characterize_corpus", self.characterize_corpus)
        workflow.add_node("check_state", self.check_state)
        workflow.add_node("plan_extraction", self.plan_extraction, destinations=("extract_branch", "check_state"))
        workflow.add_node("extract_branch", extract_branch)
        workflow.add_node("merge_and_update", self.merge_and_update)
        workflow.add_node("finalize_case", self.finalize_case)

        # Edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "scan_notes")
        workflow.add_edge("note_branch", "characterize_corpus")
        workflow.add_edge("characterize_corpus", "check_state")
        workflow.add_conditional_edges(
            "check_state",
            self.route_from_check,
            ["plan_extraction", "finalize_case"],
        )
        workflow.add_edge("extract_branch", "merge_and_update")
        workflow.add_edge("merge_and_update", "check_state")
        workflow.add_edge("finalize_case", END)

    def _build_extract_branch(self) -> CompiledStateGraph:
        """Per-group extract subgraph: deterministic hard filter + retriever soft
        filter (retrieve_notes) -> extractor (extract). Compiled once, fanned out
        from plan_extraction via Send."""
        branch = StateGraph(ExtractBranchState)
        branch.add_node("retrieve_notes", self.retrieve_notes)
        branch.add_node("extract", self.extract)
        branch.add_edge(START, "retrieve_notes")
        branch.add_edge("retrieve_notes", "extract")
        branch.add_edge("extract", END)
        return branch.compile()

    # --- Nodes (bound methods: (state) -> dict) ---
    # Initial nodes
    def initialize(self, state: CaseState) -> dict:
        """Seed the persona, the extraction plan, and one result per variable.

        Variables with a caller-supplied structured value start terminal
        (STRUCTURED_DATA) and are never sent to the extractor; the rest start
        PENDING. Any seeded value that is also a scoping fact is folded into
        case_facts up front so it can scope dependent groups even when no
        extraction ever runs for that group.
        """
        structured = state.structured_data or {}
        results: dict[int, CaseVariableResult] = {}
        for group in self._target_variables:
            for variable in group.variables:
                item_id = variable.item_id
                value = structured.get(item_id)
                if value is not None:
                    results[item_id] = CaseVariableResult(
                        item_id=item_id,
                        status=VariableStatus.STRUCTURED_DATA,
                        value=str(value),
                    )
                else:
                    results[item_id] = CaseVariableResult(item_id=item_id)
        return {
            "messages": [SystemMessage(ORCHESTRATOR_SYSTEM_PROMPT)],
            "target_variables": self._target_variables,
            "variable_results": results,
            "case_facts": derive_case_facts(state.case_facts, results),
        }

    def scan_notes(self, state: CaseState) -> Command:
        """Deploy note scanner subagents to characterize note corpus"""
        sends = [Send("note_branch", note) for note in state.note_corpus.values()]
        return Command(goto=sends)
    
    def note_branch(self, note: ClinicalNote):
        processed_note = self._scanner.run(note, progress=False)
        return {"note_corpus": {processed_note.note_id: processed_note}}
    
    def characterize_corpus(self, state: CaseState) -> dict:
        descriptors = build_corpus_descriptors(state.note_corpus)
        digests = build_corpus_digests(state.note_corpus)
        case_facts = state.case_facts
        if case_facts is None or case_facts.gross_primary_site is None:
            site_dictionary = {}
            site_dictionary_path = getattr(self, "_site_data_dictionary_path", None)
            if site_dictionary_path is not None:
                with open(site_dictionary_path, "r") as file:
                    site_dictionary = json.load(file)
            for status in ("current", "recent", "historical"):
                affected_tissues = (descriptors.affected_tissues or {}).get(
                    status, set()
                )
                if isinstance(affected_tissues, str):
                    affected_tissues = [affected_tissues]
                tissues = {
                    tissue.strip()
                    for tissue in affected_tissues
                    if tissue.strip()
                }
                if not tissues:
                    continue
                if site_dictionary:
                    resolved_sites = {
                        resolve_site_key(
                            CaseFacts(gross_primary_site=tissue), site_dictionary
                        )
                        for tissue in tissues
                    }
                    resolved_sites.discard(None)
                else:
                    resolved_sites = tissues
                if len(resolved_sites) == 1:
                    case_facts = (case_facts or CaseFacts()).model_copy(
                        update={"gross_primary_site": resolved_sites.pop()}
                    )
                break
        return {
            "note_corpus_descriptors": descriptors,
            "note_digests": digests,
            "case_facts": case_facts,
        }

    def check_state(self, state: CaseState) -> dict:
        """Loop hub (flow Step 4). Pure pass-through: the branch decision lives
        in ``route_from_check``. Kept as a node rather than an inline edge because
        both the initial characterization and every post-extraction merge route
        here, so the branch is declared once instead of at each predecessor.
        """
        return {}

    def route_from_check(
        self, state: CaseState
    ) -> Literal["plan_extraction", "finalize_case"]:
        """Continue while work remains and nothing fatal has stopped the case;
        otherwise finalize. ``plan_extraction`` owns flipping any leftover PENDING
        that can never be extracted (gated/blocked) to a terminal status, so this
        router can branch purely on the outstanding set without looping forever.
        """
        if state.fatal_blocker is not None or not state.outstanding_item_ids:
            return "finalize_case"
        return "plan_extraction"

    def _scope_group(self, group: TargetGroup, case_facts: CaseFacts | None) -> TargetGroup:
        """Fill each variable's data-dictionary metadata and site-scoped codes,
        preserving the group's gating/filter fields.

        ``build_variable_group`` returns a plain ``VariableGroupInfo`` (no gating),
        so its enriched variables are merged back onto the pending ``TargetGroup``
        by item ID; the original ordering is kept and any variable the dictionary
        does not know is left as-is rather than dropped.
        """
        enriched = build_variable_group(
            [variable.item_id for variable in group.variables],
            self._data_dictionary_path,
            case_facts=case_facts,
            site_data_dictionary_path=self._site_data_dictionary_path,
        )
        enriched_by_id = {variable.item_id: variable for variable in enriched.variables}
        return group.model_copy(
            update={
                "variables": [
                    enriched_by_id.get(variable.item_id, variable)
                    for variable in group.variables
                ]
            }
        )

    def plan_extraction(self, state: CaseState) -> Command:
        """Flow Step 5. Fan out every currently-eligible group to the extract
        branch; at the fixed point (nothing eligible) resolve the leftovers so
        check_state finalizes. Note-level filtering is intentionally NOT here — it
        is per-group and happens inside the branch.
        """
        ready = eligible_groups(
            state.target_variables,
            state.variable_results,
            state.note_corpus_descriptors,
            state.case_facts,
        )
        if ready:
            # Extract only each group's still-pending variables, so structured-data
            # seeds (or values coded on an earlier pass) are never re-extracted.
            # Enrich against the current case facts here (post-scan, and after any
            # earlier pass has coded scoping facts) so the extractor sees valid
            # codes, format, and case-scoped coding instructions.
            return Command(goto=[
                Send("extract_branch", ExtractBranchState(
                    requested_variables=self._scope_group(
                        pending_group(group, state.variable_results),
                        state.case_facts,
                    ),
                    branch_note_corpus=state.note_corpus,
                    branch_note_digests=state.note_digests,
                ))
                for group in ready
            ])
        return Command(
            goto="check_state",
            update={"variable_results": resolve_leftovers(
                state.target_variables,
                state.variable_results,
                state.note_corpus_descriptors,
                state.case_facts,
            )},
        )

    # --- Extract subgraph nodes ---
    @staticmethod
    def _retrieve_prep(
        state: ExtractBranchState,
    ) -> tuple[RetrieverInput | None, NoteSelectionProvenance]:
        """Build the retriever request and durable deterministic funnel record."""
        group = state.requested_variables
        if group.group_id is None:
            raise ValueError("A group reaching note retrieval must have a group_id.")

        candidate_note_ids: list[int | str] = []
        rejected_note_ids = {}
        unevaluated_checks: list[NoteSelectionUnevaluatedCode] = []

        for note in state.branch_note_corpus.values():
            evaluation = evaluate_note_filter(note, group.note_filter, anchor=None)
            if evaluation.passes:
                candidate_note_ids.append(note.note_id)
            else:
                rejected_note_ids[note.note_id] = evaluation.rejection_reasons
            for check in evaluation.unevaluated_checks:
                if check not in unevaluated_checks:
                    unevaluated_checks.append(check)

        # Unevaluated checks are properties of the configured funnel, so retain
        # them even when the incoming corpus itself is empty.
        if not state.branch_note_corpus and group.note_filter is not None:
            if group.note_filter.keywords:
                unevaluated_checks.append(
                    NoteSelectionUnevaluatedCode.KEYWORD_FILTER_DISABLED
                )
            if group.note_filter.within_days is not None:
                unevaluated_checks.append(
                    NoteSelectionUnevaluatedCode.TEMPORAL_ANCHOR_UNAVAILABLE
                )

        selection = NoteSelectionProvenance(
            group_id=group.group_id,
            requested_item_ids=[variable.item_id for variable in group.variables],
            candidate_note_ids=candidate_note_ids,
            rejected_note_ids=rejected_note_ids,
            unevaluated_checks=unevaluated_checks,
        )
        if not candidate_note_ids:
            return None, selection

        candidate_ids = set(candidate_note_ids)
        return RetrieverInput(
            requested_variables=group.to_variable_group(),
            available_digests={
                note_id: digest
                for note_id, digest in state.branch_note_digests.items()
                if note_id in candidate_ids
            },
        ), selection

    @staticmethod
    def _retrieve_result(
        request: RetrieverInput | None,
        selection: NoteSelectionProvenance,
        relevant_ids: list[int | str] | None,
    ) -> dict:
        """Restrict model output to offered IDs and finish the durable record."""
        offered_ids = set(request.available_digests) if request is not None else set()
        selected_ids = [
            note_id for note_id in (relevant_ids or []) if note_id in offered_ids
        ]
        discarded_ids = [
            note_id for note_id in (relevant_ids or []) if note_id not in offered_ids
        ]
        completed = selection.model_copy(
            update={
                "selected_note_ids": selected_ids,
                "discarded_note_ids": discarded_ids,
            }
        )
        return {
            "retrieved_note_ids": selected_ids,
            "note_selection": {f"group:{selection.group_id}": completed},
        }

    def retrieve_notes(self, state: ExtractBranchState) -> dict:
        """Narrow the corpus for one group through the two-stage selection funnel:
        the deterministic hard filter on the group's own NoteFilter, then the
        retriever soft filter judging relevance to the group's variables."""
        request, selection = self._retrieve_prep(state)
        if request is None:
            return self._retrieve_result(request, selection, None)
        # Soft filter: the retriever ranks the surviving digests for this group's
        # variables and returns None when nothing is plausibly relevant.
        relevant_ids = self._retriever.run(request, progress=False)
        return self._retrieve_result(request, selection, relevant_ids)

    def extract(self, state: ExtractBranchState) -> dict:
        """Extract the group's variables from the retrieved notes and fold the
        validated output into per-item orchestration results."""
        group = state.requested_variables.to_variable_group()
        if not state.retrieved_note_ids:
            # No relevant notes survived selection: nothing to read, so record a
            # clean miss for every requested variable instead of calling the LLM.
            return {"variable_results": not_found_results(
                group, "No relevant notes were selected for this variable."
            )}
        notes = [state.branch_note_corpus[note_id] for note_id in state.retrieved_note_ids]
        output = self._extractor.run(
            ExtractorInput(requested_variables=group, notes=notes),
            progress=False,
        )
        return {"variable_results": to_case_results(group, output.extracted_values)}

    def merge_and_update(self, state: CaseState) -> dict:
        """Flow Step 7. The extract branches have already merged their per-item
        results into ``variable_results`` (shared reducer); fold those newly coded
        values into ``case_facts`` so the next planning pass can scope dependent
        groups against real coded values. Returns nothing to write when no new
        fact was learned, leaving the existing facts untouched.
        """
        updated = derive_case_facts(state.case_facts, state.variable_results)
        if updated is state.case_facts:
            return {}
        return {"case_facts": updated}

    def finalize_case(self, state: CaseState) -> dict:
        """Flow Step 8. All variables are terminal (or a fatal blocker stopped the
        run); roll the results into a review report flagging errors, invalid
        extractions, and low-confidence values. ``to_case()`` then folds this into
        the durable ``Case`` snapshot returned to the caller."""
        return {"report": build_report(state.variable_results)}


    # --- Public API ---
    @staticmethod
    def _sha256_digest(path: Path | str | None) -> str | None:
        if path is None:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _retry_fingerprint(policy: Any) -> dict[str, Any]:
        values = policy._asdict() if hasattr(policy, "_asdict") else vars(policy)
        result = {}
        for name, value in values.items():
            if callable(value):
                module = getattr(value, "__module__", "")
                qualified_name = getattr(value, "__qualname__", repr(value))
                value = f"{module}.{qualified_name}" if module else qualified_name
            result[name] = value
        return result

    def _config_fingerprint(self) -> OrchestratorConfigFingerprint:
        components = {
            "orchestrator": self,
            "note_scanner": self._scanner,
            "note_retriever": self._retriever,
            "extractor": self._extractor,
        }
        agent_llm_config = {
            name: component._llm_config.model_dump(
                mode="json", exclude={"api_key", "tools"}
            )
            for name, component in components.items()
        }
        retry = {
            name: self._retry_fingerprint(component._retry_policy)
            for name, component in components.items()
        }

        prompt_dir = Path(__file__).resolve().parents[1] / "prompts"
        prompt_digests = {
            prompt_path.name: self._sha256_digest(prompt_path)
            for prompt_path in sorted(prompt_dir.glob("*.py"))
        }
        try:
            cipoc_version = version("cipoc")
        except PackageNotFoundError:
            cipoc_version = None

        return OrchestratorConfigFingerprint(
            agent_llm_config=agent_llm_config,
            retry=retry,
            max_extraction_attempts=ExtractorState.model_fields[
                "max_extraction_attempts"
            ].default,
            variable_groups_digest=self._sha256_digest(self._variable_groups_path),
            data_dictionary_digest=self._sha256_digest(self._data_dictionary_path),
            site_data_dictionary_digest=self._sha256_digest(
                self._site_data_dictionary_path
            ),
            prompt_digests=prompt_digests,
            cipoc_version=cipoc_version,
        )

    @staticmethod
    def _corpus_from_state(
        state: Mapping[str, Any] | None,
    ) -> OrchestratorRunCorpus | None:
        if state is None:
            return None
        return OrchestratorRunCorpus(
            note_corpus=state.get("note_corpus", {}),
            note_digests=state.get("note_digests", {}),
            note_corpus_descriptors=state.get("note_corpus_descriptors"),
        )

    def run(
        self,
        raw_notes: list[dict],
        structured_data: dict[int, str] | None = None,
        *,
        progress: bool = True,
        max_concurrency: int | None = None,
        capture_llm_content: bool = True,
        max_content_chars: int | None = None,
        pause_before_summary: bool = True,
        config: Mapping[str, Any] | None = None,
        event_observer: Callable[[ProgressEvent], None] | None = None,
    ) -> OrchestratorRunResult:
        """Extract the configured variable groups from ``raw_notes``.

        ``structured_data`` optionally supplies already-known coded values keyed
        by NAACCR item ID; those variables are seeded as structured-data results
        and skip extraction. Set ``progress`` to false to run without rendering
        the live progress display. ``max_concurrency`` controls LangGraph's
        parallel task limit. Prompt/response capture can be disabled independently
        from model metadata and usage collection. Returns the complete versioned
        run artifact; graph failures raise ``OrchestratorRunError`` with a partial
        failure artifact.
        """
        if not isinstance(raw_notes, list):
            raise TypeError("raw_notes must be a list.")
        if structured_data is not None and not isinstance(structured_data, Mapping):
            raise TypeError("structured_data must be a mapping or None.")
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean.")
        if not isinstance(capture_llm_content, bool):
            raise TypeError("capture_llm_content must be a boolean.")
        if not isinstance(pause_before_summary, bool):
            raise TypeError("pause_before_summary must be a boolean.")
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None.")
        if event_observer is not None and not callable(event_observer):
            raise TypeError("event_observer must be callable or None.")
        if max_concurrency is not None and (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
        ):
            raise TypeError("max_concurrency must be an integer or None.")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        if max_content_chars is not None and (
            isinstance(max_content_chars, bool)
            or not isinstance(max_content_chars, int)
        ):
            raise TypeError("max_content_chars must be an integer or None.")
        if max_content_chars is not None and max_content_chars < 0:
            raise ValueError("max_content_chars must be non-negative.")

        notes = [ClinicalNote.model_validate(note) for note in raw_notes]
        validated_input = OrchestratorInput(
            note_corpus={note.note_id: note for note in notes},
            structured_data={} if structured_data is None else structured_data,
        )
        graph_input = {
            "note_corpus": validated_input.note_corpus,
            "structured_data": validated_input.structured_data,
        }
        run_inputs = OrchestratorRunInputs(
            target_variables=self._target_variables,
            structured_data=validated_input.structured_data,
        )
        graph_config = dict(config or {})
        configured_concurrency = graph_config.get("max_concurrency")
        if max_concurrency is not None:
            graph_config["max_concurrency"] = max_concurrency
        elif configured_concurrency is not None and (
            isinstance(configured_concurrency, bool)
            or not isinstance(configured_concurrency, int)
            or configured_concurrency < 1
        ):
            raise ValueError("config max_concurrency must be a positive integer.")

        run_id = uuid4()
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        fingerprint = self._config_fingerprint()
        collector = ObservabilityCollector(
            capture_llm_content=capture_llm_content,
            max_content_chars=max_content_chars,
        )
        observed_config = collector.graph_config(graph_config)
        last_root_state: Mapping[str, Any] | None = None

        def observe(event: ProgressEvent) -> None:
            nonlocal last_root_state
            collector.observe(event)
            if event.kind == "values" and event.is_root:
                last_root_state = event.payload
            if event_observer is not None:
                event_observer(event)

        try:
            final_state = run_graph_stream(
                self._graph,
                graph_input,
                config=observed_config,
                subgraphs=True,
                progress=progress,
                description="Orchestrator",
                target_groups=self._target_variables,
                group_hierarchy=self._target_group_hierarchy,
                pause_before_summary=pause_before_summary,
                event_observer=observe,
            )
            finished_at = datetime.now(timezone.utc)
            full_state = CaseState.model_validate(final_state)
            corpus = self._corpus_from_state(final_state)
            if corpus is None:
                raise RuntimeError("Completed graph produced no corpus state.")
            return OrchestratorRunResult(
                run=OrchestratorRunInfo(
                    run_id=run_id,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=time.monotonic() - started_monotonic,
                    status="completed",
                    config_fingerprint=fingerprint,
                ),
                case=full_state.to_case(),
                inputs=run_inputs,
                corpus=corpus,
                observability=RunObservability.model_validate(collector.snapshot()),
            )
        except Exception as error:
            finished_at = datetime.now(timezone.utc)
            try:
                partial_corpus = self._corpus_from_state(last_root_state)
            except Exception:
                partial_corpus = None
            failure = OrchestratorRunFailure(
                run=OrchestratorRunInfo(
                    run_id=run_id,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=time.monotonic() - started_monotonic,
                    status="failed",
                    config_fingerprint=fingerprint,
                ),
                inputs=run_inputs,
                corpus=partial_corpus,
                observability=RunObservability.model_validate(collector.snapshot()),
                error=f"{type(error).__name__}: {error}",
            )
            raise OrchestratorRunError(failure) from error


if __name__ == "__main__":
    # End-to-end pipeline smoke run: scan -> characterize -> plan -> retrieve ->
    # extract -> finalize over the shared note bundle fixture.
    import argparse

    parser = argparse.ArgumentParser(description="Run the orchestrator end-to-end.")
    parser.add_argument(
        "--structured-data",
        default=None,
        help=(
            "Known coded values keyed by NAACCR item ID, either an inline JSON "
            "object or a path to a JSON file. Seeded as structured-data results, "
            "skipping extraction."
        ),
    )
    args = parser.parse_args()

    structured_data = None
    if args.structured_data is not None:
        raw = args.structured_data
        candidate = Path(raw)
        text = candidate.read_text() if candidate.is_file() else raw
        # JSON object keys are strings; item IDs are ints, so coerce the keys.
        structured_data = {int(k): str(v) for k, v in json.loads(text).items()}

    agent = OrchestratorAgent()
    # agent.draw(path="src/cipoc/agents/visualization/orchestrator.png")

    note_path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "note_bundle.json"
    with open(note_path, "r") as f:
        raw_notes = json.load(f)

    run_result = agent.run(raw_notes, structured_data=structured_data)
    result_path = Path(__file__).resolve().parents[3] / "tests" / "test_outputs" / "orchestrator_test.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(run_result.model_dump(mode="json"), f, indent=2)
    # print(run_result.model_dump_json(indent=2))
