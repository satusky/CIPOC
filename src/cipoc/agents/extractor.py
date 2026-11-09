import json
import re
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
    HistologyCodeOutput,
    HistologyFamilyChoice,
    HistologySelection,
    VariableGroupInfo,
    VariableGroupOutput,
    VariableInfo,
    VariableOutput,
    ValidatedVariableOutput,
    ValidatedVariableGroupOutput,
)
from cipoc.prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    EXTRACT_GROUP_VALUES_PROMPT,
    EXTRACT_VARIABLE_VALUE_PROMPT,
    REPAIR_VARIABLE_VALUE_PROMPT,
    HISTOLOGY_FAMILY_PROMPT,
    HISTOLOGY_BLOCK_PROMPT,
    HISTOLOGY_CODE_PROMPT,
)
from cipoc.tools import VariableValueValidator, build_variable_group, load_data_dictionary, load_scoping_dictionary, resolve_evidence_text, variable_group_prompt_json, variable_prompt_dict, variable_prompt_json
from cipoc.utils import CipocConfig, run_with_progress

from .base import BaseAgent

# The code step's pair fills Behavior (523) alongside the scoped Histology (522).
BEHAVIOR_ITEM = 523
_PAIR = re.compile(r"[0-9]{4}/[0-9]")


# Graph state
class ExtractorInput(BaseModel):
    requested_variables: VariableGroupInfo = Field(description="The target variable(s) to extract from the clinical notes.")
    notes: list[ClinicalNote] = Field(description="Corpus of relevant clinical notes for deriving coded value(s).")


class ExtractorOutput(BaseModel):
    extracted_values: ValidatedVariableGroupOutput | None = Field(default=None, description="The values for variable(s) extracted from the clinical notes, with per-variable validation verdicts.")


class ExtractorState(ExtractorInput, ExtractorOutput):
    messages: Annotated[list[AnyMessage], add_messages]
    max_extraction_attempts: int = Field(default=3, description="Maximum number of attempts to repair an invalid extraction.")
    variable_results: Annotated[list[ValidatedVariableOutput], add] = Field(default_factory=list)
    histology_families: HistologyFamilyChoice | None = Field(default=None, description="The histology family step's answer over the site group's families.")
    histology_blocks: HistologyFamilyChoice | None = Field(default=None, description="The fallback's answer over the ICD-O-3 blocks, when the site's families didn't fit.")


# Subgraphs
class VariableExtractionTask(BaseModel):
    variable: VariableInfo
    extraction_mode: Literal["group", "individual"]
    coding_instructions: str | None = None
    candidate: VariableOutput | None = None
    validation_errors: list[str] = Field(default_factory=list)
    extraction_attempts: int = 0
    is_valid: bool = False
    offered_values: dict[str, str] | None = Field(default=None, description="Values the histology code step offered for this variable. Validation holds the value to them, and repair shows them as the variable's codes.")


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
        variables = state.requested_variables.variables
        if not variables or len({variable.item_id for variable in variables}) != len(variables):
            raise ValueError("Extraction requires a nonempty group of unique requested item IDs.")
        for variable in variables:
            self._value_validator.preflight(variable)
        return {"messages": [SystemMessage(EXTRACTOR_SYSTEM_PROMPT)]}

    def load_notes(self, state: ExtractorState) -> dict:
        """Inject the caller-supplied notes into the conversation.

        Notes now arrive via ``ExtractorInput`` (selected upstream by the
        orchestrator's retrieve step); this node only serializes them into the
        message stream so the extraction prompts can see them.
        """
        note_string = "\n".join(note.model_dump_json() for note in state.notes)
        return {"messages": [HumanMessage("Clinical notes:\n" + note_string)]}

    def variables_to_extract(
        self, state: ExtractorState
    ) -> Literal["extract_group_values", "choose_histology_families"] | list[Send]:
        variables = state.requested_variables.variables

        if self._histology_variable(state) is not None:
            return "choose_histology_families"

        if len(variables) > 1 and state.requested_variables.extract_as_group:
            return "extract_group_values"

        return [self._individual_branch(state, variable) for variable in variables]

    def _individual_branch(self, state: ExtractorState, variable: VariableInfo) -> Send:
        return Send(
            "variable_branch",
            VariableBranchState(
                task=VariableExtractionTask(variable=variable, extraction_mode="individual"),
                notes=state.notes or [],
                messages=list(state.messages),
                max_extraction_attempts=state.max_extraction_attempts,
            ),
        )

    @staticmethod
    def _histology_variable(state: ExtractorState) -> VariableInfo | None:
        """The requested Histology (522) that carries a histology scope, if any."""
        return next(
            (variable for variable in state.requested_variables.variables if variable.histology_scope is not None),
            None,
        )

    # Two-step histology: a family step, then a code step over the chosen
    # families' pairs. "None of these" reruns the family step once over the
    # ICD-O-3 blocks, whose code step then offers every v25 pair in them.
    def choose_histology_families(self, state: ExtractorState) -> dict:
        """Choose the site group's families, or on the fallback pass the ICD-O-3 blocks."""
        scope = self._histology_variable(state).histology_scope
        # A trimmed test fixture can leave a group without families; it goes
        # straight to the blocks.
        block_mode = not scope.families or (
            state.histology_families is not None and state.histology_families.none_of_these
        )
        choices = scope.blocks if block_mode else scope.families
        heading = (
            "ICD-O-3 morphology blocks:\n"
            if block_mode
            else f"Histology families for {scope.site_group_label}:\n"
        )
        choice = self.agent.structured(
            HistologyFamilyChoice,
            state.messages
            + [
                HumanMessage(
                    "Variables to extract:\n"
                    + variable_group_prompt_json(state.requested_variables)
                ),
                HumanMessage(heading + json.dumps(choices)),
                HumanMessage(HISTOLOGY_BLOCK_PROMPT if block_mode else HISTOLOGY_FAMILY_PROMPT),
            ],
        )
        # Unknown keys are dropped. Keys that were all unknown mean nothing
        # offered fits, which is none of these.
        chosen = [key for key in dict.fromkeys(key.strip() for key in choice.families) if key in choices]
        choice = choice.model_copy(update={
            "families": chosen,
            "none_of_these": not chosen and (choice.none_of_these or bool(choice.families)),
        })
        return {"histology_blocks" if block_mode else "histology_families": choice}

    def route_histology_families(
        self, state: ExtractorState
    ) -> Literal["choose_histology_families", "choose_histology_code"] | list[Send]:
        answer = state.histology_blocks or state.histology_families
        if answer.families:
            return "choose_histology_code"
        if state.histology_blocks is None and answer.none_of_these:
            return "choose_histology_families"
        # The notes don't document a histology, or not even a block fits.
        return self._histology_sends(state, None, {})

    def choose_histology_code(self, state: ExtractorState) -> Command[Literal["variable_branch"]]:
        """Choose one code/behavior pair from the chosen families or blocks."""
        scope = self._histology_variable(state).histology_scope
        if state.histology_blocks is not None:
            source, chosen = scope.block_codes, state.histology_blocks.families
        else:
            source, chosen = scope.family_codes, state.histology_families.families
        offered = dict(sorted(
            (pair, description) for key in chosen for pair, description in source[key].items()
        ))
        output = self.agent.structured(
            HistologyCodeOutput,
            state.messages
            + [
                HumanMessage(
                    "Variables to extract:\n"
                    + variable_group_prompt_json(state.requested_variables)
                ),
                HumanMessage("Histology code/behavior pairs offered:\n" + json.dumps(offered)),
                HumanMessage(HISTOLOGY_CODE_PROMPT),
            ],
        )
        return Command(goto=self._histology_sends(state, output, offered))

    def _histology_sends(
        self, state: ExtractorState, output: HistologyCodeOutput | None, offered: dict[str, str]
    ) -> list[Send]:
        """Send 522, and 523 when requested, to validation with the code step's pair.

        With no code step (``output`` None) both get no value. Every other
        requested variable gets its usual individual branch.
        """
        histology = self._histology_variable(state)
        answer = state.histology_blocks or state.histology_families
        pair = (output.pair or "").strip() if output is not None else ""
        errors = (
            [f"The chosen pair {pair!r} isn't one of the offered code/behavior pairs."]
            if pair and pair not in offered
            else []
        )
        code, _, behavior = pair.partition("/") if _PAIR.fullmatch(pair) else (pair, "", "")
        values = {histology.item_id: code or None, BEHAVIOR_ITEM: behavior or None}
        paired = [
            variable for variable in state.requested_variables.variables
            if variable.item_id in (histology.item_id, BEHAVIOR_ITEM)
        ]
        # Each variable may only take a value the offered pairs give it: 522 a
        # code, with its pairs' descriptions, and 523 a behavior.
        behavior_codes = next(
            (variable.valid_codes for variable in paired if variable.item_id == BEHAVIOR_ITEM), None
        )
        behavior_codes = behavior_codes if isinstance(behavior_codes, dict) else {}
        offered_values: dict[int, dict[str, str]] = {histology.item_id: {}, BEHAVIOR_ITEM: {}}
        for offered_pair, description in offered.items():
            offered_code, _, offered_behavior = offered_pair.partition("/")
            codes = offered_values[histology.item_id]
            codes[offered_code] = f"{codes[offered_code]}; {description}" if offered_code in codes else description
            offered_values[BEHAVIOR_ITEM].setdefault(
                offered_behavior, str(behavior_codes.get(offered_behavior, ""))
            )

        candidates = {
            item_id: VariableOutput(
                item_id=item_id,
                value=value,
                explanation=output.explanation if output is not None else answer.reasoning,
                most_important_note=output.most_important_note if output is not None else None,
                spans=output.spans if output is not None else [],
                presence_confidence=output.presence_confidence if output is not None else answer.presence_confidence,
            )
            for item_id, value in values.items()
        }
        group_context = VariableGroupOutput(
            variables=[candidates[variable.item_id] for variable in paired]
        )
        sends = [
            Send(
                "variable_branch",
                VariableBranchState(
                    task=VariableExtractionTask(
                        variable=variable,
                        extraction_mode="group",
                        candidate=candidates[variable.item_id],
                        validation_errors=list(errors),
                        extraction_attempts=1,
                        offered_values=offered_values[variable.item_id] if offered else None,
                    ),
                    notes=state.notes or [],
                    messages=list(state.messages),
                    group_context=group_context,
                    max_extraction_attempts=state.max_extraction_attempts,
                ),
            )
            for variable in paired
        ]
        paired_ids = {variable.item_id for variable in paired}
        return sends + [
            self._individual_branch(state, variable)
            for variable in state.requested_variables.variables
            if variable.item_id not in paired_ids
        ]

    def extract_group_values(self, state: ExtractorState) -> Command[Literal["variable_branch"]]:
        group_output = self.agent.structured(
            VariableGroupOutput,
            state.messages
            + [
                HumanMessage(
                    "Variables to extract:\n"
                    + variable_group_prompt_json(state.requested_variables)
                ),
                HumanMessage(EXTRACT_GROUP_VALUES_PROMPT),
            ],
        )

        output_counts = Counter(output.item_id for output in group_output.variables)
        unexpected_ids = sorted(set(output_counts) - {
            variable.item_id for variable in state.requested_variables.variables
        })
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
                        ) + ([f"Group extraction returned unexpected item IDs: {unexpected_ids}."] if unexpected_ids else []),
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
        requested_ids = {variable.item_id for variable in state.requested_variables.variables}
        counts = Counter(result.item_id for result in state.variable_results)
        duplicates = sorted(item_id for item_id, count in counts.items() if count > 1)
        unexpected = sorted(set(counts) - requested_ids)
        if duplicates or unexpected:
            raise ValueError(
                f"Invalid completed branch IDs: duplicates={duplicates}, unexpected={unexpected}."
            )
        results_by_id = {result.item_id: result for result in state.variable_results}

        ordered_results = [
            results_by_id.get(variable.item_id)
            or ValidatedVariableOutput(
                item_id=variable.item_id,
                value=None,
                explanation="No extraction result was produced for this variable.",
                most_important_note=None,
                spans=[],
                presence_confidence=ConfidenceLevel.LOW,
                is_valid=False,
                validation_errors=["No branch result was produced for this variable."],
            )
            for variable in state.requested_variables.variables
        ]

        histology = self._histology_variable(state)
        answer = state.histology_blocks or state.histology_families
        if histology is not None and answer is not None:
            ordered_results = [
                result.model_copy(update={"histology": self._histology_selection(state, histology, results_by_id)})
                if result.item_id == histology.item_id
                else result
                for result in ordered_results
            ]

        return {"extracted_values": ValidatedVariableGroupOutput(variables=ordered_results)}

    @staticmethod
    def _histology_selection(
        state: ExtractorState,
        histology: VariableInfo,
        results_by_id: dict[int, ValidatedVariableOutput],
    ) -> HistologySelection:
        """Record how the two steps reached the 522 result.

        Whether the pair is on the site's list is read from the final values, so a
        repaired value is judged as finally coded.
        """
        scope = histology.histology_scope
        code = getattr(results_by_id.get(histology.item_id), "value", None)
        behavior = getattr(results_by_id.get(BEHAVIOR_ITEM), "value", None)
        if code is None:
            listed = None
        elif behavior is not None:
            listed = scope.listed(f"{code}/{behavior}")
        else:
            listed = any(
                pair.startswith(f"{code}/") for codes in scope.family_codes.values() for pair in codes
            )
        blocks = state.histology_blocks
        return HistologySelection(
            families=state.histology_families.families if blocks is None and state.histology_families else [],
            blocks=blocks.families if blocks is not None else [],
            reasoning="\n".join(
                answer.reasoning for answer in (state.histology_families, blocks) if answer is not None
            ),
            listed=listed,
        )


    # Variable branch
    def route_variable_entry(
        self, state: VariableBranchState
    ) -> Literal["extract_individual_value", "validate_extraction"]:
        if state.task.extraction_mode == "individual":
            return "extract_individual_value"
        return "validate_extraction"

    def extract_individual_value(self, state: VariableBranchState) -> dict:
        """Extract one variable from the clinical notes."""
        extracted_value = self.agent.structured(
            VariableOutput,
            state.messages
            + [
                HumanMessage("Variable to extract:\n" + variable_prompt_json(state.task.variable)),
                HumanMessage(EXTRACT_VARIABLE_VALUE_PROMPT),
            ],
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
        candidate = state.task.candidate
        if candidate is None:
            errors.append("No extraction candidate was returned.")
        else:
            citation = candidate.most_important_note
            if isinstance(citation, str) and not citation.strip():
                # Treat a blank optional citation as absent without mutating the raw/group response.
                candidate = candidate.model_copy(update={"most_important_note": None})
            errors.extend(
                self._value_validator.validate(
                    state.task.variable,
                    candidate,
                )
            )
            offered = state.task.offered_values
            if offered is not None and candidate.value is not None and candidate.value not in offered:
                errors.append(
                    f"Value {candidate.value!r} isn't one of the codes the histology code step offered."
                )
            if candidate.value is None and candidate.spans:
                errors.append("Supporting text spans must be empty when no value is returned.")
            elif candidate.value is not None and not candidate.spans:
                errors.append("No supporting text spans were returned.")
            citation = candidate.most_important_note
            if candidate.value is None and citation is not None:
                errors.append("Primary citation must be null when no value is returned.")
            spans = list(candidate.spans)
            for index, span in enumerate(spans, start=1):
                if not span.text.strip():
                    errors.append(f"Supporting text span {index} is empty.")
                    continue
                if "\n" in span.text or "\r" in span.text:
                    errors.append(
                        f"Supporting text span {index} contains newline characters."
                    )
                # Compare loosely on str form so an int note_id and its string
                # rendering match; the note this span cites must exist and must
                # actually contain the quoted text.
                cited_notes = [
                    note for note in state.notes if str(note.note_id) == str(span.note_id)
                ]
                if not cited_notes:
                    errors.append(
                        f"Supporting text span {index} cites note id '{span.note_id}', which "
                        "is not one of the provided notes; set note_id to the note_id the "
                        "text was copied from."
                    )
                elif not any(span.text in note.content for note in cited_notes):
                    source_text = (
                        resolve_evidence_text(span.text, cited_notes[0].content)
                        if len(cited_notes) == 1 else None
                    )
                    if source_text is None:
                        errors.append(
                            f"Supporting text span {index} is not verbatim text from note "
                            f"{span.note_id}."
                        )
                    else:
                        spans[index - 1] = span.model_copy(update={"text": source_text})
            if spans != candidate.spans:
                candidate = candidate.model_copy(update={"spans": spans})

            if citation is not None and not any(str(note.note_id) == str(citation) for note in state.notes):
                matching_notes = []
                if isinstance(citation, str) and not errors:
                    # Keep quote-as-ID recovery paired with its validated span after punctuation restoration.
                    matching_notes = [
                        note for note in state.notes
                        if any(citation in (original.text, span.text) and str(span.note_id) == str(note.note_id)
                               for original, span in zip(state.task.candidate.spans, candidate.spans))
                    ]
                if len(matching_notes) == 1:
                    candidate = candidate.model_copy(update={"most_important_note": matching_notes[0].note_id})
                else:
                    errors.append(f"Primary citation cites note id '{citation}', which is not one of the provided notes.")

        return {
            "task": state.task.model_copy(
                update={"candidate": candidate, "validation_errors": errors, "is_valid": not errors}
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
            "variable": variable_prompt_dict(state.task.variable, state.task.offered_values),
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
        repaired_value = self.agent.structured(
            VariableOutput,
            state.messages + [HumanMessage(REPAIR_VARIABLE_VALUE_PROMPT + "\nRepair context:\n" + json.dumps(repair_context))],
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
            most_important_note=None,
            spans=[],
            presence_confidence=ConfidenceLevel.LOW,
        )
        validated = ValidatedVariableOutput(
            **{**candidate.model_dump(), "item_id": state.task.variable.item_id},
            is_valid=state.task.is_valid,
            validation_errors=list(state.task.validation_errors),
            extraction_attempts=state.task.extraction_attempts,
        )
        return {"variable_results": [validated]}

    def _build_variable_branch(self):
        branch = StateGraph(VariableBranchState, output_schema=VariableBranchOutput)
        branch.add_node(
            "extract_individual_value", self.extract_individual_value, retry_policy=self.retry_policy
        )
        # validate_extraction is deterministic (data-dictionary lookup) — no retry.
        branch.add_node("validate_extraction", self.validate_extraction)
        branch.add_node(
            "repair_invalid_extraction", self.repair_invalid_extraction, retry_policy=self.retry_policy
        )
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
        workflow.add_node("load_notes", self.load_notes)
        workflow.add_node(
            "extract_group_values",
            self.extract_group_values,
            destinations=("variable_branch",),
            retry_policy=self.retry_policy,
        )
        workflow.add_node(
            "choose_histology_families",
            self.choose_histology_families,
            retry_policy=self.retry_policy,
        )
        workflow.add_node(
            "choose_histology_code",
            self.choose_histology_code,
            destinations=("variable_branch",),
            retry_policy=self.retry_policy,
        )
        # variable_branch is a subgraph; its own LLM nodes carry the retry policy.
        workflow.add_node("variable_branch", variable_branch)
        workflow.add_node("merge_variable_results", self.merge_variable_results)

        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "load_notes")
        workflow.add_conditional_edges(
            "load_notes",
            self.variables_to_extract,
            ["extract_group_values", "choose_histology_families", "variable_branch"],
        )
        workflow.add_conditional_edges(
            "choose_histology_families",
            self.route_histology_families,
            ["choose_histology_families", "choose_histology_code", "variable_branch"],
        )
        workflow.add_edge("variable_branch", "merge_variable_results")
        workflow.add_edge("merge_variable_results", END)

    # --- Public API ---
    def run(
        self,
        extractor_input: ExtractorInput | dict,
        *,
        progress: bool = True,
    ) -> ExtractorOutput:
        """Extract coded values for the requested variables from the given notes.

        Both the variables (already scoped upstream) and the relevant notes
        (already selected upstream) are supplied via ``ExtractorInput``; this
        agent no longer looks up the data dictionary or retrieves notes itself.
        """
        result = (
            run_with_progress(
                self._graph,
                extractor_input,
                subgraphs=True,
                description="Extractor",
                show_branches=True,
            )
            if progress
            else self._graph.invoke(extractor_input)
        )
        return ExtractorOutput(**result)


if __name__ == "__main__":
    agent = ExtractorAgent()
    # agent.draw(path="src/cipoc/agents/visualization/extractor.png")

    # Case facts matching tests/fixtures/note_bundle.json (left breast, dx 2025);
    # scopes valid codes from the scoping dictionary. The gross site is what note
    # characterization yields; primary_site stays unset because item 400 is one
    # of the variables being extracted here.
    facts = CaseFacts(gross_primary_site="breast", date_of_diagnosis="2025-02-24", sex="female")
    variable_group = build_variable_group(
        [400, 410, 522],  # Primary Site, Laterality, Histology
        data_dictionary=load_data_dictionary(agent._config.documents().data_dictionary_path),
        case_facts=facts,
        site_dictionary=load_scoping_dictionary(agent._config.documents().site_data_dictionary_path),
    )

    note_path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "note_bundle.json"
    with open(note_path, "r") as f:
        notes = [ClinicalNote(**note) for note in json.load(f)]

    result = agent.run(ExtractorInput(requested_variables=variable_group, notes=notes))
    # print(json.dumps(result.model_dump(), indent=2))

    output_path = Path(__file__).resolve().parents[3] / "tests" / "test_outputs" / "extractor_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)
