"""Real extractor graphs with deterministic replies and production validation."""

from collections import Counter
import json
from threading import Event, Lock
import unittest
from unittest.mock import patch

from cipoc.agents.extractor import ExtractorAgent, ExtractorInput, ExtractorOutput
from cipoc.models import ClinicalNote, VariableGroupInfo, VariableInfo, VariableStatus
from cipoc.tools import to_case_results
from cipoc.utils import CipocConfig


def candidate(item_id, value, note_id=1, **updates):
    return {
        "item_id": item_id, "value": value, "explanation": "Synthetic evidence.",
        "most_important_note": note_id if value is not None else None,
        "spans": [{"note_id": note_id, "text": "Evidence"}] if value is not None else [],
        "presence_confidence": "high", **updates,
    }


class FakeModel:
    def __init__(self, responses, group_response=None):
        self.responses = responses
        self.group_response = group_response
        self.calls = Counter()
        self.lock = Lock()

    def structured(self, schema, messages):
        if schema.__name__ == "VariableGroupOutput":
            with self.lock:
                self.calls["group"] += 1
            return schema(variables=self.group_response)
        for message in reversed(messages):
            text = message.content
            if "\nRepair context:\n" in text:
                item_id = json.loads(text.split("\nRepair context:\n", 1)[1])["variable"]["item_id"]
                break
            if text.startswith("Variable to extract:\n"):
                item_id = json.loads(text.split("\n", 1)[1])["item_id"]
                break
        else:
            raise AssertionError("No requested variable in model messages")
        with self.lock:
            index = self.calls[item_id]
            self.calls[item_id] += 1
            responses = self.responses[item_id]
            response = responses[min(index, len(responses) - 1)]
        return schema.model_validate(response)


class ExtractorRegressionTests(unittest.TestCase):
    def agent(self, model):
        return ExtractorAgent(llm=model, config=CipocConfig({"llm": {
            "model": "offline", "api_key": "synthetic", "base_url": "https://example.invalid/v1",
        }}))

    def input(self, variables, *, grouped=False, note_id=1):
        return ExtractorInput(
            requested_variables=VariableGroupInfo(variables=variables, extract_as_group=grouped),
            notes=[ClinicalNote(note_id=note_id, date="2025-01-01", note_type="pathology", content="Evidence in a synthetic note.")],
        )

    def test_source_domains_pass_in_one_call_and_placeholder_is_repaired(self):
        variables = [
            VariableInfo(item_id=820, length=2, valid_codes={"01-89": "Count", "99": "Unknown"}),
            VariableInfo(item_id=756, length=3, valid_codes={"002-988": "Size", "999": "Unknown"}),
            VariableInfo(item_id=671, length=4, data_type="mixed", format="Alphanumeric Blank",
                         allowable_values="A000, A200-A990, B000, B200-B990, Blank", valid_codes={}),
            VariableInfo(item_id=676, length=2, allowable_values="00-90, 95-99",
                         valid_codes={code: "Description" for code in ("00", "01", "02", "..", "90", "95", "96", "97", "98", "99")}),
        ]
        model = FakeModel({
            820: [candidate(820, "03")], 756: [candidate(756, "010")],
            671: [candidate(671, "ABC"), candidate(671, "A550")],
            676: [candidate(676, ".."), candidate(676, "03")],
        })
        result = self.agent(model).run(self.input(variables), progress=False)
        self.assertTrue(all(value.is_valid for value in result.extracted_values.variables))
        self.assertEqual(model.calls, {820: 1, 756: 1, 671: 2, 676: 2})

    def test_wrong_item_id_cannot_overwrite_sibling_in_either_completion_order(self):
        variables = [VariableInfo(item_id=400, valid_codes={"C509": "Breast"}),
                     VariableInfo(item_id=410, valid_codes={"1": "Right", "9": "Unknown"})]
        for first in (400, 410):
            with self.subTest(first=first):
                model = FakeModel({400: [candidate(400, "C509")], 410: [candidate(400, "C509")]})
                agent = self.agent(model)
                first_done = Event()
                completion_order = []
                complete = agent.complete_variable

                def ordered_complete(state):
                    item_id = state.task.variable.item_id
                    if item_id != first and not first_done.wait(5):
                        raise AssertionError("Sibling completion deadline exceeded")
                    output = complete(state)
                    completion_order.append(item_id)
                    if item_id == first:
                        first_done.set()
                    return output

                with patch.object(agent, "complete_variable", ordered_complete):
                    agent._graph = agent._build_graph()
                    result = agent.run(self.input(variables), progress=False)
                successful, invalid = result.extracted_values.variables
                self.assertEqual(completion_order[0], first)
                self.assertEqual((successful.item_id, successful.value, successful.is_valid), (400, "C509", True))
                self.assertEqual((invalid.item_id, invalid.value, invalid.is_valid), (410, "C509", False))
                self.assertIn("Expected item ID 410, received 400.", invalid.validation_errors)
                self.assertEqual(model.calls, {400: 1, 410: 3})

    def test_duplicate_and_unexpected_completed_branch_ids_fail_merging(self):
        variable = VariableInfo(item_id=410, valid_codes={"1": "Right"})
        for corruption in ("duplicate", "unexpected"):
            with self.subTest(corruption=corruption):
                model = FakeModel({410: [candidate(410, "1")]})
                agent = self.agent(model)
                complete = agent.complete_variable

                def corrupt_complete(state):
                    result = complete(state)
                    if corruption == "duplicate":
                        result["variable_results"] *= 2
                    else:
                        result["variable_results"][0].item_id = 400
                    return result

                with patch.object(agent, "complete_variable", corrupt_complete):
                    agent._graph = agent._build_graph()
                    with self.assertRaisesRegex(ValueError, "Invalid completed branch IDs"):
                        agent.run(self.input([variable]), progress=False)

    def test_clean_null_is_not_found_after_one_call(self):
        variable = VariableInfo(item_id=410, valid_codes={"1": "Right"})
        model = FakeModel({410: [candidate(410, None)]})
        result = self.agent(model).run(self.input([variable]), progress=False)
        extracted = result.extracted_values.variables[0]
        self.assertTrue(extracted.is_valid)
        self.assertEqual(extracted.extraction_attempts, 1)
        self.assertEqual(model.calls, {410: 1})
        self.assertEqual(to_case_results(VariableGroupInfo(variables=[variable]), result.extracted_values)[410].status,
                         VariableStatus.NOT_FOUND)

    def test_malformed_null_answers_enter_repair(self):
        variable = VariableInfo(item_id=410, valid_codes={"1": "Right"})
        for updates in ({"item_id": 400}, {"most_important_note": 1}, {"spans": [{"note_id": 1, "text": "Evidence"}]}):
            with self.subTest(updates=updates):
                model = FakeModel({410: [{**candidate(410, None), **updates}, candidate(410, None)]})
                result = self.agent(model).run(self.input([variable]), progress=False)
                self.assertTrue(result.extracted_values.variables[0].is_valid)
                self.assertEqual(model.calls, {410: 2})

    def test_missing_duplicate_and_unexpected_group_candidates_require_repair(self):
        variables = [VariableInfo(item_id=item_id, valid_codes={"1": "Present"}) for item_id in (400, 410)]
        for group, expected in (
            ([candidate(400, None)], {"group": 1, 410: 1}),
            ([candidate(400, None), candidate(410, None), candidate(410, None)], {"group": 1, 410: 1}),
            ([candidate(400, None), candidate(410, None), candidate(999, None)], {"group": 1, 400: 1, 410: 1}),
        ):
            with self.subTest(group=group):
                model = FakeModel({400: [candidate(400, None)], 410: [candidate(410, None)]}, group_response=group)
                result = self.agent(model).run(self.input(variables, grouped=True), progress=False)
                self.assertTrue(all(output.is_valid for output in result.extracted_values.variables))
                self.assertEqual(model.calls, expected)

    def test_all_metadata_is_preflighted_before_any_model_calls(self):
        valid = VariableInfo(item_id=400, valid_codes={"C509": "Breast"})
        for fields in ({}, {"length": 3}, {"data_type": "text"}, {"allowable_values": "See manual"},
                       {"valid_codes": {"A200-B990": "Unsupported"}}):
            for grouped in (False, True):
                with self.subTest(fields=fields, grouped=grouped):
                    model = FakeModel({400: [candidate(400, None)], 410: [candidate(410, None)]})
                    with self.assertRaisesRegex(ValueError, "metadata for item 410"):
                        self.agent(model).run(self.input([valid, VariableInfo(item_id=410, **fields)], grouped=grouped), progress=False)
                    self.assertEqual(model.calls, {})

    def test_empty_and_duplicate_requests_fail_before_model_calls(self):
        variable = VariableInfo(item_id=400, valid_codes={"C509": "Breast"})
        for variables in ([], [variable, variable]):
            model = FakeModel({400: [candidate(400, "C509")]})
            with self.assertRaisesRegex(ValueError, "unique requested item IDs"):
                self.agent(model).run(self.input(variables), progress=False)
            self.assertEqual(model.calls, {})

    def test_string_primary_citations_round_trip_without_scalar_conversion(self):
        variable = VariableInfo(item_id=400, valid_codes={"C509": "Breast"})
        for note_id in (1, "note-A", "001", "1"):
            with self.subTest(note_id=note_id):
                model = FakeModel({400: [candidate(400, "C509", note_id=note_id)]})
                result = self.agent(model).run(self.input([variable], note_id=note_id), progress=False)
                reloaded = ExtractorOutput.model_validate_json(result.model_dump_json())
                extracted = reloaded.extracted_values.variables[0]
                self.assertTrue(extracted.is_valid, extracted.validation_errors)
                self.assertEqual(extracted.most_important_note, note_id)
                self.assertIs(type(extracted.most_important_note), type(note_id))
                self.assertEqual(model.calls, {400: 1})

    def test_primary_citations_use_canonical_equality_without_normalizing_strings(self):
        variable = VariableInfo(item_id=400, valid_codes={"C509": "Breast"})
        for offered, citation, valid in ((1, "1", True), ("1", 1, True), ("001", 1, False),
                                         ("note-A", "note-a", False), ("note-A", " note-A", False), (1, 999, False)):
            with self.subTest(offered=offered, citation=citation):
                model = FakeModel({400: [candidate(400, "C509", note_id=offered, most_important_note=citation)]})
                result = self.agent(model).run(self.input([variable], note_id=offered), progress=False)
                extracted = result.extracted_values.variables[0]
                self.assertEqual(extracted.is_valid, valid, extracted.validation_errors)
                self.assertEqual(model.calls[400], 1 if valid else 3)
                if not valid:
                    self.assertTrue(any("Primary citation" in error for error in extracted.validation_errors))


if __name__ == "__main__":
    unittest.main()
