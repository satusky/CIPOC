"""Two-step histology extraction: a family step, then a code step, with the
ICD-O-3 block fallback. Real extractor graphs with scripted replies."""

from collections import Counter
import json
from threading import Lock
import unittest

from cipoc.agents.extractor import ExtractorAgent, ExtractorInput
from cipoc.agents.note_retriever import NoteRetrieverAgent, RetrieverInput
from cipoc.models import (
    CaseVariableResult, ClinicalNote, HistologySelection, NoteDigest, ValidatedVariableOutput,
    VariableGroupInfo, VariableInfo, VariableStatus,
)
from cipoc.tools import (
    build_histology_scope, load_scoping_dictionary, to_case_results, variable_group_prompt_json,
)
from cipoc.utils import CipocConfig

from tests.fake_orchestrator import SCOPING_DICTIONARY


SCOPING = load_scoping_dictionary(SCOPING_DICTIONARY)
# Item 522's pairs as the NAACCR dictionary holds them: every pair the fixture's
# site groups list, plus one on no site's list.
PAIRS = {
    pair: code.description
    for group in SCOPING.histology.values()
    for family in group.families.values()
    for pair, code in family.codes.items()
} | {"8100/3": "Trichilemmocarcinoma"}
BEHAVIORS = {"0": "Benign", "1": "Uncertain", "2": "In situ", "3": "Malignant"}


def histology_variables(primary_site="C504"):
    codes = {}
    for pair, description in sorted(PAIRS.items()):
        codes[pair[:4]] = f"{codes[pair[:4]]}; {description}" if pair[:4] in codes else description
    return [
        VariableInfo(item_id=522, name="Histologic Type ICD-O-3", length=4, valid_codes=codes,
                     histology_scope=build_histology_scope(SCOPING, primary_site, PAIRS)),
        VariableInfo(item_id=523, name="Behavior Code ICD-O-3", length=1, valid_codes=BEHAVIORS),
    ]


def evidence(**updates):
    return {"explanation": "Synthetic evidence.", "most_important_note": 1,
            "spans": [{"note_id": 1, "text": "Evidence"}], "presence_confidence": "high", **updates}


class ScriptedModel:
    """Replies by schema: family choices and code choices in order, and repairs
    by item ID. Records every call's messages."""

    def __init__(self, families=(), codes=(), repairs=None):
        self.families = list(families)
        self.codes = list(codes)
        self.repairs = repairs or {}
        self.calls = Counter()
        self.messages = []
        self.lock = Lock()

    def structured(self, schema, messages):
        with self.lock:
            self.messages.append((schema.__name__, [message.content for message in messages]))
            if schema.__name__ == "HistologyFamilyChoice":
                self.calls["family"] += 1
                return schema.model_validate(self.families.pop(0))
            if schema.__name__ == "HistologyCodeOutput":
                self.calls["code"] += 1
                return schema.model_validate(self.codes.pop(0))
            if schema.__name__ == "RelevantNoteIDs":
                self.calls["retrieve"] += 1
                return schema(note_ids=[1])
            text = messages[-1].content
            if "\nRepair context:\n" in text:
                item_id = json.loads(text.split("\nRepair context:\n", 1)[1])["variable"]["item_id"]
                self.calls[item_id] += 1
                return schema.model_validate({"item_id": item_id, **self.repairs[item_id]})
            raise AssertionError(f"Unexpected {schema.__name__} call")

    def prompts(self, schema_name):
        return [contents for name, contents in self.messages if name == schema_name]


def offered(contents):
    """The pairs a code-step prompt offered."""
    text = next(content for content in contents if content.startswith("Histology code/behavior pairs offered:\n"))
    return json.loads(text.split("\n", 1)[1])


class HistologyExtractionTests(unittest.TestCase):
    def agent(self, model):
        return ExtractorAgent(llm=model, config=CipocConfig({"llm": {
            "model": "offline", "api_key": "synthetic", "base_url": "https://example.invalid/v1",
        }}))

    def run_case(self, model, variables=None):
        variables = variables or histology_variables()
        result = self.agent(model).run(ExtractorInput(
            requested_variables=VariableGroupInfo(variables=variables, extract_as_group=True),
            notes=[ClinicalNote(note_id=1, date="2025-01-01", note_type="pathology",
                                content="Evidence in a synthetic note.")],
        ), progress=False)
        return {value.item_id: value for value in result.extracted_values.variables}, result

    def test_family_then_code_fills_both_items(self):
        model = ScriptedModel(
            families=[{"families": ["850"], "none_of_these": False, "reasoning": "Ductal carcinoma.",
                       "presence_confidence": "high"}],
            codes=[{"pair": "8500/3", **evidence()}],
        )
        values, _ = self.run_case(model)
        self.assertEqual((values[522].value, values[523].value), ("8500", "3"))
        self.assertTrue(values[522].is_valid and values[523].is_valid)
        self.assertEqual(values[522].histology, HistologySelection(
            families=["850"], blocks=[], reasoning="Ductal carcinoma.", listed=True))
        self.assertIsNone(values[523].histology)
        self.assertEqual(model.calls, {"family": 1, "code": 1})
        # The family step shows the list's terms; the code step the v25 descriptions.
        family_prompt = model.prompts("HistologyFamilyChoice")[0]
        self.assertTrue(any(content.startswith("Histology families for BREAST:\n") for content in family_prompt))
        self.assertEqual(set(offered(model.prompts("HistologyCodeOutput")[0])),
                         set(SCOPING.histology["breast"].families["850"].codes))
        self.assertEqual(offered(model.prompts("HistologyCodeOutput")[0])["8500/3"],
                         SCOPING.histology["breast"].families["850"].codes["8500/3"].description)

    def test_two_families_offer_the_union_of_their_pairs(self):
        model = ScriptedModel(
            families=[{"families": ["850", "852", "999"], "none_of_these": False, "reasoning": "Mixed.",
                       "presence_confidence": "medium"}],
            codes=[{"pair": "8522/3", **evidence()}],
        )
        values, _ = self.run_case(model)
        families = SCOPING.histology["breast"].families
        self.assertEqual(set(offered(model.prompts("HistologyCodeOutput")[0])),
                         set(families["850"].codes) | set(families["852"].codes))
        # The unknown key is dropped.
        self.assertEqual(values[522].histology.families, ["850", "852"])
        self.assertEqual((values[522].value, values[523].value), ("8522", "3"))

    def test_none_of_these_falls_back_to_blocks_and_an_unlisted_pair_is_recorded(self):
        model = ScriptedModel(
            families=[
                {"families": [], "none_of_these": True, "reasoning": "No breast family fits.",
                 "presence_confidence": "high"},
                {"families": ["809-811"], "none_of_these": False, "reasoning": "An adnexal carcinoma.",
                 "presence_confidence": "high"},
            ],
            codes=[{"pair": "8100/3", **evidence()}],
        )
        values, _ = self.run_case(model)
        self.assertEqual(model.calls, {"family": 2, "code": 1})
        block_prompt = model.prompts("HistologyFamilyChoice")[1]
        self.assertTrue(any(content.startswith("ICD-O-3 morphology blocks:\n") for content in block_prompt))
        self.assertEqual(set(offered(model.prompts("HistologyCodeOutput")[0])),
                         {pair for pair in PAIRS if 809 <= int(pair[:3]) <= 811})
        self.assertEqual((values[522].value, values[523].value), ("8100", "3"))
        self.assertEqual(values[522].histology, HistologySelection(
            families=[], blocks=["809-811"],
            reasoning="No breast family fits.\nAn adnexal carcinoma.", listed=False))

    def test_unknown_family_keys_count_as_none_of_these(self):
        model = ScriptedModel(
            families=[
                {"families": ["999"], "none_of_these": False, "reasoning": "Guess.", "presence_confidence": "low"},
                {"families": ["850-854"], "none_of_these": False, "reasoning": "Ductal.",
                 "presence_confidence": "high"},
            ],
            codes=[{"pair": "8500/3", **evidence()}],
        )
        values, _ = self.run_case(model)
        # The second pass ran over the blocks.
        self.assertEqual(model.calls, {"family": 2, "code": 1})
        self.assertEqual(values[522].histology.blocks, ["850-854"])
        self.assertEqual(values[522].histology.families, [])
        # The fallback's pair can still be on the site's list.
        self.assertTrue(values[522].histology.listed)

    def test_no_documented_histology_codes_nothing(self):
        for families in (
            [{"families": [], "none_of_these": False, "reasoning": "Not documented.", "presence_confidence": "high"}],
            [{"families": [], "none_of_these": True, "reasoning": "No family.", "presence_confidence": "high"},
             {"families": [], "none_of_these": True, "reasoning": "No block.", "presence_confidence": "high"}],
        ):
            with self.subTest(passes=len(families)):
                model = ScriptedModel(families=families)
                values, result = self.run_case(model)
                self.assertNotIn("code", model.calls)
                self.assertEqual((values[522].value, values[523].value), (None, None))
                self.assertTrue(values[522].is_valid and values[523].is_valid)
                self.assertIsNone(values[522].histology.listed)
                statuses = to_case_results(
                    VariableGroupInfo(variables=histology_variables()), result.extracted_values)
                self.assertEqual({statuses[522].status, statuses[523].status}, {VariableStatus.NOT_FOUND})

    def test_a_pair_that_was_not_offered_goes_to_repair(self):
        model = ScriptedModel(
            families=[{"families": ["850"], "none_of_these": False, "reasoning": "Ductal.",
                       "presence_confidence": "high"}],
            codes=[{"pair": "8140/3", **evidence()}],
            repairs={522: {"value": "8500", **evidence()}, 523: {"value": "3", **evidence()}},
        )
        values, _ = self.run_case(model)
        self.assertEqual((values[522].value, values[523].value), ("8500", "3"))
        self.assertTrue(values[522].is_valid)
        self.assertEqual(model.calls[522], 1)
        repairs = [
            json.loads(contents[-1].split("\nRepair context:\n", 1)[1])
            for contents in model.prompts("VariableOutput")
        ]
        histology = next(repair for repair in repairs if repair["variable"]["item_id"] == 522)
        self.assertIn("isn't one of the offered code/behavior pairs", " ".join(histology["validation_errors"]))
        # Repair shows the offered codes, not the full list or the scope.
        self.assertEqual(set(histology["variable"]["valid_codes"]),
                         {pair[:4] for pair in SCOPING.histology["breast"].families["850"].codes})
        self.assertNotIn("histology_scope", histology["variable"])
        self.assertTrue(values[522].histology.listed)

    def test_prompts_never_carry_the_scope_or_the_full_code_list(self):
        model = ScriptedModel(
            families=[{"families": ["850"], "none_of_these": False, "reasoning": "Ductal.",
                       "presence_confidence": "high"}],
            codes=[{"pair": "8140/3", **evidence()}],
            repairs={522: {"value": "8500", **evidence()}, 523: {"value": "3", **evidence()}},
        )
        variables = histology_variables()
        self.run_case(model, variables)
        NoteRetrieverAgent(llm=model, config=CipocConfig({"llm": {
            "model": "offline", "api_key": "synthetic", "base_url": "https://example.invalid/v1",
        }})).run(RetrieverInput(
            requested_variables=VariableGroupInfo(variables=variables, extract_as_group=True),
            available_digests={1: NoteDigest(note_id=1, note_type="pathology")},
        ), progress=False)
        # 8100/3 is on no site's list, so only the scope's block fallback holds it.
        outside, description = "8100", PAIRS["8100/3"]
        for name, contents in model.messages:
            if name == "HistologyCodeOutput":
                continue
            with self.subTest(schema=name):
                text = "\n".join(contents)
                self.assertNotIn("histology_scope", text)
                self.assertNotIn(description, text)
                self.assertNotIn(f'"{outside}"', text)
                self.assertLess(len(text), 20_000)

    def test_group_rendering_without_a_scope_is_unchanged(self):
        group = VariableGroupInfo(variables=[
            VariableInfo(item_id=410, length=1, valid_codes={"1": "Right", "2": "Left"}),
        ], extract_as_group=True)
        expected = group.model_dump(mode="json")
        for variable in expected["variables"]:
            del variable["histology_scope"]
        self.assertEqual(json.loads(variable_group_prompt_json(group)), expected)


class HistologyScopeBuilderTests(unittest.TestCase):
    def test_breast_scope_holds_its_families_and_every_block(self):
        scope = build_histology_scope(SCOPING, "C50.4", PAIRS)
        self.assertEqual((scope.site_group, scope.site_group_label), ("breast", "BREAST"))
        self.assertIn("8500/3", scope.family_codes["850"])
        self.assertTrue(scope.families["850"].startswith("DUCT CARCINOMA"))
        self.assertIn("9120/3", scope.family_codes["912"])
        self.assertTrue(scope.listed("8500/3"))
        self.assertFalse(scope.listed("8100/3"))
        self.assertEqual(
            {pair for codes in scope.block_codes.values() for pair in codes}, set(PAIRS))
        self.assertEqual(set(scope.blocks), set(scope.block_codes))

    def test_no_primary_site_uses_the_unknown_primary_group(self):
        scope = build_histology_scope(SCOPING, None, PAIRS)
        self.assertEqual(scope.site_group, "unknown")
        self.assertTrue(scope.families)


class HistologyArtifactTests(unittest.TestCase):
    def test_old_extractions_load_and_the_selection_round_trips(self):
        old = {"item_id": 522, "value": "8500", "explanation": "x", "most_important_note": 1,
               "spans": [{"note_id": 1, "text": "Evidence"}], "presence_confidence": "high",
               "is_valid": True, "validation_errors": [], "extraction_attempts": 1}
        self.assertIsNone(ValidatedVariableOutput.model_validate(old).histology)
        result = CaseVariableResult(
            item_id=522, status=VariableStatus.EXTRACTED, value="8500",
            extraction=ValidatedVariableOutput.model_validate({**old, "histology": {
                "families": ["850"], "reasoning": "Ductal.", "listed": True}}),
        )
        dumped = json.loads(result.model_dump_json())
        self.assertEqual(dumped["extraction"]["histology"],
                         {"families": ["850"], "blocks": [], "reasoning": "Ductal.", "listed": True})
        self.assertEqual(CaseVariableResult.model_validate(dumped), result)


if __name__ == "__main__":
    unittest.main()
