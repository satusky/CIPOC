"""Schema discriminators: extracted only where the case's candidate schemas differ by one."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cipoc.agents.orchestrator import OrchestratorAgent, run_recursion_limit
from cipoc.models import (
    CaseFacts,
    CaseVariableResult,
    ConfidenceLevel,
    OrchestratorRunResult,
    TargetGroup,
    VariableInfo,
    VariableOutput,
    VariableStatus,
)
from cipoc.tools import (
    VariableValueValidator,
    build_variable_group,
    derive_case_facts,
    discriminator_choice,
    discriminator_not_needed_reason,
    discriminator_results,
    discriminator_variable,
    discriminators_separating,
    load_group_hierarchy,
    load_scoping_dictionary,
    load_stages,
    load_variable_groups,
    matching_schemas,
    parse_scoping_dictionary,
    scoped_codes,
    scoped_item_codes,
    with_discriminator_variables,
)

from tests.fake_orchestrator import (
    SCOPING_DICTIONARY,
    Outcome,
    Script,
    build_fake_orchestrator,
    histology_pair_codes,
    load_notes,
    primary_site_codes,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DISCRIMINATORS = CONFIG_DIR / "variable_groups_+discriminators.json"
DISCRIMINATOR_ITEMS = (3926, 3927, 2879)


def _facts(primary_site, histology, date_of_diagnosis, **discriminators):
    return CaseFacts(
        primary_site=primary_site, histology=histology, behavior="3",
        date_of_diagnosis=date_of_diagnosis, **discriminators,
    )


HEMERETIC_OR_LYMPHOMA = _facts("C504", "9591", "20200615")
BREAST = _facts("C504", "8500", "20200615")
SOFT_TISSUE = _facts("C493", "8800", "20220615")
PERITONEUM = _facts("C481", "8441", "20120615")


class DiscriminatorChoiceTests(unittest.TestCase):
    """Whether the candidate schemas differ by a discriminator, from the fixture."""

    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)

    def needed(self, facts):
        return {
            item_id: choice.discriminator.table if choice.discriminator else None
            for item_id in DISCRIMINATOR_ITEMS
            if (choice := discriminator_choice(self.scoping, item_id, facts)).needed
        }

    def test_hemeretic_and_lymphoma_differ_by_schema_discriminator_1_for_9591(self):
        choice = discriminator_choice(self.scoping, 3926, HEMERETIC_OR_LYMPHOMA)

        self.assertEqual(choice.schemas, ["eod/hemeretic", "eod/lymphoma"])
        self.assertEqual(self.needed(HEMERETIC_OR_LYMPHOMA), {3926: "schema_discriminator_1_63995"})

    def test_a_breast_carcinoma_needs_no_discriminator(self):
        self.assertEqual(self.needed(BREAST), {})
        choice = discriminator_choice(self.scoping, 3926, BREAST)
        self.assertEqual(
            discriminator_not_needed_reason(choice, "Schema Discriminator 1 (3926)"),
            "Only eod/breast matches the case, so Schema Discriminator 1 (3926) isn't needed.",
        )

    def test_the_three_soft_tissue_schemas_differ_by_schema_discriminator_2(self):
        choice = discriminator_choice(self.scoping, 3927, SOFT_TISSUE)

        self.assertEqual(choice.schemas, [
            "eod/soft_tissue_abdomen_thoracic", "eod/soft_tissue_other", "eod/soft_tissue_trunk_extremities",
        ])
        self.assertEqual(self.needed(SOFT_TISSUE), {3927: "schema_discriminator_2_15990"})

    def test_the_peritoneum_schemas_differ_by_cs_site_specific_factor_25(self):
        choice = discriminator_choice(self.scoping, 2879, PERITONEUM)

        self.assertEqual(choice.schemas, ["cs/peritoneum", "cs/peritoneum_female_gen"])
        self.assertEqual(self.needed(PERITONEUM), {2879: "ssf25_sqp"})

    def test_the_year_picks_which_discriminators_can_apply(self):
        # CS and TNM schemas never read Schema Discriminator 1 or 2, and EOD
        # schemas never read CS Site-Specific Factor 25.
        for facts in (
            _facts("C504", "9591", "20120615"), _facts("C493", "8800", "20120615"), PERITONEUM,
        ):
            with self.subTest(facts=facts):
                for item_id in (3926, 3927):
                    self.assertFalse(discriminator_choice(self.scoping, item_id, facts).needed)
        for facts in (HEMERETIC_OR_LYMPHOMA, SOFT_TISSUE, _facts("C481", "8441", "20200615")):
            with self.subTest(facts=facts):
                self.assertFalse(discriminator_choice(self.scoping, 2879, facts).needed)

    def test_candidates_that_each_leave_it_open_dont_differ(self):
        # Without a histology, hemeretic and lymphoma also match rows that don't
        # read Schema Discriminator 1, so its value couldn't rule either out.
        facts = CaseFacts(primary_site="C504", date_of_diagnosis="20200615")
        choice = discriminator_choice(self.scoping, 3926, facts)

        self.assertFalse(choice.needed)
        self.assertIn("eod/lymphoma", choice.schemas)
        reason = discriminator_not_needed_reason(choice, "Schema Discriminator 1 (3926)")
        self.assertTrue(reason.startswith("The schemas the case matches (eod/breast, "))
        self.assertTrue(reason.endswith("don't differ by Schema Discriminator 1 (3926), so it isn't needed."))

    def test_no_candidate_needs_no_discriminator(self):
        choice = discriminator_choice(self.scoping, 3926, _facts("C504", "9591", "19990615"))

        self.assertEqual(choice.schemas, [])
        self.assertEqual(
            discriminator_not_needed_reason(choice, "Schema Discriminator 1 (3926)"),
            "No staging schema matches the case, so Schema Discriminator 1 (3926) isn't needed.",
        )

    def test_the_separating_discriminators_of_given_schemas(self):
        schemas = ["cs/peritoneum", "cs/peritoneum_female_gen"]

        self.assertEqual(discriminators_separating(self.scoping, PERITONEUM, schemas, DISCRIMINATOR_ITEMS), [2879])
        self.assertEqual(discriminators_separating(self.scoping, PERITONEUM, ["cs/peritoneum"], DISCRIMINATOR_ITEMS), [])


class DiscriminatorVariableTests(unittest.TestCase):
    """The extractor gets the table's name, notes and codes."""

    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)

    def table(self, schema, item_id):
        return self.scoping.schemas[schema].discriminators[item_id]

    def test_an_eod_discriminator_is_named_by_its_table_title(self):
        table = self.table("eod/lymphoma", 3926)
        variable = discriminator_variable(table, 3926)

        self.assertEqual(variable.name, "Schema Discriminator 1: Histology Discriminator for 9591/3")
        self.assertEqual(variable.description, table.definition.strip())
        # The blank code only says it isn't needed, so it's neither offered nor kept.
        self.assertIn("", table.codes)
        self.assertEqual(list(variable.valid_codes), ["1", "2", "3", "9"])

    def test_a_cs_discriminator_is_named_by_its_input(self):
        variable = discriminator_variable(self.table("cs/peritoneum", 2879), 2879)

        self.assertEqual(
            variable.name, "Schema Discriminator: Peritoneum/PeritoneumFemaleGen (CS Site-Specific Factor 25)"
        )
        self.assertEqual(list(variable.valid_codes), ["001", "002", "003", "004", "009", "100", "981"])

    def test_a_blank_code_that_says_more_joins_the_description(self):
        table = self.table("eod/soft_tissue_other", 3926)
        variable = discriminator_variable(table, 3926)

        self.assertEqual(table.table, "occult_head_and_neck_lymph_nodes_10277")
        self.assertNotIn("", variable.valid_codes)
        self.assertIn("When it isn't needed, the table also says:", variable.description)
        self.assertIn("Assign primary site C109", variable.description)
        self.assertIn("Assign primary site C119", variable.description)
        self.assertNotIn("Not C760", variable.description)

    def test_a_discriminator_for_another_item_fails_to_load(self):
        data = json.loads(SCOPING_DICTIONARY.read_text())
        entry = data["schemas"]["eod/lymphoma"]["discriminators"]["3926"]
        data["schemas"]["eod/lymphoma"]["discriminators"]["2880"] = entry
        with self.assertRaisesRegex(ValueError, r"eod/lymphoma has discriminators for items \[2880\]"):
            parse_scoping_dictionary(data)


class DiscriminatorFactsTests(unittest.TestCase):
    """A coded discriminator is a case fact that picks the schema."""

    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)

    def test_derive_case_facts_fills_the_discriminators(self):
        results = {
            3926: CaseVariableResult(item_id=3926, status=VariableStatus.STRUCTURED_DATA, value="3"),
            2879: CaseVariableResult(item_id=2879, status=VariableStatus.STRUCTURED_DATA, value="002"),
        }
        facts = derive_case_facts(CaseFacts(schema_discriminator_2="1"), results)

        self.assertEqual(
            (facts.schema_discriminator_1, facts.schema_discriminator_2, facts.cs_site_specific_factor_25),
            ("3", "1", "002"),
        )

    def test_a_coded_discriminator_narrows_the_schemas(self):
        for value, schemas in (("3", ["eod/lymphoma"]), ("1", ["eod/hemeretic"]), ("x", ["eod/hemeretic", "eod/lymphoma"])):
            with self.subTest(value=value):
                facts = HEMERETIC_OR_LYMPHOMA.model_copy(update={"schema_discriminator_1": value})
                self.assertEqual(matching_schemas(self.scoping, facts), schemas)
        facts = PERITONEUM.model_copy(update={"cs_site_specific_factor_25": "002"})
        self.assertEqual(matching_schemas(self.scoping, facts), ["cs/peritoneum_female_gen"])

    def test_dependent_items_take_the_picked_schemas_codes(self):
        facts = HEMERETIC_OR_LYMPHOMA.model_copy(update={"schema_discriminator_1": "3"})
        lymphoma = self.scoping.schemas["eod/lymphoma"]
        hemeretic = self.scoping.schemas["eod/hemeretic"]
        # An item the two schemas code differently, so before the discriminator
        # it can't be scoped.
        differing = [
            item_id for item_id in sorted(lymphoma.codes)
            if item_id in hemeretic.codes and scoped_codes(self.scoping, ["eod/hemeretic", "eod/lymphoma"], item_id) is None
        ]
        self.assertTrue(differing)

        scoped = scoped_item_codes(self.scoping, facts, differing)
        for item_id in differing:
            with self.subTest(item_id=item_id):
                self.assertEqual(scoped[item_id], lymphoma.codes[item_id].codes)


class DiscriminatorPlanningTests(unittest.TestCase):
    """At planning, each discriminator is closed or offered its table."""

    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)
        cls.groups = load_variable_groups(DISCRIMINATORS)
        cls.group = next(group for group in cls.groups if group.group_id == "schema_discriminators")

    def pending(self):
        return {item_id: CaseVariableResult(item_id=item_id) for item_id in DISCRIMINATOR_ITEMS}

    def test_the_discriminators_file_adds_a_discriminator_stage(self):
        self.assertEqual(load_stages(DISCRIMINATORS), ("initial", "histology", "discriminator", "dependent"))
        self.assertEqual(self.group.stage, "discriminator")
        self.assertEqual(
            [(variable.item_id, variable.name) for variable in self.group.variables],
            [(3926, "Schema Discriminator 1"), (3927, "Schema Discriminator 2"), (2879, "CS Site-Specific Factor25")],
        )
        self.assertEqual(load_group_hierarchy(DISCRIMINATORS)[1].group_id, "schema_discriminators")

    def test_only_the_discriminator_file_requests_them(self):
        for path in sorted(CONFIG_DIR.glob("variable_groups*.json")):
            with self.subTest(path=path.name):
                requested = {variable.item_id for group in load_variable_groups(path) for variable in group.variables}
                self.assertEqual(
                    bool(requested & set(DISCRIMINATOR_ITEMS)), path == DISCRIMINATORS,
                )

    def test_discriminators_the_candidates_dont_differ_by_are_closed(self):
        closed = discriminator_results([self.group], self.pending(), self.scoping, HEMERETIC_OR_LYMPHOMA)

        self.assertEqual(sorted(closed), [2879, 3927])
        self.assertEqual(closed[3927], CaseVariableResult(
            item_id=3927, status=VariableStatus.NOT_APPLICABLE,
            reason="The schemas the case matches (eod/hemeretic and eod/lymphoma) don't differ by "
                   "Schema Discriminator 2 (3927), so it isn't needed.",
        ))

    def test_a_needed_discriminator_is_offered_its_table(self):
        groups = with_discriminator_variables([self.group], self.pending(), self.scoping, HEMERETIC_OR_LYMPHOMA)
        variables = {variable.item_id: variable for variable in groups[0].variables}

        self.assertEqual(variables[3926].name, "Schema Discriminator 1: Histology Discriminator for 9591/3")
        self.assertEqual(list(variables[3926].valid_codes), ["1", "2", "3", "9"])
        self.assertEqual(variables[3927], VariableInfo(item_id=3927, name="Schema Discriminator 2"))

    def test_terminal_discriminators_and_other_groups_are_left_alone(self):
        results = {**self.pending(), 3926: CaseVariableResult(item_id=3926, status=VariableStatus.STRUCTURED_DATA, value="3")}
        other = TargetGroup(group_id="other", variables=[VariableInfo(item_id=764)])

        self.assertNotIn(3926, discriminator_results([self.group, other], results, self.scoping, BREAST))
        self.assertEqual(with_discriminator_variables([other], self.pending(), self.scoping, BREAST), [other])

    def test_without_a_site_dictionary_the_discriminators_are_closed(self):
        closed = discriminator_results([self.group], self.pending(), None, HEMERETIC_OR_LYMPHOMA)

        self.assertEqual(sorted(closed), [2879, 3926, 3927])
        self.assertEqual(closed[3926].status, VariableStatus.NOT_APPLICABLE)
        self.assertEqual(
            closed[3926].reason,
            "No site dictionary is loaded, so there are no staging schemas for Schema Discriminator 1 (3926) to pick between.",
        )

    def test_candidates_that_read_different_tables_block_the_discriminator(self):
        data = json.loads(SCOPING_DICTIONARY.read_text())
        data["schemas"]["eod/lymphoma"]["discriminators"]["3926"]["table"] = "another_table"
        closed = discriminator_results([self.group], self.pending(), parse_scoping_dictionary(data), HEMERETIC_OR_LYMPHOMA)

        self.assertEqual(closed[3926].status, VariableStatus.BLOCKED)
        self.assertIn("read it from different tables", closed[3926].reason)


class DiscriminatorScopeTests(unittest.TestCase):
    """A scoped discriminator gets its table's codes before the NAACCR entry is checked."""

    # v25's entry: no code domain at all, so it can't be offered alone.
    ENTRY = {
        "Data Item Name": "Schema Discriminator 1", "Length": 1, "Data Type": "text",
        "Allowable Values": None, "Format": None, "Code Descriptions": {},
        "Instructions for Coding": "The information recorded in Schema Discriminator differs for each anatomic site.",
    }

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base_path = Path(directory.name) / "base.json"
        base_path.write_text(json.dumps({
            "400": {"item_name": "Primary Site", "item_length": 4, "allowed_codes": primary_site_codes()},
            "522": {"item_name": "Histologic Type ICD-O-3", "item_length": 4, "allowed_codes": histology_pair_codes()},
            "3926": self.ENTRY,
        }))
        self.agent = object.__new__(OrchestratorAgent)
        self.agent._data_dictionary_path = base_path
        self.agent._site_data_dictionary_path = SCOPING_DICTIONARY
        self.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)

    def validate(self, variable, value):
        candidate = VariableOutput(
            item_id=variable.item_id, value=value, explanation="Synthetic.", most_important_note=1,
            spans=[], presence_confidence=ConfidenceLevel.HIGH,
        )
        return VariableValueValidator().validate(variable, candidate)

    def test_the_tables_name_and_codes_replace_the_generic_entry(self):
        group = TargetGroup(group_id="schema_discriminators", variables=[VariableInfo(item_id=3926, name="Schema Discriminator 1")])
        pending = {3926: CaseVariableResult(item_id=3926)}
        offered = with_discriminator_variables([group], pending, self.scoping, HEMERETIC_OR_LYMPHOMA)[0]

        variable = self.agent._scope_group(offered, HEMERETIC_OR_LYMPHOMA).variables[0]

        table = self.scoping.schemas["eod/lymphoma"].discriminators[3926]
        self.assertEqual(variable.name, table.title)
        self.assertEqual(variable.description, table.definition.strip())
        self.assertEqual((list(variable.valid_codes), variable.length), (["1", "2", "3", "9"], 1))
        self.assertIsNone(variable.coding_instructions)
        self.assertEqual(self.validate(variable, "3"), [])
        self.assertEqual(self.validate(variable, "5"), ["Value is not one of the variable's allowable codes."])

    def test_the_generic_entry_alone_has_no_codes_to_offer(self):
        with self.assertRaisesRegex(ValueError, "No usable code domain"):
            build_variable_group([3926], json.loads(self.agent._data_dictionary_path.read_text()))


class DiscriminatorRunTests(unittest.TestCase):
    """End to end with the fake orchestrator and the fixture."""

    def run_case(self, outcomes, groups_path=DISCRIMINATORS):
        agent = build_fake_orchestrator(Script(outcomes=outcomes))
        agent._variable_groups_path = groups_path
        agent._target_variables = load_variable_groups(groups_path)
        agent._stages = load_stages(groups_path)
        agent._target_group_hierarchy = load_group_hierarchy(groups_path)
        agent._site_dictionary = load_scoping_dictionary(SCOPING_DICTIONARY)
        agent._graph = agent._build_graph()
        with patch.object(agent._extractor, "run", wraps=agent._extractor.run) as extract:
            result = agent.run([note.model_dump(mode="json") for note in load_notes()], progress=False)
        sent = {
            variable.item_id: variable
            for call in extract.call_args_list
            for variable in call.args[0].requested_variables.variables
        }
        return result, sent

    @staticmethod
    def outcomes(primary_site, histology, date_of_diagnosis):
        return {
            390: Outcome(value=date_of_diagnosis), 400: Outcome(value=primary_site),
            522: Outcome(value=histology), 523: Outcome(value="3"),
        }

    def test_a_9591_case_extracts_schema_discriminator_1_and_scopes_by_it(self):
        result, sent = self.run_case(self.outcomes("C504", "9591", "20200615") | {3926: Outcome(value="3")})
        results = result.case.variable_results

        self.assertEqual((results[3926].status, results[3926].value), (VariableStatus.EXTRACTED, "3"))
        self.assertEqual(sent[3926].name, "Schema Discriminator 1: Histology Discriminator for 9591/3")
        self.assertEqual(list(sent[3926].valid_codes), ["1", "2", "3", "9"])
        for item_id in (3927, 2879):
            self.assertEqual(results[item_id].status, VariableStatus.NOT_APPLICABLE)
            self.assertNotIn(item_id, sent)
        facts = result.case.case_facts
        self.assertEqual(facts.schema_discriminator_1, "3")
        self.assertEqual(matching_schemas(load_scoping_dictionary(SCOPING_DICTIONARY), facts), ["eod/lymphoma"])

    def test_a_breast_carcinoma_closes_every_discriminator(self):
        result, sent = self.run_case(self.outcomes("C504", "8500", "20200615"))

        for item_id in DISCRIMINATOR_ITEMS:
            with self.subTest(item_id=item_id):
                self.assertEqual(result.case.variable_results[item_id].status, VariableStatus.NOT_APPLICABLE)
                self.assertIn("Only eod/breast matches the case", result.case.variable_results[item_id].reason)
                self.assertNotIn(item_id, sent)

    def test_a_coded_cs_site_specific_factor_25_settles_the_ssf(self):
        result, sent = self.run_case(self.outcomes("C481", "8441", "20120615") | {2879: Outcome(value="002")})
        results = result.case.variable_results

        self.assertEqual(list(sent[2879].valid_codes), ["001", "002", "003", "004", "009", "100", "981"])
        # cs/peritoneum_female_gen holds grade in no SSF, so Grade (440) alone stands in.
        for item_id in (3843, 3844):
            self.assertEqual(results[item_id].status, VariableStatus.NOT_APPLICABLE)
            self.assertTrue(results[item_id].reason.endswith("Grade (440) holds it for this year."))
        # It holds FIGO Stage in CS SSF2.
        self.assertEqual(result.case.ssf_equivalents[2890].schemas, ["cs/peritoneum_female_gen"])
        self.assertEqual(results[2890].status, VariableStatus.EXTRACTED)

    def test_a_missing_cs_site_specific_factor_25_blocks_the_ssf(self):
        result, _ = self.run_case(self.outcomes("C481", "8441", "20120615") | {2879: Outcome(value=None)})
        results = result.case.variable_results

        self.assertEqual(results[2879].status, VariableStatus.NOT_FOUND)
        for item_id in (3836, 3843, 3844):
            with self.subTest(item_id=item_id):
                self.assertEqual(results[item_id].status, VariableStatus.BLOCKED)
                self.assertEqual(results[item_id].blocking_item_ids, [2879])
                self.assertIn(
                    "; CS Site-Specific Factor25 (2879), which would tell them apart, has no value.",
                    results[item_id].reason,
                )
        self.assertEqual(result.case.ssf_equivalents, {})

    def test_the_default_file_extracts_no_discriminator(self):
        result, sent = self.run_case(
            self.outcomes("C504", "9591", "20200615"), groups_path=CONFIG_DIR / "variable_groups.json",
        )

        self.assertTrue(set(DISCRIMINATOR_ITEMS).isdisjoint(result.case.variable_results))
        self.assertTrue(set(DISCRIMINATOR_ITEMS).isdisjoint(sent))
        self.assertIsNone(result.case.case_facts.schema_discriminator_1)

    def test_the_recursion_limit_covers_the_fourth_stage(self):
        # A 9591/3 case settles discriminators, then SSFs and leftovers, at
        # planning; LangGraph's default 25 supersteps isn't enough for it.
        self.assertGreater(run_recursion_limit(load_stages(DISCRIMINATORS)), 25)
        self.assertLess(
            run_recursion_limit(load_stages(CONFIG_DIR / "variable_groups.json")),
            run_recursion_limit(load_stages(DISCRIMINATORS)),
        )

    def test_a_run_with_discriminators_saves_and_reloads(self):
        result, _ = self.run_case(self.outcomes("C504", "9591", "20200615") | {3926: Outcome(value="3")})

        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())

        self.assertEqual(restored.case, result.case)
        self.assertEqual(restored.inputs, result.inputs)
        self.assertEqual(restored.case.case_facts.schema_discriminator_1, "3")


if __name__ == "__main__":
    unittest.main()
