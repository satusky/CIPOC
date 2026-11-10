import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from pydantic import ValidationError

from cipoc.agents.orchestrator import CaseState
from cipoc.models import (
    CaseFacts,
    CaseVariableResult,
    ClinicalNote,
    CorpusGate,
    NoteCorpusDescriptors,
    ProcessedClinicalNote,
    SiteApplicability,
    TargetGroup,
    ValidatedVariableOutput,
    VariableInfo,
    VariableStatus,
)
from cipoc.tools.orchestration import (
    DEFAULT_STAGES,
    TREATMENT_CONCEPTS,
    build_corpus_descriptors,
    corpus_gate_passes,
    derive_case_facts,
    eligible_groups,
    load_stages,
    load_variable_groups,
    pending_group,
    resolve_leftovers,
    site_applies,
    stage_is_ready,
)
from cipoc.tools.scoping import parse_scoping_dictionary, tissue_site_codes
from cipoc.utils.progress.events import field
from tests.fake_orchestrator import (
    SCOPING_DICTIONARY,
    TISSUE_CODES,
    Outcome,
    Script,
    build_fake_orchestrator,
    graph_input,
)


VARIABLE_GROUPS = Path(__file__).resolve().parents[1] / "config" / "variable_groups.json"
MELANOMA_FAMILIES = ["872", "873", "874", "875", "876", "877", "878", "879"]


class ApplicabilityRegressionTests(unittest.TestCase):
    def test_only_restricted_facts_can_exclude(self):
        ovary = SiteApplicability(gross_primary_sites=["ovary"])
        for facts in (
            None,
            CaseFacts(histology="8500"),
            CaseFacts(primary_site="C809", histology="8500"),
            CaseFacts(primary_site="C80.9"),
            CaseFacts(primary_site="unknown"),
            CaseFacts(primary_site="C999"),
            CaseFacts(primary_site="C5.04"),
            CaseFacts(gross_primary_site="unsupported tissue", histology="8500"),
        ):
            with self.subTest(facts=facts):
                self.assertTrue(site_applies(ovary, facts, tissue_codes=TISSUE_CODES))
        self.assertTrue(site_applies(SiteApplicability(), CaseFacts(histology="8500")))
        self.assertFalse(site_applies(
            ovary, CaseFacts(primary_site="C349"), tissue_codes=TISSUE_CODES
        ))

    def test_coded_primary_overrides_conflicting_gross_in_both_directions(self):
        breast = SiteApplicability(gross_primary_sites=["breast"])
        for primary in ("C349", "C34.9", "C693"):
            with self.subTest(primary=primary):
                self.assertFalse(site_applies(
                    breast, CaseFacts(primary_site=primary, gross_primary_site="breast"),
                    tissue_codes=TISSUE_CODES,
                ))
        self.assertTrue(site_applies(
            breast, CaseFacts(primary_site="C504", gross_primary_site="lung"),
            tissue_codes=TISSUE_CODES,
        ))
        self.assertTrue(site_applies(
            breast, CaseFacts(primary_site="C809", gross_primary_site="left breast"),
            tissue_codes=TISSUE_CODES,
        ))

    def test_candidate_gross_sites_are_or(self):
        candidates = CaseFacts(candidate_gross_sites=["breast", "lung"])
        for restriction, expected in (
            (SiteApplicability(gross_primary_sites=["breast"]), True),
            (SiteApplicability(gross_primary_sites=["lung"]), True),
            (SiteApplicability(primary_sites=["C340-C349"]), True),
            (SiteApplicability(gross_primary_sites=["ovary"]), False),
            (SiteApplicability(primary_sites=["C440-C449"]), False),
        ):
            with self.subTest(restriction=restriction):
                self.assertIs(site_applies(restriction, candidates, tissue_codes=TISSUE_CODES), expected)
        # A coded primary outranks the candidates.
        self.assertFalse(site_applies(
            SiteApplicability(gross_primary_sites=["breast"]),
            CaseFacts(primary_site="C569", candidate_gross_sites=["breast", "lung"]),
            tissue_codes=TISSUE_CODES,
        ))
        # A candidate missing from the table makes the gross site unknown, so it can't exclude.
        for missing in (["breast", "kidney"], ["lung", "unspecified"]):
            with self.subTest(candidates=missing):
                self.assertTrue(site_applies(
                    SiteApplicability(gross_primary_sites=["ovary"]),
                    CaseFacts(candidate_gross_sites=missing),
                    tissue_codes=TISSUE_CODES,
                ))

    def test_newly_coded_primary_supersedes_earlier_gross_inference(self):
        facts = derive_case_facts(
            CaseFacts(gross_primary_site="breast"),
            {400: CaseVariableResult(
                item_id=400, status=VariableStatus.STRUCTURED_DATA, value="C349"
            )},
        )
        self.assertEqual(facts.primary_site, "C349")
        self.assertFalse(site_applies(
            SiteApplicability(gross_primary_sites=["breast"]), facts,
            tissue_codes=TISSUE_CODES,
        ))

    def test_explicit_unknown_primary_never_excludes_using_gross_site(self):
        for primary in ("C809", "C80.9", "malformed", "C5.04", "C999"):
            for restriction in (
                SiteApplicability(gross_primary_sites=["breast"]),
                SiteApplicability(primary_sites=["C440-C449"]),
            ):
                with self.subTest(primary=primary, restriction=restriction):
                    self.assertTrue(site_applies(
                        restriction,
                        CaseFacts(primary_site=primary, gross_primary_site="lung"),
                        tissue_codes=TISSUE_CODES,
                    ))
        self.assertFalse(site_applies(
            SiteApplicability(gross_primary_sites=["breast"]),
            CaseFacts(gross_primary_site="lung"),
            tissue_codes=TISSUE_CODES,
        ))

    def test_melanoma_family_is_numeric_and_not_site_limited(self):
        melanoma = SiteApplicability(histology_families=MELANOMA_FAMILIES)
        for histology, expected in (
            ("8719", False), ("8720", True), ("8743", True),
            ("8790", True), ("8799", True), ("8800", False), ("8500", False),
            (None, True), ("", True), ("melanoma", True), ("872", True),
            ("87200", True), ("8720/3", True),
        ):
            with self.subTest(histology=histology):
                self.assertEqual(site_applies(
                    melanoma, CaseFacts(primary_site="C693", histology=histology)
                ), expected)

    def test_histology_family_matches_a_coded_histology_by_its_first_three_digits(self):
        families = SiteApplicability(histology_families=["814", "850"])
        for histology, expected in (
            ("8140", True), ("8147", True), ("8500", True), ("8507", True),
            ("8130", False), ("8150", False), ("8720", False), ("8814", False),
        ):
            with self.subTest(histology=histology):
                self.assertEqual(
                    site_applies(families, CaseFacts(histology=histology)), expected
                )

    def test_candidate_histology_families_stand_in_for_an_uncoded_histology(self):
        melanoma = SiteApplicability(histology_families=["872"])
        for facts, expected in (
            ({"histology": "8720"}, True),
            ({"candidate_histology_families": ["872"]}, True),
            # Candidates are OR.
            ({"candidate_histology_families": ["850", "872"]}, True),
            ({"candidate_histology_families": ["850"]}, False),
            # A coded histology outranks the candidates.
            ({"histology": "8500", "candidate_histology_families": ["872"]}, False),
            ({"histology": "8720", "candidate_histology_families": ["850"]}, True),
            # Neither, a malformed candidate, or a malformed coded histology is unknown.
            ({}, True),
            ({"candidate_histology_families": ["850", "87"]}, True),
            ({"histology": "8720/3", "candidate_histology_families": ["850"]}, True),
        ):
            with self.subTest(**facts):
                self.assertEqual(site_applies(melanoma, CaseFacts(**facts)), expected)

    def test_family_labels_and_other_non_codes_fail_config_load(self):
        for family in ("melanoma", "MELANOMA", "87", "8720", " 872", "87x", "\uff18\uff17\uff12"):
            config = json.dumps({"groups": [{
                "group_id": "family", "variables": [{"item_id": 832}],
                "applies_to": {"histology_families": [family]},
            }]})
            with self.subTest(family=family):
                with patch("cipoc.tools.orchestration.open", mock_open(read_data=config)):
                    with self.assertRaisesRegex(ValueError, "Unsupported histology family"):
                        load_variable_groups("synthetic.json")

    def test_unsupported_family_fails_runtime_and_config_load_even_when_nested(self):
        for value in (
            {"histology_families": ["not-a-family"]},
            {"histology_families": ["872", "not-a-family"]},
            {"any_of": [{}, {"histology_families": ["not-a-family"]}]},
            {"all_of": [
                {"gross_primary_sites": ["breast"]},
                {"any_of": [{"histology_families": ["not-a-family"]}]},
            ]},
        ):
            restriction = SiteApplicability.model_validate(value)
            for facts in (None, CaseFacts(primary_site="C349", histology="8247")):
                with self.subTest(value=value, facts=facts):
                    with self.assertRaisesRegex(ValueError, "Unsupported histology family.*not-a-family"):
                        site_applies(restriction, facts)
            group = {
                "group_id": "unsupported", "variables": [{"item_id": 832}],
                "applies_to": value,
            }
            for config_group in (group, {"group_id": "parent", "subgroups": [group]}):
                with self.subTest(config_group=config_group):
                    config = json.dumps({"groups": [config_group]})
                    with patch("cipoc.tools.orchestration.open", mock_open(read_data=config)):
                        with self.assertRaisesRegex(ValueError, "Unsupported histology family.*not-a-family"):
                            load_variable_groups("synthetic.json")

    def test_gross_sites_outside_the_vocabulary_fail_config_load(self):
        for site in ("brest", "Breast", "left breast", "small intestine", "unspecified"):
            config = json.dumps({"groups": [{
                "group_id": "site", "variables": [{"item_id": 832}],
                "applies_to": {"any_of": [{"gross_primary_sites": ["ovary", site]}]},
            }]})
            with self.subTest(site=site):
                with patch("cipoc.tools.orchestration.open", mock_open(read_data=config)):
                    with self.assertRaisesRegex(ValueError, "Unsupported gross primary site"):
                        load_variable_groups("synthetic.json")

    def test_legacy_family_labels_round_trip_without_runtime_revalidation(self):
        legacy = {
            "group_id": "legacy", "variables": [{"item_id": 832}],
            "applies_to": {"histology_families": ["melanoma", "Sarcoma", "LYMPHOMA"]},
        }
        with patch("cipoc.tools.orchestration.site_applies", side_effect=AssertionError("Runtime evaluation during artifact parsing")):
            group = TargetGroup.model_validate(legacy)
            restored = TargetGroup.model_validate_json(group.model_dump_json())
        self.assertEqual(restored.applies_to.histology_families, ["melanoma", "Sarcoma", "LYMPHOMA"])

    def test_nested_config_families_match_without_changing_stored_codes(self):
        config = {"groups": [{
            "group_id": "melanoma", "variables": [{"item_id": 832}],
            "applies_to": {"any_of": [{"histology_families": ["879", "872"]}]},
        }]}
        with patch("cipoc.tools.orchestration.open", mock_open(read_data=json.dumps(config))):
            group = load_variable_groups("synthetic.json")[0]
        self.assertTrue(site_applies(group.applies_to, CaseFacts(histology="8720")))
        self.assertFalse(site_applies(group.applies_to, CaseFacts(histology="8247")))
        self.assertEqual(group.applies_to.any_of[0].histology_families, ["879", "872"])

    def test_leaf_alternatives_remain_or_and_unknown_keeps_scope_open(self):
        restriction = SiteApplicability(
            gross_primary_sites=["ovary"], histology_families=MELANOMA_FAMILIES
        )
        self.assertTrue(site_applies(
            restriction, CaseFacts(histology="8500"), tissue_codes=TISSUE_CODES
        ))
        self.assertTrue(site_applies(
            restriction, CaseFacts(primary_site="C349"), tissue_codes=TISSUE_CODES
        ))
        self.assertTrue(site_applies(
            restriction, CaseFacts(primary_site="C349", histology="8720"),
            tissue_codes=TISSUE_CODES,
        ))
        self.assertFalse(site_applies(
            restriction, CaseFacts(primary_site="C349", histology="8500"),
            tissue_codes=TISSUE_CODES,
        ))

    def test_explicit_boolean_operators_cover_all_truth_combinations(self):
        for operator in ("any_of", "all_of"):
            restriction = SiteApplicability.model_validate({operator: [
                {"gross_primary_sites": ["breast"]},
                {"histology_families": MELANOMA_FAMILIES},
            ]})
            for site, site_match in (("C504", True), ("C349", False), (None, None)):
                for histology, histology_match in (("8720", True), ("8500", False), (None, None)):
                    with self.subTest(operator=operator, site=site, histology=histology):
                        facts = CaseFacts(primary_site=site, histology=histology)
                        truth_values = (site_match, histology_match)
                        expected = (
                            not all(value is False for value in truth_values)
                            if operator == "any_of" else False not in truth_values
                        )
                        self.assertEqual(
                            site_applies(restriction, facts, tissue_codes=TISSUE_CODES), expected
                        )

    def test_malformed_expressions_and_ranges_fail_configuration_validation(self):
        for value in (
            {"primary_sites": ["C449-C440"]},
            {"primary_sites": ["skin"]},
            {"primary_sites": ["C44.0"]},
            {"any_of": [{}], "all_of": [{}]},
            {"any_of": [{}], "gross_primary_sites": ["breast"]},
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SiteApplicability.model_validate(value)

    def test_coarse_site_overlap_stays_open_without_overriding_coded_site(self):
        nipple = SiteApplicability(primary_sites=["C500"])
        self.assertTrue(site_applies(
            nipple, CaseFacts(gross_primary_site="breast"), tissue_codes=TISSUE_CODES
        ))
        self.assertFalse(site_applies(
            nipple, CaseFacts(primary_site="C504", gross_primary_site="breast"),
            tissue_codes=TISSUE_CODES,
        ))
        self.assertFalse(site_applies(
            nipple, CaseFacts(gross_primary_site="lung"), tissue_codes=TISSUE_CODES
        ))

    def test_unknown_site_survives_planning_and_leftovers_preserve_seeds(self):
        initial = TargetGroup(
            group_id="initial", stage="initial", gate=[CorpusGate.TREATMENT_PRESENT],
            variables=[VariableInfo(item_id=400)],
        )
        ovary = TargetGroup(
            group_id="ovary", stage="dependent",
            applies_to=SiteApplicability(gross_primary_sites=["ovary"]),
            variables=[VariableInfo(item_id=3836), VariableInfo(item_id=3837)],
        )
        seed = CaseVariableResult(
            item_id=3837, status=VariableStatus.STRUCTURED_DATA, value="seed"
        )
        results = {
            400: CaseVariableResult(item_id=400, status=VariableStatus.PENDING),
            3836: CaseVariableResult(item_id=3836, status=VariableStatus.PENDING),
            3837: seed,
        }
        corpus = NoteCorpusDescriptors(unique_flags=set())
        facts = CaseFacts(histology="8500")
        groups = [initial, ovary]
        self.assertEqual(eligible_groups(groups, results, corpus, facts), [])
        updates = resolve_leftovers(groups, results, corpus, facts)
        self.assertEqual(set(updates), {400})
        self.assertEqual(updates[400].status, VariableStatus.NOT_APPLICABLE)
        self.assertEqual(results[3836].status, VariableStatus.PENDING)
        self.assertIs(results[3837], seed)

        results.update(updates)
        self.assertTrue(stage_is_ready(ovary, groups, results))
        self.assertEqual(eligible_groups(groups, results, corpus, facts), [ovary])
        self.assertEqual([v.item_id for v in pending_group(ovary, results).variables], [3836])

        results[400] = CaseVariableResult(
            item_id=400, status=VariableStatus.NOT_FOUND,
            extraction=ValidatedVariableOutput(
                item_id=400, value=None, explanation="No primary in the notes",
                most_important_note=None, spans=[], presence_confidence="low", is_valid=True,
            ),
        )
        self.assertTrue(stage_is_ready(ovary, groups, results))
        self.assertEqual(eligible_groups(groups, results, corpus, facts), [ovary])


class TissueCodeTableTests(unittest.TestCase):
    def test_table_holds_each_tissues_primary_site_codes(self):
        raw = json.loads(SCOPING_DICTIONARY.read_text())
        table = tissue_site_codes(parse_scoping_dictionary(raw))
        self.assertEqual(set(table), set(raw["tissues"]))
        for tissue, codes in TISSUE_CODES.items():
            self.assertEqual(table[tissue], codes)

        raw["tissues"]["breast"]["codes"]["400"] = {
            "C50.0": "Nipple", "c504": "Upper-outer quadrant of breast",
        }
        raw["tissues"]["lung"]["codes"] = {}
        table = tissue_site_codes(parse_scoping_dictionary(raw))
        self.assertEqual(table["breast"], frozenset({"C500", "C504"}))
        self.assertNotIn("lung", table)
        self.assertEqual(tissue_site_codes(None), {})

        breast = SiteApplicability(gross_primary_sites=["breast"])
        self.assertTrue(site_applies(breast, CaseFacts(primary_site="C50.4"), tissue_codes=table))
        self.assertFalse(site_applies(breast, CaseFacts(primary_site="C501"), tissue_codes=table))

    def test_restrictions_and_gross_facts_both_read_the_table(self):
        # The old site dictionary's colon includes rectosigmoid and rectum, and
        # leaves out the appendix.
        table = {
            **TISSUE_CODES,
            "colon": frozenset({
                "C180", "C182", "C183", "C184", "C185", "C186", "C187", "C188", "C189",
                "C199", "C209",
            }),
            "appendix": frozenset({"C181"}),
        }
        colon = SiteApplicability(gross_primary_sites=["colon"])
        self.assertTrue(site_applies(colon, CaseFacts(primary_site="C209"), tissue_codes=table))
        self.assertFalse(site_applies(colon, CaseFacts(primary_site="C181"), tissue_codes=table))
        self.assertFalse(site_applies(
            SiteApplicability(gross_primary_sites=["appendix"]),
            CaseFacts(gross_primary_site="colon"), tissue_codes=table,
        ))
        self.assertFalse(site_applies(
            SiteApplicability(primary_sites=["C181"]),
            CaseFacts(gross_primary_site="colon"), tissue_codes=table,
        ))
        small_intestine = SiteApplicability(gross_primary_sites=["small_intestine"])
        self.assertFalse(site_applies(
            SiteApplicability(gross_primary_sites=["breast"]),
            CaseFacts(gross_primary_site="small_intestine"), tissue_codes=table,
        ))
        self.assertFalse(site_applies(
            small_intestine, CaseFacts(primary_site="C349"), tissue_codes=table
        ))
        self.assertTrue(site_applies(
            small_intestine, CaseFacts(primary_site="C171"), tissue_codes=table
        ))

    def test_a_tissue_missing_from_the_table_never_excludes(self):
        ovary = SiteApplicability(gross_primary_sites=["ovary"])
        nipple = SiteApplicability(primary_sites=["C500"])
        for table in (None, {}, {"breast": TISSUE_CODES["breast"]}):
            with self.subTest(table=table):
                self.assertTrue(site_applies(
                    ovary, CaseFacts(primary_site="C349"), tissue_codes=table
                ))
                self.assertTrue(site_applies(
                    ovary, CaseFacts(gross_primary_site="breast"), tissue_codes=table
                ))
                self.assertTrue(site_applies(
                    nipple, CaseFacts(gross_primary_site="lung"), tissue_codes=table
                ))
                # A restriction naming the case's own tissue still matches by name.
                self.assertTrue(site_applies(
                    ovary, CaseFacts(gross_primary_site="ovary"), tissue_codes=table
                ))


class StageOrderTests(unittest.TestCase):
    STAGES = ("initial", "histology", "dependent")

    @staticmethod
    def load(config):
        with patch("cipoc.tools.orchestration.open", mock_open(read_data=json.dumps(config))):
            return load_variable_groups("synthetic.json"), load_stages("synthetic.json")

    def test_unknown_stage_fails_config_load(self):
        group = {"group_id": "histology", "stage": "histology", "variables": [{"item_id": 522}]}
        inherited = {"group_id": "parent", "stage": "histology", "subgroups": [
            {"group_id": "child", "variables": [{"item_id": 522}]},
        ]}
        for config in (
            {"groups": [group]},
            {"stages": ["initial", "dependent"], "groups": [group]},
            {"stages": ["initial", "dependent"], "groups": [inherited]},
        ):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "unknown stage 'histology'"):
                self.load(config)
        for stages in ([], ["initial", "initial"], "initial"):
            with self.subTest(stages=stages), self.assertRaisesRegex(ValueError, "'stages' must be"):
                self.load({"stages": stages, "groups": []})

        self.assertEqual(self.load({"groups": []})[1], DEFAULT_STAGES)
        groups, stages = self.load({"stages": list(self.STAGES), "groups": [group]})
        self.assertEqual(stages, self.STAGES)
        self.assertEqual(groups[0].stage, "histology")

    def test_each_stage_waits_for_earlier_ones_and_leftovers_close_the_earliest(self):
        first = TargetGroup(group_id="first", stage="initial", variables=[VariableInfo(item_id=400)])
        middle = TargetGroup(group_id="middle", stage="histology", variables=[VariableInfo(item_id=522)])
        last = TargetGroup(group_id="last", stage="dependent", variables=[VariableInfo(item_id=410)])
        unstaged = TargetGroup(group_id="unstaged", variables=[VariableInfo(item_id=820)])
        groups = [last, unstaged, middle, first]
        results = {
            item_id: CaseVariableResult(item_id=item_id, status=VariableStatus.PENDING)
            for item_id in (400, 522, 410, 820)
        }
        corpus = NoteCorpusDescriptors(unique_flags=set())
        self.assertEqual(
            eligible_groups(groups, results, corpus, None, stages=self.STAGES), [unstaged, first]
        )

        results[400] = CaseVariableResult(item_id=400, status=VariableStatus.STRUCTURED_DATA, value="C504")
        self.assertTrue(stage_is_ready(middle, groups, results, stages=self.STAGES))
        self.assertFalse(stage_is_ready(last, groups, results, stages=self.STAGES))
        updates = resolve_leftovers(groups, results, corpus, None, stages=self.STAGES)
        self.assertEqual(set(updates), {522})
        self.assertEqual(updates[522].status, VariableStatus.BLOCKED)
        self.assertEqual(updates[522].reason, "Prerequisite histology-stage extraction did not complete.")
        self.assertEqual(updates[522].blocking_item_ids, [522])

        # An unstaged group never holds a stage, and closes with the last one.
        results.update(updates)
        self.assertTrue(stage_is_ready(last, groups, results, stages=self.STAGES))
        self.assertEqual(
            set(resolve_leftovers(groups, results, corpus, None, stages=self.STAGES)), {410, 820}
        )


class SentinelNodeTargetingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.group = next(
            group for group in load_variable_groups(VARIABLE_GROUPS)
            if any(variable.item_id == 832 for variable in group.variables)
        )

    def test_manual_site_ranges_and_melanoma_conjunction(self):
        # STORE 2024, item 832: breast and cutaneous melanoma only (pp. 141-142).
        # Summary-Stage_v3.3.md, MELANOMA SKIN, line 8001 supplies the site list.
        for site in (
            "C000", "C002", "C006", "C440", "C44.9", "C500", "C510", "C512",
            "C518", "C519", "C600", "C602", "C608", "C609", "C632",
        ):
            with self.subTest(site=site):
                self.assertTrue(site_applies(
                    self.group.applies_to, CaseFacts(primary_site=site, histology="8720"),
                    tissue_codes=TISSUE_CODES,
                ))
        for site in ("C003", "C005", "C009", "C520", "C690", "C693", "C349"):
            with self.subTest(site=site):
                self.assertFalse(site_applies(
                    self.group.applies_to, CaseFacts(primary_site=site, histology="8720"),
                    tissue_codes=TISSUE_CODES,
                ))

    def test_known_and_unknown_facts_follow_three_valued_logic(self):
        for site, histology, expected in (
            ("C504", "8500", True), ("C504", None, True),
            ("C449", "8720", True), ("C449", "8500", False),
            ("C449", None, True), ("C449", "malformed", True),
            ("C693", "8720", False), ("C693", None, False),
            (None, "8720", True), (None, "8500", True), (None, None, True),
            ("C809", "8500", True), ("malformed", "8500", True),
        ):
            with self.subTest(site=site, histology=histology):
                self.assertEqual(site_applies(
                    self.group.applies_to, CaseFacts(primary_site=site, histology=histology),
                    tissue_codes=TISSUE_CODES,
                ), expected)

    def test_serialized_config_keeps_explicit_conjunction(self):
        restored = SiteApplicability.model_validate_json(
            self.group.applies_to.model_dump_json()
        )
        self.assertEqual(len(restored.any_of), 2)
        self.assertEqual(len(restored.any_of[1].all_of), 2)
        self.assertEqual(restored.any_of[1].all_of[1].histology_families, MELANOMA_FAMILIES)
        self.assertEqual(self.group.stage, "dependent")


class SchedulingGraphRegressionTests(unittest.TestCase):
    def run_graph(self, script, structured_data, *, groups=None, stages=None):
        # Only the subagent responses/metadata are fake; planner, gates, stage
        # readiness, leftover resolution, and result/fact roll-up are production.
        agent = build_fake_orchestrator(script)
        if groups is not None:
            agent._target_variables = groups
        if stages is not None:
            agent._stages = stages
        note = ClinicalNote(
            note_id="note-A", date="2025-01-02", note_type="Pathology",
            content="Synthetic skin cancer pathology and sentinel node biopsy.",
        )
        states, launches = [], []
        with patch.object(agent._extractor, "run", wraps=agent._extractor.run) as extract:
            for namespace, mode, payload in agent.compiled_graph.stream(
                graph_input([note], structured_data=structured_data),
                stream_mode=["values", "tasks"], subgraphs=True,
            ):
                if namespace:
                    continue
                if mode == "values":
                    states.append(CaseState.model_validate(payload))
                elif payload["name"] == "extract_branch" and "input" in payload:
                    request = field(payload["input"], "requested_variables")
                    item_ids = tuple(v.item_id for v in request.variables)
                    launches.append((item_ids, states[-1]))
            extracted_ids = [
                variable.item_id
                for call in extract.call_args_list
                for variable in call.args[0].requested_variables.variables
            ]
        self.assertTrue(states)
        self.assertFalse(states[-1].outstanding_item_ids)
        self.assertIsNotNone(states[-1].report)
        return states, launches, extracted_ids

    def test_configured_nonmelanoma_histology_prevents_sentinel_extraction(self):
        states, launches, extracted_ids = self.run_graph(
            Script(outcomes={522: Outcome(value="8247"), 523: Outcome(value="3")}),
            {400: "C449"},
        )
        final = states[-1]
        configured = {group.group_id: group for group in final.target_variables}
        self.assertEqual(configured["site_specific_codes"].stage, "dependent")
        self.assertEqual(configured["histologic_type_and_behavior"].stage, "histology")
        self.assertEqual(final.variable_results[522].status, VariableStatus.EXTRACTED)
        self.assertEqual(final.case_facts.histology, "8247")
        self.assertEqual(final.variable_results[832].status, VariableStatus.NOT_APPLICABLE)
        self.assertNotIn(832, extracted_ids)
        self.assertFalse(any(832 in item_ids for item_ids, _ in launches))
        self.assertNotIn("group:breast_melanoma_only", final.note_selection)

    def test_configured_not_found_histology_releases_sentinel_with_unknown_scope(self):
        states, launches, extracted_ids = self.run_graph(
            Script(outcomes={
                522: Outcome(value=None), 523: Outcome(value="3"),
                832: Outcome(value="20250102"),
            }),
            {400: "C449"},
        )
        final = states[-1]
        self.assertEqual(final.variable_results[522].status, VariableStatus.NOT_FOUND)
        self.assertIsNone(final.case_facts.histology)
        self.assertEqual(final.variable_results[832].status, VariableStatus.EXTRACTED)
        self.assertEqual(extracted_ids.count(832), 1)
        sentinel_launch = next(state for item_ids, state in launches if 832 in item_ids)
        self.assertEqual(sentinel_launch.variable_results[522].status, VariableStatus.NOT_FOUND)
        self.assertEqual(sentinel_launch.variable_results[523].status, VariableStatus.EXTRACTED)

    def test_gated_initial_closure_replans_dependents_without_overwriting_seeds(self):
        groups = [
            TargetGroup(
                group_id="gated_initial", stage="initial", gate=[CorpusGate.TREATMENT_PRESENT],
                variables=[VariableInfo(item_id=item) for item in (400, 522, 523)],
            ),
            TargetGroup(
                group_id="dependent", stage="dependent",
                variables=[VariableInfo(item_id=item) for item in (832, 834)],
                applies_to=SiteApplicability(primary_sites=["C440-C449"]),
            ),
        ]
        seeds = {400: "C449", 523: "3", 834: "01"}
        states, launches, extracted_ids = self.run_graph(
            Script(concepts={"cancer": True}, outcomes={832: Outcome(value="20250102")}),
            seeds, groups=groups,
        )
        final = states[-1]
        self.assertEqual(final.variable_results[522].status, VariableStatus.NOT_APPLICABLE)
        self.assertIn("Corpus gate not met", final.variable_results[522].reason)
        self.assertEqual(final.variable_results[832].status, VariableStatus.EXTRACTED)
        self.assertEqual(extracted_ids, [832])
        self.assertTrue(any(
            state.variable_results[522].status == VariableStatus.NOT_APPLICABLE
            and state.variable_results[832].status == VariableStatus.PENDING
            for state in states if 522 in state.variable_results
        ))
        self.assertEqual(launches[0][0], (832,))
        self.assertEqual(launches[0][1].variable_results[522].status, VariableStatus.NOT_APPLICABLE)
        for item_id, value in seeds.items():
            with self.subTest(item_id=item_id):
                self.assertEqual(final.variable_results[item_id].status, VariableStatus.STRUCTURED_DATA)
                self.assertEqual(final.variable_results[item_id].value, value)

    def test_three_configured_stages_run_in_list_order(self):
        groups = [
            TargetGroup(group_id="last", stage="dependent", variables=[VariableInfo(item_id=410)]),
            TargetGroup(
                group_id="middle", stage="histology",
                variables=[VariableInfo(item_id=item) for item in (522, 523)],
            ),
            TargetGroup(group_id="first", stage="initial", variables=[VariableInfo(item_id=400)]),
        ]
        states, launches, extracted_ids = self.run_graph(
            Script(outcomes={400: Outcome(value="C504"), 522: Outcome(value="8500"), 523: Outcome(value="3")}),
            {}, groups=groups, stages=("initial", "histology", "dependent"),
        )
        self.assertEqual([item_ids for item_ids, _ in launches], [(400,), (522, 523), (410,)])
        self.assertEqual(extracted_ids, [400, 522, 523, 410])
        middle_launch, last_launch = launches[1][1], launches[2][1]
        self.assertEqual(middle_launch.variable_results[400].status, VariableStatus.EXTRACTED)
        self.assertEqual(middle_launch.variable_results[410].status, VariableStatus.PENDING)
        self.assertEqual(last_launch.variable_results[522].status, VariableStatus.EXTRACTED)
        self.assertEqual(states[-1].variable_results[410].status, VariableStatus.EXTRACTED)

    def test_configured_histology_stage_runs_between_initial_and_dependents(self):
        # A NOT_FOUND Primary Site is terminal, so it still releases the histology stage.
        for primary_site, status in (("C504", VariableStatus.EXTRACTED), (None, VariableStatus.NOT_FOUND)):
            with self.subTest(primary_site=primary_site):
                states, launches, _ = self.run_graph(
                    Script(outcomes={
                        400: Outcome(value=primary_site), 522: Outcome(value="8500"), 523: Outcome(value="3"),
                    }),
                    {},
                )
                self.assertEqual([item_ids for item_ids, _ in launches[:2]], [(390, 400, 410), (522, 523)])
                histology_launch = launches[1][1]
                self.assertEqual(histology_launch.variable_results[400].status, status)
                dependent_ids = {item_id for item_ids, _ in launches[2:] for item_id in item_ids}
                self.assertTrue(dependent_ids)
                for item_id in dependent_ids:
                    self.assertEqual(histology_launch.variable_results[item_id].status, VariableStatus.PENDING)
                for _, state in launches[2:]:
                    self.assertEqual(state.variable_results[522].status, VariableStatus.EXTRACTED)
                    self.assertEqual(state.variable_results[523].status, VariableStatus.EXTRACTED)


class TreatmentGateTests(unittest.TestCase):
    def test_each_modality_alone_opens_treatment_and_its_dates(self):
        self.assertIn("hormonal_therapy", TREATMENT_CONCEPTS)
        self.assertIn("immunotherapy", TREATMENT_CONCEPTS)
        group = next(
            group for group in load_variable_groups(VARIABLE_GROUPS)
            if group.group_id == "first_course_treatment"
        )
        results = {
            variable.item_id: CaseVariableResult(
                item_id=variable.item_id, status=VariableStatus.PENDING
            ) for variable in group.variables
        }
        for concept, item_ids in (
            ("hormonal_therapy", {710, 1230}), ("immunotherapy", {720, 1240})
        ):
            with self.subTest(concept=concept):
                note = ProcessedClinicalNote(
                    note_id="note-A", date="2025-01-01", note_type="Oncology",
                    content="Cancer-directed therapy is planned.",
                    concepts={concept: {"presence": True}},
                )
                corpus = build_corpus_descriptors({note.note_id: note})
                self.assertTrue(corpus_gate_passes([CorpusGate.TREATMENT_PRESENT], corpus))
                self.assertEqual(eligible_groups([group], results, corpus, None), [group])
                self.assertTrue(item_ids.issubset({v.item_id for v in pending_group(group, results).variables}))

    def test_no_cancer_treatment_does_not_open_gate(self):
        self.assertFalse(corpus_gate_passes(
            [CorpusGate.TREATMENT_PRESENT], NoteCorpusDescriptors(unique_flags=set())
        ))


if __name__ == "__main__":
    unittest.main()
