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
    TREATMENT_CONCEPTS,
    build_corpus_descriptors,
    corpus_gate_passes,
    derive_case_facts,
    eligible_groups,
    load_variable_groups,
    pending_group,
    resolve_leftovers,
    site_applies,
    stage_is_ready,
)
from cipoc.utils.progress.events import field
from tests.fake_orchestrator import Outcome, Script, build_fake_orchestrator, graph_input


VARIABLE_GROUPS = Path(__file__).resolve().parents[1] / "config" / "variable_groups.json"


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
                self.assertTrue(site_applies(ovary, facts))
        self.assertTrue(site_applies(SiteApplicability(), CaseFacts(histology="8500")))
        self.assertFalse(site_applies(ovary, CaseFacts(primary_site="C349")))

    def test_coded_primary_overrides_conflicting_gross_in_both_directions(self):
        breast = SiteApplicability(gross_primary_sites=["breast"])
        for primary in ("C349", "C34.9", "C693"):
            with self.subTest(primary=primary):
                self.assertFalse(site_applies(
                    breast, CaseFacts(primary_site=primary, gross_primary_site="breast")
                ))
        self.assertTrue(site_applies(
            breast, CaseFacts(primary_site="C504", gross_primary_site="lung")
        ))
        self.assertTrue(site_applies(
            breast, CaseFacts(primary_site="C809", gross_primary_site="left breast")
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
            SiteApplicability(gross_primary_sites=["breast"]), facts
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
                    ))
        self.assertFalse(site_applies(
            SiteApplicability(gross_primary_sites=["breast"]),
            CaseFacts(gross_primary_site="lung"),
        ))

    def test_melanoma_family_is_numeric_and_not_site_limited(self):
        melanoma = SiteApplicability(histology_families=["melanoma"])
        for histology, expected in (
            ("8719", False), ("8720", True), ("8743", True),
            ("8790", True), ("8791", False), ("8500", False),
            (None, True), ("", True), ("melanoma", True), ("872", True),
            ("87200", True), ("8720/3", True),
        ):
            with self.subTest(histology=histology):
                self.assertEqual(site_applies(
                    melanoma, CaseFacts(primary_site="C693", histology=histology)
                ), expected)

    def test_unsupported_family_fails_runtime_and_config_load_even_when_nested(self):
        for value in (
            {"histology_families": ["not-a-family"]},
            {"histology_families": ["melanoma", "not-a-family"]},
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

    def test_legacy_family_labels_round_trip_without_runtime_revalidation(self):
        legacy = {
            "group_id": "legacy", "variables": [{"item_id": 832}],
            "applies_to": {"histology_families": ["Sarcoma", "LYMPHOMA"]},
        }
        with patch("cipoc.tools.orchestration.site_applies", side_effect=AssertionError("Runtime evaluation during artifact parsing")):
            group = TargetGroup.model_validate(legacy)
            restored = TargetGroup.model_validate_json(group.model_dump_json())
        self.assertEqual(restored.applies_to.histology_families, ["Sarcoma", "LYMPHOMA"])

    def test_runtime_family_matching_casefolds_without_changing_stored_labels(self):
        config = {"groups": [{
            "group_id": "melanoma", "variables": [{"item_id": 832}],
            "applies_to": {"any_of": [{"histology_families": ["MeLaNoMa"]}]},
        }]}
        with patch("cipoc.tools.orchestration.open", mock_open(read_data=json.dumps(config))):
            group = load_variable_groups("synthetic.json")[0]
        self.assertTrue(site_applies(group.applies_to, CaseFacts(histology="8720")))
        self.assertFalse(site_applies(group.applies_to, CaseFacts(histology="8247")))
        self.assertEqual(group.applies_to.any_of[0].histology_families, ["MeLaNoMa"])

    def test_leaf_alternatives_remain_or_and_unknown_keeps_scope_open(self):
        restriction = SiteApplicability(
            gross_primary_sites=["ovary"], histology_families=["melanoma"]
        )
        self.assertTrue(site_applies(restriction, CaseFacts(histology="8500")))
        self.assertTrue(site_applies(restriction, CaseFacts(primary_site="C349")))
        self.assertTrue(site_applies(
            restriction, CaseFacts(primary_site="C349", histology="8720")
        ))
        self.assertFalse(site_applies(
            restriction, CaseFacts(primary_site="C349", histology="8500")
        ))

    def test_explicit_boolean_operators_cover_all_truth_combinations(self):
        for operator in ("any_of", "all_of"):
            restriction = SiteApplicability.model_validate({operator: [
                {"gross_primary_sites": ["breast"]},
                {"histology_families": ["melanoma"]},
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
                        self.assertEqual(site_applies(restriction, facts), expected)

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
        self.assertTrue(site_applies(nipple, CaseFacts(gross_primary_site="breast")))
        self.assertFalse(site_applies(
            nipple, CaseFacts(primary_site="C504", gross_primary_site="breast")
        ))
        self.assertFalse(site_applies(nipple, CaseFacts(gross_primary_site="lung")))

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
                    self.group.applies_to, CaseFacts(primary_site=site, histology="8720")
                ))
        for site in ("C003", "C005", "C009", "C520", "C690", "C693", "C349"):
            with self.subTest(site=site):
                self.assertFalse(site_applies(
                    self.group.applies_to, CaseFacts(primary_site=site, histology="8720")
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
                    self.group.applies_to, CaseFacts(primary_site=site, histology=histology)
                ), expected)

    def test_serialized_config_keeps_explicit_conjunction(self):
        restored = SiteApplicability.model_validate_json(
            self.group.applies_to.model_dump_json()
        )
        self.assertEqual(len(restored.any_of), 2)
        self.assertEqual(len(restored.any_of[1].all_of), 2)
        self.assertEqual(restored.any_of[1].all_of[1].histology_families, ["melanoma"])
        self.assertEqual(self.group.stage, "dependent")


class SchedulingGraphRegressionTests(unittest.TestCase):
    def run_graph(self, script, structured_data, *, groups=None):
        # Only the subagent responses/metadata are fake; planner, gates, stage
        # readiness, leftover resolution, and result/fact roll-up are production.
        agent = build_fake_orchestrator(script)
        if groups is not None:
            agent._target_variables = groups
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
        self.assertEqual(configured["histologic_type_and_behavior"].stage, "initial")
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
