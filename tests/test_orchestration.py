import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import mock_open, patch
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from cipoc.agents.orchestrator import CaseState, OrchestratorAgent
from cipoc.models import (
    CancerMention,
    CaseFacts,
    CaseVariableResult,
    ConfidenceLevel,
    ExtractionStop,
    HistologySelection,
    OrchestratorRunError,
    OrchestratorRunResult,
    ProcessedClinicalNote,
    ReviewFlagType,
    SsfEquivalent,
    TargetGroup,
    TextSpan,
    ValidatedVariableOutput,
    VariableInfo,
    VariableOutput,
    VariableStatus,
)
from cipoc.tools import VariableValueValidator, build_corpus_descriptors, build_report, candidate_gross_sites, derive_case_facts, load_scoping_dictionary, load_variable_groups, out_of_year_results, site_applies, ssf_equivalent_results, ssf_offered_codes, ssf_variable, stop_pending_results, with_ssf_equivalents
from cipoc.utils.progress.events import normalize

from tests.fake_orchestrator import (
    SCOPING_DICTIONARY,
    TISSUE_CODES,
    Outcome,
    Script,
    build_fake_orchestrator,
    graph_input,
    histology_pair_codes,
    load_notes,
    primary_site_codes,
)


VARIABLE_GROUPS = Path(__file__).resolve().parents[1] / "config" / "variable_groups.json"
# A coded lung adenocarcinoma. The fake extractor otherwise codes every item
# "1", which leaves the site unknown, so every CS schema is a candidate and the
# items CS held in a Site-Specific Factor end BLOCKED. CS lung holds grade in
# no SSF, so its grade items show the fixed equivalents alone.
LUNG = {400: Outcome(value="C341"), 522: Outcome(value="8140"), 523: Outcome(value="3")}
BASE_DICTIONARY = (
    Path(__file__).resolve().parents[1]
    / "documents"
    / "manuals"
    / "naaccr_data_dictionary_v25.json"
)


class SiteApplicabilityTests(unittest.TestCase):
    def test_item_832_accepts_coded_breast_primary_site(self):
        group = next(
            group
            for group in load_variable_groups(VARIABLE_GROUPS)
            if any(variable.item_id == 832 for variable in group.variables)
        )

        self.assertTrue(site_applies(
            group.applies_to, CaseFacts(primary_site="C50.4"), tissue_codes=TISSUE_CODES
        ))
        self.assertFalse(site_applies(
            group.applies_to, CaseFacts(primary_site="C34.9"), tissue_codes=TISSUE_CODES
        ))


class VariableCitationTests(unittest.TestCase):
    def test_primary_citation_preserves_original_scalar_through_json(self):
        for note_id in (1, "001", "note-A", None):
            with self.subTest(note_id=note_id):
                candidate = VariableOutput(
                    item_id=400, value="C504", explanation="Synthetic evidence",
                    most_important_note=note_id, spans=[], presence_confidence="max",
                )
                restored = VariableOutput.model_validate_json(candidate.model_dump_json())
                self.assertEqual(restored.most_important_note, note_id)
                self.assertIs(type(restored.most_important_note), type(note_id))


class CorpusCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.note = ProcessedClinicalNote(
            note_id=1,
            date="2025-02-20",
            note_type="Pathology",
            content="Left breast core biopsy.",
            cancer_mentions=[
                CancerMention(
                    presence=True,
                    confidence=ConfidenceLevel.HIGH,
                    evidence=[TextSpan(note_id=1, text="Left breast core biopsy.")],
                    status="current",
                    affected_tissue="breast",
                    metastasis=False,
                )
            ],
            cancer_status={"current"},
        )
        self.breast_note = self.note.model_copy(update={"note_id": 2})
        self.free_text_note = self.note.model_copy(
            update={
                "note_id": 3,
                "cancer_mentions": [
                    self.note.cancer_mentions[0].model_copy(
                        update={"affected_tissue": "left breast"}
                    )
                ],
            }
        )

    def agent(self, base_dictionary=None):
        if base_dictionary is None:
            directory = TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            base_dictionary = Path(directory.name) / "base.json"
            base_dictionary.write_text(json.dumps({"400": {
                "item_name": "Primary Site", "item_length": 4,
                "allowed_codes": primary_site_codes(),
            }, "522": {
                "item_name": "Histologic Type ICD-O-3", "item_length": 4,
                "allowed_codes": histology_pair_codes(),
            }}))
        agent = object.__new__(OrchestratorAgent)
        agent._data_dictionary_path = base_dictionary
        agent._site_data_dictionary_path = SCOPING_DICTIONARY
        return agent

    def test_affected_tissue_is_kept_as_a_complete_name(self):
        descriptors = build_corpus_descriptors({1: self.note, 3: self.free_text_note})

        self.assertEqual(
            descriptors.affected_tissues, {"current": {"breast", "left breast"}}
        )

    def test_characterization_sets_gross_primary_site_before_planning(self):
        state = CaseState(note_corpus={1: self.note, 2: self.breast_note})

        update = self.agent().characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    def test_single_cancer_mention_sets_gross_primary_site(self):
        state = CaseState(note_corpus={1: self.note})

        update = self.agent().characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    def test_a_name_outside_the_vocabulary_counts_as_unspecified(self):
        for corpus in ({3: self.free_text_note}, {1: self.note, 3: self.free_text_note}):
            with self.subTest(notes=list(corpus)):
                update = self.agent().characterize_corpus(CaseState(note_corpus=corpus))

                self.assertIsNone(update["case_facts"])

    def corpus(self, *mentions):
        """One note holding (status, tissue, behavior, diagnostic_certainty) mentions."""
        template = self.note.cancer_mentions[0]
        return {1: self.note.model_copy(update={"cancer_mentions": [
            template.model_copy(update={
                "status": status, "affected_tissue": tissue,
                "behavior": behavior, "diagnostic_certainty": certainty,
            })
            for status, tissue, behavior, certainty in mentions
        ]})}

    def test_several_tissues_become_candidates_without_a_gross_site(self):
        corpus = self.corpus(
            ("current", "ovary", "malignant", "definite"),
            ("current", "breast", "malignant", "probable"),
        )
        self.assertEqual(candidate_gross_sites(corpus), ["breast", "ovary"])

        facts = self.agent().characterize_corpus(CaseState(note_corpus=corpus))["case_facts"]
        self.assertEqual(facts.candidate_gross_sites, ["breast", "ovary"])
        self.assertIsNone(facts.gross_primary_site)

    def test_one_candidate_also_sets_the_gross_site(self):
        facts = self.agent().characterize_corpus(CaseState(note_corpus={1: self.note}))["case_facts"]
        self.assertEqual(facts.candidate_gross_sites, ["breast"])
        self.assertEqual(facts.gross_primary_site, "breast")

    def test_a_current_benign_tumor_blocks_an_older_cancer(self):
        for behavior in ("benign", "unknown"):
            with self.subTest(behavior=behavior):
                corpus = self.corpus(
                    ("current", "meninges", behavior, "definite"),
                    ("historical", "breast", "malignant", "definite"),
                )
                self.assertEqual(candidate_gross_sites(corpus), [])
                self.assertIsNone(self.agent().characterize_corpus(CaseState(note_corpus=corpus))["case_facts"])

    def test_a_current_possible_mention_doesnt_block_history(self):
        corpus = self.corpus(
            ("current", "lung", "malignant", "possible"),
            ("recent", "breast", "malignant", "definite"),
            ("historical", "ovary", "malignant", "definite"),
        )
        self.assertEqual(candidate_gross_sites(corpus), ["breast"])

    def test_an_unspecified_tissue_or_no_tissue_gives_no_candidates(self):
        self.assertEqual(candidate_gross_sites(self.corpus(
            ("current", "breast", "malignant", "definite"),
            ("current", "unspecified", "malignant", "definite"),
            ("historical", "lung", "malignant", "definite"),
        )), [])
        self.assertEqual(candidate_gross_sites({}), [])

    def test_known_site_facts_are_kept(self):
        for facts in (
            CaseFacts(primary_site="C341"),
            CaseFacts(gross_primary_site="lung"),
            CaseFacts(candidate_gross_sites=["lung", "ovary"]),
        ):
            with self.subTest(facts=facts):
                state = CaseState(note_corpus={1: self.note}, case_facts=facts)
                self.assertIs(self.agent().characterize_corpus(state)["case_facts"], facts)

    def test_characterization_does_not_need_the_site_dictionary(self):
        agent = object.__new__(OrchestratorAgent)
        agent._site_data_dictionary_path = None
        state = CaseState(note_corpus={1: self.note})

        update = agent.characterize_corpus(state)

        self.assertEqual(update["case_facts"].gross_primary_site, "breast")

    def test_initial_group_uses_breast_codes_after_characterization(self):
        if not BASE_DICTIONARY.exists():
            self.skipTest(f"{BASE_DICTIONARY} not present")
        state = CaseState(note_corpus={1: self.note, 2: self.breast_note})
        agent = self.agent(BASE_DICTIONARY)
        facts = agent.characterize_corpus(state)["case_facts"]
        initial_group = load_variable_groups(VARIABLE_GROUPS)[0]

        scoped = agent._scope_group(initial_group, facts)
        primary_site = next(
            variable for variable in scoped.variables if variable.item_id == 400
        )

        # Breast's nine codes, plus C809 for an unknown primary.
        self.assertEqual(len(primary_site.valid_codes), 10)
        self.assertIn("C504", primary_site.valid_codes)
        self.assertIn("C809", primary_site.valid_codes)
        self.assertNotIn("C341", primary_site.valid_codes)

    def test_characterization_and_scoping_read_each_dictionary_once(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base_path = Path(directory.name) / "base.json"
        site_path = Path(directory.name) / "site.json"
        base_path.write_text(json.dumps({"400": {
            "item_name": "Primary Site", "item_length": 4,
            "allowed_codes": primary_site_codes(),
        }, "522": {
            "item_name": "Histologic Type ICD-O-3", "item_length": 4,
            "allowed_codes": histology_pair_codes(),
        }}))
        site_path.write_text(SCOPING_DICTIONARY.read_text())
        agent = object.__new__(OrchestratorAgent)
        agent._data_dictionary_path = base_path
        agent._site_data_dictionary_path = site_path
        group = TargetGroup(group_id="initial", variables=[VariableInfo(item_id=400)])

        with patch("builtins.open", wraps=open) as opened:
            for _ in range(2):
                state = CaseState(note_corpus={2: self.breast_note})
                facts = agent.characterize_corpus(state)["case_facts"]
                scoped = agent._scope_group(group, facts)

        for path in (base_path, site_path):
            reads = [call for call in opened.call_args_list if Path(call.args[0]) == path]
            self.assertEqual(len(reads), 1, path.name)
        self.assertEqual(facts.gross_primary_site, "breast")
        self.assertEqual(set(scoped.variables[0].valid_codes), TISSUE_CODES["breast"] | {"C809"})


class HistologyScopeWiringTests(unittest.TestCase):
    def agent(self, pairs=None):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base_path = Path(directory.name) / "base.json"
        base_path.write_text(json.dumps({
            "400": {"item_name": "Primary Site", "item_length": 4, "allowed_codes": primary_site_codes()},
            "522": {"item_name": "Histologic Type ICD-O-3", "item_length": 4,
                    "allowed_codes": histology_pair_codes() if pairs is None else pairs},
            "523": {"item_name": "Behavior Code ICD-O-3", "item_length": 1,
                    "allowed_codes": {"0": "Benign", "1": "Borderline", "2": "In situ", "3": "Malignant"}},
        }))
        agent = object.__new__(OrchestratorAgent)
        agent._data_dictionary_path = base_path
        agent._site_data_dictionary_path = SCOPING_DICTIONARY
        return agent

    def scoped(self, agent, facts):
        group = TargetGroup(
            group_id="histologic_type_and_behavior", stage="histology", extract_as_group=True,
            variables=[VariableInfo(item_id=522), VariableInfo(item_id=523)],
        )
        return {variable.item_id: variable for variable in agent._scope_group(group, facts).variables}

    def test_a_coded_breast_site_scopes_histology_to_the_breast_list(self):
        agent = self.agent()
        for primary in ("C504", "C50.4"):
            with self.subTest(primary=primary):
                variables = self.scoped(agent, CaseFacts(primary_site=primary))
                scope = variables[522].histology_scope
                self.assertEqual(scope.site_group, "breast")
                reachable = {pair for codes in scope.family_codes.values() for pair in codes}
                # The old breast list missed angiosarcoma and malignant phyllodes.
                self.assertTrue({"9120/3", "9020/3"} <= reachable)
                self.assertIsNone(variables[523].histology_scope)

    def test_the_diagnosis_year_limits_the_histology_offered(self):
        agent = self.agent()
        offered = {}
        for date in (None, "20120315"):
            scope = self.scoped(agent, CaseFacts(primary_site="C504", date_of_diagnosis=date))[522].histology_scope
            offered[date] = {pair for codes in (*scope.family_codes.values(), *scope.block_codes.values()) for pair in codes}
        # CS (2004-2015) rejects solid papillary carcinoma and NUT carcinoma.
        self.assertTrue({"8509/3", "8023/3"} <= offered[None])
        self.assertTrue({"8509/3", "8023/3"}.isdisjoint(offered["20120315"]))

    def test_an_uncoded_site_scopes_histology_to_the_unknown_primary_list(self):
        agent = self.agent()
        for facts in (None, CaseFacts(), CaseFacts(gross_primary_site="breast")):
            with self.subTest(facts=facts):
                self.assertEqual(self.scoped(agent, facts)[522].histology_scope.site_group, "unknown")

    def test_without_a_site_dictionary_histology_has_no_scope(self):
        agent = self.agent()
        agent._site_data_dictionary_path = None
        self.assertIsNone(self.scoped(agent, CaseFacts(primary_site="C504"))[522].histology_scope)

    def test_the_site_lists_must_only_offer_the_dictionarys_histology_pairs(self):
        pairs = histology_pair_codes()
        del pairs["9120/3"]
        with self.assertRaisesRegex(ValueError, "9120/3"):
            self.agent(pairs)._site_dictionary


def _extracted(item_id, value, confidence=ConfidenceLevel.HIGH):
    return CaseVariableResult(
        item_id=item_id, status=VariableStatus.EXTRACTED, value=value,
        extraction=ValidatedVariableOutput(
            item_id=item_id, value=value, explanation="Coded.", most_important_note=1,
            spans=[], presence_confidence=confidence, is_valid=True,
        ),
    )


def _histology(value, *, families=(), blocks=(), listed=None, status=VariableStatus.EXTRACTED, confidence=ConfidenceLevel.HIGH):
    """A 522 result as the two histology steps leave it."""
    return CaseVariableResult(
        item_id=522, status=status, value=value,
        extraction=ValidatedVariableOutput(
            item_id=522, value=value, explanation="Coded.", most_important_note=1 if value else None,
            spans=[], presence_confidence=confidence, is_valid=True,
            histology=HistologySelection(
                families=list(families), blocks=list(blocks), reasoning="Chosen.", listed=listed,
            ),
        ),
    )


class HistologyFamilyFactsTests(unittest.TestCase):
    def test_chosen_families_reach_the_case_facts(self):
        facts = derive_case_facts(CaseFacts(), {522: _histology("8500", families=["850", "852"], listed=True)})
        self.assertEqual((facts.histology, facts.candidate_histology_families), ("8500", ["850", "852"]))

    def test_families_from_a_not_found_histology_still_count(self):
        result = _histology(None, families=["967", "968"], status=VariableStatus.NOT_FOUND)
        facts = derive_case_facts(None, {522: result})
        self.assertEqual((facts.histology, facts.candidate_histology_families), (None, ["967", "968"]))

    def test_block_fallback_and_known_candidates_add_nothing(self):
        before = CaseFacts(histology="9120")
        self.assertIs(derive_case_facts(before, {522: _histology("9120", blocks=["912-916"], listed=False)}), before)
        known = CaseFacts(candidate_histology_families=["850"])
        self.assertIs(derive_case_facts(known, {522: _histology(None, families=["967"], status=VariableStatus.NOT_FOUND)}), known)

    def test_old_case_facts_without_candidate_families_load(self):
        facts = CaseFacts.model_validate({"primary_site": "C504", "histology": "8500"})
        self.assertEqual(facts.candidate_histology_families, [])
        restored = CaseFacts.model_validate_json(
            CaseFacts(candidate_histology_families=["967"]).model_dump_json()
        )
        self.assertEqual(restored.candidate_histology_families, ["967"])


class OutOfYearResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)
        cls.groups = [
            TargetGroup(group_id="initial", stage="initial", variables=[VariableInfo(item_id=390)]),
            TargetGroup(group_id="site_specific_codes", stage="dependent", variables=[
                VariableInfo(item_id=670, name="RX Hosp--Surg Prim Site 03-2022"),
                VariableInfo(item_id=671, name="RX Hosp--Surg Prim Site 2023"),
                VariableInfo(item_id=3843, name="Grade Clinical"),
                VariableInfo(item_id=3844, name="Grade Pathological"),
            ]),
        ]

    def closed(self, date, results=None):
        if results is None:
            results = {
                390: CaseVariableResult(item_id=390, status=VariableStatus.STRUCTURED_DATA, value=date or "99999999"),
                **{item_id: CaseVariableResult(item_id=item_id) for item_id in (670, 671, 3843, 3844)},
            }
        return out_of_year_results(self.groups, results, self.scoping, date)

    def test_each_item_in_a_group_is_decided_on_its_own(self):
        closed = self.closed("20150610")
        # CS held grade in a Site-Specific Factor for some schemas, so 3843 and
        # 3844 wait for their group's planning (ssf_equivalent_results).
        self.assertEqual(sorted(closed), [671])
        self.assertEqual(
            closed[671].reason, "Not collected for 2015 diagnoses: RX Hosp--Surg Prim Site 2023 (671) applies from 2023."
        )
        self.assertTrue(all(result.status == VariableStatus.NOT_APPLICABLE for result in closed.values()))

    def test_the_surgery_items_swap_in_2023(self):
        self.assertEqual(sorted(self.closed("20220601")), [671])
        closed = self.closed("20230601")
        self.assertEqual(sorted(closed), [670])
        self.assertEqual(
            closed[670].reason,
            "Not collected for 2023 diagnoses: RX Hosp--Surg Prim Site 03-2022 (670) applies through 2022.",
        )

    def test_an_unknown_year_or_no_site_dictionary_closes_nothing(self):
        self.assertEqual(self.closed(None), {})
        self.assertEqual(self.closed("2000"), {})
        results = {item_id: CaseVariableResult(item_id=item_id) for item_id in (390, 670, 671, 3843, 3844)}
        self.assertEqual(out_of_year_results(self.groups, results, None, "20150610"), {})

    def test_terminal_results_are_left_alone(self):
        results = {
            390: CaseVariableResult(item_id=390, status=VariableStatus.STRUCTURED_DATA, value="20230601"),
            670: CaseVariableResult(item_id=670, status=VariableStatus.STRUCTURED_DATA, value="00"),
            671: CaseVariableResult(item_id=671, status=VariableStatus.NOT_APPLICABLE, reason="Not reportable."),
            3843: CaseVariableResult(item_id=3843, status=VariableStatus.STRUCTURED_DATA, value="2"),
            3844: CaseVariableResult(item_id=3844),
        }
        # 670 ends in 2022, but a value already supplied is kept.
        self.assertEqual(self.closed("20230601", results), {})
        results[670] = CaseVariableResult(item_id=670)
        self.assertEqual(sorted(self.closed("20230601", results)), [670])


class ItemYearsRunTests(unittest.TestCase):
    """Once 390 is coded, the items its year doesn't collect are never extracted."""

    def run_case(self, outcomes, structured_data=None, groups=None):
        agent = build_fake_orchestrator(Script(outcomes={**LUNG, **outcomes}))
        if groups is not None:
            agent._target_variables = groups
            agent._target_group_hierarchy = []
            agent._graph = agent._build_graph()
        # The fake's _scope_group is a pass-through, so only planning reads it.
        agent._site_dictionary = load_scoping_dictionary(SCOPING_DICTIONARY)
        with patch.object(agent._extractor, "run", wraps=agent._extractor.run) as extract:
            result = agent.run(
                [note.model_dump(mode="json") for note in load_notes()],
                structured_data=structured_data, progress=False,
            )
        extracted = {
            variable.item_id
            for call in extract.call_args_list
            for variable in call.args[0].requested_variables.variables
        }
        return result.case.variable_results, extracted

    def test_a_2015_diagnosis_closes_grade_and_extracts_the_rest(self):
        results, extracted = self.run_case({390: Outcome(value="20150610")})

        self.assertEqual(results[390].value, "20150610")
        for item_id in (3843, 3844):
            self.assertEqual(results[item_id].status, VariableStatus.NOT_APPLICABLE)
            self.assertIn("Not collected for 2015 diagnoses", results[item_id].reason)
        # The default config holds Grade (440) for the years before 3843.
        self.assertEqual(
            results[3843].reason,
            "Not collected for 2015 diagnoses: Grade Clinical (3843) applies from 2018; "
            "Grade (440) holds it for this year.",
        )
        # Summary Stage 2018, Tumor Size Summary and Mets at DX start later too.
        for item_id in (671, 764, 756, 1112, 682, 832):
            self.assertEqual(results[item_id].status, VariableStatus.NOT_APPLICABLE, item_id)
        self.assertTrue({671, 764, 756, 1112, 682, 832, 3843, 3844}.isdisjoint(extracted))
        # Their older equivalents are extracted in their place.
        for item_id in (670, 1182, 820, 830, 3280, 522, 690, 440, 759, 2800, 2850, 2854):
            self.assertEqual(results[item_id].status, VariableStatus.EXTRACTED, item_id)

    def test_the_surgery_item_follows_the_diagnosis_year(self):
        for date, closed, kept in (("20220601", 671, 670), ("20230601", 670, 671)):
            with self.subTest(date=date):
                results, extracted = self.run_case({390: Outcome(value=date)})
                self.assertEqual(results[closed].status, VariableStatus.NOT_APPLICABLE)
                self.assertNotIn(closed, extracted)
                self.assertEqual(results[kept].status, VariableStatus.EXTRACTED)
                self.assertIn(kept, extracted)

    def test_an_uncoded_diagnosis_date_extracts_both(self):
        results, extracted = self.run_case({390: Outcome(value=None)})

        self.assertEqual(results[390].status, VariableStatus.NOT_FOUND)
        self.assertTrue({670, 671, 3843, 3844} <= extracted)
        self.assertEqual(
            {results[item_id].status for item_id in (670, 671, 3843, 3844)}, {VariableStatus.EXTRACTED}
        )

    def test_a_structured_value_outside_its_years_is_kept(self):
        results, extracted = self.run_case({}, structured_data={390: "20150610", 3843: "2"})

        self.assertEqual((results[3843].status, results[3843].value), (VariableStatus.STRUCTURED_DATA, "2"))
        self.assertEqual(results[3844].status, VariableStatus.NOT_APPLICABLE)
        self.assertTrue({390, 3843, 3844}.isdisjoint(extracted))

    def test_a_stage_left_with_nothing_to_extract_releases_the_next(self):
        # Neither item has an equivalent, so both close as soon as 390 is coded.
        # (Grade would wait for its group: CS breast held it in an SSF.)
        groups = [
            TargetGroup(group_id="initial", stage="initial", variables=[VariableInfo(item_id=390)]),
            TargetGroup(group_id="dates", stage="histology", variables=[
                VariableInfo(item_id=682, name="Date Regional Lymph Node Dissection"),
                VariableInfo(item_id=832, name="Date of Sentinel Lymph Node Biopsy"),
            ]),
            TargetGroup(group_id="nodes", stage="dependent", variables=[VariableInfo(item_id=820)]),
        ]
        results, extracted = self.run_case({390: Outcome(value="20150610")}, groups=groups)

        self.assertEqual(
            {item_id: result.status for item_id, result in results.items()},
            {390: VariableStatus.EXTRACTED, 682: VariableStatus.NOT_APPLICABLE,
             832: VariableStatus.NOT_APPLICABLE, 820: VariableStatus.EXTRACTED},
        )
        self.assertEqual(extracted, {390, 820})


class OlderEquivalentConfigTests(unittest.TestCase):
    """The older equivalents sit in the groups of the items they stand in for."""

    MAPPING = {
        "site_specific_codes": {440: [3843, 3844]},
        "other": {759: [764], 2800: [756]},
        "metastases": {
            2850: [1112, 1113, 1114, 1115, 1116, 1117],
            2851: [1112], 2852: [1113], 2853: [1115], 2854: [1116],
        },
    }

    @staticmethod
    def load(variables):
        config = json.dumps({"groups": [{"group_id": "grade", "variables": variables}]})
        with patch("cipoc.tools.orchestration.open", mock_open(read_data=config)):
            return load_variable_groups("synthetic.json")

    def test_the_default_config_maps_each_older_item_to_its_replacements(self):
        replaces = {
            group.group_id: group.replaces
            for group in load_variable_groups(VARIABLE_GROUPS)
            if group.replaces
        }
        self.assertEqual(replaces, self.MAPPING)

    def test_the_older_items_carry_their_v25_names(self):
        path = Path(__file__).resolve().parents[1] / "documents" / "manuals" / "naaccr_data_dictionary_v25.json"
        if not path.exists():
            raise unittest.SkipTest(f"{path} isn't present; it's gitignored reference data.")
        dictionary = json.loads(path.read_text())
        for group in load_variable_groups(VARIABLE_GROUPS):
            for variable in group.variables:
                if variable.item_id in group.replaces:
                    with self.subTest(item_id=variable.item_id):
                        self.assertEqual(variable.name, dictionary[str(variable.item_id)]["Data Item Name"])

    def test_replaces_is_read_into_the_group(self):
        group, = self.load([
            {"item_id": 3843, "name": "Grade Clinical"},
            {"item_id": 440, "name": "Grade", "replaces": [3843]},
        ])
        self.assertEqual(group.replaces, {440: [3843]})
        self.assertEqual([variable.item_id for variable in group.variables], [3843, 440])

    def test_replaces_must_name_other_items_in_the_group(self):
        for replaces, message in (
            ([], "is empty"),
            ([440], "can't replace itself"),
            ([3843, 3844], r"replaces \[3844\], which aren't in the group"),
        ):
            with self.subTest(replaces=replaces), self.assertRaisesRegex(ValueError, message):
                self.load([
                    {"item_id": 3843, "name": "Grade Clinical"},
                    {"item_id": 440, "name": "Grade", "replaces": replaces},
                ])


class OlderEquivalentReasonTests(unittest.TestCase):
    """Each reason names the equivalent that's collected that year, in both directions."""

    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)
        cls.groups = [
            group for group in load_variable_groups(VARIABLE_GROUPS)
            if group.group_id in ("site_specific_codes", "other", "metastases", "lymph_node_removal")
        ]

    def reasons(self, date, groups=None):
        groups = groups or self.groups
        results = {
            variable.item_id: CaseVariableResult(item_id=variable.item_id)
            for group in groups for variable in group.variables
        }
        return {item_id: result.reason for item_id, result in out_of_year_results(groups, results, self.scoping, date).items()}

    def test_a_newer_item_names_the_older_one_that_holds_it(self):
        reasons = self.reasons("20150610")
        # Grade waits for its group's planning, since CS held it in an SSF for
        # some schemas. CS lung didn't, so only Grade (440) holds it.
        self.assertNotIn(3844, reasons)
        lung = CaseFacts(primary_site="C341", histology="8140", behavior="3", date_of_diagnosis="20150610")
        results = {
            variable.item_id: CaseVariableResult(item_id=variable.item_id)
            for group in self.groups for variable in group.variables
        }
        settled, equivalents = ssf_equivalent_results(self.groups, self.groups, results, self.scoping, lung, {})
        self.assertEqual(equivalents, {})
        self.assertEqual(
            settled[3844].reason,
            "Not collected for 2015 diagnoses: Grade Pathological (3844) applies from 2018; "
            "Grade (440) holds it for this year.",
        )
        self.assertEqual(
            reasons[764],
            "Not collected for 2015 diagnoses: Summary Stage 2018 (764) applies from 2018; "
            "SEER Summary Stage 2000 (759) holds it for this year.",
        )
        self.assertNotIn(440, reasons)

    def test_an_older_item_names_the_newer_ones_that_replace_it(self):
        reasons = self.reasons("20200610")
        self.assertEqual(
            reasons[440],
            "Not collected for 2020 diagnoses: Grade (440) applies through 2017; "
            "Grade Clinical (3843) and Grade Pathological (3844) replace it.",
        )
        self.assertEqual(
            reasons[2800],
            "Not collected for 2020 diagnoses: CS Tumor Size (2800) applies from 2003 through 2015; "
            "Tumor Size Summary (756) replaces it.",
        )
        self.assertEqual(
            reasons[2850],
            "Not collected for 2020 diagnoses: CS Mets at DX (2850) applies from 2003 through 2015; "
            "Mets at DX-Bone (1112), Mets at DX-Brain (1113), Mets at Dx-Distant LN (1114), "
            "Mets at DX-Liver (1115), Mets at DX-Lung (1116) and Mets at DX-Other (1117) replace it.",
        )

    def test_only_equivalents_collected_that_year_are_named(self):
        # CS Mets at Dx-Bone (2851) starts in 2010, CS Mets at DX (2850) in 2003.
        self.assertEqual(
            self.reasons("20120610")[1112],
            "Not collected for 2012 diagnoses: Mets at DX-Bone (1112) applies from 2016; "
            "CS Mets at DX (2850) and CS Mets at Dx-Bone (2851) hold it for this year.",
        )
        reasons = self.reasons("20080610")
        self.assertEqual(
            reasons[1112],
            "Not collected for 2008 diagnoses: Mets at DX-Bone (1112) applies from 2016; "
            "CS Mets at DX (2850) holds it for this year.",
        )
        self.assertEqual(
            reasons[2851],
            "Not collected for 2008 diagnoses: CS Mets at Dx-Bone (2851) applies from 2010 through 2015.",
        )

    def test_only_equivalents_requested_in_the_run_are_named(self):
        grade = TargetGroup(
            group_id="grade",
            variables=[VariableInfo(item_id=3843, name="Grade Clinical"), VariableInfo(item_id=440, name="Grade")],
            replaces={440: [3843, 3844]},
        )
        self.assertEqual(
            self.reasons("20200610", [grade])[440],
            "Not collected for 2020 diagnoses: Grade (440) applies through 2017; Grade Clinical (3843) replaces it.",
        )

    def test_an_item_without_an_equivalent_keeps_the_plain_reason(self):
        self.assertEqual(
            self.reasons("20150610")[682],
            "Not collected for 2015 diagnoses: Date Regional Lymph Node Dissection (682) applies from 2018.",
        )


class OlderEquivalentRunTests(unittest.TestCase):
    """End to end on the default config: each year extracts one of each pair."""

    def run_case(self, date):
        agent = build_fake_orchestrator(Script(outcomes={**LUNG, 390: Outcome(value=date)}))
        agent._site_dictionary = load_scoping_dictionary(SCOPING_DICTIONARY)
        with patch.object(agent._extractor, "run", wraps=agent._extractor.run) as extract:
            result = agent.run([note.model_dump(mode="json") for note in load_notes()], progress=False)
        extracted = {
            variable.item_id
            for call in extract.call_args_list
            for variable in call.args[0].requested_variables.variables
        }
        return result.case.variable_results, extracted

    def assert_pair(self, results, extracted, closed, kept):
        self.assertEqual(results[closed].status, VariableStatus.NOT_APPLICABLE, closed)
        self.assertNotIn(closed, extracted)
        self.assertEqual(results[kept].status, VariableStatus.EXTRACTED, kept)
        self.assertIn(kept, extracted)

    def test_2015_codes_grade_in_440(self):
        results, extracted = self.run_case("20150610")
        for item_id in (3843, 3844):
            self.assert_pair(results, extracted, item_id, 440)
            self.assertIn("Grade (440) holds it for this year.", results[item_id].reason)

    def test_tumor_size_swaps_between_2014_and_2017(self):
        results, extracted = self.run_case("20140610")
        self.assert_pair(results, extracted, 756, 2800)
        self.assertIn("CS Tumor Size (2800) holds it for this year.", results[756].reason)
        results, extracted = self.run_case("20170610")
        self.assert_pair(results, extracted, 2800, 756)
        self.assertIn("Tumor Size Summary (756) replaces it.", results[2800].reason)

    def test_2012_codes_summary_stage_in_759(self):
        results, extracted = self.run_case("20120610")
        self.assert_pair(results, extracted, 764, 759)
        self.assertIn("SEER Summary Stage 2000 (759) holds it for this year.", results[764].reason)

    def test_2012_codes_mets_in_the_cs_items(self):
        results, extracted = self.run_case("20120610")
        for item_id in (1112, 1113, 1114, 1115, 1116, 1117):
            self.assertEqual(results[item_id].status, VariableStatus.NOT_APPLICABLE, item_id)
            self.assertNotIn(item_id, extracted)
        self.assertIn("CS Mets at DX (2850) holds it for this year.", results[1114].reason)
        for item_id in (2850, 2851, 2852, 2853, 2854):
            self.assertEqual(results[item_id].status, VariableStatus.EXTRACTED, item_id)
            self.assertIn(item_id, extracted)

    def test_an_unknown_year_extracts_current_and_older_items(self):
        results, extracted = self.run_case(None)
        self.assertEqual(results[390].status, VariableStatus.NOT_FOUND)
        both = {3843, 3844, 440, 764, 759, 756, 2800, *range(1112, 1118), *range(2850, 2855)}
        self.assertTrue(both <= extracted)
        self.assertEqual({results[item_id].status for item_id in both}, {VariableStatus.EXTRACTED})


def _cases(primary_site, histology, date_of_diagnosis):
    """Fake-extractor outcomes coding the case's site, histology and year."""
    return {
        390: Outcome(value=date_of_diagnosis), 400: Outcome(value=primary_site),
        522: Outcome(value=histology), 523: Outcome(value="3"),
    }


class SsfEquivalentResultsTests(unittest.TestCase):
    """At planning, each item CS held in an SSF is replaced, blocked or closed."""

    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)
        cls.groups = load_variable_groups(VARIABLE_GROUPS)
        cls.by_id = {group.group_id: group for group in cls.groups}

    def settle(self, facts, group_ids=("site_specific_codes",), results=None, structured_data=None, equivalents=None):
        if results is None:
            results = {
                variable.item_id: CaseVariableResult(item_id=variable.item_id)
                for group in self.groups for variable in group.variables
            }
        ready = [self.by_id[group_id] for group_id in group_ids]
        return ssf_equivalent_results(
            ready, self.groups, results, self.scoping, facts, equivalents or {}, structured_data,
        )

    @staticmethod
    def facts(primary_site, histology, date_of_diagnosis="20120615"):
        return CaseFacts(primary_site=primary_site, histology=histology, behavior="3", date_of_diagnosis=date_of_diagnosis)

    def test_both_grade_items_share_the_breast_grade_ssf(self):
        settled, equivalents = self.settle(self.facts("C504", "8500"))

        self.assertEqual(sorted(settled), [2861, 3843, 3844])
        self.assertEqual(settled[2861], CaseVariableResult(item_id=2861))
        for item_id, name in ((3843, "Grade Clinical"), (3844, "Grade Pathological")):
            self.assertEqual(settled[item_id].status, VariableStatus.NOT_APPLICABLE)
            self.assertEqual(
                settled[item_id].reason,
                f"Not collected for 2012 diagnoses: {name} ({item_id}) applies from 2018; Grade (440) and "
                "CS SSF7 (2861, Nottingham or Bloom-Richardson (BR) Score/Grade) hold it for this year.",
            )
        self.assertEqual(equivalents, {2861: SsfEquivalent(
            item_id=2861, label="CS SSF7", name="Nottingham or Bloom-Richardson (BR) Score/Grade",
            replaces=[3843, 3844], schemas=["cs/breast"], group_id="site_specific_codes",
        )})

    def test_disagreeing_schemas_block_the_item(self):
        settled, equivalents = self.settle(self.facts("C481", "8441"))

        self.assertEqual(equivalents, {})
        self.assertEqual(sorted(settled), [3843, 3844])
        result = settled[3843]
        self.assertEqual((result.status, result.blocking_item_ids), (VariableStatus.BLOCKED, []))
        self.assertEqual(
            result.reason,
            "Not collected for 2012 diagnoses: Grade Clinical (3843) applies from 2018; Grade (440) holds it "
            "for this year; the candidate schemas disagree on the CS Site-Specific Factor that holds it: "
            "cs/peritoneum holds it in CS SSF1 (2880, Grade for Sarcomas); cs/peritoneum_female_gen holds "
            "no SSF for it.",
        )

    def test_a_schema_without_an_ssf_closes_the_item_by_year(self):
        settled, equivalents = self.settle(self.facts("C569", "8441", "20080615"), group_ids=("ovary_only",))

        self.assertEqual(equivalents, {})
        self.assertEqual(settled, {3836: CaseVariableResult(
            item_id=3836, status=VariableStatus.NOT_APPLICABLE,
            reason="Not collected for 2008 diagnoses: FIGO Stage (3836) applies from 2018.",
        )})

    def test_only_ready_groups_are_settled(self):
        settled, _ = self.settle(self.facts("C569", "8441"), group_ids=("ovary_only",))
        self.assertEqual(sorted(settled), [2890, 3836])
        self.assertEqual(self.settle(self.facts("C569", "8441"), group_ids=("other",)), ({}, {}))

    def test_an_eod_year_or_no_site_dictionary_settles_nothing(self):
        self.assertEqual(self.settle(self.facts("C504", "8500", "20200615")), ({}, {}))
        results = {3843: CaseVariableResult(item_id=3843)}
        self.assertEqual(
            ssf_equivalent_results(self.groups, self.groups, results, None, self.facts("C504", "8500"), {}),
            ({}, {}),
        )

    def test_the_ssf_starts_from_structured_data_or_an_existing_result(self):
        settled, _ = self.settle(self.facts("C504", "8500"), structured_data={2861: "050"})
        self.assertEqual(
            settled[2861], CaseVariableResult(item_id=2861, status=VariableStatus.STRUCTURED_DATA, value="050")
        )
        results = {
            3843: CaseVariableResult(item_id=3843),
            2861: CaseVariableResult(item_id=2861, status=VariableStatus.STRUCTURED_DATA, value="070"),
        }
        settled, equivalents = self.settle(self.facts("C504", "8500"), results=results)
        self.assertEqual(sorted(settled), [3843])
        self.assertEqual(equivalents[2861].replaces, [3843])

    def test_the_ssf_joins_its_group_for_planning(self):
        _, equivalents = self.settle(self.facts("C504", "8500"))
        planned = with_ssf_equivalents(self.groups, equivalents, self.scoping)

        grade = {group.group_id: group for group in planned}["site_specific_codes"]
        self.assertEqual([variable.item_id for variable in grade.variables], [670, 671, 3843, 3844, 440, 2861])
        self.assertEqual(grade.variables[-1], ssf_variable(self.scoping.schemas["cs/breast"].ssf[3843]))
        others = [group for group in planned if group.group_id != "site_specific_codes"]
        self.assertEqual(others, [group for group in self.groups if group.group_id != "site_specific_codes"])
        # The configured groups are never changed.
        self.assertNotIn(2861, [v.item_id for v in self.by_id["site_specific_codes"].variables])
        self.assertEqual(with_ssf_equivalents(self.groups, {}, self.scoping), self.groups)
        self.assertEqual(with_ssf_equivalents(self.groups, equivalents, None), self.groups)

    def test_an_ssf_the_config_requests_stays_in_its_own_group(self):
        requested = TargetGroup(group_id="ssfs", variables=[VariableInfo(item_id=2861, name="CS SSF7")])
        _, equivalents = self.settle(self.facts("C504", "8500"))
        planned = with_ssf_equivalents([*self.groups, requested], equivalents, self.scoping)
        self.assertEqual(
            [variable.item_id for variable in planned[[g.group_id for g in planned].index("site_specific_codes")].variables],
            [670, 671, 3843, 3844, 440],
        )

    def test_a_reportability_stop_closes_a_pending_ssf(self):
        results = {
            3836: CaseVariableResult(item_id=3836, status=VariableStatus.NOT_APPLICABLE, reason="Replaced."),
            2890: CaseVariableResult(item_id=2890),
        }
        self.assertEqual(sorted(stop_pending_results(results, ExtractionStop.NOT_REPORTABLE)), [2890])


class SsfScopeTests(unittest.TestCase):
    """A scoped SSF keeps its schema's name, notes and codes, with the NAACCR format."""

    SSF_ENTRY = {
        "Data Item Name": "CS Site-Specific Factor {n}", "Length": "3", "Data Type": "digits",
        "Allowable Values": "000-999 (site-specific)", "Format": "Right justified, zero filled",
        "Instructions for Coding": "The information recorded in CS Site-Specific Factor {n} differs for each anatomic site.",
        "Code Descriptions": {},
    }

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        base_path = Path(directory.name) / "base.json"
        entries = {
            "400": {"item_name": "Primary Site", "item_length": 4, "allowed_codes": primary_site_codes()},
            "522": {"item_name": "Histologic Type ICD-O-3", "item_length": 4, "allowed_codes": histology_pair_codes()},
        }
        for item_id, n in ((2862, 8), (2890, 2)):
            entries[str(item_id)] = {
                key: value.format(n=n) if isinstance(value, str) else value for key, value in self.SSF_ENTRY.items()
            }
        base_path.write_text(json.dumps(entries))
        self.agent = object.__new__(OrchestratorAgent)
        self.agent._data_dictionary_path = base_path
        self.agent._site_data_dictionary_path = SCOPING_DICTIONARY
        self.scoping = load_scoping_dictionary(SCOPING_DICTIONARY)

    def scoped(self, schema, item_id):
        mapping = self.scoping.schemas[schema].ssf[item_id]
        group = TargetGroup(group_id="ssf", variables=[ssf_variable(mapping)])
        facts = CaseFacts(date_of_diagnosis="20120615")
        return mapping, self.agent._scope_group(group, facts).variables[0]

    def validate(self, variable, value):
        candidate = VariableOutput(
            item_id=variable.item_id, value=value, explanation="Synthetic.", most_important_note=1,
            spans=[], presence_confidence=ConfidenceLevel.HIGH,
        )
        return VariableValueValidator().validate(variable, candidate)

    def test_the_schemas_name_and_codes_replace_the_generic_ssf(self):
        mapping, variable = self.scoped("cs/ovary", 3836)

        self.assertEqual((variable.item_id, variable.name), (2890, "FIGO Stage"))
        self.assertEqual(variable.description, mapping.definition.strip())
        self.assertEqual(variable.valid_codes, ssf_offered_codes(mapping))
        self.assertEqual((variable.length, variable.format), (3, "Right justified, zero filled"))
        self.assertIsNone(variable.coding_instructions)
        self.assertEqual(self.validate(variable, "310"), [])
        self.assertEqual(self.validate(variable, "888"), ["Value is not one of the variable's allowable codes."])

    def test_a_code_range_accepts_the_values_inside_it(self):
        _, variable = self.scoped("cs/prostate", 3840)

        self.assertEqual(variable.name, "Gleason's Score on Needle Core Biopsy/Transurethral Resection of Prostate (TURP)")
        self.assertIn("002-010", variable.valid_codes)
        for value in ("002", "007", "010", "999"):
            with self.subTest(value=value):
                self.assertEqual(self.validate(variable, value), [])
        for value in ("011", "001"):
            with self.subTest(value=value):
                self.assertEqual(self.validate(variable, value), ["Value is not one of the variable's allowable codes."])


class SsfRunTests(unittest.TestCase):
    """End to end on the default config: an SSF stands in for the items CS held in one."""

    def run_case(self, outcomes, structured_data=None):
        agent = build_fake_orchestrator(Script(outcomes=outcomes))
        agent._site_dictionary = load_scoping_dictionary(SCOPING_DICTIONARY)
        with patch.object(agent._extractor, "run", wraps=agent._extractor.run) as extract:
            result = agent.run(
                [note.model_dump(mode="json") for note in load_notes()],
                structured_data=structured_data, progress=False,
            )
        sent = [
            variable
            for call in extract.call_args_list
            for variable in call.args[0].requested_variables.variables
        ]
        return result, sent

    def test_a_2012_ovary_case_extracts_figo_stage_in_ssf2(self):
        result, sent = self.run_case(_cases("C569", "8441", "20120615"))
        results = result.case.variable_results

        self.assertEqual(results[3836].status, VariableStatus.NOT_APPLICABLE)
        self.assertEqual(
            results[3836].reason,
            "Not collected for 2012 diagnoses: FIGO Stage (3836) applies from 2018; "
            "CS SSF2 (2890, FIGO Stage) holds it for this year.",
        )
        self.assertEqual(results[2890].status, VariableStatus.EXTRACTED)
        offered = [variable for variable in sent if variable.item_id == 2890]
        self.assertEqual(len(offered), 1)
        cs_ovary = load_scoping_dictionary(SCOPING_DICTIONARY).schemas["cs/ovary"].ssf[3836]
        self.assertEqual((offered[0].name, offered[0].valid_codes), ("FIGO Stage", ssf_offered_codes(cs_ovary)))
        self.assertNotIn(3836, [variable.item_id for variable in sent])
        self.assertEqual(result.case.ssf_equivalents, {2890: SsfEquivalent(
            item_id=2890, label="CS SSF2", name="FIGO Stage", replaces=[3836],
            schemas=["cs/ovary"], group_id="ovary_only",
        )})
        self.assertEqual(result.case.note_selection["group:ovary_only"].requested_item_ids, [2890])
        # It wasn't requested, so the run's inputs don't list it.
        self.assertNotIn(2890, [v.item_id for group in result.inputs.target_variables for v in group.variables])

    def test_a_2012_breast_case_extracts_grade_once_in_ssf7(self):
        result, sent = self.run_case(_cases("C504", "8500", "20120615"))
        results = result.case.variable_results

        for item_id in (3843, 3844):
            self.assertEqual(results[item_id].status, VariableStatus.NOT_APPLICABLE)
            self.assertIn("Grade (440) and CS SSF7 (2861, Nottingham", results[item_id].reason)
        self.assertEqual([variable.item_id for variable in sent].count(2861), 1)
        self.assertEqual((results[2861].status, results[440].status), (VariableStatus.EXTRACTED,) * 2)
        self.assertEqual(result.case.ssf_equivalents[2861].replaces, [3843, 3844])

    def test_a_2012_peritoneum_case_blocks_grade(self):
        result, sent = self.run_case(_cases("C481", "8441", "20120615"))
        results = result.case.variable_results

        for item_id in (3843, 3844):
            self.assertEqual(results[item_id].status, VariableStatus.BLOCKED)
            self.assertIn("cs/peritoneum_female_gen holds no SSF for it", results[item_id].reason)
            self.assertEqual(results[item_id].blocking_item_ids, [])
        self.assertEqual(result.case.ssf_equivalents, {})
        self.assertTrue({2880, 3843, 3844}.isdisjoint(variable.item_id for variable in sent))
        self.assertEqual(results[440].status, VariableStatus.EXTRACTED)

    def test_a_structured_ssf_is_used_and_not_extracted(self):
        result, sent = self.run_case(_cases("C569", "8441", "20120615"), structured_data={2890: "310"})

        self.assertEqual(
            (result.case.variable_results[2890].status, result.case.variable_results[2890].value),
            (VariableStatus.STRUCTURED_DATA, "310"),
        )
        self.assertNotIn(2890, [variable.item_id for variable in sent])
        self.assertIn("CS SSF2 (2890, FIGO Stage)", result.case.variable_results[3836].reason)

    def test_an_unknown_year_substitutes_nothing(self):
        result, sent = self.run_case(_cases("C569", "8441", None))

        self.assertEqual(result.case.ssf_equivalents, {})
        self.assertEqual(result.case.variable_results[3836].status, VariableStatus.EXTRACTED)
        self.assertNotIn(2890, result.case.variable_results)

    def test_a_pending_ssf_keeps_the_run_open(self):
        agent = build_fake_orchestrator()
        results = {
            3836: CaseVariableResult(item_id=3836, status=VariableStatus.NOT_APPLICABLE, reason="Replaced."),
            2890: CaseVariableResult(item_id=2890),
        }
        state = CaseState(variable_results=results)
        self.assertEqual(state.outstanding_item_ids, {2890})
        self.assertEqual(agent.route_from_check(state), "plan_extraction")
        done = CaseState(variable_results={**results, 2890: CaseVariableResult(
            item_id=2890, status=VariableStatus.NOT_APPLICABLE, reason="Not reportable.",
        )})
        self.assertEqual(agent.route_from_check(done), "finalize_case")

    def test_a_run_with_an_ssf_result_saves_and_reloads(self):
        result, _ = self.run_case(_cases("C569", "8441", "20120615"))

        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())

        self.assertEqual(restored.case, result.case)
        self.assertEqual(restored.inputs, result.inputs)
        self.assertEqual(restored.case.variable_results[2890].status, VariableStatus.EXTRACTED)
        self.assertEqual(restored.case.ssf_equivalents[2890].replaces, [3836])


class BuildReportTests(unittest.TestCase):
    RESULTS = {400: _extracted(400, "C504"), 410: _extracted(410, "1", ConfidenceLevel.LOW)}

    def flags(self, results, facts):
        return [(flag.item_id, flag.flag_type) for flag in build_report(results, facts).flags]

    def test_no_facts_or_one_candidate_raises_no_site_flag(self):
        expected = [(410, ReviewFlagType.LOW_CONFIDENCE)]
        for facts in (
            None,
            CaseFacts(gross_primary_site="breast"),
            CaseFacts(candidate_gross_sites=["breast"], gross_primary_site="breast"),
        ):
            with self.subTest(facts=facts):
                self.assertEqual(self.flags(self.RESULTS, facts), expected)

    def test_two_candidates_flag_primary_site(self):
        report = build_report(self.RESULTS, CaseFacts(candidate_gross_sites=["breast", "ovary"]))
        self.assertEqual(
            [(flag.item_id, flag.flag_type) for flag in report.flags],
            [(400, ReviewFlagType.MULTIPLE_CANDIDATE_SITES), (410, ReviewFlagType.LOW_CONFIDENCE)],
        )
        self.assertIn("breast, ovary", report.flags[0].detail)

    def test_site_flag_follows_primary_sites_own_flag(self):
        results = {400: _extracted(400, "C504", ConfidenceLevel.LOW)}
        self.assertEqual(
            self.flags(results, CaseFacts(candidate_gross_sites=["breast", "ovary"])),
            [(400, ReviewFlagType.LOW_CONFIDENCE), (400, ReviewFlagType.MULTIPLE_CANDIDATE_SITES)],
        )

    def test_site_flag_is_raised_when_primary_site_was_not_requested(self):
        results = {410: _extracted(410, "1"), 380: _extracted(380, "00")}
        self.assertEqual(
            self.flags(results, CaseFacts(candidate_gross_sites=["breast", "ovary"])),
            [(400, ReviewFlagType.MULTIPLE_CANDIDATE_SITES)],
        )
        self.assertEqual(self.flags(results, CaseFacts(candidate_gross_sites=["breast"])), [])

    def test_histology_off_the_sites_list_is_flagged(self):
        results = {
            522: _histology("9120", blocks=["912-916"], listed=False),
            523: _extracted(523, "3"),
        }
        report = build_report(results)
        self.assertEqual(
            [(flag.item_id, flag.flag_type) for flag in report.flags],
            [(522, ReviewFlagType.UNLISTED_SITE_HISTOLOGY)],
        )
        self.assertIn("9120/3", report.flags[0].detail)

    def test_listed_or_uncoded_histology_raises_no_histology_flag(self):
        for result in (
            _histology("8500", families=["850"], listed=True),
            _histology("8500", families=["850"], listed=None),
            _histology(None, families=["850"], listed=False, status=VariableStatus.NOT_FOUND),
            _extracted(522, "8500"),
        ):
            with self.subTest(result=result):
                self.assertEqual(self.flags({522: result}, None), [])

    def test_histology_flag_follows_its_own_result_flag(self):
        results = {522: _histology("9120", blocks=["912-916"], listed=False, confidence=ConfidenceLevel.LOW)}
        self.assertEqual(
            self.flags(results, None),
            [(522, ReviewFlagType.LOW_CONFIDENCE), (522, ReviewFlagType.UNLISTED_SITE_HISTOLOGY)],
        )


class OrchestratorRunTests(unittest.TestCase):
    class _Renderer:
        min_interval = 0

        def paint(self, snapshot, **kwargs):
            return True

        def close(self):
            pass

    @staticmethod
    def raw_notes():
        return [note.model_dump(mode="json") for note in load_notes()]

    @staticmethod
    def stable_dump(result):
        value = result.model_dump(mode="json")
        for field in ("run_id", "started_at", "finished_at", "duration_seconds"):
            value["run"].pop(field)
        return value

    def test_run_returns_result_from_full_root_state(self):
        agent = build_fake_orchestrator()

        result = agent.run(self.raw_notes(), structured_data={390: "20250101"}, progress=False)

        self.assertIsInstance(result, OrchestratorRunResult)
        # Re-run the independent full-state stream with the same structured seed.
        expected_with_structured = None
        expected_agent = build_fake_orchestrator()
        for raw_event in expected_agent._graph.stream(
            graph_input(structured_data={390: "20250101"}),
            stream_mode=["values", "tasks"],
            subgraphs=True,
        ):
            event = normalize(raw_event, subgraphs=True)
            if event is not None and event.kind == "values" and event.is_root:
                expected_with_structured = event.payload
        self.assertEqual(result.case, CaseState(**expected_with_structured).to_case())
        self.assertEqual(result.inputs.structured_data, {390: "20250101"})
        self.assertEqual(result.inputs.target_variables, agent._target_variables)
        self.assertEqual(len(result.corpus.note_corpus), len(load_notes()))
        self.assertTrue(
            all(
                isinstance(note, ProcessedClinicalNote)
                and note.summary is not None
                and note.concepts
                for note in result.corpus.note_corpus.values()
            )
        )
        restored = OrchestratorRunResult.model_validate_json(result.model_dump_json())
        restored_note = next(iter(restored.corpus.note_corpus.values()))
        self.assertIsInstance(restored_note, ProcessedClinicalNote)
        self.assertIsNotNone(restored_note.summary)
        self.assertTrue(restored_note.concepts)

    def test_run_info_and_fingerprint_have_identity_and_no_secret(self):
        result = build_fake_orchestrator().run(self.raw_notes(), progress=False)

        self.assertEqual(result.run.run_id.version, 4)
        self.assertLessEqual(result.run.started_at, result.run.finished_at)
        self.assertGreaterEqual(result.run.duration_seconds, 0)
        fingerprint = result.run.config_fingerprint
        self.assertTrue(fingerprint.variable_groups_digest.startswith("sha256:"))
        self.assertTrue(fingerprint.prompt_digests)
        self.assertEqual(fingerprint.max_extraction_attempts, 3)
        serialized = json.dumps(fingerprint.model_dump(mode="json"))
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("fake-api-key", serialized)
        self.assertEqual(
            fingerprint.retry["extractor"]["retry_on"],
            "cipoc.llm.retry.retry_on_transient",
        )

    def test_progress_modes_have_identical_non_identity_result(self):
        with patch(
            "cipoc.utils.progress.runner._select_renderer",
            return_value=self._Renderer(),
        ):
            displayed = build_fake_orchestrator().run(
                self.raw_notes(), progress=True, pause_before_summary=False
            )
        headless = build_fake_orchestrator().run(self.raw_notes(), progress=False)

        self.assertEqual(self.stable_dump(displayed), self.stable_dump(headless))

    def test_existing_callback_and_concurrency_survive_collector_attachment(self):
        agent = build_fake_orchestrator()
        original_graph = agent._graph

        class RecordingGraph:
            def __init__(self):
                self.kwargs = None

            def stream(self, graph_input, **kwargs):
                self.kwargs = kwargs
                yield from original_graph.stream(graph_input, **kwargs)

        graph = RecordingGraph()
        agent._graph = graph
        existing_callback = BaseCallbackHandler()
        observed = []

        agent.run(
            self.raw_notes(),
            progress=False,
            config={
                "callbacks": [existing_callback],
                "max_concurrency": 7,
                "tags": ["caller"],
            },
            event_observer=observed.append,
        )

        forwarded = graph.kwargs["config"]
        self.assertEqual(forwarded["max_concurrency"], 7)
        self.assertEqual(forwarded["tags"], ["caller"])
        self.assertIs(forwarded["callbacks"][0], existing_callback)
        self.assertEqual(len(forwarded["callbacks"]), 2)
        self.assertTrue(observed)
        self.assertEqual(graph.kwargs["stream_mode"], ["values", "tasks"])
        self.assertTrue(graph.kwargs["subgraphs"])

    def test_content_disabled_and_no_llm_runs_are_explicit(self):
        no_llm_result = build_fake_orchestrator().run(
            self.raw_notes(),
            progress=False,
        )

        self.assertEqual(no_llm_result.observability.llm_exchanges, {})
        self.assertTrue(no_llm_result.observability.variable_attempts)
        self.assertEqual(
            no_llm_result.observability.llm_usage_summary.model_invocations, 0
        )
        self.assertEqual(no_llm_result.observability.llm_usage_summary.total_tokens, 0)

        agent = build_fake_orchestrator()
        final_state = None
        for raw_event in agent._graph.stream(
            graph_input(), stream_mode=["values", "tasks"], subgraphs=True
        ):
            event = normalize(raw_event, subgraphs=True)
            if event is not None and event.kind == "values" and event.is_root:
                final_state = event.payload

        class ObservedGraph:
            def stream(self, graph_input, **kwargs):
                group_scope = ("extract_branch:group",)
                retrieve_scope = (*group_scope, "retrieve_notes:retrieve")
                yield (), "tasks", {
                    "id": "group",
                    "name": "extract_branch",
                    "input": {"requested_variables": {"group_id": "initial"}},
                }
                yield group_scope, "tasks", {
                    "id": "retrieve",
                    "name": "retrieve_notes",
                    "input": {},
                }
                yield retrieve_scope, "tasks", {
                    "id": "llm",
                    "name": "identify_relevant_notes",
                    "input": {},
                }
                callback = kwargs["config"]["callbacks"][-1]
                callback_run_id = uuid4()
                callback.on_chat_model_start(
                    {},
                    [[HumanMessage(content="sensitive prompt")]],
                    run_id=callback_run_id,
                    metadata={
                        "langgraph_node": "identify_relevant_notes",
                        "langgraph_checkpoint_ns": "|".join(
                            (*retrieve_scope, "identify_relevant_notes:llm")
                        ),
                        "ls_model_name": "fake-model",
                    },
                )
                callback.on_llm_end(
                    {
                        "generations": [[{
                            "message": {
                                "content": '{"relevant_note_ids":[50]}',
                                "usage_metadata": {
                                    "input_tokens": 4,
                                    "output_tokens": 2,
                                    "total_tokens": 6,
                                },
                            }
                        }]]
                    },
                    run_id=callback_run_id,
                )
                yield (), "values", final_state

        agent._graph = ObservedGraph()
        result = agent.run(
            self.raw_notes(),
            progress=False,
            capture_llm_content=False,
            max_content_chars=12,
        )

        exchange = result.observability.llm_exchanges["group:initial"][0]
        self.assertFalse(result.observability.llm_content_captured)
        self.assertEqual(result.observability.max_content_chars, 12)
        self.assertIsNone(exchange.prompt_messages)
        self.assertIsNone(exchange.response)
        self.assertEqual(exchange.usage.total_tokens, 6)
        self.assertEqual(result.observability.llm_usage_summary.total_tokens, 6)

    def test_graph_error_raises_with_partial_corpus_and_chained_cause(self):
        agent = build_fake_orchestrator()
        original_graph = agent._graph

        class FailingGraph:
            def stream(self, graph_input, **kwargs):
                for raw_event in original_graph.stream(graph_input, **kwargs):
                    yield raw_event
                    event = normalize(raw_event, subgraphs=True)
                    if (
                        event is not None
                        and event.kind == "values"
                        and event.is_root
                        and event.payload.get("note_corpus_descriptors") is not None
                    ):
                        group_scope = ("extract_branch:group",)
                        retrieve_scope = (*group_scope, "retrieve_notes:retrieve")
                        yield (), "tasks", {
                            "id": "group",
                            "name": "extract_branch",
                            "input": {
                                "requested_variables": {"group_id": "initial"}
                            },
                        }
                        yield group_scope, "tasks", {
                            "id": "retrieve",
                            "name": "retrieve_notes",
                            "input": {},
                        }
                        yield retrieve_scope, "tasks", {
                            "id": "llm",
                            "name": "identify_relevant_notes",
                            "input": {},
                        }
                        callback = kwargs["config"]["callbacks"][-1]
                        callback_run_id = uuid4()
                        callback.on_chat_model_start(
                            {},
                            [[HumanMessage(content="select notes")]],
                            run_id=callback_run_id,
                            metadata={
                                "langgraph_node": "identify_relevant_notes",
                                "langgraph_checkpoint_ns": "|".join(
                                    (*retrieve_scope, "identify_relevant_notes:llm")
                                ),
                            },
                        )
                        callback.on_llm_error(
                            TimeoutError("endpoint failed"), run_id=callback_run_id
                        )
                        raise RuntimeError("scripted graph failure")

        agent._graph = FailingGraph()
        with self.assertRaises(OrchestratorRunError) as raised:
            agent.run(self.raw_notes(), progress=False)

        failure = raised.exception.failure
        self.assertEqual(failure.run.status, "failed")
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertIn("scripted graph failure", failure.error)
        self.assertIsNotNone(failure.corpus)
        self.assertEqual(len(failure.corpus.note_corpus), len(load_notes()))
        self.assertNotIn("case", failure.model_dump())
        self.assertEqual(failure.observability.llm_usage_summary.model_invocations, 1)
        self.assertEqual(failure.observability.llm_usage_summary.failed_invocations, 1)
        self.assertIn("group:initial", failure.observability.llm_exchanges)

    def test_result_assembly_error_raises_with_failure_artifact(self):
        agent = build_fake_orchestrator()

        with patch.object(
            agent,
            "_corpus_from_state",
            side_effect=ValueError("invalid completed corpus"),
        ):
            with self.assertRaises(OrchestratorRunError) as raised:
                agent.run(self.raw_notes(), progress=False)

        failure = raised.exception.failure
        self.assertEqual(failure.run.status, "failed")
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIn("invalid completed corpus", failure.error)
        self.assertIsNotNone(failure.corpus)
        self.assertEqual(len(failure.corpus.note_corpus), len(load_notes()))

    def test_run_validates_stream_options(self):
        agent = build_fake_orchestrator()
        with self.assertRaises(ValueError):
            agent.run([], progress=False, max_concurrency=0)
        with self.assertRaises(TypeError):
            agent.run([], progress=False, max_concurrency=True)
        with self.assertRaises(ValueError):
            agent.run([], progress=False, max_content_chars=-1)
        with self.assertRaises(TypeError):
            agent.run([], progress=False, capture_llm_content="yes")
        with self.assertRaises(TypeError):
            agent.run([], structured_data=[], progress=False)


if __name__ == "__main__":
    unittest.main()
