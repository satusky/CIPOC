import copy
import json
import unittest
from pathlib import Path
from unittest import mock
from typing import get_args

from cipoc.models import CaseFacts, GrossSite
from cipoc.tools import (
    build_variable_group,
    check_histology_pairs,
    check_primary_site_coverage,
    histology_block_listing,
    histology_code_listing,
    histology_family_listing,
    histology_pair_listed,
    histology_pairs,
    histology_site_group,
    load_data_dictionary,
    load_scoping_dictionary,
    matching_schemas,
    parse_scoping_dictionary,
    gross_site_candidates,
    primary_site_tissues,
    scoped_codes,
    scoped_item_codes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "scoping_dictionary.json"
BASE_DICTIONARY = REPO_ROOT / "documents" / "manuals" / "naaccr_data_dictionary_v25.json"
TISSUES = [name for name in get_args(GrossSite) if name != "unspecified"]


def fixture_codes(scoping) -> list[str]:
    return [code for entry in scoping.tissues.values() for code in entry.codes[400]]


class ScopingDictionaryLoadTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(FIXTURE.read_text())

    def test_fixture_tissues_are_grosssite_and_hold_distinct_codes(self):
        scoping = load_scoping_dictionary(FIXTURE)

        self.assertEqual(list(scoping.tissues), TISSUES)
        self.assertEqual(scoping.tissues["breast"].codes[400]["C500"], "Nipple & areola")
        codes = fixture_codes(scoping)
        self.assertEqual(len(codes), len(set(codes)))
        self.assertIn("C809", codes)

    def test_old_tissue_keyed_format_points_at_the_generator(self):
        old = {
            "breast": {"400": {"allowed_codes": [{"code": "C500", "description": "Nipple"}]}},
            "lung": {"400": {"allowed_codes": []}, "522": {"allowed_codes": []}},
        }
        with self.assertRaisesRegex(ValueError, "old tissue-keyed format.*canonical_naaccr.cipoc_scoping"):
            parse_scoping_dictionary(old)

    def test_tissue_names_must_equal_grosssite(self):
        missing = copy.deepcopy(self.raw)
        del missing["tissues"]["kidney"]
        with self.assertRaisesRegex(ValueError, r"missing \['kidney'\]"):
            parse_scoping_dictionary(missing)

        extra = copy.deepcopy(self.raw)
        extra["tissues"]["colorectal"] = {"codes": {"400": {}}}
        with self.assertRaisesRegex(ValueError, r"not in GrossSite \['colorectal'\]"):
            parse_scoping_dictionary(extra)

    def test_a_code_in_two_tissues_names_both(self):
        duplicated = copy.deepcopy(self.raw)
        duplicated["tissues"]["skin"]["codes"]["400"]["C50.0"] = "Nipple"
        with self.assertRaisesRegex(ValueError, "C500 in skin and breast"):
            parse_scoping_dictionary(duplicated)

    def test_sections_from_a_newer_generator_still_load(self):
        newer = copy.deepcopy(self.raw)
        newer["algorithms"] = {"eod": {"years": "2018-"}}
        newer["tissues"]["breast"]["notes"] = "added later"
        newer["schemas"]["cs/breast"]["ssf"] = {"3836": "2880"}
        newer["histology"]["breast"]["notes"] = "added later"

        scoping = parse_scoping_dictionary(newer)

        self.assertEqual(list(scoping.tissues), TISSUES)
        self.assertEqual(list(scoping.schemas), list(self.raw["schemas"]))
        self.assertEqual(list(scoping.histology), list(self.raw["histology"]))

    def test_schemas_select_only_on_keys_the_matcher_reads(self):
        unknown_key = copy.deepcopy(self.raw)
        unknown_key["schemas"]["eod/lung"]["select"][0]["grade"] = "1"
        with self.assertRaisesRegex(ValueError, r"eod/lung selects on keys CIPOC doesn't read: \['grade'\].*canonical_naaccr"):
            parse_scoping_dictionary(unknown_key)

        unparseable = copy.deepcopy(self.raw)
        unparseable["schemas"]["eod/lung"]["select"][0]["year_dx"] = "2018 onward"
        with self.assertRaisesRegex(ValueError, r"eod/lung: Unsupported year_dx value '2018 onward'.*canonical_naaccr"):
            parse_scoping_dictionary(unparseable)

    def test_override_sources_must_be_listed(self):
        unlisted = copy.deepcopy(self.raw)
        unlisted["schemas"]["eod/lung"]["codes"]["820"]["source"] = "seer_manual"
        with self.assertRaisesRegex(ValueError, r"eod/lung item 820 names source 'seer_manual'.*canonical_naaccr"):
            parse_scoping_dictionary(unlisted)

    def test_a_file_without_schemas_is_a_stale_build(self):
        stale = copy.deepcopy(self.raw)
        del stale["schemas"]
        with self.assertRaisesRegex(ValueError, "schemas"):
            parse_scoping_dictionary(stale)

    def test_a_file_without_histology_or_blocks_names_the_section(self):
        for sections in (["histology"], ["blocks"], ["histology", "blocks"]):
            with self.subTest(sections=sections):
                stale = copy.deepcopy(self.raw)
                for section in sections:
                    del stale[section]
                with self.assertRaisesRegex(
                    ValueError, f"no {' or '.join(sections)} section.*canonical_naaccr.cipoc_scoping"
                ):
                    parse_scoping_dictionary(stale)


class HistologyLoadTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads(FIXTURE.read_text())

    def test_fixture_keeps_every_group_and_block(self):
        scoping = load_scoping_dictionary(FIXTURE)

        self.assertEqual(len(scoping.histology), 82)
        self.assertEqual(len(scoping.blocks), 49)
        self.assertEqual(scoping.histology["breast"].sites, "C500-C506,C508-C509")
        self.assertEqual(scoping.blocks["814-838"].label, "Adenomas and adenocarcinomas")
        # The trimmed fixture keeps the families of only a few groups.
        self.assertEqual(sum(1 for group in scoping.histology.values() if not group.families), 76)

    def test_a_code_in_two_groups_names_both(self):
        overlapping = copy.deepcopy(self.raw)
        overlapping["histology"]["breast"]["sites"] = "C500-C506,C508-C509,C649"
        with self.assertRaisesRegex(ValueError, "C649 in breast and kidney"):
            parse_scoping_dictionary(overlapping)

    def test_every_primary_site_code_needs_a_group(self):
        uncovered = copy.deepcopy(self.raw)
        uncovered["histology"]["breast"]["sites"] = "C500-C504"
        with self.assertRaisesRegex(ValueError, r"in no histology group: \['C505', 'C506', 'C508', 'C509'\]"):
            parse_scoping_dictionary(uncovered)

    def test_unknown_primary_needs_a_group(self):
        missing = copy.deepcopy(self.raw)
        missing["histology"]["unknown"]["sites"] = "C809"
        missing["tissues"]["unknown_primary_site"]["codes"]["400"] = {}
        with self.assertRaisesRegex(ValueError, "No histology group holds C809"):
            parse_scoping_dictionary(missing)

    def test_sites_must_parse_like_a_schema_site(self):
        unparseable = copy.deepcopy(self.raw)
        unparseable["histology"]["breast"]["sites"] = "C50.0-C50.9 (breast)"
        with self.assertRaisesRegex(ValueError, r"Histology group breast: Unsupported site value.*canonical_naaccr"):
            parse_scoping_dictionary(unparseable)

    def test_families_and_pairs_must_be_well_formed(self):
        families = self.raw["histology"]["breast"]["families"]
        for family, pair, message in (
            ("85", "8500/3", "group breast has family '85', which isn't 3 digits"),
            ("850", "85003", r"group breast family 850 holds \['85003'\]"),
            ("850", "8520/3", r"group breast family 850 holds \['8520/3'\]"),
        ):
            with self.subTest(family=family, pair=pair):
                malformed = copy.deepcopy(self.raw)
                malformed["histology"]["breast"]["families"] = {
                    family: {"label": "DUCT CARCINOMA", "codes": {pair: families["850"]["codes"]["8500/3"]}},
                }
                with self.assertRaisesRegex(ValueError, message):
                    parse_scoping_dictionary(malformed)

    def test_blocks_must_be_ranges_that_dont_overlap(self):
        for blocks, message in (
            ({"814-838": {"label": "Adenomas"}, "838-842": {"label": "Adnexal"}}, "blocks 814-838 and 838-842 overlap"),
            ({"8140-8389": {"label": "Adenomas"}}, "block '8140-8389' isn't a 3-digit prefix or range"),
            ({"838-814": {"label": "Adenomas"}}, "block '838-814' isn't a 3-digit prefix or range"),
        ):
            with self.subTest(blocks=blocks):
                malformed = copy.deepcopy(self.raw)
                malformed["blocks"] = blocks
                with self.assertRaisesRegex(ValueError, message):
                    parse_scoping_dictionary(malformed)


class PrimarySiteCoverageTests(unittest.TestCase):
    def setUp(self):
        self.scoping = load_scoping_dictionary(FIXTURE)
        self.codes = fixture_codes(self.scoping)

    def test_matching_codes_pass_in_either_notation(self):
        check_primary_site_coverage(self.scoping, self.codes)
        check_primary_site_coverage(
            self.scoping, [f"{code[:3]}.{code[3:]}".lower() for code in self.codes]
        )

    def test_missing_and_extra_codes_are_listed(self):
        with self.assertRaisesRegex(ValueError, r"in item 400 but no tissue \['C999'\]"):
            check_primary_site_coverage(self.scoping, [*self.codes, "C999"])
        with self.assertRaisesRegex(ValueError, r"in a tissue but not item 400 \['C500', 'C649'\]"):
            check_primary_site_coverage(
                self.scoping, [code for code in self.codes if code not in ("C500", "C649")]
            )


class RealPrimarySiteCoverageTests(unittest.TestCase):
    """The fixture's tissues cover the v25 dictionary's 330 Primary Site codes."""

    @classmethod
    def setUpClass(cls):
        if not BASE_DICTIONARY.exists():
            raise unittest.SkipTest(f"{BASE_DICTIONARY} not present")
        cls.primary_site = build_variable_group(400, load_data_dictionary(BASE_DICTIONARY)).variables[0]

    def test_fixture_covers_item_400(self):
        self.assertEqual(len(self.primary_site.valid_codes), 330)
        check_primary_site_coverage(load_scoping_dictionary(FIXTURE), self.primary_site.valid_codes)


LUNG_SCHEMAS = ["cs/lung", "tnm/lung", "eod/lung", "eod/lung_v9_2025"]


def lung(date_of_diagnosis=None) -> CaseFacts:
    return CaseFacts(primary_site="C34.1", histology="8140/3", date_of_diagnosis=date_of_diagnosis)


class SchemaMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(FIXTURE)

    def match(self, facts):
        return matching_schemas(self.scoping, facts)

    def test_unknown_facts_match_every_schema(self):
        self.assertEqual(self.match(None), list(self.scoping.schemas))
        self.assertEqual(self.match(CaseFacts()), list(self.scoping.schemas))
        self.assertEqual(self.match(CaseFacts(histology="8140")), [
            "cs/breast", "cs/lung", "cs/peritoneum_female_gen",
            "tnm/breast", "tnm/lung", "tnm/peritoneum_female_gen",
            "eod/breast", "eod/colon_rectum", "eod/corpus_carcinoma", "eod/lung",
            "eod/lung_v9_2025", "eod/primary_peritoneal_carcinoma",
        ])

    def test_diagnosis_year_picks_the_algorithm(self):
        for date_of_diagnosis, expected in (
            ("2012-06-15", ["cs/lung"]),
            ("2016-06-15", ["tnm/lung"]),
            ("2020-06-15", ["eod/lung"]),
            ("2025-06-15", ["eod/lung_v9_2025"]),
            (None, LUNG_SCHEMAS),
            ("2000-06-15", []),
        ):
            with self.subTest(date_of_diagnosis=date_of_diagnosis):
                self.assertEqual(self.match(lung(date_of_diagnosis)), expected)

    def test_unknown_year_gives_both_eod_lung_versions(self):
        matched = self.match(lung())
        self.assertIn("eod/lung", matched)
        self.assertIn("eod/lung_v9_2025", matched)

    def test_year_is_parsed_from_naaccr_and_iso_dates(self):
        for date_of_diagnosis in ("20120615", "2012-06-15", "20120600", "201206", "2012-06", "2012"):
            with self.subTest(date_of_diagnosis=date_of_diagnosis):
                self.assertEqual(self.match(lung(date_of_diagnosis)), ["cs/lung"])
        for date_of_diagnosis in ("06/15/2012", "2012-0615", "20121315", "00000000", "unknown"):
            with self.subTest(date_of_diagnosis=date_of_diagnosis):
                self.assertEqual(self.match(lung(date_of_diagnosis)), LUNG_SCHEMAS)

    def test_gross_tissue_stands_in_for_an_uncoded_site(self):
        facts = {"histology": "8140", "date_of_diagnosis": "2020-06-15"}
        self.assertEqual(self.match(CaseFacts(gross_primary_site="lung", **facts)), ["eod/lung"])
        self.assertEqual(
            self.match(CaseFacts(primary_site="C504", gross_primary_site="lung", **facts)),
            ["eod/breast"],
        )
        # Neither an unspecified tissue nor a malformed coded site narrows the site.
        for site_facts in (
            {"gross_primary_site": "unspecified"},
            {"primary_site": "C5", "gross_primary_site": "lung"},
        ):
            with self.subTest(**site_facts):
                matched = self.match(CaseFacts(**site_facts, **facts))
                self.assertIn("eod/breast", matched)
                self.assertIn("eod/lung", matched)

    def test_candidate_tissues_match_as_or(self):
        facts = {"histology": "8140", "date_of_diagnosis": "2020-06-15"}
        both = self.match(CaseFacts(candidate_gross_sites=["breast", "lung"], **facts))
        self.assertEqual(both, ["eod/breast", "eod/lung"])
        self.assertEqual(
            set(both),
            set(self.match(CaseFacts(gross_primary_site="breast", **facts)))
            | set(self.match(CaseFacts(gross_primary_site="lung", **facts))),
        )
        # One candidate outside the file makes the site unknown.
        self.assertEqual(
            self.match(CaseFacts(candidate_gross_sites=["breast", "left lung"], **facts)),
            self.match(CaseFacts(**facts)),
        )

    def test_candidate_histology_families_stand_in_for_an_uncoded_histology(self):
        breast = {"primary_site": "C504", "date_of_diagnosis": "2020-06-15"}
        everything = self.match(CaseFacts(**breast))
        self.assertEqual(everything, ["eod/breast", "eod/hemeretic", "eod/lymphoma"])
        for facts, expected in (
            # Lymphoma families steer an uncoded histology to the lymphoma schema.
            ({"candidate_histology_families": ["967", "968"]}, ["eod/lymphoma"]),
            ({"candidate_histology_families": ["850"]}, ["eod/breast"]),
            # A coded histology outranks the candidates.
            ({"histology": "8500", "candidate_histology_families": ["967"]}, ["eod/breast"]),
            # A malformed candidate leaves the histology unknown.
            ({"candidate_histology_families": ["96"]}, everything),
        ):
            with self.subTest(**facts):
                self.assertEqual(self.match(CaseFacts(**breast, **facts)), expected)

    def test_c809_is_a_site_code_not_an_unknown_site(self):
        self.assertEqual(
            self.match(CaseFacts(primary_site="C80.9", histology="8500", date_of_diagnosis="2020-06-15")),
            [],
        )

    def test_behavior_comes_from_the_fact_or_the_histology_suffix(self):
        brain = {"primary_site": "C710", "date_of_diagnosis": "2020-06-15"}
        for facts, expected in (
            ({"histology": "9591/3"}, ["eod/hemeretic", "eod/lymphoma"]),
            ({"histology": "9591", "behavior": "3"}, ["eod/hemeretic", "eod/lymphoma"]),
            ({"histology": "9591/1"}, []),
            ({"histology": "9591/3", "behavior": "1"}, []),
        ):
            with self.subTest(**facts):
                self.assertEqual(self.match(CaseFacts(**brain, **facts)), expected)

    def test_sex_at_birth_filters_schemas_for_one_sex(self):
        peritoneum = {"primary_site": "C481", "histology": "8461"}
        for date_of_diagnosis, sex, expected in (
            ("2020-06-15", "2", ["eod/primary_peritoneal_carcinoma"]),
            ("2020-06-15", "1", []),
            ("2020-06-15", "9", ["eod/primary_peritoneal_carcinoma"]),
            ("2016-06-15", "1", []),
            # CS selects on Site-Specific Factor 25, which is always unknown.
            ("2012-06-15", "1", ["cs/peritoneum_female_gen"]),
        ):
            with self.subTest(date_of_diagnosis=date_of_diagnosis, sex=sex):
                self.assertEqual(
                    self.match(CaseFacts(**peritoneum, sex=sex, date_of_diagnosis=date_of_diagnosis)),
                    expected,
                )


class ScopedCodesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(FIXTURE.read_text())
        cls.scoping = parse_scoping_dictionary(cls.raw)

    def test_several_matches_give_the_union(self):
        matched = matching_schemas(self.scoping, CaseFacts(primary_site="C504", date_of_diagnosis="2020-06-15"))
        self.assertEqual(matched, ["eod/breast", "eod/hemeretic", "eod/lymphoma"])

        grade = scoped_codes(self.scoping, matched, 3843)

        breast = self.scoping.schemas["eod/breast"].codes[3843].codes
        self.assertEqual(set(grade), {*breast, "8"})
        self.assertEqual(grade["1"], breast["1"])

    def test_conflicting_descriptions_leave_the_item_unscoped(self):
        # Breast grades by Nottingham, corpus carcinoma by FIGO.
        self.assertIsNone(scoped_codes(self.scoping, ["eod/breast", "eod/corpus_carcinoma"], 3843))
        matched = matching_schemas(self.scoping, CaseFacts(histology="8500", date_of_diagnosis="2020-06-15"))
        self.assertIn("eod/corpus_carcinoma", matched)
        self.assertIsNone(scoped_codes(self.scoping, matched, 3843))

    def test_whitespace_and_case_differences_are_not_conflicts(self):
        restyled = copy.deepcopy(self.raw)
        codes = restyled["schemas"]["eod/lung_v9_2025"]["codes"]["3843"]["codes"]
        codes["9"] = "  " + codes["9"].upper().replace(" ", "\n")
        scoping = parse_scoping_dictionary(restyled)

        grade = scoped_codes(scoping, ["eod/lung", "eod/lung_v9_2025"], 3843)

        self.assertEqual(grade["9"], self.scoping.schemas["eod/lung"].codes[3843].codes["9"])

    def test_a_schema_without_an_override_keeps_the_naaccr_codes(self):
        self.assertEqual(
            scoped_codes(self.scoping, ["eod/lung"], 830),
            self.scoping.schemas["eod/lung"].codes[830].codes,
        )
        self.assertNotIn(830, self.scoping.schemas["tnm/lung"].codes)
        self.assertIsNone(scoped_codes(self.scoping, ["tnm/lung", "eod/lung"], 830))
        self.assertIsNone(scoped_codes(self.scoping, ["eod/lung"], 832))

    def test_an_unknown_year_scopes_only_what_every_era_agrees_on(self):
        matched = matching_schemas(self.scoping, lung())
        self.assertEqual(matched, LUNG_SCHEMAS)

        # CS's own LVI table and the carry-over are the same four codes.
        self.assertEqual(set(scoped_codes(self.scoping, matched, 1182)), {"0", "1", "8", "9"})
        # Grade is only overridden from 2018, so earlier schemas keep NAACCR's.
        self.assertIsNone(scoped_codes(self.scoping, matched, 3843))

    def test_no_matching_schema_leaves_items_unscoped(self):
        matched = matching_schemas(self.scoping, lung("2000-06-15"))
        self.assertEqual(matched, [])
        self.assertIsNone(scoped_codes(self.scoping, matched, 820))


class PrimarySiteTissuesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(FIXTURE)

    def tissues(self, **facts):
        return primary_site_tissues(self.scoping, CaseFacts(**facts))

    def test_candidates_or_else_the_gross_site(self):
        self.assertEqual(gross_site_candidates(CaseFacts(candidate_gross_sites=["breast", "ovary"])),
                         ["breast", "ovary"])
        self.assertEqual(gross_site_candidates(CaseFacts(gross_primary_site="breast")), ["breast"])
        self.assertEqual(gross_site_candidates(CaseFacts()), [])
        self.assertEqual(gross_site_candidates(None), [])

    def test_a_coded_primary_picks_its_tissue_over_the_candidates(self):
        for code, tissue in (("C504", "breast"), ("c34.1", "lung"), ("C181", "appendix"), ("C209", "rectum")):
            with self.subTest(code=code):
                self.assertEqual(self.tissues(primary_site=code, gross_primary_site="ovary"), [tissue])
                self.assertEqual(
                    self.tissues(primary_site=code, candidate_gross_sites=["ovary", "kidney"]), [tissue]
                )

    def test_unknown_malformed_or_unlisted_primaries_pick_none(self):
        for code in ("C809", "C80.9", "C34", "C3491", "unknown", " ", "C999"):
            with self.subTest(code=code):
                self.assertEqual(self.tissues(primary_site=code, gross_primary_site="breast"), [])
                self.assertEqual(self.tissues(primary_site=code, candidate_gross_sites=["breast", "ovary"]), [])

    def test_the_gross_site_counts_only_as_a_tissue_name(self):
        self.assertEqual(self.tissues(gross_primary_site="unknown_primary_site"), ["unknown_primary_site"])
        self.assertEqual(self.tissues(gross_primary_site="breast"), ["breast"])
        for gross_site in ("unspecified", "left breast", "Breast", None):
            with self.subTest(gross_site=gross_site):
                self.assertEqual(self.tissues(gross_primary_site=gross_site), [])
        self.assertEqual(primary_site_tissues(self.scoping, None), [])

    def test_every_candidate_counts_unless_one_is_unknown(self):
        self.assertEqual(self.tissues(candidate_gross_sites=["breast", "ovary"]), ["breast", "ovary"])
        for candidates in (["breast", "unspecified"], ["breast", "left breast"]):
            with self.subTest(candidates=candidates):
                self.assertEqual(self.tissues(candidate_gross_sites=candidates), [])


class ScopedItemCodesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(FIXTURE)

    def test_primary_site_takes_its_tissues_codes_plus_unknown_primary(self):
        tissues = self.scoping.tissues
        unknown = {"C809": "Unknown primary site"}
        for facts, expected in (
            (CaseFacts(gross_primary_site="breast"), {**tissues["breast"].codes[400], **unknown}),
            (CaseFacts(primary_site="C341", gross_primary_site="breast"), {**tissues["lung"].codes[400], **unknown}),
            (
                CaseFacts(candidate_gross_sites=["breast", "ovary"]),
                {**tissues["breast"].codes[400], **tissues["ovary"].codes[400], **unknown},
            ),
            (CaseFacts(gross_primary_site="unknown_primary_site"), unknown),
        ):
            with self.subTest(facts=facts):
                self.assertEqual(scoped_item_codes(self.scoping, facts, [400]), {400: expected})

    def test_primary_site_is_unscoped_when_a_candidate_is_unspecified(self):
        facts = CaseFacts(candidate_gross_sites=["breast", "unspecified"])
        self.assertEqual(scoped_item_codes(self.scoping, facts, [400]), {})

    def test_histology_is_never_scoped(self):
        for facts in (CaseFacts(primary_site="C504"), CaseFacts(gross_primary_site="breast"), None):
            with self.subTest(facts=facts):
                self.assertNotIn(522, scoped_item_codes(self.scoping, facts, [522]))

    def test_other_items_take_the_schema_match(self):
        facts = CaseFacts(primary_site="C349", date_of_diagnosis="20200115", histology="8070")
        scoped = scoped_item_codes(self.scoping, facts, [400, 410, 522, 764, 3843])

        self.assertEqual(set(scoped), {400, 764, 3843})
        lung = self.scoping.schemas["eod/lung"].codes
        self.assertEqual(scoped[764], lung[764].codes)
        self.assertEqual(scoped[3843], lung[3843].codes)

    def test_items_left_unscoped_are_left_out(self):
        self.assertEqual(scoped_item_codes(self.scoping, None, [400, 522, 764, 3843]), {})
        self.assertEqual(
            scoped_item_codes(self.scoping, CaseFacts(primary_site="C809", date_of_diagnosis="2000"), [400, 820]),
            {},
        )

    def test_schemas_are_matched_once_per_call(self):
        with mock.patch("cipoc.tools.scoping.matching_schemas", wraps=matching_schemas) as matched:
            scoped_item_codes(self.scoping, CaseFacts(gross_primary_site="breast"), [400, 522, 764, 1182, 3843])
            scoped_item_codes(self.scoping, CaseFacts(gross_primary_site="breast"), [400, 522])
        self.assertEqual(matched.call_count, 1)


class HistologyScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(FIXTURE)

    def test_a_coded_primary_picks_its_site_group(self):
        for code, group in (("C50.4", "breast"), ("C504", "breast"), ("c34.9", "lung_and_bronchus"), ("C649", "kidney")):
            with self.subTest(code=code):
                self.assertEqual(histology_site_group(self.scoping, code), group)

    def test_an_uncoded_malformed_or_unknown_primary_gets_the_unknown_group(self):
        # NOT_FOUND and ERROR leave the primary uncoded; C507 isn't an ICD-O-3 code.
        for code in (None, "", "C50", "breast", "C809", "C80.9", "C507"):
            with self.subTest(code=code):
                self.assertEqual(histology_site_group(self.scoping, code), "unknown")

    def test_breast_reaches_angiosarcoma_and_malignant_phyllodes(self):
        group = histology_site_group(self.scoping, "C50.4")
        families = histology_family_listing(self.scoping, group)

        self.assertIn("912", families)
        self.assertIn("902", families)
        self.assertEqual(histology_code_listing(self.scoping, group, ["912"]), {"9120/3": "Hemangiosarcoma; Angiosarcoma"})
        self.assertIn("9020/3", histology_code_listing(self.scoping, group, ["902"]))

    def test_the_family_listing_shows_the_lists_terms(self):
        families = histology_family_listing(self.scoping, "breast")

        self.assertEqual(len(families), 63)
        self.assertEqual(
            families["852"],
            "LOBULAR AND OTHER DUCTAL CA.: Lobular carcinoma in situ; Lobular carcinoma, NOS; "
            "Infiltrating ductular carcinoma; Intraductal and lobular in situ carcinoma; "
            "Infiltrating duct and lobular carcinoma; Infiltr. duct mixed with other types of carcinoma; "
            "Infiltrating lobular mixed with other types of carc.; Polymorphous low grade adenocarcinoma",
        )
        self.assertEqual(histology_family_listing(self.scoping, "stomach"), {})

    def test_the_code_listing_shows_v25_descriptions(self):
        codes = histology_code_listing(self.scoping, "breast", ["850"])

        self.assertTrue(codes["8500/3"].startswith("Infiltrating duct carcinoma, NOS; Invasive breast carcinoma"))
        self.assertNotIn("Invasive carcinoma of no special type", codes.values())

    def test_two_families_give_the_union_of_their_codes(self):
        duct = histology_code_listing(self.scoping, "breast", ["850"])
        lobular = histology_code_listing(self.scoping, "breast", ["852"])

        both = histology_code_listing(self.scoping, "breast", ["852", "850", "852"])

        self.assertEqual(both, {**duct, **lobular})
        self.assertEqual(list(both), sorted(both))

    def test_an_unknown_family_or_no_family_raises(self):
        with self.assertRaisesRegex(ValueError, r"breast has no families \['999'\]"):
            histology_code_listing(self.scoping, "breast", ["850", "999"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            histology_code_listing(self.scoping, "breast", [])

    def test_whether_a_pair_is_on_the_groups_list(self):
        self.assertTrue(histology_pair_listed(self.scoping, "breast", "8500/3"))
        self.assertTrue(histology_pair_listed(self.scoping, "breast", " 9120/3 "))
        # Renal cell carcinoma and glioblastoma are on other sites' lists.
        for pair in ("8500/1", "8312/3", "9440/3", "8500", ""):
            with self.subTest(pair=pair):
                self.assertFalse(histology_pair_listed(self.scoping, "breast", pair))
        self.assertFalse(histology_pair_listed(self.scoping, "stomach", "8140/3"))


class HistologyFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoping = load_scoping_dictionary(FIXTURE)

    def test_pairs_are_grouped_by_block_in_block_order(self):
        pairs = {"9993/3": "Myelodysplastic syndrome", "8500/3": "Duct carcinoma", "8140/3": "Adenocarcinoma, NOS",
                 "8000/3": "Neoplasm, malignant", "8141/3": "Scirrhous adenocarcinoma"}

        blocks = histology_block_listing(self.scoping, pairs)

        self.assertEqual(list(blocks), ["800", "814-838", "850-854", "998-999"])
        self.assertEqual(blocks["814-838"], {"8140/3": "Adenocarcinoma, NOS", "8141/3": "Scirrhous adenocarcinoma"})

    def test_a_pair_in_no_block_or_malformed_raises(self):
        with self.assertRaisesRegex(ValueError, r"7999/3 is in morphology blocks \[\]"):
            histology_block_listing(self.scoping, {"7999/3": "Not a neoplasm"})
        with self.assertRaisesRegex(ValueError, "'8500' isn't a code/behavior pair"):
            histology_block_listing(self.scoping, {"8500": "Duct carcinoma"})

    def test_histology_pairs_are_read_uncollapsed(self):
        dictionary = {"522": {"item_length": 4, "allowed_codes": {"8500/2": "Intraductal", "8500/3": "Duct"}}}
        self.assertEqual(histology_pairs(dictionary), {"8500/2": "Intraductal", "8500/3": "Duct"})
        for broken in ({}, {"522": {"allowed_codes": {"8500": "Duct"}}}):
            with self.subTest(broken=broken):
                with self.assertRaisesRegex(ValueError, "item 522 has no code/behavior pairs"):
                    histology_pairs(broken)

    def test_listed_pairs_must_be_item_522_pairs_in_a_block(self):
        pairs = {
            pair
            for group in self.scoping.histology.values()
            for family in group.families.values()
            for pair in family.codes
        }
        check_histology_pairs(self.scoping, pairs)
        with self.assertRaisesRegex(ValueError, r"aren't in item 522: \['8500/3'\]"):
            check_histology_pairs(self.scoping, pairs - {"8500/3"})
        with self.assertRaisesRegex(ValueError, "7999/3 is in morphology blocks"):
            check_histology_pairs(self.scoping, pairs | {"7999/3"})


class RealHistologyPairsTests(unittest.TestCase):
    """The fixture offers only v25 pairs, and every v25 pair has a block."""

    @classmethod
    def setUpClass(cls):
        if not BASE_DICTIONARY.exists():
            raise unittest.SkipTest(f"{BASE_DICTIONARY} not present")
        cls.pairs = histology_pairs(load_data_dictionary(BASE_DICTIONARY))
        cls.scoping = load_scoping_dictionary(FIXTURE)

    def test_only_v25_pairs_are_offered(self):
        self.assertEqual(len(self.pairs), 1143)
        check_histology_pairs(self.scoping, self.pairs)
        # On the list for stomach, pancreas and lymph nodes, but not in v25.
        self.assertNotIn("8157/3", self.pairs)

    def test_the_fallback_offers_every_v25_pair_once(self):
        blocks = histology_block_listing(self.scoping, self.pairs)

        self.assertEqual(sum(len(codes) for codes in blocks.values()), len(self.pairs))
        self.assertEqual(max(blocks, key=lambda key: len(blocks[key])), "814-838")
        self.assertEqual(blocks["850-854"]["8500/3"], self.pairs["8500/3"])


if __name__ == "__main__":
    unittest.main()
