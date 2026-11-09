import copy
import json
import unittest
from pathlib import Path
from typing import get_args

from cipoc.models import CaseFacts, GrossSite
from cipoc.tools import (
    build_variable_group,
    check_primary_site_coverage,
    load_data_dictionary,
    load_scoping_dictionary,
    matching_schemas,
    parse_scoping_dictionary,
    scoped_codes,
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
        newer["blocks"] = {"800-800": {"name": "Neoplasms, NOS"}}
        newer["tissues"]["breast"]["notes"] = "added later"
        newer["schemas"]["cs/breast"]["ssf"] = {"3836": "2880"}

        scoping = parse_scoping_dictionary(newer)

        self.assertEqual(list(scoping.tissues), TISSUES)
        self.assertEqual(list(scoping.schemas), list(self.raw["schemas"]))

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


if __name__ == "__main__":
    unittest.main()
