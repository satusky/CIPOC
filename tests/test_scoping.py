import copy
import json
import unittest
from pathlib import Path
from typing import get_args

from cipoc.models import GrossSite
from cipoc.tools import (
    build_variable_group,
    check_primary_site_coverage,
    load_data_dictionary,
    load_scoping_dictionary,
    parse_scoping_dictionary,
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
        newer["schemas"] = {"eod/breast": {"select": {"site": "C500-C509"}, "codes": {}}}
        newer["tissues"]["breast"]["notes"] = "added later"

        scoping = parse_scoping_dictionary(newer)

        self.assertEqual(list(scoping.tissues), TISSUES)


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


if __name__ == "__main__":
    unittest.main()
