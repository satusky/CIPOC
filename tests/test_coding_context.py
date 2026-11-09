"""Live dictionary scoping tests, independent of gitignored source manuals."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cipoc.models import CaseFacts
from cipoc.tools import (
    build_variable_group,
    load_data_dictionary,
    load_scoping_dictionary,
    lookup_variable_info,
    primary_site_tissues,
)

from tests.fake_orchestrator import SCOPING_DICTIONARY, primary_site_codes


BREAST_2020 = CaseFacts(gross_primary_site="breast", date_of_diagnosis="20200115", histology="8500/3")


class SiteDataDictionaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sites = load_scoping_dictionary(SCOPING_DICTIONARY)
        # Narrowing by tissue always allows C809, unknown primary.
        unknown = cls.sites.tissues["unknown_primary_site"].codes[400]
        cls.breast = {**cls.sites.tissues["breast"].codes[400], **unknown}
        cls.lung = {**cls.sites.tissues["lung"].codes[400], **unknown}

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base_path = Path(directory.name) / "base.json"
        self.base = {
            "400": {"Data Item Name": "Primary Site", "Length": 4, "Code Descriptions": primary_site_codes(),
                    "Instructions for Coding": "Retain base instructions."},
            "410": {"item_name": "Laterality", "item_data_type": "digits", "item_length": 1,
                    "allowed_codes": {"1": "Right: origin of primary", "2": "Left"}},
            "522": {"Data Item Name": "Histologic Type ICD-O-3", "Code Descriptions": {"8500": "Duct carcinoma"}},
            "523": {"Data Item Name": "Behavior Code ICD-O-3", "Code Descriptions": {"3": "Malignant"}},
            "764": {"Code Descriptions": [{"code": "0", "description": "Base in situ"},
                                          {"code": "1", "description": "Base localized"}]},
            "3843": {"Code Descriptions": {"1": "Base G1", "9": "Base unknown"}},
        }
        self.base_path.write_text(json.dumps(self.base))

    def build_group(self, item_ids, case_facts=None):
        return build_variable_group(item_ids, load_data_dictionary(self.base_path), case_facts=case_facts,
                                    site_dictionary=self.sites)

    def base_codes(self, item_id):
        return build_variable_group(item_id, load_data_dictionary(self.base_path)).variables[0].valid_codes

    def test_breast_gross_site_uses_tissue_codes_and_base_instructions(self):
        variable = self.build_group(400, CaseFacts(gross_primary_site="breast")).variables[0]
        self.assertEqual(variable.valid_codes, self.breast)
        self.assertEqual(variable.valid_codes["C500"], "Nipple & areola")
        self.assertEqual(variable.coding_instructions, "Retain base instructions.")

    def test_row_oriented_base_codes_are_normalized(self):
        self.assertEqual(self.build_group(764).variables[0].valid_codes,
                         {"0": "Base in situ", "1": "Base localized"})

    def test_dependent_items_take_the_matched_schema_codes(self):
        variables = {variable.item_id: variable for variable in self.build_group([764, 3843], BREAST_2020).variables}
        self.assertEqual(set(variables[764].valid_codes), {"0", "1", "2", "3", "4", "7", "9"})
        self.assertIn("Nottingham", variables[3843].valid_codes["1"])

    def test_dependent_items_keep_naaccr_codes_without_a_scoping_match(self):
        # With nothing known, every schema matches, and some don't override the item.
        self.assertEqual(self.build_group(3843).variables[0].valid_codes, self.base_codes(3843))
        self.assertEqual(self.build_group(523, BREAST_2020).variables[0].name, "Behavior Code ICD-O-3")
        self.assertEqual(self.build_group(410, BREAST_2020).variables[0].valid_codes, self.base_codes(410))

    def test_histology_is_never_narrowed_by_site(self):
        for facts in (BREAST_2020, CaseFacts(primary_site="C504")):
            with self.subTest(facts=facts):
                self.assertEqual(self.build_group(522, facts).variables[0].valid_codes, self.base_codes(522))

    def test_names_outside_the_vocabulary_and_unspecified_use_base(self):
        for gross_site in ("upper outer left breast", "Breast", "unspecified", ""):
            with self.subTest(gross_site=gross_site):
                facts = CaseFacts(gross_primary_site=gross_site)
                self.assertEqual(primary_site_tissues(self.sites, facts), [])
                self.assertEqual(self.build_group(400, facts).variables[0].valid_codes, self.base_codes(400))

    def test_primary_site_outranks_conflicting_gross_site(self):
        for code in ("C349", "C34.1", "c34.9", " C340 "):
            with self.subTest(code=code):
                facts = CaseFacts(primary_site=code, gross_primary_site="breast")
                self.assertEqual(primary_site_tissues(self.sites, facts), ["lung"])
                self.assertEqual(self.build_group(400, facts).variables[0].valid_codes, self.lung)

    def test_coded_primary_in_no_tissue_uses_base_not_gross(self):
        for code in ("C507", "C999"):
            with self.subTest(code=code):
                facts = CaseFacts(primary_site=code, gross_primary_site="breast")
                self.assertEqual(primary_site_tissues(self.sites, facts), [])
                self.assertEqual(self.build_group(400, facts).variables[0].valid_codes, self.base_codes(400))

    def test_unknown_or_malformed_primary_uses_base_not_gross(self):
        for code in ("C809", "C80.9", "unknown", "C34", "C.3.4.9", " "):
            with self.subTest(code=code):
                facts = CaseFacts(primary_site=code, gross_primary_site="breast")
                self.assertEqual(primary_site_tissues(self.sites, facts), [])
                self.assertEqual(self.build_group(400, facts).variables[0].valid_codes, self.base_codes(400))

    def test_absent_primary_allows_gross_site_fallback(self):
        for code in (None, ""):
            with self.subTest(code=code):
                facts = CaseFacts(primary_site=code, gross_primary_site="breast")
                self.assertEqual(primary_site_tissues(self.sites, facts), ["breast"])
                self.assertEqual(self.build_group(400, facts).variables[0].valid_codes, self.breast)

    def test_lookup_scopes_like_the_group_builder(self):
        data_dictionary = load_data_dictionary(self.base_path)
        for item_id in (400, 522, 764, 3843):
            with self.subTest(item_id=item_id):
                self.assertEqual(
                    lookup_variable_info(item_id, data_dictionary, site_dictionary=self.sites,
                                         case_facts=BREAST_2020).valid_codes,
                    self.build_group(item_id, BREAST_2020).variables[0].valid_codes,
                )

    def test_snake_case_metadata_is_preserved(self):
        variable = self.build_group(410).variables[0]
        self.assertEqual((variable.name, variable.data_type, variable.length), ("Laterality", "digits", 1))
        self.assertEqual(variable.valid_codes["1"], "Right: origin of primary")

    def test_unknown_ids_raise_instead_of_silently_disappearing(self):
        with self.assertRaisesRegex(ValueError, "item 99999"):
            self.build_group([400, 99999])
        with self.assertRaisesRegex(ValueError, "item 99999"):
            lookup_variable_info(99999, load_data_dictionary(self.base_path))

    def test_group_metadata_preflight_rejects_missing_validation_path(self):
        self.base["99999"] = {"Data Item Name": "Unsupported", "Length": 3}
        self.base_path.write_text(json.dumps(self.base))
        with self.assertRaisesRegex(ValueError, "metadata for item 99999"):
            self.build_group([400, 99999])


class NaaccrDictionaryScopingTests(unittest.TestCase):
    BASE_DICTIONARY = Path(__file__).resolve().parents[1] / "documents" / "manuals" / "naaccr_data_dictionary_v25.json"

    @classmethod
    def setUpClass(cls):
        if not cls.BASE_DICTIONARY.exists():
            raise unittest.SkipTest(f"{cls.BASE_DICTIONARY} not present")
        cls.data_dictionary = load_data_dictionary(cls.BASE_DICTIONARY)
        cls.sites = load_scoping_dictionary(SCOPING_DICTIONARY)

    def build_group(self, item_ids, case_facts=None):
        return build_variable_group(
            item_ids,
            self.data_dictionary,
            case_facts=case_facts,
            site_dictionary=self.sites,
        )

    def test_breast_primary_site_uses_tissue_codes(self):
        group = self.build_group(
            400,
            CaseFacts(
                gross_primary_site="breast",
                date_of_diagnosis="2025-02-24",
                sex="female",
            ),
        )

        site = group.variables[0]
        self.assertEqual(site.valid_codes["C504"], "Upper-outer quadrant of breast")
        # Breast's nine codes, plus C809 for an unknown primary.
        self.assertEqual(len(site.valid_codes), 10)
        self.assertIn("C809", site.valid_codes)
        self.assertNotIn("C340", site.valid_codes)
        self.assertIsNotNone(site.coding_instructions)

    def test_summary_stage_takes_the_breast_schema_codes(self):
        group = self.build_group(764, BREAST_2020)

        summary_stage = group.variables[0]
        self.assertEqual(set(summary_stage.valid_codes), {"0", "1", "2", "3", "4", "7", "9"})
        self.assertIn("In situ", summary_stage.valid_codes["0"])

    def test_item_missing_from_site_dictionary_falls_back_to_naaccr(self):
        group = self.build_group(523, CaseFacts(gross_primary_site="breast"))

        behavior = group.variables[0]
        self.assertEqual(behavior.name, "Behavior Code ICD-O-3")
        self.assertTrue(behavior.valid_codes)

    def test_primary_site_code_selects_the_lung_schema(self):
        group = self.build_group(
            764, CaseFacts(primary_site="C34.1", date_of_diagnosis="20200115", histology="8070")
        )

        summary_stage = group.variables[0]
        self.assertIn("Atelectasis", summary_stage.valid_codes["2"])

    def test_unspecified_site_uses_unscoped_naaccr_codes(self):
        group = self.build_group(400, CaseFacts(gross_primary_site="unspecified"))

        site = group.variables[0]
        self.assertIn("C341", site.valid_codes)
        self.assertIn("C504", site.valid_codes)

    def test_snake_case_base_entry_populates_variable_info(self):
        group = build_variable_group(410, self.data_dictionary)

        variable = group.variables[0]
        self.assertEqual(variable.name, "Laterality")
        self.assertEqual(variable.data_type, "digits")
        self.assertEqual(variable.length, 1)
        self.assertEqual(variable.valid_codes["1"], "Right: origin of primary")


if __name__ == "__main__":
    unittest.main()
