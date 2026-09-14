"""Live dictionary scoping tests, independent of gitignored source manuals."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cipoc.models import CaseFacts
from cipoc.tools import build_variable_group, lookup_variable_info, resolve_site_key


class SiteDataDictionaryTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base_path = Path(directory.name) / "base.json"
        self.site_path = Path(directory.name) / "sites.json"
        self.base = {
            "400": {"Data Item Name": "Primary Site", "Length": 4, "Code Descriptions": {
                "C341": "Lung", "C349": "Lung NOS", "C504": "Breast", "C649": "Kidney",
            }, "Instructions for Coding": "Retain base instructions."},
            "410": {"item_name": "Laterality", "item_data_type": "digits", "item_length": 1,
                    "allowed_codes": {"1": "Right: origin of primary", "2": "Left"}},
            "523": {"Data Item Name": "Behavior Code ICD-O-3", "Code Descriptions": {"3": "Malignant"}},
            "764": {"Code Descriptions": {"0": "Base in situ", "1": "Base localized", "2": "Base regional"}},
        }
        self.sites = {
            "breast": {
                "400": {"allowed_codes": {"C500": "Nipple", "C504": "Upper-outer quadrant of breast"}},
                "764": {"allowed_codes": [{"code": "0", "description": "In situ"}, {"code": "1", "description": "Localized"}]},
            },
            "lung": {
                "400": {"allowed_codes": {"C341": "Upper lobe", "C349": "Lung NOS"}},
                "764": {"allowed_codes": {"2": "Atelectasis"}},
            },
        }
        self.base_path.write_text(json.dumps(self.base))
        self.site_path.write_text(json.dumps(self.sites))

    def build_group(self, item_ids, case_facts=None):
        return build_variable_group(item_ids, self.base_path, case_facts=case_facts,
                                    site_data_dictionary_path=self.site_path)

    def test_breast_gross_site_uses_site_codes_and_base_instructions(self):
        variable = self.build_group(400, CaseFacts(gross_primary_site="upper outer left breast")).variables[0]
        self.assertEqual(variable.valid_codes["C500"], "Nipple")
        self.assertNotIn("C341", variable.valid_codes)
        self.assertEqual(variable.coding_instructions, "Retain base instructions.")

    def test_row_oriented_site_codes_are_normalized(self):
        variable = self.build_group(764, CaseFacts(gross_primary_site="left breast")).variables[0]
        self.assertEqual(variable.valid_codes, {"0": "In situ", "1": "Localized"})

    def test_missing_site_item_and_unknown_gross_site_use_base(self):
        self.assertEqual(self.build_group(523, CaseFacts(gross_primary_site="breast")).variables[0].name,
                         "Behavior Code ICD-O-3")
        self.assertEqual(self.build_group(400, CaseFacts(gross_primary_site="kidney")).variables[0].valid_codes,
                         self.base["400"]["Code Descriptions"])

    def test_primary_site_outranks_conflicting_gross_site(self):
        for code in ("C349", "C34.1", "c34.9"):
            with self.subTest(code=code):
                facts = CaseFacts(primary_site=code, gross_primary_site="breast")
                self.assertEqual(resolve_site_key(facts, self.sites), "lung")
                self.assertEqual(self.build_group(764, facts).variables[0].valid_codes, {"2": "Atelectasis"})

    def test_known_primary_without_matching_dictionary_uses_base_not_gross(self):
        facts = CaseFacts(primary_site="C649", gross_primary_site="breast")
        self.assertIsNone(resolve_site_key(facts, self.sites))
        self.assertEqual(self.build_group(400, facts).variables[0].valid_codes, self.base["400"]["Code Descriptions"])
        self.sites.pop("lung")
        self.site_path.write_text(json.dumps(self.sites))
        facts.primary_site = "C349"
        self.assertEqual(self.build_group(400, facts).variables[0].valid_codes, self.base["400"]["Code Descriptions"])

    def test_unknown_or_malformed_primary_uses_base_not_gross(self):
        for code in ("C809", "C80.9", "unknown", "C34", "C.3.4.9", "C999", " "):
            with self.subTest(code=code):
                facts = CaseFacts(primary_site=code, gross_primary_site="breast")
                self.assertIsNone(resolve_site_key(facts, self.sites))
                self.assertEqual(self.build_group(400, facts).variables[0].valid_codes,
                                 self.base["400"]["Code Descriptions"])

    def test_absent_primary_allows_gross_site_fallback(self):
        for code in (None, ""):
            with self.subTest(code=code):
                facts = CaseFacts(primary_site=code, gross_primary_site="breast")
                self.assertEqual(resolve_site_key(facts, self.sites), "breast")
                self.assertEqual(self.build_group(400, facts).variables[0].valid_codes,
                                 self.sites["breast"]["400"]["allowed_codes"])

    def test_snake_case_metadata_is_preserved(self):
        variable = self.build_group(410).variables[0]
        self.assertEqual((variable.name, variable.data_type, variable.length), ("Laterality", "digits", 1))
        self.assertEqual(variable.valid_codes["1"], "Right: origin of primary")

    def test_unknown_ids_raise_instead_of_silently_disappearing(self):
        with self.assertRaisesRegex(ValueError, "item 99999"):
            self.build_group([400, 99999])
        with self.assertRaisesRegex(ValueError, "item 99999"):
            lookup_variable_info(99999, self.base_path)

    def test_group_metadata_preflight_rejects_missing_validation_path(self):
        self.base["99999"] = {"Data Item Name": "Unsupported", "Length": 3}
        self.base_path.write_text(json.dumps(self.base))
        with self.assertRaisesRegex(ValueError, "metadata for item 99999"):
            self.build_group([400, 99999])


class SiteDataDictionaryTests(unittest.TestCase):
    BASE_DICTIONARY = "documents/manuals/naaccr_data_dictionary_v25.json"
    SITE_DICTIONARY = "documents/cipoc_data_dictionary.json"

    def build_group(self, item_ids, case_facts):
        return build_variable_group(
            item_ids,
            self.BASE_DICTIONARY,
            case_facts=case_facts,
            site_data_dictionary_path=self.SITE_DICTIONARY,
        )

    def test_breast_primary_site_uses_site_allowed_codes(self):
        group = self.build_group(
            400,
            CaseFacts(
                gross_primary_site="upper outer left breast",
                date_of_diagnosis="2025-02-24",
                sex="female",
            ),
        )

        site = group.variables[0]
        self.assertEqual(site.valid_codes["C500"], "Nipple")
        self.assertEqual(site.valid_codes["C504"], "Upper-outer quadrant of breast")
        self.assertNotIn("C340", site.valid_codes)
        self.assertIsNotNone(site.coding_instructions)

    def test_row_oriented_site_codes_are_normalized(self):
        group = self.build_group(764, CaseFacts(gross_primary_site="left breast"))

        summary_stage = group.variables[0]
        self.assertEqual(set(summary_stage.valid_codes), {"0", "1", "2", "3", "4", "7", "9"})
        self.assertIn("In situ", summary_stage.valid_codes["0"])

    def test_item_missing_from_site_dictionary_falls_back_to_naaccr(self):
        group = self.build_group(523, CaseFacts(gross_primary_site="breast"))

        behavior = group.variables[0]
        self.assertEqual(behavior.name, "Behavior Code ICD-O-3")
        self.assertTrue(behavior.valid_codes)

    def test_primary_site_code_can_select_tissue(self):
        group = self.build_group(764, CaseFacts(primary_site="C34.1"))

        summary_stage = group.variables[0]
        self.assertIn("Atelectasis", summary_stage.valid_codes["2"])

    def test_unknown_site_uses_unscoped_naaccr_codes(self):
        group = build_variable_group(
            400,
            self.BASE_DICTIONARY,
            case_facts=CaseFacts(gross_primary_site="kidney"),
            site_data_dictionary_path=self.SITE_DICTIONARY,
        )

        site = group.variables[0]
        self.assertIn("C341", site.valid_codes)
        self.assertIn("C504", site.valid_codes)

    def test_snake_case_base_entry_populates_variable_info(self):
        group = build_variable_group(410, self.BASE_DICTIONARY)

        variable = group.variables[0]
        self.assertEqual(variable.name, "Laterality")
        self.assertEqual(variable.data_type, "digits")
        self.assertEqual(variable.length, 1)
        self.assertEqual(variable.valid_codes["1"], "Right: origin of primary")


if __name__ == "__main__":
    unittest.main()
