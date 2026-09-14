import unittest

from cipoc.models import CaseFacts
from cipoc.tools import build_variable_group, load_rule_store


class CodingContextTests(unittest.TestCase):
    def test_breast_primary_site_uses_scoped_rule_descriptions(self):
        group = build_variable_group(
            400,
            "documents/manuals/naaccr_data_dictionary_v25.json",
            case_facts=CaseFacts(
                gross_primary_site="upper outer left breast",
                date_of_diagnosis="2025-02-24",
                sex="female",
            ),
            rule_store=load_rule_store("documents/rules", use_cache=False),
        )

        site = group.variables[0]
        self.assertEqual(site.valid_codes["C500"], "Nipple")
        self.assertEqual(site.valid_codes["C504"], "Upper outer quadrant of breast")


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
