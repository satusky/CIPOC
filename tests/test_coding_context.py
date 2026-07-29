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


if __name__ == "__main__":
    unittest.main()
