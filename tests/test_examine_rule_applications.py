import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.examine_rule_applications import (
    build_parser,
    case_facts_from_args,
    main,
)
from cipoc.tools import load_rule_store


RULES_DIR = Path("tests/fixtures/rules")


class ExamineRuleApplicationsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dictionary = Path(self.temp_dir.name) / "dictionary.json"
        self.data_dictionary.write_text(
            json.dumps(
                {
                    "390": {
                        "Data Item Name": "Date of Diagnosis",
                        "Data Type": "date",
                        "Length": 8,
                        "Format": "YYYYMMDD",
                    },
                    "410": {
                        "Data Item Name": "Laterality",
                        "Length": 1,
                        "Code Descriptions": {"0": "Not a paired site", "1": "Right"},
                    },
                    "522": {
                        "Data Item Name": "Histologic Type ICD-O-3",
                        "Length": 4,
                        "Code Descriptions": {
                            "8500/2": "Ductal carcinoma in situ",
                            "8500/3": "Invasive ductal carcinoma",
                            "8520/2": "Lobular carcinoma in situ",
                            "8522/3": "Invasive duct and lobular carcinoma",
                            "9999/3": "Unrelated fixture code",
                        },
                    },
                    "523": {
                        "Data Item Name": "Behavior Code ICD-O-3",
                        "Length": 1,
                        "Code Descriptions": {"2": "In situ", "3": "Malignant"},
                    },
                    "999": {"Data Item Name": "Rule-free fixture item"},
                }
            )
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_inspector(self, *arguments: str) -> str:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(
                [
                    *arguments,
                    "--rules-dir",
                    str(RULES_DIR),
                    "--data-dictionary",
                    str(self.data_dictionary),
                ]
            )
        self.assertEqual(result, 0)
        return stdout.getvalue()

    def test_maps_every_case_fact_option(self):
        args = build_parser().parse_args(
            [
                "522",
                "--primary-site",
                "C509",
                "--gross-primary-site",
                "left breast",
                "--histology",
                "8500",
                "--behavior",
                "3",
                "--sex",
                "female",
                "--date-of-diagnosis",
                "2021-06-01",
            ]
        )

        self.assertEqual(
            case_facts_from_args(args).model_dump(),
            {
                "primary_site": "C509",
                "gross_primary_site": "left breast",
                "histology": "8500",
                "behavior": "3",
                "sex": "female",
                "date_of_diagnosis": "2021-06-01",
            },
        )

    def test_multiple_items_show_final_precedence_selection_and_provenance(self):
        output = self.run_inspector(
            "390", "410", "--date-of-diagnosis", "2024-06-01"
        )

        self.assertIn("Requested item IDs: 390, 410", output)
        self.assertIn("spcsm_2024:date_of_diagnosis", output)
        self.assertNotIn("store_2024:date_of_initial_diagnosis", output)
        self.assertIn("store_2024:laterality", output)
        self.assertIn(
            str((RULES_DIR / "spcsm_2024/general.json").resolve()), output
        )
        self.assertIn("rule_id: spcsm_2024:date_of_diagnosis", output)
        self.assertIn("documents/markdown/SPCSM_2024_MainDoc.md", output)
        self.assertIn("anchor: date-of-diagnosis", output)

    def test_case_applicability_filters_nonmatching_rules(self):
        output = self.run_inspector(
            "523",
            "--gross-primary-site",
            "breast",
            "--histology",
            "9999",
            "--date-of-diagnosis",
            "2021-06-01",
        )

        self.assertIn("solid_tumor_rules:breast:general", output)
        self.assertNotIn("solid_tumor_rules:breast:behavior_epc", output)

    def test_scoping_review_reasons_are_rendered(self):
        output = self.run_inspector("410")

        self.assertIn("Scoping review reasons:", output)
        self.assertIn("Item 410: unknown_dx_date_wide_scope", output)

    def test_code_table_and_scoped_variable_group_are_not_truncated(self):
        output = self.run_inspector(
            "522",
            "--gross-primary-site",
            "breast",
            "--date-of-diagnosis",
            "2021-06-01",
        )

        self.assertIn("Rule: solid_tumor_rules:breast:table3 (code_table)", output)
        self.assertIn("8500/2: Ductal carcinoma in situ", output)
        self.assertIn("Section path: Breast > Terms and Definitions > Table 3", output)
        self.assertIn("documents/markdown/SolidTumorRules_Combined.md", output)
        self.assertIn("VariableGroupInfo:\n{", output)
        self.assertIn('"coding_instructions": "-', output)
        self.assertIn('"8500":', output)
        self.assertNotIn('"9999":', output)

    def test_single_item_with_no_rules_prints_explicit_none(self):
        output = self.run_inspector("999")

        self.assertIn("Requested item IDs: 999", output)
        self.assertIn("Item 999:\n  (none)", output)
        self.assertIn('"item_id": 999', output)

    def test_reports_all_item_ids_missing_from_data_dictionary(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(
                [
                    "777",
                    "888",
                    "--rules-dir",
                    str(RULES_DIR),
                    "--data-dictionary",
                    str(self.data_dictionary),
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "item IDs absent from the data dictionary: 777, 888", stderr.getvalue()
        )

    def test_loader_records_compiled_source_file(self):
        store = load_rule_store(RULES_DIR, use_cache=False)

        self.assertEqual(
            store.source_files_by_rule_id["solid_tumor_rules:breast:h4"],
            (RULES_DIR / "solid_tumor_rules/breast.json").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
