import json
import unittest

from scripts.convert_data_dictionary import convert_dictionary, convert_item


class ConvertDataDictionaryTests(unittest.TestCase):
    def test_converts_item_fields_and_code_mapping(self):
        converted = convert_item(
            {
                "Data Item Number": 410,
                "Data Item Name": "Laterality",
                "Data Type": "digits",
                "Length": 1,
                "Parent XML Element": "Tumor",
                "Description": ["First sentence. ", "Second sentence."],
                "Allowable Values": "0-5, 9",
                "Code Descriptions": {"0": "Not a paired site", "1": "Right"},
            },
            naaccr_version="24",
        )

        self.assertEqual(converted["naaccr_version"], "24")
        self.assertEqual(converted["item_number"], 410)
        self.assertEqual(converted["xml_parent_id"], "Tumor")
        self.assertEqual(converted["description"], "First sentence. Second sentence.")
        self.assertEqual(converted["allowable_values"], "0-5, 9")
        self.assertEqual(converted["retired"], "No")
        self.assertEqual(
            converted["allowed_codes"],
            [
                {"code": "0", "description": "Not a paired site"},
                {"code": "1", "description": "Right"},
            ],
        )

    def test_converts_row_oriented_code_tables(self):
        converted = convert_item(
            {
                "Data Item Number": 764,
                "Code Descriptions": [
                    {"SS2018": "0", "Description": "In situ"},
                    {"SS2018": "1", "Description": "Localized"},
                ],
            },
            naaccr_version="25",
        )

        self.assertEqual(
            converted["allowed_codes"],
            [
                {"code": "0", "description": "In situ"},
                {"code": "1", "description": "Localized"},
            ],
        )

    def test_converts_partial_chunk_without_inventing_metadata(self):
        converted = convert_item(
            {"Code Descriptions": {"0": "No", "1": "Yes"}},
            naaccr_version="24",
        )

        self.assertEqual(
            converted,
            {
                "allowed_codes": [
                    {"code": "0", "description": "No"},
                    {"code": "1", "description": "Yes"},
                ]
            },
        )

    def test_converts_the_complete_tissue_dictionary(self):
        with open("documents/cipoc_data_dictionary.json", "r") as source_file:
            source = json.load(source_file)

        converted = convert_dictionary(source)

        self.assertEqual(set(converted), set(source))
        self.assertEqual(
            sum(len(items) for items in converted.values()),
            sum(len(items) for items in source.values()),
        )
        laterality = converted["breast"]["410"]
        self.assertEqual(laterality["item_name"], "Laterality")
        self.assertIsInstance(laterality["allowed_codes"], list)
        self.assertEqual(laterality["allowed_codes"][0]["code"], "0")

    def test_complete_flat_dictionary_is_idempotent(self):
        with open(
            "documents/manuals/naaccr_data_dictionary_v25.json", "r"
        ) as source_file:
            source = json.load(source_file)

        converted = convert_dictionary(source, naaccr_version="25")

        self.assertEqual(converted, source)
        self.assertEqual(len(converted), 783)
        self.assertEqual(converted["10"]["naaccr_version"], "25")


if __name__ == "__main__":
    unittest.main()
