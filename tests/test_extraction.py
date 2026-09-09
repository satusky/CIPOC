import unittest

from cipoc.models import ConfidenceLevel, VariableInfo, VariableOutput
from cipoc.tools import VariableValueValidator


class VariableValueValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = VariableValueValidator()
        self.date_variable = VariableInfo(
            item_id=1280,
            name="RX Date DX/Stg Proc",
            data_type="date",
            length=8,
            format="YYYYMMDD",
            valid_codes={"CCYYMMDD": "CCYYMMDD", "Blank": "Blank"},
        )

    @staticmethod
    def candidate(value: str | None) -> VariableOutput:
        return VariableOutput(
            item_id=1280,
            value=value,
            explanation="Test candidate.",
            most_important_note=1,
            spans=[],
            presence_confidence=ConfidenceLevel.HIGH,
        )

    def test_date_format_token_is_not_accepted_as_a_value(self):
        errors = self.validator.validate(
            self.date_variable, self.candidate("CCYYMMDD")
        )

        self.assertIn(
            "Date must contain exactly eight ASCII digits in YYYYMMDD form.", errors
        )

    def test_date_validation_ignores_malformed_scoped_code_table(self):
        errors = self.validator.validate(
            self.date_variable, self.candidate("20250318")
        )

        self.assertEqual(errors, [])

    def test_row_oriented_code_table_is_used_for_membership_validation(self):
        variable = VariableInfo(
            item_id=1280,
            valid_codes=[
                {"code": "C500", "description": "Nipple"},
                {"code": "C509", "description": "Breast, NOS"},
            ],
        )

        errors = self.validator.validate(variable, self.candidate("C50"))

        self.assertIn("Value is not one of the variable's allowable codes.", errors)

    def test_row_oriented_code_table_accepts_a_listed_code(self):
        variable = VariableInfo(
            item_id=1280,
            valid_codes=[{"code": "C500", "description": "Nipple"}],
        )

        errors = self.validator.validate(variable, self.candidate("C500"))

        self.assertEqual(errors, [])

    def assert_values(self, variable, accepted, rejected):
        self.validator.preflight(variable)
        for value in accepted + rejected:
            with self.subTest(item_id=variable.item_id, value=value):
                candidate = self.candidate(value).model_copy(update={"item_id": variable.item_id})
                errors = self.validator.validate(variable, candidate)
                self.assertEqual(not errors, value in accepted, errors)

    def test_numeric_ranges_preserve_width_boundaries_and_special_codes(self):
        self.assert_values(
            VariableInfo(item_id=820, length=2, data_type="digits", valid_codes={
                "00": "Negative", "01-89": "Count", "90": "90 or more",
                "95": "Aspiration", "97": "Unspecified", "98": "Not examined", "99": "Unknown",
            }),
            ["00", "01", "03", "89", "90", "95", "97", "98", "99"],
            ["0", "3", "003", "91", "96", "01-89", " 03", "\u0660\u0663"],
        )
        self.assert_values(
            VariableInfo(item_id=756, length=3, valid_codes={
                "000": "No mass", "001": "1 mm", "002-988": "Exact size",
                "989": "989 or larger", "990": "Microscopic", "998": "Special", "999": "Unknown",
            }),
            ["000", "002", "010", "988", "989", "990", "998", "999"],
            ["10", "0010", "991", "997", "002-988"],
        )

    def test_empty_tables_use_item_671_declared_prefixed_ranges(self):
        for codes in (None, {}, []):
            with self.subTest(codes=codes):
                self.assert_values(
                    VariableInfo(
                        item_id=671, length=4, data_type="mixed", format="Alphanumeric Blank",
                        allowable_values="A000, A200-A990, B000, B200-B990, Blank", valid_codes=codes,
                    ),
                    ["A000", "A200", "A550", "A990", "B000", "B200", "B990"],
                    ["ABC", "A199", "A991", "B199", "C200", "a200", "A20", "A0200", "Blank", ""],
                )

    def test_item_676_known_abbreviation_uses_only_explicit_source_domain(self):
        codes = {code: "Description" for code in ("00", "01", "02", "..", "90", "95", "96", "97", "98", "99")}
        variable = VariableInfo(item_id=676, length=2, data_type="digits",
                                allowable_values="00-90, 95-99", valid_codes=codes)
        self.assert_values(variable, ["00", "03", "89", "90", "95", "99"],
                           ["..", "...", "3", "003", "91", "94"])
        for changes in ({"item_id": 830}, {"allowable_values": "00-99"}, {"allowable_values": None}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "metadata"):
                self.validator.preflight(variable.model_copy(update=changes))

    def test_scoped_tables_are_not_broadened_by_base_allowable_values(self):
        for item_id, codes, declaration, rejected in (
            (676, {"00": "None", "02": "Two", "99": "Unknown"}, "00-90, 95-99", "03"),
            (671, {"A200": "Scoped", "A990": "Unknown"}, "A000, A200-A990, B000, B200-B990", "A550"),
        ):
            self.assert_values(VariableInfo(item_id=item_id, valid_codes=codes, allowable_values=declaration),
                               list(codes), [rejected])

    def test_length_is_maximum_unless_format_or_range_requires_width(self):
        self.assert_values(
            VariableInfo(item_id=3836, length=5, data_type="text", format="Left justified",
                         allowable_values="Valid alphanumeric codes for FIGO stages; 97-99",
                         valid_codes={"1": "I", "1A": "IA", "3A11": "IIIA1i", "99": "Unknown"}),
            ["1", "1A", "3A11", "99"], ["00001", "1a", "3A1111"],
        )
        self.assert_values(VariableInfo(item_id=1, data_type="digits", length=3), ["1", "123"], ["1234", "A"])
        self.assert_values(VariableInfo(item_id=1, data_type="digits", length=3, format="Right justified, zero filled"),
                           ["001", "123"], ["1", "12", "1234"])

    def test_unsupported_domains_and_missing_metadata_fail_preflight(self):
        for fields in (
            {}, {"length": 3}, {"data_type": "text", "length": 5},
            {"length": 3, "format": "Right justified, zero filled"},
            {"data_type": "digits", "format": "Two decimal places"},
            {"length": 1, "valid_codes": {"999": "Too wide"}},
            {"data_type": "digits", "valid_codes": {"ABC": "Not numeric"}},
            {"valid_codes": {"A200-B990": "Mixed prefix"}},
            {"valid_codes": {"1-09": "Mixed width"}},
            {"valid_codes": {"09-01": "Reversed"}},
            {"valid_codes": {"1 to 9": "Unsupported"}, "allowable_values": "1-9"},
            {"valid_codes": {"..": "Ellipsis"}, "allowable_values": "00-99"},
            {"valid_codes": "See manual", "data_type": "digits"},
            {"allowable_values": "00-90, see manual", "data_type": "digits"},
            {"allowable_values": "Valid alphanumeric codes for FIGO stages; 97-99", "format": "Alphanumeric"},
        ):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "metadata for item 1"):
                self.validator.preflight(VariableInfo(item_id=1, **fields))

    def test_code_only_metadata_and_exact_case_literals_remain_valid(self):
        self.assert_values(VariableInfo(item_id=1, valid_codes={"X6": "Unknown", "cT1": "Stage", "N/A": "Special"}),
                           ["X6", "cT1", "N/A"], ["x6", "CT1", "n/a"])

    def test_duplicate_documentation_rows_do_not_remove_valid_codes(self):
        self.assert_values(VariableInfo(item_id=820, valid_codes=[
            {"code": "00", "description": "No positive nodes"},
            {"code": "00", "description": "All examined nodes negative"},
            {"code": "01-89", "description": "Count"},
        ]), ["00", "03"], ["90", "01-89"])

    def test_explicit_free_text_allows_spaces_but_not_controls_or_blank_strings(self):
        self.assert_values(VariableInfo(item_id=310, data_type="text", length=100, format="Free text",
                                       allowable_values="Neither carriage return nor line feed characters allowed"),
                           ["Registered nurse", "Unknown"], ["", " ", "Nurse\n", "Nurse\t", "x" * 101])

    def test_calendar_dates_and_format_tokens_take_precedence(self):
        for format_value in ("YYYYMMDD", "CCYYMMDD", "YYYYMMDD\\r\\nFixed-length, left-justified, space filled."):
            self.assert_values(self.date_variable.model_copy(update={"data_type": None, "format": format_value}),
                               ["20240229", "20250200"], ["20250229", "20251301", "00000101", "CCYYMMDD", "202501"])
        errors = VariableValueValidator(allow_unknown_date_day=False).validate(self.date_variable, self.candidate("20250100"))
        self.assertTrue(errors)

    def test_null_is_not_an_error_but_does_not_bypass_metadata_or_item_identity(self):
        candidate = self.candidate(None)
        self.assertEqual(self.validator.validate(self.date_variable, candidate), [])
        self.assertTrue(self.validator.validate(self.date_variable, candidate.model_copy(update={"item_id": 1})))
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.validator.validate(VariableInfo(item_id=1280), candidate)


if __name__ == "__main__":
    unittest.main()
