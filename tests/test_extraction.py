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
    def candidate(value: str) -> VariableOutput:
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


if __name__ == "__main__":
    unittest.main()
