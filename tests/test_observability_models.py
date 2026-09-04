import unittest

from pydantic import ValidationError

from cipoc.models import (
    LLMExchange,
    LLMPromptMessage,
    LLMUsageBucket,
    LLMUsageSummary,
    NormalizedTokenUsage,
    RunObservability,
    TokenDetails,
    VariableAttempt,
)


def usage_bucket(**overrides):
    values = {
        "logical_calls": 1,
        "model_invocations": 2,
        "successful_invocations": 1,
        "failed_invocations": 1,
        "retry_invocations": 1,
        "usage_reported_invocations": 1,
        "missing_usage_invocations": 1,
        "input_tokens": 12,
        "output_tokens": 5,
        "total_tokens": 17,
        "input_token_details": {"cache_read": 4, "provider_special": 2},
        "output_token_details": {"reasoning": 3},
    }
    values.update(overrides)
    return LLMUsageBucket(**values)


class ObservabilityModelTests(unittest.TestCase):
    def test_current_artifact_fields_validate_and_entity_key_is_retained(self):
        observability = RunObservability(
            llm_content_captured=True,
            variable_attempts={
                "group:initial/variable:390": [
                    {
                        "attempt": 1,
                        "mode": "group",
                        "candidate": {"item_id": 390, "value": "1"},
                        "validation_errors": [],
                        "is_valid": True,
                    }
                ]
            },
            llm_exchanges={
                "note:ABC-1": [
                    {
                        "agent": "note_scanner",
                        "node": "summarize_note",
                        "attempt": 1,
                        "prompt_messages": [
                            {"role": "human", "content": "summarize"}
                        ],
                        "response": {"summary": "visible"},
                        "model": "test-model",
                        "usage": {
                            "input_tokens": 8,
                            "output_tokens": 3,
                            "total_tokens": 11,
                        },
                        "error": None,
                    }
                ]
            },
        )

        exchange = observability.llm_exchanges["note:ABC-1"][0]
        self.assertEqual(exchange.entity_key, "note:ABC-1")
        self.assertEqual(exchange.attempt, 1)
        self.assertIsNone(exchange.retry_ordinal)
        self.assertEqual(exchange.usage.input_token_details.root, {})
        self.assertEqual(
            observability.variable_attempts[
                "group:initial/variable:390"
            ][0].candidate,
            {"item_id": 390, "value": "1"},
        )

    def test_complete_contract_round_trips_through_json(self):
        bucket = usage_bucket()
        observability = RunObservability(
            llm_content_captured=True,
            max_content_chars=5,
            content_truncated=True,
            variable_attempts={
                "group:initial/variable:390": [
                    VariableAttempt(
                        attempt=2,
                        mode="repair",
                        candidate={"item_id": 390, "value": None},
                        validation_errors=["missing value"],
                        is_valid=False,
                    )
                ]
            },
            llm_exchanges={
                "group:initial/variable:390": [
                    LLMExchange(
                        entity_key="group:initial/variable:390",
                        agent="extractor",
                        node="repair_invalid_extraction",
                        attempt=2,
                        retry_ordinal=1,
                        model="gpt-test",
                        prompt_messages=[
                            LLMPromptMessage(
                                role="human",
                                content="abcde",
                                truncated=True,
                                original_char_count=12,
                            )
                        ],
                        response={"item_id": 390, "value": None},
                        usage=NormalizedTokenUsage(
                            input_tokens=12,
                            output_tokens=5,
                            total_tokens=17,
                            input_token_details=TokenDetails(
                                root={"cache_read": 4, "provider_special": 2}
                            ),
                            output_token_details={"reasoning": 3},
                        ),
                    )
                ]
            },
            llm_usage_summary=LLMUsageSummary(
                **bucket.model_dump(),
                by_agent={"extractor": bucket},
                by_node={"repair_invalid_extraction": bucket},
                by_model={"gpt-test": bucket},
            ),
        )

        restored = RunObservability.model_validate_json(observability.model_dump_json())

        self.assertEqual(restored, observability)
        dumped = restored.model_dump(mode="json")
        self.assertEqual(
            dumped["llm_usage_summary"]["input_token_details"],
            {"cache_read": 4, "provider_special": 2},
        )
        self.assertEqual(
            dumped["llm_exchanges"]["group:initial/variable:390"][0][
                "response"
            ],
            {"item_id": 390, "value": None},
        )

    def test_usage_bucket_rejects_inconsistent_counts_and_negative_details(self):
        with self.assertRaises(ValidationError):
            usage_bucket(model_invocations=3)
        with self.assertRaises(ValidationError):
            TokenDetails(root={"cache_read": -1})

    def test_contract_rejects_non_json_values_and_unknown_attempt_modes(self):
        with self.assertRaises(ValidationError):
            VariableAttempt(
                attempt=1,
                mode="fallback",
                candidate=None,
                is_valid=False,
            )
        with self.assertRaises(ValidationError):
            VariableAttempt(
                attempt=1,
                mode="individual",
                candidate=object(),
                is_valid=False,
            )

    def test_content_metadata_must_match_capture_and_truncation_settings(self):
        exchange = {
            "entity_key": "note:1",
            "agent": "note_scanner",
            "node": "summarize_note",
            "attempt": 1,
            "prompt_messages": [{"role": "human", "content": "visible"}],
        }
        with self.assertRaises(ValidationError):
            RunObservability(
                llm_content_captured=False,
                llm_exchanges={"note:1": [exchange]},
            )
        with self.assertRaises(ValidationError):
            RunObservability(
                llm_content_captured=True,
                max_content_chars=3,
                llm_exchanges={"note:1": [exchange]},
            )
        with self.assertRaises(ValidationError):
            LLMPromptMessage(
                role="human",
                content="full",
                truncated=True,
                original_char_count=4,
            )

        metadata_only = RunObservability(
            llm_content_captured=False,
            llm_exchanges={
                "note:1": [
                    {
                        "entity_key": "note:1",
                        "agent": "note_scanner",
                        "node": "summarize_note",
                        "attempt": 1,
                        "model": "test-model",
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                ]
            },
        )
        self.assertIsNone(metadata_only.llm_exchanges["note:1"][0].prompt_messages)


if __name__ == "__main__":
    unittest.main()
