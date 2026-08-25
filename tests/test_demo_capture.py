"""Tap 2: LLMCaptureHandler folds callback events into correlated LLMCalls."""

import unittest
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cipoc.demo.capture import LLMCaptureHandler


def _metadata(node="summarize_note", ns="note_branch:1|summarize_note:2", **extra):
    return {"langgraph_node": node, "langgraph_checkpoint_ns": ns, **extra}


def _llm_result(message: AIMessage) -> LLMResult:
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class CaptureLifecycleTests(unittest.TestCase):
    def test_start_then_end_produces_one_correlated_call(self):
        handler = LLMCaptureHandler()
        run_id = uuid4()
        handler.on_chat_model_start(
            {},
            [[SystemMessage(content="sys"), HumanMessage(content="do it")]],
            run_id=run_id,
            metadata=_metadata(ls_model_name="gpt-oss-120b"),
        )
        self.assertEqual(handler.snapshot(), [])  # nothing until the end fires

        message = AIMessage(
            content='{"value": "1"}',
            usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        )
        handler.on_llm_end(_llm_result(message), run_id=run_id)

        calls = handler.snapshot()
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call.node, "summarize_note")
        self.assertEqual(call.namespace, ("note_branch:1", "summarize_note:2"))
        self.assertEqual(call.model, "gpt-oss-120b")
        self.assertEqual(
            call.prompt_messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "human", "content": "do it"},
            ],
        )
        self.assertEqual(call.response, '{"value": "1"}')
        self.assertEqual(call.usage, {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13})
        self.assertIsNone(call.error)

    def test_namespace_is_empty_without_metadata(self):
        handler = LLMCaptureHandler()
        run_id = uuid4()
        handler.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=run_id)
        handler.on_llm_end(_llm_result(AIMessage(content="y")), run_id=run_id)
        call = handler.snapshot()[0]
        self.assertEqual(call.namespace, ())
        self.assertEqual(call.node, "")

    def test_multimodal_content_blocks_flatten_to_text(self):
        handler = LLMCaptureHandler()
        run_id = uuid4()
        human = HumanMessage(content=[{"type": "text", "text": "part-a"}, {"type": "text", "text": "part-b"}])
        handler.on_chat_model_start({}, [[human]], run_id=run_id, metadata=_metadata())
        handler.on_llm_end(_llm_result(AIMessage(content="ok")), run_id=run_id)
        self.assertEqual(handler.snapshot()[0].prompt_messages[0]["content"], "part-apart-b")


class ReasoningExtractionTests(unittest.TestCase):
    def _capture_with(self, **message_kwargs):
        handler = LLMCaptureHandler()
        run_id = uuid4()
        handler.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=run_id, metadata=_metadata())
        handler.on_llm_end(_llm_result(AIMessage(content="ok", **message_kwargs)), run_id=run_id)
        return handler.snapshot()[0]

    def test_plain_string_reasoning_in_response_metadata(self):
        call = self._capture_with(response_metadata={"reasoning_content": "because reasons"})
        self.assertEqual(call.reasoning, "because reasons")

    def test_structured_summary_reasoning_in_additional_kwargs(self):
        call = self._capture_with(
            additional_kwargs={"reasoning": {"summary": [{"text": "step one"}, {"text": "step two"}]}}
        )
        self.assertEqual(call.reasoning, "step one\nstep two")

    def test_absent_reasoning_stays_none(self):
        self.assertIsNone(self._capture_with().reasoning)


class ErrorAndOrphanTests(unittest.TestCase):
    def test_llm_error_is_recorded_on_the_call(self):
        handler = LLMCaptureHandler()
        run_id = uuid4()
        handler.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=run_id, metadata=_metadata())
        handler.on_llm_error(ValueError("boom"), run_id=run_id)
        call = handler.snapshot()[0]
        self.assertEqual(call.error, "ValueError: boom")
        self.assertEqual(call.node, "summarize_note")

    def test_end_without_start_still_records_something(self):
        handler = LLMCaptureHandler()
        run_id = uuid4()
        handler.on_llm_end(_llm_result(AIMessage(content="orphan")), run_id=run_id)
        calls = handler.snapshot()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].response, "orphan")
        self.assertEqual(calls[0].node, "")


class ByAgentTests(unittest.TestCase):
    def test_by_agent_counts_calls_per_owning_agent(self):
        handler = LLMCaptureHandler()

        def fire(node, ns):
            run_id = uuid4()
            handler.on_chat_model_start(
                {}, [[HumanMessage(content="x")]], run_id=run_id,
                metadata=_metadata(node=node, ns=ns),
            )
            handler.on_llm_end(_llm_result(AIMessage(content="y")), run_id=run_id)

        fire("summarize_note", "note_branch:1|summarize_note:2")
        fire("identify_relevant_notes", "retrieve_notes:3|identify_relevant_notes:4")
        fire("extract_individual_value", "extract:5|variable_branch:6")
        fire("plan_extraction", "plan_extraction:7")

        self.assertEqual(
            handler.by_agent(),
            {"scanner": 1, "retriever": 1, "extractor": 1, "orchestrator": 1},
        )


if __name__ == "__main__":
    unittest.main()
