import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import run_case_state as cli
from cipoc.models import (
    Case,
    LLMUsageSummary,
    OrchestratorRunError,
    OrchestratorRunFailure,
    OrchestratorRunResult,
    RunObservability,
)
from tests.test_run_models import corpus, inputs, run_info


def usage_summary(*, partial=False, details=False):
    invocations = 3 if partial else 2
    values = {
        "logical_calls": 2,
        "model_invocations": invocations,
        "successful_invocations": 2,
        "failed_invocations": 1 if partial else 0,
        "retry_invocations": 1 if partial else 0,
        "usage_reported_invocations": 2,
        "missing_usage_invocations": 1 if partial else 0,
        "input_tokens": 1200,
        "output_tokens": 300,
        "total_tokens": 1500,
    }
    if details:
        values["input_token_details"] = {"cache_read": 200, "unused": 0}
        values["output_token_details"] = {"reasoning": 75}
    return LLMUsageSummary(**values)


def result_with_usage(summary=None):
    return OrchestratorRunResult(
        run=run_info(),
        case=Case(),
        inputs=inputs(),
        corpus=corpus(),
        observability=RunObservability(
            llm_content_captured=False,
            llm_usage_summary=summary or LLMUsageSummary(),
        ),
    )


class RunCaseStateCliTests(unittest.TestCase):
    def test_run_case_state_calls_only_the_agent_public_run_api(self):
        result = result_with_usage()
        agent = Mock()
        agent.run.return_value = result
        notes = [{"note_id": "A7"}]

        with patch.object(cli, "OrchestratorAgent", return_value=agent):
            actual = cli.run_case_state(
                notes,
                structured_data={390: "20260101"},
                max_concurrency=4,
                progress=False,
                capture_llm_content=False,
                max_content_chars=500,
            )

        self.assertIs(actual, result)
        agent.run.assert_called_once_with(
            notes,
            structured_data={390: "20260101"},
            max_concurrency=4,
            progress=False,
            capture_llm_content=False,
            max_content_chars=500,
        )

    def test_cli_serializes_result_schema_directly_and_forwards_capture_options(self):
        result = result_with_usage(usage_summary(details=True))
        note = {
            "note_id": "note-A",
            "date": "2026-09-03",
            "note_type": "Pathology",
            "content": "Clinical text",
        }
        with tempfile.TemporaryDirectory() as directory:
            notes_path = Path(directory) / "notes.json"
            output_path = Path(directory) / "result.json"
            notes_path.write_text(json.dumps([note]), encoding="utf-8")
            stdout = io.StringIO()
            with (
                patch.object(cli, "run_case_state", return_value=result) as run,
                redirect_stdout(stdout),
            ):
                exit_code = cli.main(
                    [
                        "--notes",
                        str(notes_path),
                        "--output",
                        str(output_path),
                        "--no-progress",
                        "--no-llm-content-capture",
                        "--max-content-chars",
                        "123",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                result.model_dump_json(indent=2),
            )
            self.assertEqual(
                OrchestratorRunResult.model_validate_json(
                    output_path.read_text(encoding="utf-8")
                ),
                result,
            )
            run.assert_called_once_with(
                [note],
                structured_data=None,
                max_concurrency=None,
                progress=False,
                capture_llm_content=False,
                max_content_chars=123,
            )
            self.assertIn("Token details: input.cache_read=200", stdout.getvalue())

    def test_usage_rendering_handles_complete_partial_and_zero_usage(self):
        complete = cli.usage_lines(usage_summary(details=True))
        partial = cli.usage_lines(usage_summary(partial=True))
        zero = cli.usage_lines(LLMUsageSummary())

        self.assertEqual(complete[0], "Tokens: input=1,200 output=300 total=1,500")
        self.assertEqual(complete[1], "Calls: logical=2 invocations=2 retries=0")
        self.assertEqual(complete[2], "Usage coverage: reported=2 missing=0")
        self.assertEqual(
            complete[3],
            "Token details: input.cache_read=200, output.reasoning=75",
        )
        self.assertEqual(partial[1], "Calls: logical=2 invocations=3 retries=1")
        self.assertEqual(partial[2], "Usage coverage: reported=2 missing=1")
        self.assertEqual(
            zero,
            [
                "Tokens: input=0 output=0 total=0",
                "Calls: logical=0 invocations=0 retries=0",
                "Usage coverage: reported=0 missing=0",
            ],
        )

    def test_cli_writes_partial_failure_before_nonzero_exit(self):
        failure = OrchestratorRunFailure(
            run=run_info(status="failed"),
            inputs=inputs(),
            corpus=None,
            observability=RunObservability(
                llm_content_captured=False,
                llm_usage_summary=usage_summary(partial=True),
            ),
            error="RuntimeError: endpoint unavailable",
        )
        note = {
            "note_id": 1,
            "date": "2026-09-03",
            "note_type": "Pathology",
            "content": "Clinical text",
        }
        with tempfile.TemporaryDirectory() as directory:
            notes_path = Path(directory) / "notes.json"
            output_path = Path(directory) / "failure.json"
            notes_path.write_text(json.dumps([note]), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(
                    cli,
                    "run_case_state",
                    side_effect=OrchestratorRunError(failure),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main(
                    ["--notes", str(notes_path), "--output", str(output_path)]
                )

            self.assertEqual(exit_code, 1)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                failure.model_dump_json(indent=2),
            )
            self.assertNotIn("case", json.loads(output_path.read_text(encoding="utf-8")))
            self.assertIn("missing=1", stdout.getvalue())
            self.assertIn("endpoint unavailable", stderr.getvalue())

    def test_script_has_no_parallel_graph_or_observability_driver(self):
        source = Path(cli.__file__).read_text(encoding="utf-8")

        self.assertNotIn("compiled_graph", source)
        self.assertNotIn("ObservabilityCollector", source)
        self.assertNotIn("normalize(", source)
        self.assertNotIn("_workbench_note_selection", source)
        self.assertNotIn("_retriever_offered", source)

    def test_capture_help_warns_that_corpus_phi_remains(self):
        parser = cli.build_parser()

        self.assertFalse(parser.parse_args([]).no_llm_content_capture)
        self.assertTrue(
            parser.parse_args(["--no-llm-capture"]).no_llm_content_capture
        )
        self.assertIsNone(parser.parse_args([]).max_content_chars)
        self.assertIn("does not de-identify", parser.format_help())


if __name__ == "__main__":
    unittest.main()
