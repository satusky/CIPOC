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
            collection_status="complete",
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
                collection_status="complete",
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

    def test_derived_timing_and_scalar_coverage_include_diagnostic_invocations_once(self):
        obs = RunObservability(
            llm_content_captured=False, collection_status="partial",
            collection_issues=[{"code": "missing_task_binding", "message": "Diagnostic call."}],
            llm_usage_summary=usage_summary(partial=True),
            llm_exchanges={"note:1": [{
                "agent": "note_scanner", "node": "summarize_note", "attempt": 1,
                "service_seconds": 0, "queue_seconds": 0,
                "usage": {}, "usage_reported_fields": ["input_tokens", "output_tokens"],
            }, {
                "agent": "note_scanner", "node": "detect_concepts", "attempt": 1,
                "usage_reported_fields": [],
            }]},
            unattributed_exchanges=[{
                "invocation_id": "diagnostic", "agent": "unknown", "node": "unknown",
                "service_seconds": 3.5, "queue_seconds": 1.25,
            }],
        )
        before = obs.model_dump_json()
        lines = cli.usage_lines(obs.llm_usage_summary, observability=obs)
        self.assertIn("Timing records: retained=3 diagnostic=1 summary_invocations=3; collection=partial", lines)
        self.assertIn("Summed invocation time: 3.500s mean=1.750s max=3.500s; recorded=2 missing=1", lines)
        self.assertIn("Summed queue wait: 1.250s mean=0.625s max=1.250s; recorded=2 missing=1", lines)
        self.assertIn("Scalar provenance (input/output): verified=1 incomplete=1 unknown=1", lines)
        self.assertEqual(obs.model_dump_json(), before)

        payload = obs.model_dump()
        payload["collection_issues"].append({"code": "invalid_invocation_sequence", "message": "Duplicate identity."})
        obs = RunObservability.model_validate(payload)
        lines = cli.usage_lines(None, observability=obs)
        self.assertIn("Usage: unavailable; telemetry did not produce a valid summary.", lines)
        self.assertIn("Timing/provenance aggregates: unavailable; invalid invocation identity or sequence.", lines)
        self.assertFalse(any("Summed invocation time:" in line for line in lines))

    def test_legacy_missing_and_measured_zero_timing_remain_distinct(self):
        obs = RunObservability(
            llm_content_captured=False,
            llm_exchanges={"note:1": [{
                "agent": "note_scanner", "node": "summarize_note", "attempt": 1,
            }]},
        )
        lines = cli.usage_lines(None, observability=obs)
        self.assertIn("Summed invocation time: unavailable; recorded=0 missing=1", lines)
        self.assertIn("Summed queue wait: unavailable; recorded=0 missing=1", lines)
        self.assertIn("Scalar provenance (input/output): verified=0 incomplete=0 unknown=1", lines)
        self.assertIn("summary_invocations=unavailable; collection=unknown", "\n".join(lines))
        obs.llm_exchanges["note:1"][0].service_seconds = 0
        lines = cli.usage_lines(None, observability=obs)
        self.assertIn("Summed invocation time: 0.000s mean=0.000s max=0.000s; recorded=1 missing=0", lines)

    def test_script_has_no_parallel_graph_or_observability_driver(self):
        source = Path(cli.__file__).read_text(encoding="utf-8")

        self.assertNotIn("compiled_graph", source)
        self.assertNotIn("ObservabilityCollector", source)
        self.assertNotIn("normalize(", source)
        self.assertNotIn("_workbench_note_selection", source)
        self.assertNotIn("_retriever_offered", source)

    def test_cli_reports_unavailable_telemetry_without_zero_usage_or_crashing(self):
        failure = OrchestratorRunFailure(
            run=run_info(status="failed"), inputs=inputs(), corpus=None,
            observability=RunObservability(
                llm_content_captured=False, collection_status="unavailable",
                collection_issues=[{"code": "telemetry_finalization_error", "message": "Summary unavailable."}],
                llm_usage_summary=None,
            ),
            error="RuntimeError: original graph failure",
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure.json"
            with (
                patch.object(cli, "_load_notes", return_value=[{"note_id": "synthetic"}]),
                patch.object(cli, "run_case_state", side_effect=OrchestratorRunError(failure)),
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                code = cli.main(["--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(OrchestratorRunFailure.model_validate_json(output.read_text()), failure)
        self.assertIn("Telemetry collection: unavailable", stdout.getvalue())
        self.assertIn("Usage: unavailable", stdout.getvalue())
        self.assertNotIn("total=0", stdout.getvalue())
        self.assertIn("original graph failure", stderr.getvalue())

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
