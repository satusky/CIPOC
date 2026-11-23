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
from cipoc.tools import load_variable_groups
from cipoc.utils import CipocConfig
from tests.test_run_models import corpus, inputs, run_info


VARIABLE_GROUPS = Path(__file__).resolve().parents[1] / "config" / "variable_groups.json"


def usage_summary(*, partial=False):
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

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(cli, "OrchestratorAgent", return_value=agent),
        ):
            output = Path(directory) / "nested" / "result.json"
            actual = cli.run_case_state(
                notes,
                str(output),
                structured_data={390: "20260101"},
                max_concurrency=4,
                progress=False,
                capture_llm_content=False,
                max_content_chars=500,
            )
            written = output.read_text(encoding="utf-8")

        self.assertIs(actual, result)
        self.assertEqual(written, result.model_dump_json(indent=2))
        agent.run.assert_called_once_with(
            notes,
            structured_data={390: "20260101"},
            max_concurrency=4,
            progress=False,
            capture_llm_content=False,
            max_content_chars=500,
        )

    def test_run_case_state_writes_the_failure_artifact_before_raising(self):
        failure = OrchestratorRunFailure(
            run=run_info(status="failed"), inputs=inputs(), corpus=None,
            observability=RunObservability(llm_content_captured=False, collection_status="complete"),
            error="RuntimeError: endpoint unavailable",
        )
        agent = Mock(**{"run.side_effect": OrchestratorRunError(failure)})

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(cli, "OrchestratorAgent", return_value=agent),
        ):
            output = Path(directory) / "failure.json"
            with self.assertRaises(OrchestratorRunError):
                cli.run_case_state([{"note_id": "A7"}], output, progress=False)
            self.assertEqual(OrchestratorRunFailure.model_validate_json(output.read_text()), failure)

    def test_item_ids_trim_the_plan_and_keep_structured_items(self):
        result = result_with_usage()
        agent = Mock()
        agent.run.return_value = result
        plans = []

        def build_agent(*, config):
            plans.append(load_variable_groups(config.documents().variable_groups_path))
            return agent

        config = CipocConfig({"documents": {"variable_groups_path": str(VARIABLE_GROUPS)}})
        notes = [{"note_id": "A7"}]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(cli, "load_config", return_value=config),
            patch.object(cli, "OrchestratorAgent", side_effect=build_agent),
        ):
            actual = cli.run_case_state(
                notes,
                Path(directory) / "result.json",
                item_ids=[764],
                structured_data={400: "C50.4"},
                progress=False,
            )

        self.assertIs(actual, result)
        [plan] = plans
        self.assertEqual(
            [(group.group_id, [v.item_id for v in group.variables]) for group in plan],
            [("initial_llm_extraction", [400]), ("other", [764])],
        )
        agent.run.assert_called_once_with(
            notes,
            structured_data={400: "C50.4"},
            max_concurrency=None,
            progress=False,
            capture_llm_content=True,
            max_content_chars=None,
        )

    def test_cli_serializes_result_schema_directly_and_forwards_capture_options(self):
        result = result_with_usage(usage_summary())
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
            agent = Mock()
            agent.run.return_value = result
            with (
                patch.object(cli, "OrchestratorAgent", return_value=agent),
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
            agent.run.assert_called_once_with(
                [note],
                structured_data=None,
                max_concurrency=None,
                progress=False,
                capture_llm_content=False,
                max_content_chars=123,
            )
            self.assertIn("Usage: 1,500 tokens over 2 model calls.", stdout.getvalue())

    def test_usage_line_reports_totals_and_never_shows_unknown_usage_as_zero(self):
        unreported = LLMUsageSummary(
            logical_calls=1, model_invocations=1, successful_invocations=1, missing_usage_invocations=1
        )

        self.assertEqual(cli.usage_line(usage_summary()), "Usage: 1,500 tokens over 2 model calls.")
        self.assertEqual(
            cli.usage_line(usage_summary(partial=True)),
            "Usage: 1,500 tokens over 3 model calls (1 without reported usage).",
        )
        self.assertEqual(cli.usage_line(LLMUsageSummary()), "Usage: 0 tokens over 0 model calls.")
        self.assertEqual(cli.usage_line(unreported), "Usage: unavailable.")
        self.assertEqual(cli.usage_line(None), "Usage: unavailable.")

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
            agent = Mock()
            agent.run.side_effect = OrchestratorRunError(failure)
            with (
                patch.object(cli, "OrchestratorAgent", return_value=agent),
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
            self.assertIn("1 without reported usage", stdout.getvalue())
            self.assertIn("endpoint unavailable", stderr.getvalue())

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
                patch.object(cli, "OrchestratorAgent", return_value=Mock(**{"run.side_effect": OrchestratorRunError(failure)})),
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                code = cli.main(["--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(OrchestratorRunFailure.model_validate_json(output.read_text()), failure)
        self.assertIn("Telemetry collection: unavailable", stdout.getvalue())
        self.assertIn("Usage: unavailable", stdout.getvalue())
        self.assertNotIn("0 tokens", stdout.getvalue())
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
