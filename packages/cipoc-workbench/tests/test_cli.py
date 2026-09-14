"""CLI wiring and standalone imports, without starting a network server."""

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cipoc_workbench import EXAMPLE_DIR
from cipoc_workbench.__main__ import main
from cipoc_workbench.server import build_app


class WorkbenchCliTests(unittest.TestCase):
    def test_feedback_options_are_mutually_exclusive(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), patch("uvicorn.run") as run:
            with self.assertRaises(SystemExit) as error:
                main(["serve", "--feedback", "feedback.json", "--feedback-dir", "reviews"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("not allowed with argument", stderr.getvalue())
        run.assert_not_called()

    def test_feedback_modes_are_forwarded_without_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for option, key, label in (
                (None, None, "read-only"),
                ("--feedback", "feedback_path", "startup run only"),
                ("--feedback-dir", "feedback_dir", "one file per run UUID"),
            ):
                with self.subTest(option=option):
                    destination = Path(directory) / "new" / "reviews"
                    argv = ["serve", "--host", "127.0.0.2", "--port", "8123"]
                    state = EXAMPLE_DIR / "case_state.json" if option == "--feedback" else None
                    if state is not None:
                        argv.extend(["--state", str(state)])
                    if option:
                        argv.extend([option, str(destination)])
                    stdout = io.StringIO()
                    with redirect_stdout(stdout), patch("uvicorn.run") as run, \
                            patch("cipoc_workbench.server.build_app", wraps=build_app) as build:
                        result = main(argv)
                    self.assertEqual(result, 0)
                    expected = dict(state_path=state, ground_truth_path=None, feedback_path=None, feedback_dir=None)
                    if key:
                        expected[key] = destination
                    build.assert_called_once_with(**expected)
                    run.assert_called_once()
                    self.assertEqual(run.call_args.kwargs, {"host": "127.0.0.2", "port": 8123, "log_level": "warning"})
                    self.assertIn(label, stdout.getvalue())
                    if state is None:
                        self.assertIn("State:        none - use Load Run...", stdout.getvalue())
                        self.assertIn("returns to empty (no --state)", stdout.getvalue())
                    else:
                        self.assertIn(f"State:        {state}", stdout.getvalue())
                        self.assertIn("reloads the configured startup artifact", stdout.getvalue())
                    self.assertNotIn("bundled example", stdout.getvalue())
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_help_explains_binding_local_files_and_phi(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as error:
            main(["serve", "--help"])
        self.assertEqual(error.exception.code, 0)
        help_text = " ".join(stdout.getvalue().split())
        for text in ("--feedback-dir", "startup run only", "without uploading", "PHI", "trusted interface"):
            self.assertIn(text, help_text)
        self.assertIn("Reference values for the initial startup run only", help_text)
        for text in ("Start empty", "Load Run...", "default: no artifact", "auto-load at startup",
                     "Refresh returns to empty without --state", "reloads the configured startup artifact",
                     "Requires --state", "with or without --state"):
            self.assertIn(text, help_text)
        self.assertNotIn("bundled example", help_text)

    def test_startup_bound_options_require_explicit_state(self):
        for options in (
            ["--feedback", "feedback.json"],
            ["--ground-truth", "truth.json"],
            ["--feedback", "feedback.json", "--ground-truth", "truth.json"],
            ["--feedback-dir", "reviews", "--ground-truth", "truth.json"],
        ):
            stderr = io.StringIO()
            stdout = io.StringIO()
            with self.subTest(options=options), redirect_stderr(stderr), redirect_stdout(stdout), \
                    patch("uvicorn.run") as run, patch("cipoc_workbench.server.build_app") as build:
                self.assertEqual(main(["serve", *options]), 1)
            self.assertIn("require an explicit --state", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            build.assert_not_called()
            run.assert_not_called()

    def test_explicit_state_and_ground_truth_are_forwarded(self):
        state = EXAMPLE_DIR / "case_state.json"
        truth = EXAMPLE_DIR / "ground_truth.json"
        with redirect_stdout(io.StringIO()), patch("uvicorn.run") as run, \
                patch("cipoc_workbench.server.build_app", wraps=build_app) as build:
            self.assertEqual(main(["serve", "--state", str(state), "--ground-truth", str(truth)]), 0)
        build.assert_called_once_with(state_path=state, ground_truth_path=truth, feedback_path=None, feedback_dir=None)
        run.assert_called_once()

    def test_missing_configured_input_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for option in ("--state", "--ground-truth"):
                stderr = io.StringIO()
                argv = ["serve", option, str(Path(directory) / "missing.json")]
                if option == "--ground-truth":
                    argv.extend(["--state", str(EXAMPLE_DIR / "case_state.json")])
                with self.subTest(option=option), redirect_stderr(stderr), patch("uvicorn.run") as run:
                    result = main(argv)
                    self.assertEqual(result, 1)
                    self.assertIn(f"{option}:", stderr.getvalue())
                    self.assertIn("does not exist", stderr.getvalue())
                    run.assert_not_called()

    def test_malformed_startup_still_allows_server_and_browser_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "run.json"
            state.write_bytes(b"{malformed")
            for options in (
                [], ["--feedback", str(Path(directory) / "review.json")],
                ["--ground-truth", str(EXAMPLE_DIR / "ground_truth.json")],
                ["--feedback-dir", str(Path(directory) / "reviews")],
            ):
                with self.subTest(options=options), redirect_stdout(io.StringIO()), patch("uvicorn.run") as run:
                    self.assertEqual(main(["serve", "--state", str(state), *options]), 0)
                run.assert_called_once()
            self.assertEqual(state.read_bytes(), b"{malformed")

    def test_server_and_cli_import_without_runtime(self):
        code = """
import sys

class BlockRuntime:
    def find_spec(self, fullname, *args):
        if fullname == 'cipoc' or fullname.startswith('cipoc.'):
            raise AssertionError('Workbench must not import the runtime: ' + fullname)

sys.meta_path.insert(0, BlockRuntime())
from cipoc_workbench.server import build_app
from cipoc_workbench.__main__ import main
build_app()
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
