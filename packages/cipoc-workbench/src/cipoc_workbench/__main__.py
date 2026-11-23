"""Serve a CIPOC extraction run in the review workbench.

    cipoc-workbench serve \
        --state tests/test_outputs/case_state.json \
        --ground-truth gt/case01.json \
        --feedback feedback/case01.json

With no --state, the workbench starts empty; use Load Run... to open an artifact.
An optional --state auto-loads a startup artifact. --ground-truth and --feedback
require an explicit --state. Ground truth is a JSON object of ``{item_id: value}``.
With --ground-truth-dir, each opened run file loads the file of the same name
from that directory, if there is one.
Use ``serve --feedback-dir path/to/reviews`` to save feedback for multiple runs
opened locally in browser tabs without restarting the server. Selected artifacts
are not uploaded; refresh returns to empty without --state, or reloads the
configured startup artifact. Artifacts and feedback may contain PHI: serve only
on a trusted interface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import build_app

    if args.state is None and (args.feedback is not None or args.ground_truth is not None):
        print("--feedback and --ground-truth require an explicit --state.", file=sys.stderr)
        return 1

    for label, path in (("--state", args.state), ("--ground-truth", args.ground_truth)):
        if path is not None and not path.is_file():
            print(f"{label}: {path} does not exist.", file=sys.stderr)
            return 1
    if args.ground_truth_dir is not None and not args.ground_truth_dir.is_dir():
        print(f"--ground-truth-dir: {args.ground_truth_dir} is not a directory.", file=sys.stderr)
        return 1

    app = build_app(
        state_path=args.state,
        ground_truth_path=args.ground_truth,
        ground_truth_dir=args.ground_truth_dir,
        feedback_path=args.feedback,
        feedback_dir=args.feedback_dir,
    )

    print(f"State:        {args.state or 'none - use Load Run... to open an artifact'}")
    if args.ground_truth is not None:
        print(f"Ground truth: {args.ground_truth} (startup run)")
    if args.ground_truth_dir is not None:
        print(f"Ground truth: {args.ground_truth_dir} (matched by run file name)")
    if args.ground_truth is None and args.ground_truth_dir is None:
        print("Ground truth: none - use Ground truth... to load a file")
    if args.feedback_dir is not None:
        print(f"Feedback:     {args.feedback_dir} (one file per run UUID)")
    elif args.feedback is not None:
        print(f"Feedback:     {args.feedback} (startup run only)")
    else:
        print("Feedback:     none - the annotation form is read-only")
    print("Refresh:      " + ("reloads the configured startup artifact" if args.state is not None else "returns to empty (no --state)"))
    print(f"\nOpen http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cipoc-workbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve", help="Serve the workbench frontend.",
        description="Start empty and use Load Run... to open local run files in browser tabs without uploading them. Refresh returns to empty without --state, or reloads the configured startup artifact.",
        epilog="Artifacts and feedback may contain PHI. Serve only on a trusted interface.",
    )
    serve.add_argument("--state", type=Path, default=None,
                       help="Optional OrchestratorRunResult JSON to auto-load at startup (default: no artifact).")
    serve.add_argument("--ground-truth", type=Path, default=None,
                       help="Reference values for the initial startup run only, as a JSON object of {item_id: value}. Requires --state.")
    serve.add_argument("--ground-truth-dir", type=Path, default=None,
                       help="Directory of ground-truth files. Opening a run loads the file here with the same name as the run file, if any.")
    feedback = serve.add_mutually_exclusive_group()
    feedback.add_argument("--feedback", type=Path, default=None,
                          help="Reviewer annotations for the startup run only; accepts legacy feedback. Requires --state. Created on first save.")
    feedback.add_argument("--feedback-dir", type=Path, default=None,
                          help="Save feedback for every opened run as DIRECTORY/<run-uuid>.json, with or without --state. Created on first save.")
    # Localhost by default: the workbench renders raw note text and model output.
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
