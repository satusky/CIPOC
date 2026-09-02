"""Serve a CIPOC extraction run in the review workbench.

    cipoc-workbench serve \
        --state tests/test_outputs/case_state.json \
        --ground-truth gt/case01.json \
        --feedback feedback/case01.json

Every path is optional. With none, this serves the example bundled with the
package. The ground-truth file is a JSON object of ``{item_id: value}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import build_app

    for label, path in (("--state", args.state), ("--ground-truth", args.ground_truth)):
        if path is not None and not path.is_file():
            print(f"{label}: {path} does not exist.", file=sys.stderr)
            return 1

    app = build_app(
        state_path=args.state,
        ground_truth_path=args.ground_truth,
        feedback_path=args.feedback,
    )

    print(f"State:        {args.state or 'the bundled example'}")
    print(f"Ground truth: {args.ground_truth or 'none — comparison features stay hidden'}")
    print(f"Feedback:     {args.feedback or 'none — the annotation form is read-only'}")
    print(f"\nOpen http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cipoc-workbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Serve the workbench frontend.")
    serve.add_argument("--state", type=Path, default=None,
                       help="Orchestrator state JSON (default: the copy committed in web/).")
    serve.add_argument("--ground-truth", type=Path, default=None,
                       help="Reference values as a JSON object of {item_id: value}.")
    serve.add_argument("--feedback", type=Path, default=None,
                       help="Where to read and write reviewer annotations. Created on first save.")
    # Localhost by default: the workbench renders raw note text and model output.
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
