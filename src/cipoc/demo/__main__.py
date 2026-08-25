"""Phase 2 — the demo CLI: ``record`` a trace, or ``serve`` the dashboard.

Two subcommands::

    python -m cipoc.demo record --notes NOTES.json --out trace.jsonl
    python -m cipoc.demo serve --replay trace.jsonl
    python -m cipoc.demo serve --live --notes NOTES.json [--record trace.jsonl]

``record`` and ``serve --live`` drive the real orchestrator over live LLM calls,
so they need model credentials and network access. ``serve --replay`` needs only
a recorded trace file and runs fully offline — the primary presentation path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from cipoc.models import ClinicalNote

from .stream import record_demo_trace, run_demo_stream
from .trace import read_trace


def _load_notes(path: Path) -> list[ClinicalNote]:
    return [ClinicalNote(**note) for note in json.loads(path.read_text())]


def _graph_input(notes: Iterable[ClinicalNote]) -> dict[str, Any]:
    return {
        "note_corpus": {note.note_id: note for note in notes},
        "structured_data": {},
    }


def _build_orchestrator():
    """Import and construct the real orchestrator (deferred: needs credentials)."""
    from cipoc.agents.orchestrator import OrchestratorAgent

    return OrchestratorAgent()


def _session_hints(agent: Any) -> dict[str, Any]:
    """Seed the demo state with the agent's planned groups/hierarchy if exposed."""
    hints: dict[str, Any] = {}
    target_groups = getattr(agent, "_target_variables", None)
    hierarchy = getattr(agent, "_target_group_hierarchy", None)
    if target_groups:
        hints["target_groups"] = target_groups
    if hierarchy:
        hints["group_hierarchy"] = hierarchy
    return hints


def cmd_record(args: argparse.Namespace) -> int:
    notes = _load_notes(args.notes)
    agent = _build_orchestrator()
    print(f"Recording a demo trace over {len(notes)} note(s) → {args.out} (live LLM)…")
    count = record_demo_trace(
        agent.compiled_graph,
        _graph_input(notes),
        args.out,
    )
    print(f"Wrote {count} events to {args.out}.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import DemoSession, LiveDemoSession, build_app

    if args.replay:
        events = read_trace(args.replay)
        session: DemoSession = DemoSession(
            events, description=args.description or f"Replay of {Path(args.replay).name}"
        )
        print(f"Serving replay of {args.replay} ({len(events)} events).")
    else:
        notes = _load_notes(args.notes)
        agent = _build_orchestrator()
        record_path = str(args.record) if args.record else None
        stream = run_demo_stream(
            agent.compiled_graph,
            _graph_input(notes),
            record_path=record_path,
        )
        session = LiveDemoSession(
            stream,
            description=args.description or "CIPOC extraction (live)",
            **_session_hints(agent),
        )
        where = f" (recording to {record_path})" if record_path else ""
        print(f"Serving a live run over {len(notes)} note(s){where}.")

    # Build the app first: for a live session this wires the push-on-event
    # listener, so start the graph only after it is connected (no missed steps).
    app = build_app(session)
    if isinstance(session, LiveDemoSession):
        session.start()
    print(f"Open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cipoc.demo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Record a demo trace (live LLM).")
    record.add_argument("--notes", type=Path, required=True, help="JSON list of clinical notes.")
    record.add_argument("--out", type=Path, required=True, help="Output trace path (.jsonl).")
    record.set_defaults(func=cmd_record)

    serve = sub.add_parser("serve", help="Serve the demo dashboard.")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path, help="Serve a recorded trace file.")
    source.add_argument("--live", action="store_true", help="Serve a live orchestrator run.")
    serve.add_argument("--notes", type=Path, help="JSON list of notes (required with --live).")
    serve.add_argument("--record", type=Path, default=None, help="Also record the live run here.")
    serve.add_argument("--description", default=None, help="Human label for the run.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    if args.command == "serve" and args.live and not args.notes:
        parser.error("--live requires --notes.")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
