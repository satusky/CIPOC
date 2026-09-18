"""Record or present a CIPOC run locally. Live serving waits for Start Run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stream import DemoRun, DemoSetupError


def build_agent(config_path: Path, base_dir: Path):
    """Resolve resources once without changing process CWD or environment."""
    from cipoc.agents import OrchestratorAgent
    from cipoc.models import CaseFacts
    from cipoc.utils import load_config
    from cipoc.utils.utils import CipocConfig

    try:
        config = load_config(config_path)
    except KeyError as error:
        raise DemoSetupError(str(error)) from None  # loader reports the missing variable name only
    except FileNotFoundError:
        raise DemoSetupError(f"Configuration file not found: {config_path}") from None
    documents = config.documents().model_dump()
    for key, value in documents.items():
        if key.endswith("_path") and value is not None:
            path = Path(value)
            documents[key] = path if path.is_absolute() else (base_dir / path).resolve()
    for key in ("data_dictionary_path", "site_data_dictionary_path", "variable_groups_path"):
        path = documents.get(key)
        if path is None or not path.is_file():
            raise DemoSetupError(f"Missing configured resource: {key} ({path})")
    agents = {}
    for name in ("orchestrator", "note_scanner", "note_retriever", "extractor"):
        settings = config.agent_settings(name)
        # Explicit demo defaults; existing config values take precedence. Node
        # retry remains owned by LangGraph, avoiding hidden whole-run retries.
        settings.setdefault("timeout", settings.get("request_timeout", 120))
        settings.setdefault("max_retries", 0)
        agents[name] = settings
    config = CipocConfig({"llm": config.defaults, "agents": agents, "documents": documents})
    agent = OrchestratorAgent(config=config)
    for group in agent._target_variables:
        agent._scope_group(group, CaseFacts())
    return agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cipoc-demo", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Run once and save a replay plus canonical artifact.")
    record.add_argument("--out", type=Path, required=True, help="New JSONL recording path.")
    serve = commands.add_parser("serve", help="Serve a replay or an explicitly started live run.")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path)
    source.add_argument("--live", action="store_true")
    serve.add_argument("--record", type=Path, help="Optional new JSONL recording path.")
    serve.add_argument("--artifact", type=Path, help="Canonical result accompanying a replay.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)
    serve.add_argument("--description", default="CIPOC extraction")
    for sub in (record, serve):
        sub.add_argument("--notes", type=Path, help="JSON list of clinical notes.")
        sub.add_argument("--structured-data", type=Path, help="JSON object of known item values.")
        sub.add_argument("--config", type=Path, default=Path("config/config.yaml"))
        sub.add_argument("--resource-root", type=Path, default=Path.cwd(), help="Base for config resource paths (default: launch directory).")
        sub.add_argument("--output-dir", type=Path, default=Path("demo-runs"))
        sub.add_argument("--max-concurrency", type=int, help="Graph concurrency, at least 2. Model capacity is configured separately.")
        sub.add_argument("--no-llm-content-capture", action="store_true")
        sub.add_argument("--max-content-chars", type=int)
    args = parser.parse_args(argv)
    live = args.command == "record" or args.live
    if live and args.notes is None:
        parser.error("Live execution requires --notes.")
    if args.max_concurrency is not None and args.max_concurrency < 2:
        parser.error("--max-concurrency must be at least 2.")
    if args.max_content_chars is not None and args.max_content_chars < 0:
        parser.error("--max-content-chars must be nonnegative.")
    try:
        if live:
            notes = json.loads(args.notes.read_text(encoding="utf-8"))
            structured = json.loads(args.structured_data.read_text(encoding="utf-8")) if args.structured_data else None
            job = DemoRun(
                notes, structured_data=structured,
                agent_factory=lambda: build_agent(args.config.resolve(), args.resource_root.resolve()),
                output_dir=args.output_dir,
                record_path=args.out if args.command == "record" else args.record,
                max_concurrency=args.max_concurrency,
                capture_llm_content=not args.no_llm_content_capture,
                max_content_chars=args.max_content_chars,
            )
            job.prepare()
            if args.command == "record":
                if job.status == "initialization_failed":
                    print(job.error)
                    return 1
                job.execute(lambda event: None)
                print(f"{job.status}: {job.output_path}")
                for issue in job.issues:
                    print(issue)
                return 0 if job.status == "completed" and job.artifact_saved and not job.issues else 1
        else:
            job = None
        import uvicorn
        from .server import LiveDemoSession, build_app, load_replay_session
        session = LiveDemoSession(job, description=args.description) if job else load_replay_session(
            args.replay, description=args.description, artifact_path=args.artifact,
        )
        print(f"Open http://{args.host}:{args.port}/" + (" and click Start Run." if job else ""))
        uvicorn.run(build_app(session), host=args.host, port=args.port, log_level="warning")
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"Unable to open demo inputs ({type(error).__name__}). Check paths and JSON formats.\n")


if __name__ == "__main__":
    raise SystemExit(main())
