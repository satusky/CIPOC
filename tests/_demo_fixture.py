"""Generate a current, offline rehearsal without endpoint credentials.

The committed legacy demo_trace.jsonl stays unchanged as a compatibility fixture.
This generator uses real orchestration and deterministic fake subagents; its
canonical artifact correctly reports no model invocations.
"""

import argparse
from pathlib import Path

from cipoc.demo.stream import DemoRun
from tests.fake_orchestrator import Outcome, Script, build_fake_orchestrator, load_notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("demo-runs"))
    args = parser.parse_args()
    job = DemoRun(
        [note.model_dump() for note in load_notes()],
        agent_factory=lambda: build_fake_orchestrator(Script(outcomes={674: Outcome(repairs=1)})),
        record_path=args.out, output_dir=args.output_dir,
    )
    job.execute(lambda event: None)
    print(job.status, job.output_path)
    if job.status != "completed" or job.issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
