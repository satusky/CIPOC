"""SQLite store for test-set evaluation runs, and the dashboard data bundle.

The store is the system of record; the dashboard never opens it. ``export_bundle``
flattens it into ``web/data.js``, a plain script that assigns one global, so the
dashboard works from ``file://`` with no server and no WASM SQLite build.

Stdlib only. Nothing here imports ``cipoc``.

    python evalstore.py init        data/eval.sqlite
    python evalstore.py load-truth  data/eval.sqlite ground_truth.csv [--names variable_groups.json]
    python evalstore.py export      data/eval.sqlite web/data.js
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS variables (
    item_id    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    group_id   TEXT,
    group_name TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    tissue     TEXT NOT NULL,
    dx_year    INTEGER NOT NULL,
    n_notes    INTEGER NOT NULL
);
-- One row per ground-truth cell. A NULL value is a blank cell: the variable is
-- expected to stay uncoded for that patient. A variable with no column in the
-- ground-truth file has no rows at all and is not scored.
CREATE TABLE IF NOT EXISTS ground_truth (
    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    item_id    INTEGER NOT NULL REFERENCES variables(item_id),
    value      TEXT,
    PRIMARY KEY (patient_id, item_id)
) WITHOUT ROWID;
-- A run is one sweep of the orchestrator over the test set. The orchestrator's
-- own run_id is per case, so it lives on run_patients as case_run_id.
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    label         TEXT,
    model         TEXT,
    cipoc_version TEXT,
    config_digest TEXT,
    notes         TEXT
);
-- Usage is NULL when the provider did not report it, never zero.
CREATE TABLE IF NOT EXISTS run_patients (
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    patient_id        TEXT NOT NULL REFERENCES patients(patient_id),
    case_run_id       TEXT,
    status            TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    model_invocations INTEGER,
    duration_seconds  REAL,
    PRIMARY KEY (run_id, patient_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS predictions (
    run_id     TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    item_id    INTEGER NOT NULL,
    value      TEXT,
    status     TEXT,
    PRIMARY KEY (run_id, patient_id, item_id),
    FOREIGN KEY (run_id, patient_id) REFERENCES run_patients(run_id, patient_id)
) WITHOUT ROWID;
"""

PATIENT_COLUMNS = ("patient_id", "tissue", "year", "n_notes")

# One character per variable in the bundle's status strings. "-" means no
# prediction was recorded (the run did not plan the variable) and "?" a
# prediction with an unrecognized status. The dashboard doesn't score "-" or "s".
STATUS_CODES = {
    "extracted": "e",
    "structured_data": "s",
    "not_found": "n",
    "not_applicable": "a",
    "error": "x",
    "blocked": "b",
}


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def variable_names(variable_groups_path: str | Path) -> dict[int, dict]:
    """Flatten a CIPOC variable groups file to ``{item_id: name/group/order}``."""
    found: dict[int, dict] = {}

    def walk(group: dict) -> None:
        for variable in group.get("variables", []):
            found[variable["item_id"]] = {
                "name": variable["name"],
                "group_id": group["group_id"],
                "group_name": group["name"],
                "sort_order": len(found),
            }
        for subgroup in group.get("subgroups", []):
            walk(subgroup)

    for group in json.loads(Path(variable_groups_path).read_text())["groups"]:
        walk(group)
    return found


def load_ground_truth(
    conn: sqlite3.Connection,
    csv_path: str | Path,
    names: dict[int, dict] | None = None,
) -> None:
    """Load the wide ground-truth file: patient columns, then one per NAACCR ID."""
    names = names or {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in PATIENT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Ground truth is missing column(s): {', '.join(missing)}")
        item_columns = [c for c in reader.fieldnames if c not in PATIENT_COLUMNS]
        for order, column in enumerate(item_columns):
            item_id = int(column)
            meta = names.get(item_id, {})
            conn.execute(
                "INSERT OR REPLACE INTO variables VALUES (?, ?, ?, ?, ?)",
                (
                    item_id,
                    meta.get("name", f"Item {item_id}"),
                    meta.get("group_id"),
                    meta.get("group_name"),
                    meta.get("sort_order", order),
                ),
            )
        for row in reader:
            conn.execute(
                "INSERT OR REPLACE INTO patients VALUES (?, ?, ?, ?)",
                (row["patient_id"], row["tissue"], int(row["year"]), int(row["n_notes"])),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO ground_truth VALUES (?, ?, ?)",
                [
                    (row["patient_id"], int(c), (row[c] or "").strip() or None)
                    for c in item_columns
                ],
            )
    conn.commit()


def record_run(conn: sqlite3.Connection, run_id: str, started_at: str, **meta) -> None:
    """Register a sweep. ``meta``: label, model, cipoc_version, config_digest, notes."""
    conn.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            started_at,
            meta.get("label"),
            meta.get("model"),
            meta.get("cipoc_version"),
            meta.get("config_digest"),
            meta.get("notes"),
        ),
    )


def record_patient(
    conn: sqlite3.Connection,
    run_id: str,
    patient_id: str,
    results: dict[int, tuple[str | None, str | None]],
    *,
    status: str = "completed",
    case_run_id: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    model_invocations: int | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Store one patient's outcome for a sweep. ``results``: item_id -> (value, status)."""
    conn.execute(
        "INSERT OR REPLACE INTO run_patients VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            patient_id,
            case_run_id,
            status,
            input_tokens,
            output_tokens,
            model_invocations,
            duration_seconds,
        ),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?)",
        [
            (run_id, patient_id, item_id, value, variable_status)
            for item_id, (value, variable_status) in results.items()
        ],
    )


def record_case_artifact(
    conn: sqlite3.Connection, run_id: str, patient_id: str, artifact: dict
) -> None:
    """Store one orchestrator run artifact (the ``OrchestratorRunResult`` JSON)."""
    results = {
        int(item_id): (result.get("value"), result.get("status"))
        for item_id, result in artifact["case"]["variable_results"].items()
    }
    usage = (artifact.get("observability") or {}).get("llm_usage_summary") or {}
    # A summary with no usage-reporting invocation carries zero-filled counts.
    reported = bool(usage.get("usage_reported_invocations"))
    record_patient(
        conn,
        run_id,
        patient_id,
        results,
        case_run_id=artifact["run"]["run_id"],
        input_tokens=usage.get("input_tokens") if reported else None,
        output_tokens=usage.get("output_tokens") if reported else None,
        model_invocations=usage.get("model_invocations"),
        duration_seconds=artifact["run"].get("duration_seconds"),
    )


def config_digest(artifact: dict) -> str:
    """Short digest of an artifact's config fingerprint, for ``record_run``.

    Sweeps that share a digest are treated as replicates by the dashboard.
    """
    fingerprint = json.dumps(artifact["run"]["config_fingerprint"], sort_keys=True, default=str)
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:6]


def build_bundle(conn: sqlite3.Connection) -> dict:
    """Flatten the store into the positional matrices the dashboard reads."""
    variables = [
        {"item_id": r[0], "name": r[1], "group_id": r[2] or "ungrouped", "group_name": r[3] or "Ungrouped"}
        for r in conn.execute(
            "SELECT item_id, name, group_id, group_name FROM variables ORDER BY sort_order, item_id"
        )
    ]
    patients = [
        {"patient_id": r[0], "tissue": r[1], "year": r[2], "n_notes": r[3]}
        for r in conn.execute(
            "SELECT patient_id, tissue, dx_year, n_notes FROM patients ORDER BY patient_id"
        )
    ]
    v_index = {v["item_id"]: i for i, v in enumerate(variables)}
    p_index = {p["patient_id"]: i for i, p in enumerate(patients)}

    truth = [[None] * len(variables) for _ in patients]
    for patient_id, item_id, value in conn.execute("SELECT * FROM ground_truth"):
        truth[p_index[patient_id]][v_index[item_id]] = value

    runs = []
    for run_id, started_at, label, model, version, digest, notes in conn.execute(
        "SELECT * FROM runs ORDER BY started_at, run_id"
    ).fetchall():
        pred: list = [None] * len(patients)
        status: list = [None] * len(patients)
        usage: list = [None] * len(patients)
        for patient_id, case_status, tokens_in, tokens_out, calls, seconds in conn.execute(
            "SELECT patient_id, status, input_tokens, output_tokens, model_invocations,"
            " duration_seconds FROM run_patients WHERE run_id = ?",
            (run_id,),
        ):
            p = p_index[patient_id]
            usage[p] = [tokens_in, tokens_out, calls, seconds]
            if case_status == "completed":
                pred[p] = [None] * len(variables)
                status[p] = ["-"] * len(variables)
        for patient_id, item_id, value, variable_status in conn.execute(
            "SELECT patient_id, item_id, value, status FROM predictions WHERE run_id = ?",
            (run_id,),
        ):
            p, v = p_index[patient_id], v_index.get(item_id)
            if v is None or pred[p] is None:
                continue
            pred[p][v] = value
            status[p][v] = STATUS_CODES.get(variable_status, "?")
        runs.append(
            {
                "run_id": run_id,
                "started_at": started_at,
                "label": label,
                "model": model,
                "cipoc_version": version,
                "config_digest": digest,
                "notes": notes,
                "pred": pred,
                "status": [None if s is None else "".join(s) for s in status],
                "usage": usage,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status_codes": {code: name for name, code in STATUS_CODES.items()},
        "variables": variables,
        "patients": patients,
        "truth": truth,
        "runs": runs,
    }


def export_bundle(conn: sqlite3.Connection, out_path: str | Path) -> None:
    payload = json.dumps(build_bundle(conn), separators=(",", ":"))
    Path(out_path).write_text(f"window.CIPOC_EVAL = {payload};\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init").add_argument("db")
    load = commands.add_parser("load-truth")
    load.add_argument("db")
    load.add_argument("csv")
    load.add_argument("--names", help="variable groups JSON, for names and grouping")
    export = commands.add_parser("export")
    export.add_argument("db")
    export.add_argument("out")
    args = parser.parse_args()

    conn = connect(args.db)
    if args.command == "load-truth":
        load_ground_truth(conn, args.csv, variable_names(args.names) if args.names else None)
    elif args.command == "export":
        export_bundle(conn, args.out)
    conn.close()


if __name__ == "__main__":
    main()
