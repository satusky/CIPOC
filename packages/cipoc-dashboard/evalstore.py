"""SQLite store for test-set evaluation runs, and the dashboard data bundle.

The store is the system of record; the dashboard never opens it. ``export_bundle``
flattens it into ``web/data.js``, a plain script that assigns one global, so the
dashboard works from ``file://`` with no server and no WASM SQLite build.

Stdlib only. Nothing here imports ``cipoc``.

    python evalstore.py init        data/eval.sqlite
    python evalstore.py load-truth  data/eval.sqlite ground_truth.csv [--names variable_groups.json]
    python evalstore.py load-truth  data/eval.sqlite ground_truth/    [--names variable_groups.json]
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
-- NULL until known. JSON ground truth carries no tissue or note count; the
-- patient's first recorded artifact fills them in.
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    tissue     TEXT,
    dx_year    INTEGER,
    n_notes    INTEGER
);
-- One row per ground-truth cell. A NULL value is a blank cell: the variable is
-- expected to stay uncoded for that patient. A missing row is untested and not
-- scored: the variable had no CSV column, or no key in the patient's JSON file.
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
DATE_OF_DIAGNOSIS = 390

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


def _record_variable(conn: sqlite3.Connection, item_id: int, meta: dict, order: int) -> None:
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


def _truth_text(value) -> str | None:
    """A JSON ground-truth value as stored: text, or None for a blank."""
    if value is None:
        return None
    # Pandas writes integer columns with blanks as floats; 2.0 means "2".
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip() or None


def _diagnosis_year(date: str | None) -> int | None:
    """Year of a YYYYMMDD or YYYY-MM-DD date; None if missing or unknown (9999)."""
    year = int(date[:4]) if date and date[:4].isdigit() else None
    return year if year is not None and 1900 <= year <= 2100 else None


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
            _record_variable(conn, item_id, names.get(item_id, {}), order)
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


def load_ground_truth_dir(
    conn: sqlite3.Connection,
    directory: str | Path,
    names: dict[int, dict] | None = None,
) -> None:
    """Load the Workbench's ground truth: one ``<patient_id>.json`` per patient.

    Each file is ``{item_id: value}``. A null or empty value is a blank cell. An
    item missing from a patient's file is untested for that patient, and keys
    that aren't item IDs, such as ``mrn``, are ignored. The diagnosis year comes
    from item 390; tissue and note count come from the first recorded artifact.
    """
    names = names or {}
    truth: dict[str, dict[int, str | None]] = {}
    for path in sorted(Path(directory).glob("*.json")):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path} is not a JSON object of {{item_id: value}}.")
        truth[path.stem] = {int(k): _truth_text(v) for k, v in raw.items() if k.strip().isdigit()}
    if not truth:
        raise ValueError(f"No .json files in {directory}.")
    for order, item_id in enumerate(sorted({i for values in truth.values() for i in values})):
        _record_variable(conn, item_id, names.get(item_id, {}), order)
    for patient_id, values in truth.items():
        conn.execute(
            "INSERT INTO patients (patient_id, dx_year) VALUES (?, ?)"
            " ON CONFLICT (patient_id) DO UPDATE SET dx_year = COALESCE(excluded.dx_year, dx_year)",
            (patient_id, _diagnosis_year(values.get(DATE_OF_DIAGNOSIS))),
        )
        # The file is the patient's whole reference, so an item removed from it
        # becomes untested rather than keeping its old value.
        conn.execute("DELETE FROM ground_truth WHERE patient_id = ?", (patient_id,))
        conn.executemany(
            "INSERT INTO ground_truth VALUES (?, ?, ?)",
            [(patient_id, item_id, value) for item_id, value in values.items()],
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
    """Store one orchestrator run artifact (the ``OrchestratorRunResult`` JSON).

    Also fills in patient details the ground truth left unknown. Tissue is the
    run's ``gross_primary_site``, which is unset when the run already knew the
    primary site (for example from structured data).
    """
    facts = artifact["case"].get("case_facts") or {}
    notes = (artifact.get("corpus") or {}).get("note_corpus") or {}
    conn.execute(
        "UPDATE patients SET tissue = COALESCE(tissue, ?), dx_year = COALESCE(dx_year, ?),"
        " n_notes = COALESCE(n_notes, ?) WHERE patient_id = ?",
        (facts.get("gross_primary_site"), _diagnosis_year(facts.get("date_of_diagnosis")), len(notes) or None, patient_id),
    )
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
        {"patient_id": r[0], "tissue": r[1] or "unknown", "year": r[2], "n_notes": r[3]}
        for r in conn.execute(
            "SELECT patient_id, tissue, dx_year, n_notes FROM patients ORDER BY patient_id"
        )
    ]
    v_index = {v["item_id"]: i for i, v in enumerate(variables)}
    p_index = {p["patient_id"]: i for i, p in enumerate(patients)}

    truth = [[None] * len(variables) for _ in patients]
    tested = set()
    for patient_id, item_id, value in conn.execute("SELECT * FROM ground_truth"):
        truth[p_index[patient_id]][v_index[item_id]] = value
        tested.add((p_index[patient_id], v_index[item_id]))
    untested = [[p, v] for p in range(len(patients)) for v in range(len(variables)) if (p, v) not in tested]

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
        "untested": untested,
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
    load.add_argument("path", help="wide CSV, or a directory of <patient_id>.json files")
    load.add_argument("--names", help="variable groups JSON, for names and grouping")
    export = commands.add_parser("export")
    export.add_argument("db")
    export.add_argument("out")
    args = parser.parse_args()

    conn = connect(args.db)
    if args.command == "load-truth":
        names = variable_names(args.names) if args.names else None
        if Path(args.path).is_dir():
            load_ground_truth_dir(conn, args.path, names)
        else:
            load_ground_truth(conn, args.path, names)
    elif args.command == "export":
        export_bundle(conn, args.out)
    conn.close()


if __name__ == "__main__":
    main()
