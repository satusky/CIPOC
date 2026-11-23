"""Serve the evaluation dashboard straight from run artifacts and ground-truth files.

    python dashboard.py serve                # http://127.0.0.1:8001
    python dashboard.py export web/data.js   # one static bundle, for opening web/index.html from file://

``serve`` rebuilds ``data.js`` whenever a page asks for it, so a refresh shows new
artifacts. Each file is parsed once and re-read only when its size or mtime
changes. Paths come from flags, then ``dashboard.json`` next to this file, then
``data/sweeps``, ``data/ground_truth`` and the repository's variable groups.

Stdlib only. Nothing here imports ``cipoc``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
CONFIG_PATH = HERE / "dashboard.json"
DEFAULT_PATHS = {
    "sweeps": HERE / "data" / "sweeps",
    "truth": HERE / "data" / "ground_truth",
    "names": HERE.parents[1] / "config" / "variable_groups.json",
}
SWEEP_META = "sweep.json"
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


# ---------------------------------------------------------------- reading files


def variable_names(variable_groups: dict) -> dict[int, dict]:
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

    for group in variable_groups["groups"]:
        walk(group)
    return found


def _truth_text(value) -> str | None:
    """A ground-truth value as the dashboard compares it: text, or None for a blank."""
    if value is None:
        return None
    # Pandas writes integer columns with blanks as floats; 2.0 means "2".
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip() or None


def _diagnosis_year(date: str | None) -> int | None:
    """Year of a YYYYMMDD or YYYY-MM-DD date; None if missing or unknown (9999)."""
    year = int(date[:4]) if isinstance(date, str) and date[:4].isdigit() else None
    return year if year is not None and 1900 <= year <= 2100 else None


def read_truth(raw: dict) -> dict[int, str | None]:
    """One patient's ``{item_id: value}``, the format the Workbench loads.

    A null or empty value is a blank cell. Keys that aren't item IDs, such as
    ``mrn``, are ignored.
    """
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object of {item_id: value}")
    return {int(k): _truth_text(v) for k, v in raw.items() if k.strip().isdigit()}


def config_digest(fingerprint: dict) -> str:
    """Short digest of a run's config fingerprint. Equal digests are replicates."""
    text = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:6]


def _from_state_dump(state: dict) -> dict:
    """Read a file from before versioned artifacts (schema 1.0) as a completed run.

    Those files are the orchestrator's final state at top level, with usage only
    on each model call and no run block, so they carry no date, duration or
    config fingerprint. The sweep's ``sweep.json`` supplies the date.
    """
    calls = [call for calls in (state.get("llm_exchanges") or {}).values() for call in calls]
    usage = [call.get("usage") or {} for call in calls]
    reported = [u for u in usage if u.get("input_tokens") is not None]
    return {
        "run": {"status": "completed"},
        "case": {"variable_results": state["variable_results"], "case_facts": state.get("case_facts")},
        "corpus": {
            "note_corpus": state.get("note_corpus"),
            "note_corpus_descriptors": state.get("note_corpus_descriptors"),
        },
        "observability": {"llm_usage_summary": {
            "input_tokens": sum(u["input_tokens"] for u in reported),
            "output_tokens": sum(u.get("output_tokens") or 0 for u in reported),
            "model_invocations": len(calls),
            "usage_reported_invocations": len(reported),
            "by_model": {call["model"]: {} for call in calls if call.get("model")},
        }},
    }


def summarize_artifact(artifact: dict) -> dict:
    """Keep what the dashboard needs from one ``OrchestratorRunResult`` or failure.

    Artifacts hold every note, so the server caches this summary, not the file.
    """
    if not isinstance(artifact, dict) or ("run" not in artifact and "variable_results" not in artifact):
        raise ValueError("not a CIPOC run artifact")
    if "run" not in artifact:
        artifact = _from_state_dump(artifact)
    run = artifact["run"]
    if run["status"] not in ("completed", "failed"):
        raise ValueError(f"unexpected run status {run['status']!r}")
    fingerprint = run.get("config_fingerprint") or {}
    usage = (artifact.get("observability") or {}).get("llm_usage_summary") or {}
    models = {
        settings["model"]
        for settings in (fingerprint.get("agent_llm_config") or {}).values()
        if isinstance(settings, dict) and settings.get("model")
    } or set(usage.get("by_model") or {})
    # A summary with no usage-reporting invocation carries zero-filled counts.
    reported = bool(usage.get("usage_reported_invocations"))
    case = artifact.get("case") or {}
    facts = case.get("case_facts") or {}
    corpus = artifact.get("corpus") or {}
    n_notes = (corpus.get("note_corpus_descriptors") or {}).get("note_count") or len(corpus.get("note_corpus") or {})
    return {
        "completed": run["status"] == "completed",
        "started_at": run.get("started_at"),
        "config_digest": config_digest(fingerprint) if fingerprint else None,
        "cipoc_version": fingerprint.get("cipoc_version"),
        "model": ", ".join(sorted(models)) or None,
        "results": {
            int(item_id): (result.get("value"), result.get("status"))
            for item_id, result in (case.get("variable_results") or {}).items()
        },
        "usage": [
            usage.get("input_tokens") if reported else None,
            usage.get("output_tokens") if reported else None,
            usage.get("model_invocations"),
            run.get("duration_seconds"),
        ],
        "tissue": facts.get("gross_primary_site"),
        "dx_year": _diagnosis_year(facts.get("date_of_diagnosis")),
        "n_notes": n_notes or None,
    }


def _signature(path: Path) -> tuple[int, int]:
    """(mtime, size), or (-1, -1) for a file deleted since the folder was listed."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (-1, -1)
    return (stat.st_mtime_ns, stat.st_size)


class FileCache:
    """Parsed files keyed by path, re-read only when size or mtime changes.

    A file that fails to parse is cached as None and warned about once, so a
    half-written artifact is skipped until it changes again.
    """

    def __init__(self) -> None:
        self._entries: dict[Path, tuple[tuple[int, int], object]] = {}

    def get(self, path: Path, parse):
        signature = _signature(path)
        cached = self._entries.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        try:
            value = parse(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(f"Skipping {path}: {error}", file=sys.stderr)
            value = None
        self._entries[path] = (signature, value)
        return value

    def forget_missing(self, seen: set[Path]) -> None:
        for path in self._entries.keys() - seen:
            del self._entries[path]


def _json_files(directory: Path) -> list[Path]:
    # Dot files are editor and macOS debris, never artifacts.
    return sorted(p for p in directory.glob("*.json") if p.is_file() and not p.name.startswith("."))


# ------------------------------------------------------------------- the bundle


def build_bundle(
    truth: dict[str, dict[int, str | None]],
    sweeps: list[tuple[str, dict, dict[str, dict]]],
    names: dict[int, dict] | None = None,
) -> dict:
    """Flatten ground truth and sweeps into the positional matrices the pages read.

    ``truth`` is ``{patient_id: {item_id: value}}``. ``sweeps`` is a list of
    ``(sweep_id, sweep.json contents, {patient_id: artifact summary})``. A sweep's
    date is ``started_at`` in its sweep.json, else its earliest artifact's.

    Patients are the ground-truth files; variables are every item they mention.
    An item missing from a patient's file is untested for that patient. The
    diagnosis year comes from item 390, then the artifacts; tissue and note count
    come from the earliest artifact that has them. Tissue is the run's
    ``gross_primary_site``, which a run leaves unset when it already knows the
    primary site.
    """
    names = names or {}
    item_ids = sorted({item for values in truth.values() for item in values})
    # Variable-groups order, then anything the groups file doesn't name.
    item_ids.sort(key=lambda item: (0, names[item]["sort_order"]) if item in names else (1, item))
    variables = [
        {
            "item_id": item,
            "name": names.get(item, {}).get("name", f"Item {item}"),
            "group_id": names.get(item, {}).get("group_id") or "ungrouped",
            "group_name": names.get(item, {}).get("group_name") or "Ungrouped",
        }
        for item in item_ids
    ]
    v_index = {item: v for v, item in enumerate(item_ids)}
    patient_ids = sorted(truth)
    p_index = {patient_id: p for p, patient_id in enumerate(patient_ids)}

    def started(sweep):
        if sweep[1].get("started_at"):
            return str(sweep[1]["started_at"])
        times = [s["started_at"] for s in sweep[2].values() if s["started_at"]]
        return min(times) if times else ""

    sweeps = sorted((s for s in sweeps if s[2]), key=lambda s: (started(s), s[0]))
    for sweep_id, meta, summaries in sweeps:
        if not started((sweep_id, meta, summaries)):
            print(f"Sweep {sweep_id} has no date, so it sorts first. Set started_at in its {SWEEP_META}.", file=sys.stderr)

    patients = []
    for patient_id in patient_ids:
        summaries = [sweep[2][patient_id] for sweep in sweeps if patient_id in sweep[2]]
        first = lambda key: next((s[key] for s in summaries if s[key] is not None), None)  # noqa: E731
        patients.append({
            "patient_id": patient_id,
            "tissue": first("tissue") or "unknown",
            "year": _diagnosis_year(truth[patient_id].get(DATE_OF_DIAGNOSIS)) or first("dx_year"),
            "n_notes": first("n_notes"),
        })

    truth_matrix = [[truth[pid].get(item) for item in item_ids] for pid in patient_ids]
    untested = [
        [p, v] for p, pid in enumerate(patient_ids) for v, item in enumerate(item_ids) if item not in truth[pid]
    ]

    runs, unmatched = [], set()
    for sweep_id, meta, summaries in sweeps:
        pred: list = [None] * len(patient_ids)
        status: list = [None] * len(patient_ids)
        usage: list = [None] * len(patient_ids)
        for patient_id, summary in summaries.items():
            p = p_index.get(patient_id)
            if p is None:
                unmatched.add(patient_id)
                continue
            usage[p] = summary["usage"]
            if not summary["completed"]:
                continue
            pred[p] = [None] * len(item_ids)
            codes = ["-"] * len(item_ids)
            for item, (value, variable_status) in summary["results"].items():
                v = v_index.get(item)
                if v is not None:
                    pred[p][v] = value
                    codes[v] = STATUS_CODES.get(variable_status, "?")
            status[p] = "".join(codes)
        reference = next((s for s in summaries.values() if s["completed"]), next(iter(summaries.values())))
        runs.append({
            "run_id": sweep_id,
            "started_at": started((sweep_id, meta, summaries)),
            "label": meta.get("label"),
            "model": meta.get("model") or reference["model"],
            "cipoc_version": reference["cipoc_version"],
            "config_digest": reference["config_digest"],
            "notes": meta.get("notes"),
            "pred": pred,
            "status": status,
            "usage": usage,
        })
    if unmatched:
        example = ", ".join(sorted(unmatched)[:3])
        print(f"Ignoring artifacts for {len(unmatched)} patient(s) with no ground-truth file: {example}", file=sys.stderr)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status_codes": {code: name for name, code in STATUS_CODES.items()},
        "variables": variables,
        "patients": patients,
        "truth": truth_matrix,
        "untested": untested,
        "runs": runs,
    }


class Bundler:
    """Rebuilds the bundle from the folders, reusing every unchanged file."""

    def __init__(self, sweeps_dir: Path, truth_dir: Path, names_path: Path | None) -> None:
        self.sweeps_dir, self.truth_dir, self.names_path = sweeps_dir, truth_dir, names_path
        self._cache = FileCache()
        self._lock = threading.Lock()
        self._signature = None
        self._script = b""

    def script(self) -> bytes:
        """``data.js``: the bundle as one global, rebuilt only when a file changed."""
        with self._lock:
            truth_files = _json_files(self.truth_dir)
            sweep_dirs = sorted(d for d in self.sweeps_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
            sweep_files = {d: _json_files(d) for d in sweep_dirs}
            files = [*truth_files, *(p for paths in sweep_files.values() for p in paths)]
            if self.names_path is not None:
                files.append(self.names_path)
            signature = [(p, *_signature(p)) for p in files]
            if signature == self._signature:
                return self._script

            truth = {}
            for path in truth_files:
                values = self._cache.get(path, read_truth)
                if values is not None:
                    truth[path.stem] = values
            sweeps = []
            for directory, paths in sweep_files.items():
                meta, summaries = {}, {}
                for path in paths:
                    if path.name == SWEEP_META:
                        meta = self._cache.get(path, lambda raw: raw if isinstance(raw, dict) else {}) or {}
                    else:
                        summary = self._cache.get(path, summarize_artifact)
                        if summary is not None:
                            summaries[path.stem] = summary
                sweeps.append((directory.name, meta, summaries))
            names = self._cache.get(self.names_path, variable_names) if self.names_path is not None else None
            self._cache.forget_missing(set(files))

            payload = json.dumps(build_bundle(truth, sweeps, names), separators=(",", ":"))
            self._script = f"window.CIPOC_EVAL = {payload};\n".encode()
            self._signature = signature
            return self._script


# ------------------------------------------------------------------ command line


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    """Flags, then ``dashboard.json``, then the defaults. Exits on a missing folder."""
    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    paths = {}
    for key in ("sweeps", "truth", "names"):
        if getattr(args, key) is not None:
            paths[key] = getattr(args, key)
        elif key in config:
            # Relative paths in the config file are relative to the file.
            paths[key] = (CONFIG_PATH.parent / Path(config[key]).expanduser())
        else:
            paths[key] = DEFAULT_PATHS[key]
    for key in ("sweeps", "truth"):
        if not paths[key].is_dir():
            sys.exit(f"{key}: {paths[key]} is not a directory. Set it with --{key} or in {CONFIG_PATH.name}.")
    names = paths["names"]
    if not names.is_file():
        if names != DEFAULT_PATHS["names"]:
            sys.exit(f"names: {names} does not exist.")
        names = None
    return paths["sweeps"], paths["truth"], names


def make_handler(bundler: Bundler):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlsplit(self.path).path != "/data.js":
                return super().do_GET()
            body = bundler.script()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def guess_type(self, path) -> str:
            # Say so explicitly, so no browser falls back to a local code page.
            kind = super().guess_type(path)
            return f"{kind}; charset=utf-8" if kind.startswith("text/") or kind.endswith("javascript") else kind

        def log_message(self, format, *args) -> None:
            pass

    return partial(Handler, directory=str(WEB_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    paths = argparse.ArgumentParser(add_help=False)
    paths.add_argument("--sweeps", type=Path, help="folder with one subfolder of artifacts per sweep")
    paths.add_argument("--truth", type=Path, help="folder of <patient_id>.json ground-truth files")
    paths.add_argument("--names", type=Path, help="variable groups JSON, for variable names and groups")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", parents=[paths], help="serve the dashboard, rebuilt on every page load")
    # Localhost by default: the bundle holds patient IDs and extracted values.
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)
    export = commands.add_parser("export", parents=[paths], help="write data.js once")
    export.add_argument("out", type=Path)
    args = parser.parse_args()

    bundler = Bundler(*resolve_paths(args))
    if args.command == "export":
        args.out.write_bytes(bundler.script())
        return
    print(f"Sweeps: {bundler.sweeps_dir}\nTruth:  {bundler.truth_dir}\nNames:  {bundler.names_path or 'none'}", flush=True)
    bundler.script()  # Parse everything once now, not on the first page load.
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bundler))
    print(f"\nOpen http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
