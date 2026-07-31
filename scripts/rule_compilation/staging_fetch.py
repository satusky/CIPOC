"""Fetch and read one SEER*RSA algorithm release ZIP.

stdlib only — ``urllib.request`` + ``zipfile`` + ``json``. One request replaces
the ~1,700 page loads scraping ``staging.seer.cancer.gov`` would need, and adds
no dependency to a DBR-18.2-pinned project. ``extract_rules/create_datadict.ipynb``
already pulls ``eod_public-3.3.zip`` the same way, so the precedent is in-repo.

The ZIP is read into memory by default. ``--cache-dir`` writes it once so
re-runs, and the airgapped case, work with no network at all.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .staging_index import StagingAlgorithm

USER_AGENT = "cipoc-rule-compilation (+https://github.com/RENCI/CIPOC)"
TIMEOUT_SECONDS = 300


@dataclass
class StagingRelease:
    """Parsed-on-demand view of one algorithm ZIP.

    Keeps the ``ZipFile`` rather than eagerly decoding ~1,200 members: a run
    filtered to one schema reads a handful of them. Decoded tables are memoized
    because they are heavily shared — a full run visits ~140 schemas whose ~40
    inputs each resolve to a few hundred distinct tables between them.
    """

    algorithm: StagingAlgorithm
    archive: zipfile.ZipFile
    _tables: dict[str, dict] = field(default_factory=dict, repr=False)

    def schema_ids(self) -> list[str]:
        return sorted(
            Path(name).stem
            for name in self.archive.namelist()
            if name.startswith("schemas/") and name.endswith(".json")
        )

    def schema(self, schema_id: str) -> dict:
        return json.loads(self.archive.read(f"schemas/{schema_id}.json"))

    def table(self, table_id: str) -> dict:
        if table_id not in self._tables:
            self._tables[table_id] = json.loads(self.archive.read(f"tables/{table_id}.json"))
        return self._tables[table_id]

    def tables_for_schema(self, schema: dict) -> dict[str, dict]:
        """Every table the unit builder may need for one schema, keyed by id.

        That is each input's code table plus the schema-selection table. Tables
        named only by ``mappings`` (the staging computation itself) are not
        coding value sets and are not read.
        """
        wanted = {inp["table"] for inp in schema.get("inputs", []) if inp.get("table")}
        selection = schema.get("schema_selection_table")
        if selection:
            wanted.add(selection)
        return {table_id: self.table(table_id) for table_id in sorted(wanted)}


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def fetch_algorithm(
    algorithm: StagingAlgorithm, *, cache_dir: Path | None = None
) -> StagingRelease:
    """Return the algorithm's release ZIP, downloading it unless it is cached.

    The declared version is verified against the data rather than trusted: a
    release tag whose asset name says 3.3 but whose schemas say otherwise means
    the pin in ``staging_index`` no longer describes what is being compiled, and
    every unit would carry a wrong manifest key. Fails loudly instead.
    """
    payload: bytes | None = None
    if cache_dir is not None:
        cached = Path(cache_dir) / algorithm.asset
        if cached.exists():
            payload = cached.read_bytes()
        else:
            payload = _download(algorithm.url)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(payload)
    if payload is None:
        payload = _download(algorithm.url)

    release = StagingRelease(algorithm, zipfile.ZipFile(io.BytesIO(payload)))
    verify_release(release)
    return release


def verify_release(release: StagingRelease) -> None:
    """Check the asset's declared algorithm/version against a schema it contains."""
    schema_ids = release.schema_ids()
    if not schema_ids:
        raise ValueError(f"{release.algorithm.asset} contains no schemas/*.json members.")
    schema = release.schema(schema_ids[0])
    algorithm = release.algorithm
    actual = (schema.get("algorithm"), schema.get("version"))
    expected = (algorithm.algorithm_id, algorithm.version)
    if actual != expected:
        raise ValueError(
            f"{algorithm.asset} at {algorithm.url} declares algorithm/version {actual}, "
            f"expected {expected}. The pinned release tag no longer matches "
            f"staging_index.ALGORITHMS[{algorithm.name!r}]."
        )
