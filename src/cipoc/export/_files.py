"""Same-filesystem staging for OMOP flat files."""

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator


@contextmanager
def staged_output_paths(
    output_directory: Path, filenames: tuple[str, ...]
) -> Iterator[tuple[Path, ...]]:
    """Publish only after the caller has written and closed every staged file.

    Validation and staging failures leave existing destinations untouched. Each
    os.replace is individual: a later replacement failure or crash can leave a
    mixed bundle. This is not a multi-file transaction or a durability guarantee.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".omop-", dir=output_directory) as directory:
        paths = tuple(Path(directory) / name for name in filenames)
        yield paths
        for path, name in zip(paths, filenames):
            os.replace(path, output_directory / name)
