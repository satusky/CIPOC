"""Deterministic markdown segmentation for the rule-compilation pipeline.

Pure code, no LLM. Splits a source manual's markdown on its heading structure
into candidate ``Section`` objects carrying a heading trail and a line range.
These provide the provenance anchors (``section_path`` and ``anchor``) that the
downstream LLM tagging pass attaches to every ``RuleUnit`` and that the
validation pass fuzzy-matches back against the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass(frozen=True)
class Section:
    """One heading-delimited candidate section of a source manual.

    ``start_line``/``end_line`` are 0-indexed and inclusive, spanning the
    heading line through the line before the next section's heading.
    """
    heading: str
    level: int
    section_path: tuple[str, ...]
    start_line: int
    end_line: int
    body: str
    anchor: str = field(compare=False, default="")

    @property
    def text(self) -> str:
        """Heading plus body, as it reads in the source (for the fidelity check)."""
        return f"{self.heading}\n{self.body}".strip()


def slugify(text: str, max_words: int = 8) -> str:
    """A stable, human-readable anchor slug from heading text."""
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return "-".join(words[:max_words])


def _clean_heading(raw: str) -> str:
    """Strip trailing markdown emphasis/punctuation noise from a heading."""
    return raw.strip().strip("*").strip()


def segment_markdown(text: str, *, max_heading_level: int = 3) -> list[Section]:
    """Split markdown into sections on headings up to ``max_heading_level``.

    Headings deeper than ``max_heading_level`` are folded into their parent
    section's body rather than starting new sections — this keeps deeply nested
    table-row pseudo-headings (the Solid Tumor Rules markdown has hundreds of
    ``#####`` fragments) from exploding into noise while preserving them as
    retrievable text under their real parent heading.

    Returns sections in document order. Content before the first heading is
    dropped (front matter with no heading anchor).
    """
    lines = text.splitlines()

    # Collect the headings that will actually delimit sections.
    boundaries: list[tuple[int, int, str]] = []  # (line_index, level, heading text)
    for i, line in enumerate(lines):
        match = _HEADING.match(line)
        if match and len(match.group(1)) <= max_heading_level:
            boundaries.append((i, len(match.group(1)), _clean_heading(match.group(2))))

    sections: list[Section] = []
    trail: list[tuple[int, str]] = []  # (level, heading) ancestor stack
    for index, (line_no, level, heading) in enumerate(boundaries):
        end_line = (boundaries[index + 1][0] - 1) if index + 1 < len(boundaries) else len(lines) - 1

        while trail and trail[-1][0] >= level:
            trail.pop()
        trail.append((level, heading))
        section_path = tuple(h for _, h in trail)

        body = "\n".join(lines[line_no + 1 : end_line + 1]).strip()
        sections.append(
            Section(
                heading=heading,
                level=level,
                section_path=section_path,
                start_line=line_no,
                end_line=end_line,
                body=body,
                anchor=f"L{line_no + 1}-L{end_line + 1}:{slugify(heading)}",
            )
        )
    return sections


def select_subtree(
    sections: list[Section],
    root_heading_contains: str,
    *,
    boundary_heading_contains: str | None = None,
) -> list[Section]:
    """Return the sections of the first subtree whose root heading matches.

    Matching is case-insensitive substring on the heading. The subtree runs from
    the matching section up to (but excluding) its end boundary:

    - By default the boundary is the next section at or above the root's level.
    - When ``boundary_heading_contains`` is given, the boundary is instead the
      next section whose heading contains that marker. Use this when the source
      markdown has spurious same-level headings inside the subtree (the Solid
      Tumor Rules markdown wraps site names onto false ``#`` lines like
      ``# Breast NOS C509.``); the true site-group roots all carry
      'Equivalent Terms and Definitions', so passing that marker cleanly bounds
      one site group regardless of interior heading noise.
    """
    needle = root_heading_contains.casefold()
    start = next(
        (i for i, s in enumerate(sections) if needle in s.heading.casefold()),
        None,
    )
    if start is None:
        return []

    end = len(sections)
    if boundary_heading_contains is not None:
        marker = boundary_heading_contains.casefold()
        for i in range(start + 1, len(sections)):
            if marker in sections[i].heading.casefold():
                end = i
                break
    else:
        root_level = sections[start].level
        for i in range(start + 1, len(sections)):
            if sections[i].level <= root_level:
                end = i
                break
    return sections[start:end]
