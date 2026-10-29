"""Rules that decide what must not be auto-priced and what a human should look at.

Policyholder corrections arrive from the live agent as assertions ("that's a first edition",
"skip that shelf, those aren't mine", "it's a print"). They can raise a flag or exclude a line;
they can never create a price.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

APPRAISAL_TEXT = re.compile(
    r"\b(first edition|1st edition|first printing|1st printing|signed|autographed|inscribed|"
    r"limited edition|numbered|antiquarian|rare|leather ?bound|folio society)\b", re.I)
ANTIQUARIAN_BEFORE = 1950

FLAG_KINDS = {"first_edition", "signed", "rare", "original"}
KINDS = FLAG_KINDS | {"skip_shelf", "not_mine", "is_print"}


@dataclass
class Assertion:
    kind: str
    target: str = ""     # a book or item id, if the agent could tie it to one
    shelf: str = ""      # or a whole shelf
    said: str = ""       # the policyholder's words, for the adjuster
    at_s: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown assertion kind {self.kind!r}")
        if not (self.target or self.shelf):
            raise ValueError("an assertion must point at a line or a shelf")

    def applies_to(self, line_id: str, shelf: str = "") -> bool:
        return (self.target and self.target == line_id) or (self.shelf and shelf and self.shelf == shelf)


def appraisal_reasons(*, spine_text: str = "", year: int | None = None, looks_antique: bool = False,
                      replacement: float | None = None, used: float | None = None, threshold: float = 150.0,
                      said: list[Assertion] = ()) -> list[str]:
    """year must be the year of the copy's edition, known only when the spine named the edition.
    A work's first-publication year says nothing about the copy: a 2005 paperback of an 1871
    novel is not antiquarian."""
    reasons = []
    hit = APPRAISAL_TEXT.search(spine_text or "")
    if hit:
        reasons.append(f"spine reads '{hit.group(0)}'")
    if year is not None and 0 < year < ANTIQUARIAN_BEFORE:
        reasons.append(f"edition published {year}; may be antiquarian")
    if looks_antique:
        reasons.append("spine looks antiquarian (binding or wear seen on camera)")
    top = max((v for v in (replacement, used) if v is not None), default=None)
    if top is not None and top >= threshold:
        reasons.append(f"market price {top:.2f} at or above appraisal threshold {threshold:.0f}")
    for a in said:
        if a.kind in FLAG_KINDS:
            reasons.append(f"policyholder said: {a.said or a.kind.replace('_', ' ')}")
    return reasons


def excluded(line_id: str, shelf: str, said: list[Assertion]) -> Assertion | None:
    for a in said:
        if a.kind in ("skip_shelf", "not_mine") and a.applies_to(line_id, shelf):
            return a
    return None


def says_print(line_id: str, said: list[Assertion]) -> bool:
    """The latest statement wins: 'it's a print' then 'actually it's the original' -> original."""
    verdict = False
    for a in sorted((a for a in said if a.applies_to(line_id)), key=lambda a: a.at_s or 0):
        if a.kind == "is_print":
            verdict = True
        elif a.kind == "original":
            verdict = False
    return verdict
