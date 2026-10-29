"""Spine text -> a specific catalogue record, or an honest "unidentified".

The read text is searched in Open Library (and Google Books when a key is set). Each
candidate is scored by how well its title and author match what was actually printed on
the spine. Only a candidate that clears the threshold AND clearly beats any different work
is accepted. Confident wrong answers cost more than blanks, so ties go to "unidentified".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .fetch import FetchError, GetJson
from .vision import SpineText

OPEN_LIBRARY = "https://openlibrary.org/search.json"
GOOGLE_BOOKS = "https://www.googleapis.com/books/v1/volumes"
DEFAULT_THRESHOLD = 0.80
MIN_MARGIN = 0.08  # best candidate must beat the best *different* work by this much


@dataclass
class Candidate:
    title: str
    authors: list[str]
    year: int | None          # year of this record: first publication of the work (Open Library)
    isbns: list[str]          # or of one edition (Google Books); never the age of the copy on the shelf
    publishers: list[str]
    source: str
    url: str
    work_key: str = ""


@dataclass
class Resolution:
    status: str  # identified | unidentified
    confidence: float
    candidate: Candidate | None = None
    edition: str = ""
    isbn: str = ""
    reason: str = ""
    runner_up: Candidate | None = None
    sources: list[str] = field(default_factory=list)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"['`]", "", s)  # "DONT" on a spine must match "Don't" in the catalogue
    s = re.sub(r"[^a-z0-9 ]+", " ", s.casefold())
    return " ".join(s.split())


def main_title(title: str) -> str:
    """Title without subtitle or catalogue notes: "X: a novel", "X (Book 1)", "X [Nepali language]"."""
    return re.split(r"[:(\[]", title or "", maxsplit=1)[0].strip().rstrip(".").strip()


def isbn13(isbns: list[str]) -> str:
    for i in isbns:
        digits = re.sub(r"[^0-9Xx]", "", i)
        if len(digits) == 13:
            return digits
    for i in isbns:
        digits = re.sub(r"[^0-9Xx]", "", i)
        if len(digits) == 10:
            return digits
    return ""


STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "on", "for"}


def significant_words(title: str) -> list[str]:
    return [w for w in norm(main_title(title)).split() if w not in STOPWORDS]


def query_text(read: SpineText, with_author: bool = True) -> str:
    parts = [read.title or read.text, read.author if with_author else ""]
    return " ".join(p for p in parts if p)[:200]


def records(data, key: str, limit: int = 8) -> list[dict]:
    """The list under `key`, keeping only dict entries. Catalogue payloads are not trusted."""
    rows = data.get(key) if isinstance(data, dict) else None
    return [r for r in rows[:limit] if isinstance(r, dict)] if isinstance(rows, list) else []


def open_library(read: SpineText, get_json: GetJson) -> list[Candidate]:
    fields = "key,title,author_name,first_publish_year,isbn,publisher"
    docs = records(get_json(OPEN_LIBRARY, {"q": query_text(read), "limit": 8, "fields": fields}).data, "docs")
    if not docs and read.author and read.title:
        # A half-read author ("NATASHA IONS") can make the search return nothing; the
        # title alone usually still finds the work, and the author check happens in score().
        docs = records(get_json(OPEN_LIBRARY, {"q": query_text(read, False), "limit": 8, "fields": fields}).data, "docs")
    return [Candidate(d["title"], d.get("author_name", [])[:3], d.get("first_publish_year"),
                      (d.get("isbn") or [])[:20], (d.get("publisher") or [])[:5],
                      "Open Library", f"https://openlibrary.org{d.get('key', '')}", d.get("key", ""))
            for d in docs if d.get("title")]


def google_books(read: SpineText, get_json: GetJson, key: str) -> list[Candidate]:
    out = []
    for q in dict.fromkeys([query_text(read), query_text(read, False)]):
        for it in records(get_json(GOOGLE_BOOKS, {"q": q, "maxResults": 8, "key": key}).data, "items"):
            v = it.get("volumeInfo") if isinstance(it.get("volumeInfo"), dict) else {}
            if not v.get("title"):
                continue
            year = (v.get("publishedDate") or "")[:4]
            ids = [x.get("identifier", "") for x in v.get("industryIdentifiers", []) if isinstance(x, dict)]
            out.append(Candidate(v["title"], v.get("authors", [])[:3], int(year) if year.isdigit() else None,
                                 ids, [v["publisher"]] if v.get("publisher") else [], "Google Books",
                                 v.get("canonicalVolumeLink") or v.get("infoLink", ""), it.get("id", "")))
        if out:
            break
    return out


def search(read: SpineText, get_json: GetJson, google_key: str = "") -> tuple[list[Candidate], list[str]]:
    """Candidates from every catalogue we can reach. A catalogue being down is a note, not a crash."""
    found, notes = [], []
    try:
        found += open_library(read, get_json)
    except FetchError as exc:
        notes.append(f"Open Library unavailable: {exc}")
    if google_key:
        try:
            found += google_books(read, get_json, google_key)
        except FetchError as exc:
            notes.append(f"Google Books unavailable: {exc}")
    return found, notes


def identify(read: SpineText, get_json: GetJson, google_key: str = "",
             threshold: float = DEFAULT_THRESHOLD) -> tuple[Resolution, list[str]]:
    """Cheapest path first: one catalogue, and the second only if the first can't settle it.
    With a Books key, Google Books goes first (keyed quota); Open Library is free but rate
    limited and has blocked bursts. Without a key, Open Library alone."""
    if not read.readable:
        return resolve(read, [], threshold), []
    notes, found = [], []
    sources = ([("Google Books", lambda: google_books(read, get_json, google_key))] if google_key else []) + \
              [("Open Library", lambda: open_library(read, get_json))]
    res = resolve(read, [], threshold)
    for name, fetch in sources:
        try:
            found += fetch()
        except FetchError as exc:
            notes.append(f"{name} unavailable: {exc}")
            continue
        res = resolve(read, found, threshold)
        if res.status == "identified":
            break
    return res, notes


def score(read: SpineText, c: Candidate) -> float:
    """0..1. Title evidence dominates; author agreement confirms; a clear author mismatch vetoes."""
    spine_title = norm(read.title or read.text)
    cand = norm(main_title(c.title))
    full = norm(c.title)
    if not spine_title or not cand:
        return 0.0
    # Without a separate title, the spine text also holds author and publisher words,
    # so token-set matching is needed. With a clean title, plain ratio is stricter.
    if read.title:
        t = max(fuzz.ratio(spine_title, cand), fuzz.ratio(spine_title, full)) / 100
        # Spines get cut off at the frame edge: "THESE WITCHES DONT" is the start of the title,
        # not a different book. Only for reads long enough that a prefix is distinctive.
        if len(spine_title) >= 12 and (cand.startswith(spine_title) or full.startswith(spine_title)):
            t = max(t, 0.95)
    else:
        t = fuzz.token_set_ratio(spine_title, cand) / 100
        if len(cand) < 6:
            t *= 0.85  # "It" or "Emma" inside a long spine string is weak evidence
    a = author_evidence(read, c)
    if a is None:
        return t * (0.9 if read.author else 0.92)  # no author to confirm with
    if a < 0.5:
        return t * 0.5  # same title, different author: a different book
    return 0.75 * t + 0.25 * a


def name_keys(name: str) -> list[str]:
    words = [w for w in name.split() if len(w) > 1]
    if not words:
        return []
    keys = [words[-1]]
    if len(words) >= 3:
        keys.append(max(words, key=len))
    return keys


def on_spine(key: str, spine: str) -> bool:
    return any(fuzz.ratio(t, key) >= 80 or (len(t) >= 4 and fuzz.partial_ratio(t, key) >= 75) for t in spine.split())


def author_match(spine: str, record: str) -> float:
    """Spine authors are often partial ("NICH SPARKS", "SHADID"); partial matching accepts those."""
    a, b = norm(spine), norm(record)
    if not a or not b:
        return 0.0
    # A shared first name is not agreement ("Jane Austen" vs "Jane Ellison"): the record's surname
    # must be on the spine, allowing for truncation and OCR slips ("IONS" read off a half-covered
    # "SOLOMONS"). Catalogues also write long names surname-first ("Prabhupada A. C. Bhaktivedanta
    # Swami"), so for names of 3+ words the most distinctive word counts as well.
    if not any(on_spine(n, a) for n in name_keys(b)):
        return 0.3
    score = fuzz.token_set_ratio(a, b)
    partial = fuzz.partial_ratio(a, b) if len(a) >= 5 else 0
    return max(score, partial if partial >= 85 else 0) / 100


def author_evidence(read: SpineText, c: Candidate) -> float | None:
    """How well the spine's author agrees with the record: from the author field if the reader
    filled it, else from the record's author surname appearing anywhere in the spine text.
    None means the spine shows no author at all."""
    if read.author:
        return max((author_match(read.author, x) for x in c.authors), default=None)
    words = set(norm(read.text).split())
    for name in c.authors:
        if any(len(k) >= 3 and k in words for k in name_keys(norm(name))):
            return 1.0
    return None


def title_coverage(read: SpineText, c: Candidate) -> int:
    """Words of the record's full title found on the spine, minus words that aren't."""
    on_spine = set(norm(f"{read.text} {read.title}").split())
    words = set(norm(c.title).split()) - STOPWORDS
    return len(words & on_spine) - len(words - on_spine)


def same_work(a: Candidate, b: Candidate) -> bool:
    """Catalogues hold many records per work (editions, translations, messy author lists).
    Same main title and any author in common is the same work, not a rival."""
    if fuzz.ratio(norm(main_title(a.title)), norm(main_title(b.title))) < 90:
        return False
    if not a.authors or not b.authors:
        return True
    # Same person written differently ("A. C. Bhaktivedanta Swami Prabhupada" / "Prabhupada A. C.
    # Bhaktivedanta Swami"), or one record also lists a narrator: a surname-like word of one appears in
    # the other. Short surnames count ("Amy Tan"); a shared first name alone does not.
    def words(names):
        return {w for n in names for w in norm(n).split()}

    def keys(names):
        return {k for n in names for k in name_keys(norm(n))}
    return bool(keys(a.authors) & words(b.authors) or keys(b.authors) & words(a.authors))


def resolve(read: SpineText, candidates: list[Candidate], threshold: float = DEFAULT_THRESHOLD) -> Resolution:
    if not read.readable:
        return Resolution("unidentified", 0.0, reason="spine not legible")
    if not candidates:
        return Resolution("unidentified", 0.0, reason="no catalogue match for the spine text")
    # Ties go to the record whose full title best matches everything on the spine, so a spine
    # reading "... A GUIDED JOURNAL" picks the journal, not the book it is based on.
    scored = sorted(((score(read, c), title_coverage(read, c), c) for c in candidates),
                    key=lambda x: (x[0], x[1]), reverse=True)
    best_s, _, best = scored[0]
    rival = next(((s, c) for s, _, c in scored[1:] if not same_work(best, c)), None)
    confidence = round(best_s * min(1.0, read.legibility / 0.6), 3)

    title_words_on_spine = set(significant_words(best.title.replace(":", " "))) & set(norm(f"{read.title} {read.text}").split())
    if author_evidence(read, best) is None and len(title_words_on_spine) < 3:
        # "The Raven" with no author legible could be Poe or Stiefvater's "The Raven Boys" cut
        # off at the frame edge. Short titles need the author to be told apart.
        return Resolution("unidentified", confidence, best,
                          reason=f"title-only match on a short title ('{best.title}'); author not legible")
    if confidence < threshold:
        return Resolution("unidentified", confidence, best, reason=f"best match '{best.title}' below threshold ({confidence:.2f})")
    if rival and best_s - rival[0] < MIN_MARGIN:
        return Resolution("unidentified", confidence, best, runner_up=rival[1],
                          reason=f"ambiguous: '{best.title}' vs '{rival[1].title}'")
    # Edition only when the spine names a publisher that the record also lists.
    edition = ""
    if read.publisher and any(fuzz.partial_ratio(norm(read.publisher), norm(p)) >= 85 for p in best.publishers):
        edition = f"{read.publisher} edition"
    return Resolution("identified", confidence, best, edition=edition, isbn=isbn13(best.isbns) if edition else "",
                      runner_up=rival[1] if rival else None)
