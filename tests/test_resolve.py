import pytest

from libraryclaim.resolve import Candidate, norm, resolve, score, search
from libraryclaim.vision import SpineText
from helpers import FakeWeb


def read(text="", title="", author="", publisher="", legibility=0.9):
    return SpineText(text, title, author, publisher, legibility)


def cand(title, authors=("Frank Herbert",), year=1965, isbns=("9780441172719",), publishers=("Ace",)):
    return Candidate(title, list(authors), year, list(isbns), list(publishers), "Open Library", "https://openlibrary.org/works/X")


def test_clean_match_identified():
    r = resolve(read(title="Dune", author="Frank Herbert"), [cand("Dune")])
    assert r.status == "identified" and r.confidence > 0.9


def test_subtitle_on_record_still_matches():
    r = resolve(read(title="Sapiens", author="Yuval Noah Harari"),
                [cand("Sapiens: A Brief History of Humankind", ["Yuval Noah Harari"])])
    assert r.status == "identified"


def test_unreadable_spine_never_searched_or_named():
    r = resolve(read(text="", legibility=0), [cand("Dune")])
    assert r.status == "unidentified" and r.reason == "spine not legible"


def test_no_candidates():
    assert resolve(read(title="Zzqx Unknown"), []).status == "unidentified"


def test_same_title_different_author_rejected():
    # "Emma" by Austen on the spine; catalogue only offers a different Emma.
    r = resolve(read(title="Emma", author="Jane Austen"), [cand("Emma", ["Someone Else"])])
    assert r.status == "unidentified"


def test_ambiguous_title_without_author_is_not_guessed():
    r = resolve(read(title="Foundation"),
                [cand("Foundation", ["Isaac Asimov"]), cand("Foundation", ["Mark Lawrence"])])
    assert r.status == "unidentified" and ("ambiguous" in r.reason or "short title" in r.reason)


def test_editions_of_same_work_are_not_ambiguity():
    r = resolve(read(title="Dune", author="Frank Herbert"),
                [cand("Dune", year=1965), cand("Dune", year=2005, isbns=("9780441013593",))])
    assert r.status == "identified"


def test_partial_garbled_read_stays_below_threshold():
    r = resolve(read(text="D NE HERB", legibility=0.4), [cand("Dune")])
    assert r.status == "unidentified"


def test_low_legibility_scales_confidence_down():
    hi = resolve(read(title="Dune", author="Frank Herbert", legibility=0.9), [cand("Dune")])
    lo = resolve(read(title="Dune", author="Frank Herbert", legibility=0.3), [cand("Dune")])
    assert lo.confidence < hi.confidence and lo.status == "unidentified"


def test_edition_and_isbn_only_when_publisher_matches():
    with_pub = resolve(read(title="Dune", author="Frank Herbert", publisher="ACE"), [cand("Dune")])
    assert with_pub.edition and with_pub.isbn == "9780441172719"
    no_pub = resolve(read(title="Dune", author="Frank Herbert"), [cand("Dune")])
    assert no_pub.edition == "" and no_pub.isbn == ""
    wrong_pub = resolve(read(title="Dune", author="Frank Herbert", publisher="Penguin"), [cand("Dune")])
    assert wrong_pub.isbn == ""


def test_spine_text_only_with_author_words_mixed_in():
    r = resolve(read(text="THE HOBBIT J.R.R. TOLKIEN HARPERCOLLINS"),
                [cand("The Hobbit", ["J.R.R. Tolkien"])])
    assert r.status == "identified"


def test_short_title_buried_in_text_is_weak():
    s = score(read(text="IT STEPHEN KING HODDER"), cand("It", ["Stephen King"]))
    assert s < 0.9


def test_unicode_and_punctuation_normalised():
    assert norm("Les Misérables — Victor Hugo!") == "les miserables victor hugo"
    r = resolve(read(title="LES MISERABLES", author="VICTOR HUGO"), [cand("Les Misérables", ["Victor Hugo"])])
    assert r.status == "identified"


def test_search_survives_catalogue_outage():
    web = FakeWeb(fail=["openlibrary"])
    found, notes = search(read(title="Dune"), web)
    assert found == [] and "Open Library unavailable" in notes[0]


def test_search_parses_both_catalogues_and_skips_untitled():
    web = FakeWeb({
        "openlibrary": {"docs": [{"key": "/works/OL1W", "title": "Dune", "author_name": ["Frank Herbert"], "isbn": ["0441172717"]},
                                 {"key": "/works/OL2W"}]},
        "googleapis": {"items": [{"id": "g1", "volumeInfo": {"title": "Dune", "authors": ["Frank Herbert"],
                       "publishedDate": "2005-08-02", "industryIdentifiers": [{"identifier": "9780441013593"}]}},
                       {"id": "g2", "volumeInfo": {}}]},
    })
    found, notes = search(read(title="Dune"), web, google_key="k")
    assert [c.source for c in found] == ["Open Library", "Google Books"]
    assert found[1].year == 2005 and not notes


@pytest.mark.parametrize("payload", [None, {"docs": None}, {"docs": [None]}, []])
def test_search_tolerates_malformed_catalogue_payloads(payload):
    web = FakeWeb({"openlibrary": payload})
    try:
        found, _ = search(read(title="Dune"), web)
    except (AttributeError, TypeError) as exc:
        pytest.fail(f"crashed on malformed payload: {exc}")
    assert found == []


# Regressions from the first real run (web shelf photos, Oct 28).

def test_real_raven_boys_cut_off_is_not_poe():
    r = resolve(read(text="THE RAVEN CYCLE BOOK 1 THE RAVEN FIC STI S", title="THE RAVEN", legibility=0.85),
                [cand("The Raven", ["Edgar Allan Poe"]), cand("The Raven Boys", ["Maggie Stiefvater"])])
    assert r.status == "unidentified" and "short title" in r.reason


def test_real_half_read_author_retries_title_only_search():
    def ol(url, params, headers):
        if "IONS" in params["q"]:
            return {"docs": []}
        return {"docs": [{"key": "/works/G", "title": "House of Gold", "author_name": ["Natasha Solomons"]}]}
    web = FakeWeb({"openlibrary": ol})
    text = read(text="HOUSE OF GOLD NATASHA IONS FIC SOL PUTNAM", title="HOUSE OF GOLD", author="NATASHA IONS", legibility=0.85)
    found, _ = search(text, web)
    assert len(web.calls) == 2 and found
    assert resolve(text, found).status == "identified"


def test_real_partial_author_on_spine():
    r = resolve(read(text="True Believer NICH FIC SPA SPARKS", title="True Believer", author="NICH SPARKS", legibility=0.85),
                [cand("True Believer", ["Nicholas Sparks"])])
    assert r.status == "identified"


def test_real_two_records_of_one_work_are_not_ambiguous():
    r = resolve(read(title="The Notebook", author="NICHOLAS SPARKS", legibility=0.85),
                [cand("The Notebook", ["Nicholas Sparks"]), cand("The Notebook", ["Editorial Universe", "Sparks,Nicholas"])])
    assert r.status == "identified"


def test_real_truncated_title_matches_by_prefix():
    r = resolve(read(text="THESE WITCHES DONT N FIC STE", title="THESE WITCHES DONT", legibility=0.7),
                [cand("These Witches Don't Burn", ["Isabel Sterling"])])
    assert r.status == "identified"


def test_short_prefix_is_not_enough():
    r = resolve(read(title="THE NIGHT", author="Someone", legibility=0.9), [cand("The Night Circus", ["Someone"])])
    assert r.status == "unidentified"


def test_real_subtitle_on_spine_breaks_the_tie():
    r = resolve(read(text="PARKER STRONG IS THE NEW PRETTY A GUIDED JOURNAL FOR GIRLS", title="STRONG IS THE NEW PRETTY",
                     author="PARKER", legibility=0.95),
                [cand("Strong is the new pretty", ["Kate T. Parker"]),
                 cand("Strong Is the New Pretty: A Guided Journal for Girls", ["Kate T. Parker"])])
    assert "Journal" in r.candidate.title


def test_partial_author_match_needs_to_be_strong():
    r = resolve(read(title="Emma", author="Jane Austen"), [cand("Emma", ["Jane Ellison"])])
    assert r.status == "unidentified"


def test_identify_uses_one_catalogue_when_it_settles_it():
    from libraryclaim.resolve import identify
    gb = {"items": [{"id": "g", "volumeInfo": {"title": "Dune", "authors": ["Frank Herbert"]}}]}
    web = FakeWeb({"googleapis": gb, "openlibrary": {"docs": []}})
    res, notes = identify(read(title="Dune", author="Frank Herbert"), web, google_key="k")
    assert res.status == "identified" and res.candidate.source == "Google Books"
    assert not any("openlibrary" in u for u, _, _ in web.calls)


def test_identify_falls_back_when_first_catalogue_misses_or_fails():
    from libraryclaim.resolve import identify
    ol = {"docs": [{"key": "/works/D", "title": "Dune", "author_name": ["Frank Herbert"]}]}
    for gb_route in ({"googleapis": {"items": []}}, {}):
        web = FakeWeb({**gb_route, "openlibrary": ol}, fail=() if gb_route else ["googleapis"])
        res, notes = identify(read(title="Dune", author="Frank Herbert"), web, google_key="k")
        assert res.status == "identified" and res.candidate.source == "Open Library"
    assert "Google Books unavailable" in notes[0]


def test_identify_without_key_or_text():
    from libraryclaim.resolve import identify
    web = FakeWeb()
    res, _ = identify(read(text="", legibility=0), web)
    assert res.status == "unidentified" and web.calls == []


# Regressions from the second phone sweep (Oct 28): legible spines refused by over-cautious rules.

def test_real_language_note_record_is_not_a_rival_work():
    r = resolve(read(text="BHAGAVAD GITA AS IT IS HIS DIVINE GRACE A. C. BHAKTIVEDANTA SWAMI PRABHUPADA",
                     title="BHAGAVAD GĪTĀ AS IT IS", author="HIS DIVINE GRACE A. C. BHAKTIVEDANTA SWAMI PRABHUPĀDA", legibility=0.95),
                [cand("Bhagavad-Gita As It Is", ["Prabhupada A. C. Bhaktivedanta Swami"]),
                 cand("BHAGAVAD GITA AS IT IS [NEPALI LANGUAGE].", [])])
    assert r.status == "identified"


def test_real_surname_first_author_listing_is_the_same_person():
    r = resolve(read(text="THE SCIENCE OF Self-Realization His Divine Grace A. C. Bhaktivedanta Swami Prabhupada",
                     title="THE SCIENCE OF Self-Realization", author="His Divine Grace A. C. Bhaktivedanta Swami Prabhupada",
                     legibility=0.95),
                [cand("The Science of Self Realization", ["A. C. Bhaktivedanta Swami Prabhupāda"]),
                 cand("The Science of Self-realization", ["Prabhupada A. C. Bhaktivedanta Swami"])])
    assert r.status == "identified"


def test_real_subtitle_printed_on_spine_makes_a_short_title_distinctive():
    r = resolve(read(text="COMING BACK THE SCIENCE OF REINCARNATION", title="COMING BACK THE SCIENCE OF REINCARNATION",
                     legibility=0.85), [cand("Coming Back : the Science of Reincarnation", [])])
    assert r.status == "identified"


def test_still_rejected_after_loosening():
    from libraryclaim.resolve import author_match
    assert author_match("Stephen King", "Stephen Fry") < 0.5        # two-word names: surname still required
    assert author_match("Jane Austen", "Jane Ellison") < 0.5
    raven = resolve(read(text="THE RAVEN", title="THE RAVEN", legibility=0.9),
                    [cand("The Raven", ["Edgar Allan Poe"]), cand("The Raven Boys", ["Maggie Stiefvater"])])
    assert raven.status == "unidentified"


def test_real_short_surname_with_narrator_record_is_same_work():
    r = resolve(read(text="THE JOY LUCK CLUB AMY TAN PUTNAM", title="THE JOY LUCK CLUB", author="AMY TAN", legibility=0.9),
                [cand("The Joy Luck Club", ["Amy Tan", "Gwendoline Christie"]), cand("The Joy Luck Club", ["Amy Tan"])])
    assert r.status == "identified"


def test_shared_first_name_is_still_a_different_author():
    from libraryclaim.resolve import same_work
    assert not same_work(cand("Emma", ["Jane Austen"]), cand("Emma", ["Jane Ellison"]))


GRISHAM = ("John Grisham",)


def test_series_name_alone_is_not_a_specific_book():
    """Spine read "THEODORE BOONE": the catalogue offers several books of the series."""
    r = resolve(read("JOHN GRISHAM THEODORE BOONE", "THEODORE BOONE", "JOHN GRISHAM"),
                [cand("Theodore Boone: The Abduction", GRISHAM), cand("Theodore Boone: Kid Lawyer", GRISHAM),
                 cand("Theodore Boone: The Accused", GRISHAM)])
    assert r.status == "unidentified" and "shared by several books" in r.reason


def test_series_book_named_on_the_spine_is_identified():
    r = resolve(read("JOHN GRISHAM THEODORE BOONE the accused", "THEODORE BOONE the accused", "JOHN GRISHAM"),
                [cand("Theodore Boone: The Abduction", GRISHAM), cand("Theodore Boone: The Accused", GRISHAM)])
    assert r.status == "identified" and r.candidate.title == "Theodore Boone: The Accused"


def test_two_editions_of_one_book_are_not_a_series():
    authors = ("Joseph Grenny", "Kerry Patterson")
    r = resolve(read("INFLUENCER GRENNY PATTERSON", "Influencer", "Grenny"),
                [cand("Influencer: The Power to Change Anything", authors),
                 cand("Influencer: The New Science of Leading Change", authors)])
    assert r.status == "identified"


def test_year_on_the_spine_rules_out_another_annual_edition():
    spine = read("Sunday Missal - Prayerbook and Hymnal for 2019", "Sunday Missal - Prayerbook and Hymnal for 2019")
    wrong = cand("St. Joseph Sunday Missal Prayerbook and Hymnal for 2026", ("Catholic Book Publishing",))
    assert score(spine, wrong) == 0.0
    assert resolve(spine, [wrong]).status == "unidentified"
