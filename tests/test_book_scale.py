import pytest

from libraryclaim.book_scale import consensus_scale, edition_heights, format_scale, parse_cm
from helpers import FakeWeb


@pytest.mark.parametrize("text,cm", [("24.00 cm", 24.0), ("9.5 in", 24.13), ("230 mm", 23.0), ("", None),
                                      ("tall", None), ("400 cm", None), (None, None)])
def test_parse_cm(text, cm):
    assert parse_cm(text) == (pytest.approx(cm) if cm else None)


def test_consensus_ignores_wrong_editions_and_misreads():
    true_k = 1 / 0.75                      # depth model reads every spine 25% short
    books = [24.1, 21.0, 23.4, 17.5, 22.9, 24.0]
    pairs = [(h * 0.75, [h, 17.3]) for h in books]          # each also lists a paperback edition
    pairs.append((30 * 0.75, [12.0]))                       # a misread: no edition fits
    k, n = consensus_scale(pairs)
    assert k == pytest.approx(true_k, rel=0.02) and n >= 6


def test_too_few_agreeing_books_gives_no_scale():
    assert consensus_scale([(18.0, [24.0]), (15.7, [21.0])]) is None
    assert consensus_scale([]) is None


def test_format_fallback_needs_a_real_shelf():
    assert format_scale([16, 17, 18]) is None
    assert format_scale([16.1, 15.8, 17.6, 18.0, 16.2]) == pytest.approx(21.5 / 16.2)


def test_edition_heights_from_volume_lookups():
    def books(url, params, headers):
        if url.endswith("/volumes"):
            return {"items": [{"id": "a", "volumeInfo": {"title": "The Witchwood Crown"}},
                              {"id": "b", "volumeInfo": {"title": "The Witchwood Crown: Book One"}},
                              {"id": "x", "volumeInfo": {"title": "Something Else"}},
                              {"id": "c", "volumeInfo": {"title": "The Witchwood Crown"}}]}
        dims = {"a": {"height": "24.10 cm"}, "b": {"height": "17.30 cm"}, "c": {}}[url.rsplit("/", 1)[1]]
        return {"volumeInfo": {"dimensions": dims}}
    web = FakeWeb({"googleapis.com/books/v1/volumes": books})
    assert edition_heights("The Witchwood Crown", "Tad Williams", web, "k") == [17.3, 24.1]
    assert not any(u.endswith("/x") for u, _, _ in web.calls)  # other titles never looked up
    assert edition_heights("The Witchwood Crown", "Tad Williams", web, "") == []
    assert edition_heights("X", "Y", FakeWeb(fail=["googleapis"]), "k") == []
