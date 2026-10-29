import pytest

from libraryclaim.config import Locale
from libraryclaim.report import render
from libraryclaim.schema import Book, BookPrice, ClaimPacket, Sweep
from libraryclaim.second_country import pick_books, price_second_country
from helpers import FakeWeb

GOOD = dict(source="eBay US", url="https://www.ebay.com/sch/x", retrieved_at="2026-10-28T20:00:00+00:00")


def packet(n=12):
    books = [Book(id=f"A1-{i:02d}", status="identified", title=f"Book {i}", author="Some Author",
                  replacement_cost=BookPrice(amount=10 + i, **GOOD) if i % 3 else BookPrice()) for i in range(1, n + 1)]
    books.append(Book(id="A1-99", status="unidentified"))
    books.append(Book(id="A1-98", status="needs_appraisal", title="Rare"))
    return ClaimPacket(sweep=Sweep(country="US", currency="USD"), books=books)


def gb_listings(url, params, headers):
    assert headers["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_GB"
    return {"itemSummaries": [{"title": params.get("q", ""), "price": {"value": v, "currency": "GBP"}} for v in (8, 9, 10)]}


def test_ten_books_priced_in_the_other_market_and_currency():
    sc = price_second_country(packet(), Locale("GB", "GBP"), FakeWeb({"ebay": gb_listings}), "t")
    assert sc.country == "GB" and len(sc.lines) == 10
    assert all(l.replacement_cost.amount == 9 and not l.replacement_cost.converted for l in sc.lines)
    assert all("ebay.co.uk" in l.replacement_cost.url for l in sc.lines)


def test_only_identified_books_and_priced_ones_first():
    chosen = pick_books(packet(), 10)
    assert all(b.status == "identified" for b in chosen)
    assert [b.replacement_cost.amount is not None for b in chosen][:8] == [True] * 8


def test_same_country_is_refused():
    with pytest.raises(ValueError):
        price_second_country(packet(), Locale("US", "USD"), FakeWeb(), "t")


def test_no_credentials_gives_empty_prices_with_a_note_not_numbers():
    sc = price_second_country(packet(), Locale("GB", "GBP"), FakeWeb(), "")
    assert all(l.replacement_cost.amount is None and "not configured" in l.note for l in sc.lines)


def test_report_shows_the_table_and_totals_ignore_it():
    p = packet()
    p.second_country = price_second_country(p, Locale("GB", "GBP"), FakeWeb({"ebay": gb_listings}), "t")
    html = render(p)
    assert "Same books in GB" in html and "GBP 9.00" in html
    assert p.totals.books_replacement_cost == 0  # untouched: totals are only ever computed from the claim's own lines
