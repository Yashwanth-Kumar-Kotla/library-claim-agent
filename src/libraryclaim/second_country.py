"""Price the same books for a second country, to show the locale is a setting and not hardcoded.

Runs the exact pricing code used for the claim with a different Locale: a different eBay
marketplace and currency, with the same fallback-and-convert rules. Prices here never enter the
claim totals.
"""

from __future__ import annotations

from .config import Locale
from .fetch import GetJson
from .pricing import Quote, price_book
from .schema import BookPrice, ClaimPacket, SecondCountry, SecondCountryLine


def as_price(q: Quote | None) -> BookPrice:
    if q is None:
        return BookPrice()
    return BookPrice(amount=q.amount, source=q.source, url=q.url, retrieved_at=q.retrieved_at,
                     converted=q.converted, condition_assumed=q.condition)


def pick_books(packet: ClaimPacket, n: int) -> list:
    """Identified books, those already priced at home first, in shelf order."""
    ok = [b for b in packet.books if b.status == "identified" and b.title]
    priced = [b for b in ok if b.replacement_cost.amount is not None]
    rest = [b for b in ok if b.replacement_cost.amount is None]
    return (priced + rest)[:n]


def price_second_country(packet: ClaimPacket, locale: Locale, get_json: GetJson, token: str, n: int = 10) -> SecondCountry:
    home = (packet.sweep.country, packet.sweep.currency)
    if (locale.country, locale.currency) == home:
        raise ValueError("the second country must differ from the claim's own country and currency")
    out = SecondCountry(country=locale.country, currency=locale.currency)
    for b in pick_books(packet, n):
        prices = price_book(get_json, token, locale, b.title, b.author.split(",")[0], b.isbn)
        out.lines.append(SecondCountryLine(book_id=b.id, title=b.title, isbn=b.isbn,
                                           replacement_cost=as_price(prices.replacement), used_value=as_price(prices.used),
                                           note="; ".join(n for n in prices.notes if n)))
    return out
