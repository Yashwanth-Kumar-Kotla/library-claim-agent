"""Prices from live listings, never from a model.

Books: eBay Browse API in the policyholder's marketplace. New fixed-price listings give the
replacement cost, used ones the market value; each is the median of the listings that survive
filtering. ISBN search is preferred (exact edition); title search filters out listings that
aren't the book (study guides, lots, sequels).

When the local marketplace has nothing, the same search runs in the fallback market and the
figure is converted at the ECB rate and marked converted. When nothing is found anywhere the
price stays empty and the line is excluded from totals.

The eBay Browse API only exposes active listings, so these are asking prices; the source text
says so.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from urllib.parse import quote_plus

from .config import EBAY_MARKETS, FALLBACK_COUNTRY, Locale
from .fetch import FetchError, GetJson
from .fx import get_rate
from .resolve import main_title, norm

EBAY_SEARCH = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_BOOKS_CATEGORY = "267"
EBAY_SITES = {
    "EBAY_US": "www.ebay.com", "EBAY_GB": "www.ebay.co.uk", "EBAY_CA": "www.ebay.ca", "EBAY_AU": "www.ebay.com.au",
    "EBAY_IE": "www.ebay.ie", "EBAY_DE": "www.ebay.de", "EBAY_FR": "www.ebay.fr", "EBAY_IT": "www.ebay.it",
    "EBAY_ES": "www.ebay.es", "EBAY_NL": "www.ebay.nl", "EBAY_AT": "www.ebay.at", "EBAY_BE": "www.benl.ebay.be",
    "EBAY_CH": "www.ebay.ch", "EBAY_PL": "www.ebay.pl", "EBAY_HK": "www.ebay.com.hk", "EBAY_SG": "www.ebay.com.sg",
}
MIN_LISTINGS = 3
NOT_THE_BOOK = re.compile(
    r"\b(lot|lots|bundle|set of|collection|box set|cliffs ?notes|sparknotes|study guide|summary of|workbook|"
    r"analysis|poster|dvd|blu ?ray|audiobook|audio cd|cd|mp3|kindle|ebook|pdf|bookmark|t ?shirt)\b", re.I)
NO_CREDENTIALS = "price source not configured (no eBay credentials); left unpriced"
CONDITION_ASSUMED ={"new": "new", "used": "used, good (median of used listings)"}


@dataclass
class Quote:
    amount: float
    currency: str
    source: str
    url: str
    retrieved_at: str
    n_listings: int
    converted: bool = False
    condition: str = ""
    low: float | None = None
    high: float | None = None
    api_url: str = ""
    note: str = ""


@dataclass
class Listing:
    title: str
    price: float
    currency: str
    url: str


def ebay_token(client_id: str, client_secret: str) -> str:
    """Application token (client-credentials grant) for the Browse API. Valid ~2 hours."""
    import httpx

    resp = httpx.post("https://api.ebay.com/identity/v1/oauth2/token", auth=(client_id, client_secret),
                      data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
                      timeout=15)
    resp.raise_for_status()
    return resp.json()["access_token"]


def site_search_url(market: str, query: str, condition: str) -> str:
    cond = {"new": "1000", "used": "3000"}[condition]
    return f"https://{EBAY_SITES.get(market, 'www.ebay.com')}/sch/i.html?_nkw={quote_plus(query)}&LH_ItemCondition={cond}&LH_BIN=1"


class SerpApi(str):
    """A SerpApi key, passed where an eBay OAuth token would go: the same eBay searches, through
    SerpApi's eBay engine, for when eBay's own developer programme is not available."""


SERPAPI = "https://serpapi.com/search"
# eBay condition ids. Used books are listed as Like New, Very Good, Good or Acceptable, rarely plain "Used".
CONDITION_IDS = {"new": "1000", "used": "2750|3000|4000|5000|6000"}


def serpapi_listings(get_json: GetJson, key: str, market: str, q: str, condition: str, category: str | None):
    currency = next(c for m, c in EBAY_MARKETS.values() if m == market)
    params = {"engine": "ebay", "_nkw": q, "ebay_domain": EBAY_SITES[market].removeprefix("www."),
              "LH_ItemCondition": CONDITION_IDS[condition], "buying_format": "BIN", "_ipg": 60, "api_key": key}
    if category:
        params["category_id"] = category
    resp = get_json(SERPAPI, params)
    data = resp.data if isinstance(resp.data, dict) else {}
    out = []
    for it in data.get("organic_results") or []:
        if not isinstance(it, dict):
            continue
        price = it.get("price") if isinstance(it.get("price"), dict) else {}
        if "from" in price or "to" in price:
            lo, hi = (price.get("from") or {}).get("extracted"), (price.get("to") or {}).get("extracted")
            value = lo if lo == hi else None  # a price range is a multi-variation listing: no single price
        else:
            value = price.get("extracted")
        if not isinstance(value, (int, float)):
            continue
        out.append(Listing(str(it.get("title", "")), float(value), currency, str(it.get("link", ""))))
    return out, resp


def ebay_listings(get_json: GetJson, token: str, market: str, *, q: str = "", gtin: str = "",
                  condition: str, category: str | None = None):
    if isinstance(token, SerpApi):
        return serpapi_listings(get_json, token, market, gtin or q, condition, category)
    params = {"limit": 50, "filter": f"buyingOptions:{{FIXED_PRICE}},conditions:{{{condition.upper()}}}"}
    if gtin:
        params["gtin"] = gtin
    else:
        params["q"] = q
    if category:
        params["category_ids"] = category
    resp = get_json(EBAY_SEARCH, params, {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": market})
    data = resp.data if isinstance(resp.data, dict) else {}
    out = []
    for it in data.get("itemSummaries") or []:
        if not isinstance(it, dict):
            continue
        price = it.get("price") or {}
        try:
            value = float(price.get("value"))
        except (TypeError, ValueError):
            continue
        out.append(Listing(str(it.get("title", "")), value, str(price.get("currency", "")), str(it.get("itemWebUrl", ""))))
    return out, resp


def listing_matches_book(listing_title: str, title: str) -> bool:
    if NOT_THE_BOOK.search(listing_title):
        return False
    want = set(norm(main_title(title)).split()) - {"the", "a", "an", "of", "and"}
    have = set(norm(listing_title).split())
    return bool(want) and want <= have


def robust_prices(prices: list[float]) -> list[float]:
    """Drop zero prices and anything more than 3x off the median: mispriced or mislisted."""
    prices = [p for p in prices if p > 0]
    if len(prices) < MIN_LISTINGS:
        return prices
    mid = statistics.median(prices)
    return [p for p in prices if mid / 3 <= p <= mid * 3]


def quote_from(listings: list[Listing], currency: str, resp, market: str, query: str, condition: str,
               keep=lambda l: True) -> Quote | None:
    prices = robust_prices([l.price for l in listings if l.currency == currency and keep(l)])
    if len(prices) < MIN_LISTINGS:
        return None
    q = statistics.quantiles(prices, n=4) if len(prices) >= 4 else [min(prices), 0, max(prices)]
    return Quote(
        amount=round(statistics.median(prices), 2), currency=currency,
        source=(f"eBay {market.removeprefix('EBAY_')}{' (via SerpApi)' if 'serpapi.com' in resp.url else ''}: "
                f"median asking price of {len(prices)} {condition} fixed-price listings"),
        url=site_search_url(market, query, condition), retrieved_at=resp.retrieved_at, n_listings=len(prices),
        condition=CONDITION_ASSUMED[condition], low=round(q[0], 2), high=round(q[-1], 2), api_url=resp.url,
    )


def search_book(get_json: GetJson, token: str, country: str, title: str, author: str, isbn: str,
                condition: str) -> tuple[Quote | None, str]:
    market, currency = EBAY_MARKETS[country]
    tried = []
    try:
        if isbn:
            listings, resp = ebay_listings(get_json, token, market, gtin=isbn, condition=condition)
            quote = quote_from(listings, currency, resp, market, isbn, condition)
            if quote:
                return quote, ""
            tried.append("ISBN")
        query = f"{main_title(title)} {author}".strip()
        listings, resp = ebay_listings(get_json, token, market, q=query, condition=condition, category=EBAY_BOOKS_CATEGORY)
        quote = quote_from(listings, currency, resp, market, query, condition,
                           keep=lambda l: listing_matches_book(l.title, title))
        if quote:
            return quote, ""
        tried.append("title")
        return None, f"fewer than {MIN_LISTINGS} matching {condition} listings on eBay {country} ({'/'.join(tried)} search)"
    except FetchError as exc:
        return None, f"eBay {country} lookup failed: {exc}"


def convert(quote: Quote, to: str, get_json: GetJson) -> Quote | None:
    rate = get_rate(get_json, quote.currency, to)
    if rate is None:
        return None
    f = lambda v: round(v * rate.rate, 2) if v is not None else None
    return Quote(
        amount=f(quote.amount), currency=to, source=f"{quote.source}; converted {quote.currency}->{to} at ECB rate {rate.rate} ({rate.date})",
        url=quote.url, retrieved_at=quote.retrieved_at, n_listings=quote.n_listings, converted=True,
        condition=quote.condition, low=f(quote.low), high=f(quote.high), api_url=quote.api_url, note=rate.url,
    )


@dataclass
class BookPrices:
    replacement: Quote | None = None
    used: Quote | None = None
    notes: list[str] = field(default_factory=list)


def price_book(get_json: GetJson, token: str, locale: Locale, title: str, author: str = "", isbn: str = "") -> BookPrices:
    out = BookPrices()
    if not title and not isbn:
        out.notes.append("nothing to search for")
        return out
    if not token:
        out.notes.append(NO_CREDENTIALS)
        return out
    for condition in ("new", "used"):
        quote, note = (None, f"no eBay marketplace for {locale.country}")
        if locale.country in EBAY_MARKETS:
            quote, note = search_book(get_json, token, locale.country, title, author, isbn, condition)
            if quote and quote.currency != locale.currency:
                # e.g. policyholder chose GB but wants USD: convert, and say so.
                quote = convert(quote, locale.currency, get_json)
                note = "" if quote else f"no exchange rate to {locale.currency}"
        if quote is None and locale.country != FALLBACK_COUNTRY:
            out.notes.append(note)
            fb, fb_note = search_book(get_json, token, FALLBACK_COUNTRY, title, author, isbn, condition)
            quote = convert(fb, locale.currency, get_json) if fb else None
            note = fb_note if fb is None else ("" if quote else f"no exchange rate to {locale.currency}")
        if quote is None:
            out.notes.append(note)
        setattr(out, "replacement" if condition == "new" else "used", quote)
    if out.replacement and out.used and out.used.amount > out.replacement.amount * 1.5:
        out.notes.append("used asking price is well above new; may be a collectible edition")
    return out


@dataclass
class ItemPrice:
    status: str  # priced | range | needs_appraisal
    quote: Quote | None = None
    note: str = ""


RANGE_NOTE = "brand/model not visible; interquartile range of similar new items"


def has_model(brand_model: str) -> bool:
    """'Sony WH-1000XM4', 'Herman Miller Aeron' name a product; 'LG' or 'IKEA' only name a maker."""
    return bool(re.search(r"\d", brand_model)) or len(brand_model.split()) > 1


def price_item(get_json: GetJson, token: str, locale: Locale, category: str, description: str,
               brand_model: str = "", is_art: bool = False, user_says_print: bool = False,
               name: str = "") -> ItemPrice:
    """Known brand/model -> priced at the median of new listings for it.
    Unknown model -> a sourced range (interquartile) for new items of that kind, searched by its short
    shop name ("office chair"), which finds comparable listings and lets similar objects share a search.
    Art -> appraisal unless the policyholder said it is a print."""
    if is_art and not user_says_print:
        return ItemPrice("needs_appraisal", note="art defaults to appraisal; the policyholder has not said it is a print")
    # A bare brand ("LG") is not a model: searched alone it matches anything the brand sells.
    # Brand plus the kind of object ("LG monitor") finds comparable listings, priced as a range.
    if brand_model and not has_model(brand_model):
        brand_model, name = "", f"{brand_model} {name or description}".strip()
    query = brand_model or name or description
    if is_art:
        query = f"{description} art print".strip()
    if not query:
        return ItemPrice("range", note="no description to search for")
    if not token:
        return ItemPrice("range", note=NO_CREDENTIALS)
    countries = [locale.country] if locale.country in EBAY_MARKETS else []
    if locale.country != FALLBACK_COUNTRY:
        countries.append(FALLBACK_COUNTRY)
    note = ""
    for country in countries:
        market, currency = EBAY_MARKETS[country]
        try:
            listings, resp = ebay_listings(get_json, token, market, q=query, condition="new")
        except FetchError as exc:
            note = f"eBay {country} lookup failed: {exc}"
            continue
        quote = quote_from(listings, currency, resp, market, query, "new",
                           keep=lambda l: not re.search(r"\b(lot|parts|for parts|broken|replacement part)\b", l.title, re.I))
        if quote is None:
            note = f"fewer than {MIN_LISTINGS} new listings for '{query}' on eBay {country}"
            continue
        if quote.currency != locale.currency:
            quote = convert(quote, locale.currency, get_json)
            if quote is None:
                note = f"no exchange rate to {locale.currency}"
                continue
        if brand_model:
            quote.low = quote.high = quote.amount
            return ItemPrice("priced", quote)
        return ItemPrice("range", quote, note=RANGE_NOTE)
    return ItemPrice("range", None, note=note)
