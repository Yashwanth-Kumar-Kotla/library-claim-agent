import pytest

from libraryclaim.config import Locale, Settings
from libraryclaim.fx import get_rate
from libraryclaim.pricing import listing_matches_book, price_book, price_item, robust_prices
from libraryclaim.review import Assertion, appraisal_reasons, excluded, says_print
from helpers import FakeWeb

US = Locale("US", "USD")
GB = Locale("GB", "GBP")
IN = Locale("IN", "INR")


def listings(*prices, currency="USD", title="Dune by Frank Herbert paperback"):
    return {"itemSummaries": [{"title": title, "price": {"value": str(p), "currency": currency},
                               "itemWebUrl": f"https://www.ebay.com/itm/{i}"} for i, p in enumerate(prices)]}


def ebay(by_market_and_condition):
    """Route eBay calls by (marketplace header, condition in filter)."""
    def route(url, params, headers):
        cond = "NEW" if "NEW" in params.get("filter", "") else "USED"
        return by_market_and_condition.get((headers.get("X-EBAY-C-MARKETPLACE-ID"), cond), {})
    return route


FX = {"frankfurter": lambda u, p, h: {"date": "2026-10-28", "rates": {p["to"]: {"USD": 0.75, "GBP": 1.33}.get(p["from"], 2.0) if p["to"] != "INR" else 88.0}}}


def test_us_book_new_and_used():
    web = FakeWeb({"ebay": ebay({("EBAY_US", "NEW"): listings(18, 19, 20, 21),
                                 ("EBAY_US", "USED"): listings(6, 7, 8, 9, 10)})})
    p = price_book(web, "t", US, "Dune", "Frank Herbert", "9780441172719")
    assert p.replacement.amount == 19.5 and p.used.amount == 8
    assert not p.replacement.converted and p.replacement.retrieved_at
    assert "ebay.com/sch" in p.replacement.url and p.replacement.n_listings == 4
    assert "asking price" in p.replacement.source


def test_isbn_search_preferred_then_title_fallback():
    calls = []
    def route(url, params, headers):
        calls.append("gtin" if "gtin" in params else "q")
        return listings(10, 11, 12) if "q" in params else {}
    p = price_book(FakeWeb({"ebay": route}), "t", US, "Dune", "Frank Herbert", "9780441172719")
    assert calls[:2] == ["gtin", "q"] and p.replacement.amount == 11


def test_title_search_rejects_study_guides_lots_and_sequels():
    assert not listing_matches_book("CliffsNotes on Herbert's Dune", "Dune")
    assert not listing_matches_book("Dune lot of 6 books", "Dune")
    assert not listing_matches_book("Children of Dune audiobook CD", "Dune")
    assert listing_matches_book("DUNE - Frank Herbert - Ace paperback 1990", "Dune")
    assert not listing_matches_book("Herbert paperback", "Dune")
    noisy = {"itemSummaries": [
        {"title": t, "price": {"value": v, "currency": "USD"}} for t, v in
        [("Dune Frank Herbert", 10), ("Dune Frank Herbert", 12), ("Dune Herbert pb", 11),
         ("Dune Study Guide", 3), ("Dune lot of 5", 60)]]}
    p = price_book(FakeWeb({"ebay": noisy}), "t", US, "Dune", "Frank Herbert")
    assert p.replacement.amount == 11 and p.replacement.n_listings == 3


def test_too_few_listings_means_no_price_not_a_guess():
    web = FakeWeb({"ebay": ebay({("EBAY_US", "NEW"): listings(20, 21)})})
    p = price_book(web, "t", US, "Obscure Book")
    assert p.replacement is None and p.used is None
    assert any("fewer than 3" in n for n in p.notes)


def test_outliers_and_garbage_prices_dropped():
    assert robust_prices([10, 11, 12, 500, 0, -3]) == [10, 11, 12]
    web = FakeWeb({"ebay": {"itemSummaries": [
        {"title": "Dune", "price": {"value": "abc", "currency": "USD"}},
        {"title": "Dune", "price": None}, None, "x",
        {"title": "Dune", "price": {"value": 10, "currency": "USD"}},
        {"title": "Dune", "price": {"value": 11, "currency": "USD"}},
        {"title": "Dune", "price": {"value": 12, "currency": "USD"}}]}})
    assert price_book(web, "t", US, "Dune").replacement.amount == 11


def test_foreign_currency_listings_ignored():
    web = FakeWeb({"ebay": listings(10, 11, 12, currency="EUR")})
    assert price_book(web, "t", US, "Dune").replacement is None


def test_second_country_is_a_setting_not_a_conversion():
    web = FakeWeb({"ebay": ebay({("EBAY_GB", "NEW"): listings(9, 10, 11, currency="GBP"),
                                 ("EBAY_GB", "USED"): listings(4, 5, 6, currency="GBP")})})
    p = price_book(web, "t", GB, "Dune", "Frank Herbert")
    assert p.replacement.currency == "GBP" and not p.replacement.converted
    assert "ebay.co.uk" in p.replacement.url
    assert all(h.get("X-EBAY-C-MARKETPLACE-ID") == "EBAY_GB" for _, _, h in web.calls)


def test_no_local_market_converts_from_fallback_and_labels_it():
    web = FakeWeb({"ebay": ebay({("EBAY_US", "NEW"): listings(10, 10, 10),
                                 ("EBAY_US", "USED"): listings(4, 4, 4)}), **FX})
    p = price_book(web, "t", IN, "Dune")
    assert p.replacement.converted and p.replacement.currency == "INR"
    assert p.replacement.amount == 880.0 and "ECB" in p.replacement.source
    assert any("no eBay marketplace for IN" in n for n in p.notes)


def test_local_market_empty_falls_back_and_converts():
    web = FakeWeb({"ebay": ebay({("EBAY_US", "NEW"): listings(20, 20, 20)}), **FX})
    p = price_book(web, "t", GB, "Dune")
    assert p.replacement.converted and p.replacement.amount == 15.0


def test_fx_unavailable_means_no_price():
    web = FakeWeb({"ebay": ebay({("EBAY_US", "NEW"): listings(20, 20, 20)})}, fail=["frankfurter"])
    assert price_book(web, "t", GB, "Dune").replacement is None


def test_ebay_outage_is_a_note_not_a_crash():
    p = price_book(FakeWeb(fail=["ebay"]), "t", US, "Dune")
    assert p.replacement is None and "lookup failed" in p.notes[0]


def test_nothing_to_search():
    assert price_book(FakeWeb(), "t", US, "").notes == ["nothing to search for"]


def test_used_far_above_new_noted():
    web = FakeWeb({"ebay": ebay({("EBAY_US", "NEW"): listings(10, 10, 10),
                                 ("EBAY_US", "USED"): listings(90, 95, 100)})})
    assert any("collectible" in n for n in price_book(web, "t", US, "Dune").notes)


@pytest.mark.parametrize("payload", [None, [], {"rates": None}, {"rates": {"GBP": "x"}}, {"rates": {"GBP": -1}}])
def test_fx_bad_payloads(payload):
    assert get_rate(FakeWeb({"frankfurter": payload}), "USD", "GBP") is None


def test_fx_same_currency_needs_no_call():
    web = FakeWeb()
    assert get_rate(web, "usd", "USD").rate == 1.0 and web.calls == []


# items

def test_item_with_brand_is_priced():
    web = FakeWeb({"ebay": listings(99, 100, 101, title="Breville Barista Express")})
    p = price_item(web, "t", US, "appliance", "espresso machine", "Breville BES870XL")
    assert p.status == "priced" and p.quote.low == p.quote.high == 100


def test_item_without_brand_gets_sourced_range():
    web = FakeWeb({"ebay": listings(20, 30, 40, 50, 60, title="brass floor lamp")})
    p = price_item(web, "t", US, "lighting", "brass floor lamp")
    assert p.status == "range" and p.quote.low < p.quote.amount < p.quote.high


def test_art_defaults_to_appraisal_and_is_not_searched():
    web = FakeWeb()
    p = price_item(web, "t", US, "art", "framed portrait of a man", is_art=True)
    assert p.status == "needs_appraisal" and web.calls == []


def test_art_print_is_priced_as_print():
    web = FakeWeb({"ebay": listings(30, 40, 50, title="framed portrait art print")})
    p = price_item(web, "t", US, "art", "framed portrait", is_art=True, user_says_print=True)
    assert p.status == "range" and "art print" in web.calls[0][1]["q"]


def test_item_parts_listings_ignored_and_none_found_is_range_without_price():
    web = FakeWeb({"ebay": listings(5, 6, 7, title="lamp for parts broken")})
    p = price_item(web, "t", US, "lighting", "floor lamp")
    assert p.status == "range" and p.quote is None and "fewer than" in p.note


# review rules

def test_appraisal_reasons():
    assert appraisal_reasons(spine_text="DUNE  FIRST EDITION") == ["spine reads 'FIRST EDITION'"]
    assert "antiquarian" in appraisal_reasons(year=1923)[0]
    assert appraisal_reasons(year=1951) == []
    assert appraisal_reasons(year=0) == []
    assert "threshold" in appraisal_reasons(used=150, threshold=150)[0]
    assert appraisal_reasons(replacement=149.99, threshold=150) == []
    said = [Assertion("signed", target="b1", said="that one is signed")]
    assert appraisal_reasons(said=said) == ["policyholder said: that one is signed"]


def test_signed_word_inside_other_words_not_a_flag():
    assert appraisal_reasons(spine_text="DESIGNED FOR LIFE") == []


def test_skip_shelf_and_not_mine():
    said = [Assertion("skip_shelf", shelf="B2"), Assertion("not_mine", target="i7")]
    assert excluded("b10", "B2", said) and not excluded("b11", "B3", said)
    assert excluded("i7", "", said)


def test_latest_print_statement_wins():
    said = [Assertion("is_print", target="i3", at_s=10), Assertion("original", target="i3", at_s=40)]
    assert not says_print("i3", said)
    assert says_print("i3", said[:1])
    assert not says_print("i9", said)


@pytest.mark.parametrize("kwargs", [{"kind": "first_edition"}, {"kind": "made_up", "target": "b1"}])
def test_bad_assertions_rejected(kwargs):
    with pytest.raises(ValueError):
        Assertion(**kwargs)


def test_no_credentials_means_no_calls_and_an_honest_note():
    web = FakeWeb()
    p = price_book(web, "", US, "Dune", "Frank Herbert")
    assert p.replacement is None and "not configured" in p.notes[0] and web.calls == []
    i = price_item(web, "", US, "lighting", "floor lamp")
    assert i.status == "range" and i.quote is None and "not configured" in i.note and web.calls == []
    # Art still defaults to appraisal without credentials.
    assert price_item(web, "", US, "art", "portrait", is_art=True).status == "needs_appraisal"


# SerpApi: eBay's listings through SerpApi when eBay's developer programme isn't available.

def serp(prices_by_cond, title="Dune Frank Herbert paperback"):
    def route(url, params, headers):
        assert params["engine"] == "ebay" and params["buying_format"] == "BIN" and params["api_key"] == "sk"
        cond = "new" if params["LH_ItemCondition"] == "1000" else "used"
        assert cond == "new" or "4000" in params["LH_ItemCondition"]   # used covers Very Good, Good, ...
        return {"organic_results": [{"title": title, "price": {"raw": f"${v}", "extracted": v}, "link": "https://www.ebay.com/itm/1"}
                                    for v in prices_by_cond.get((params["ebay_domain"], cond), ())]}
    return route


def test_serpapi_prices_new_and_used_with_the_same_rules():
    from libraryclaim.pricing import SerpApi
    web = FakeWeb({"serpapi.com/search": serp({("ebay.com", "new"): (18, 19, 20, 21), ("ebay.com", "used"): (6, 7, 8)})})
    p = price_book(web, SerpApi("sk"), US, "Dune", "Frank Herbert")
    assert p.replacement.amount == 19.5 and p.used.amount == 7
    assert "via SerpApi" in p.replacement.source and "ebay.com/sch" in p.replacement.url


def test_serpapi_second_country_uses_that_countrys_site_and_currency():
    from libraryclaim.pricing import SerpApi
    web = FakeWeb({"serpapi.com/search": serp({("ebay.co.uk", "new"): (9, 10, 11), ("ebay.co.uk", "used"): (4, 5, 6)})})
    p = price_book(web, SerpApi("sk"), GB, "Dune", "Frank Herbert")
    assert p.replacement.currency == "GBP" and p.replacement.amount == 10 and not p.replacement.converted


def test_serpapi_skips_price_ranges_and_junk():
    from libraryclaim.pricing import SerpApi

    def route(url, params, headers):
        return {"organic_results": [
            {"title": "Dune Herbert", "price": {"from": {"extracted": 5}, "to": {"extracted": 40}}},   # variations
            {"title": "Dune Herbert", "price": {"from": {"extracted": 12}, "to": {"extracted": 12}}},
            {"title": "Dune Herbert", "price": {"extracted": 11}}, {"title": "Dune Herbert", "price": {"extracted": 13}},
            {"title": "Dune Herbert", "price": "n/a"}, None, "x"]}
    p = price_book(FakeWeb({"serpapi.com/search": route}), SerpApi("sk"), US, "Dune")
    assert p.replacement.amount == 12 and p.replacement.n_listings == 3


def test_price_token_prefers_ebay_then_serpapi_then_nothing():
    from libraryclaim.pricing import SerpApi
    assert Settings(locale=US).price_token() == ""
    t = Settings(locale=US, serpapi_key="sk").price_token()
    assert isinstance(t, SerpApi) and t == "sk"


def test_preflight_reports_serpapi_quota(monkeypatch):
    import httpx
    from libraryclaim import preflight
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"plan_searches_left": 240}))
    c = preflight.prices(Settings(locale=US, serpapi_key="sk"))
    assert c.status == preflight.OK and "240 searches left" in c.detail
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"plan_searches_left": 40}))
    assert preflight.prices(Settings(locale=US, serpapi_key="sk")).status == preflight.WARN
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"error": "Invalid API key."}))
    assert preflight.prices(Settings(locale=US, serpapi_key="bad")).status == preflight.FAIL
