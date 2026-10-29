"""uv run python -m libraryclaim run data/sweeps/<id> [--country GB --currency GBP]"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import MODEL_PRICES, REPO_ROOT, Locale, Settings
from .depth import DepthPro
from .fetch import Fetcher
from .pipeline import Deps, run_pipeline
from . import report
from .pricing import ebay_token
from .schema import ClaimPacket
from .second_country import price_second_country
from .vision import Meter, cached, gemini_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="libraryclaim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="build claim_packet.json and report.html for a saved sweep")
    r.add_argument("sweep", type=Path)
    r.add_argument("--country")
    r.add_argument("--currency")
    r.add_argument("--no-cache", action="store_true",
                   help="call the models even for images seen before (use when measuring cost or variance)")
    rp = sub.add_parser("reprice", help="price the claim's books for a second country (shown in the report, not totalled)")
    rp.add_argument("sweep", type=Path)
    rp.add_argument("--country", required=True)
    rp.add_argument("--currency", required=True)
    rp.add_argument("--books", type=int, default=10)
    args = ap.parse_args(argv)

    if args.cmd == "reprice":
        return reprice(args)

    settings = Settings.from_env()
    if not settings.google_api_key:
        print("GOOGLE_API_KEY is not set (see .env.example)", file=sys.stderr)
        return 2
    if args.country or args.currency:
        settings = replace(settings, locale=Locale.parse(args.country or settings.locale.country,
                                                         args.currency or settings.locale.currency))
        meta_path = args.sweep / "sweep.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta.update(country=settings.locale.country, currency=settings.locale.currency)
        meta_path.write_text(json.dumps(meta, indent=2))

    token = ""
    if settings.ebay_client_id and settings.ebay_client_secret:
        token = ebay_token(settings.ebay_client_id, settings.ebay_client_secret)
    else:
        print("eBay keys not set: every book and item will be left unpriced", file=sys.stderr)

    meter = Meter(MODEL_PRICES)
    vision = gemini_model(settings.google_api_key, settings.vision_model, meter=meter)
    # The closer read is optional; if it is slow, the read from detection stands.
    escalate = gemini_model(settings.google_api_key, settings.vision_escalation_model, meter=meter, timeout_s=15)
    if not args.no_cache:
        vision = cached(vision, REPO_ROOT / "data/cache/model", settings.vision_model)
        escalate = cached(escalate, REPO_ROOT / "data/cache/model", settings.vision_escalation_model)
    deps = Deps(
        vision=vision,
        escalate=escalate,
        get_json=Fetcher(cache_dir=REPO_ROOT / "data/cache/http"),
        ebay_token=token,
        google_books_key=settings.google_books_key,
        meter=meter,
        depth=DepthPro(),  # loaded only if the sweep has a room scan or room views
    )
    packet = run_pipeline(args.sweep, deps, settings)
    t = packet.totals
    cost = meter.cost()
    print(f"{t.book_count} books ({t.books_identified} identified), {len(packet.items)} items, "
          f"{len(packet.review_queue)} review lines -> {args.sweep / 'claim_packet.json'}")
    print(f"model spend this run: ${cost['usd']:.4f}" + ("" if not args.no_cache else " (uncached)"))
    return 0


def reprice(args) -> int:
    settings = Settings.from_env()
    packet_path = args.sweep / "claim_packet.json"
    packet = ClaimPacket.model_validate_json(packet_path.read_text())
    token = ebay_token(settings.ebay_client_id, settings.ebay_client_secret) \
        if settings.ebay_client_id and settings.ebay_client_secret else ""
    packet.second_country = price_second_country(packet, Locale.parse(args.country, args.currency),
                                                 Fetcher(cache_dir=REPO_ROOT / "data/cache/http"), token, args.books)
    packet_path.write_text(packet.model_dump_json(indent=2))
    report.write(packet, args.sweep)
    priced = sum(l.replacement_cost.amount is not None for l in packet.second_country.lines)
    print(f"{len(packet.second_country.lines)} books priced for {args.country.upper()}: {priced} with a replacement price")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
