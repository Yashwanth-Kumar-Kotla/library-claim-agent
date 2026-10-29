"""Settings from the environment. Country and currency are inputs, never baked into a stage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# eBay marketplace and its listing currency, by ISO country code.
EBAY_MARKETS = {
    "US": ("EBAY_US", "USD"),
    "GB": ("EBAY_GB", "GBP"),
    "CA": ("EBAY_CA", "CAD"),
    "AU": ("EBAY_AU", "AUD"),
    "IE": ("EBAY_IE", "EUR"),
    "DE": ("EBAY_DE", "EUR"),
    "FR": ("EBAY_FR", "EUR"),
    "IT": ("EBAY_IT", "EUR"),
    "ES": ("EBAY_ES", "EUR"),
    "NL": ("EBAY_NL", "EUR"),
    "AT": ("EBAY_AT", "EUR"),
    "BE": ("EBAY_BE", "EUR"),
    "CH": ("EBAY_CH", "CHF"),
    "PL": ("EBAY_PL", "PLN"),
    "HK": ("EBAY_HK", "HKD"),
    "SG": ("EBAY_SG", "SGD"),
}

# Gemini API paid tier, USD per 1M tokens (input, output incl. thinking).
# Source: https://ai.google.dev/gemini-api/docs/pricing, retrieved 2026-10-28.
# gemini-3.8-flash rises to (1.50, 7.50) on 2027-01-01.
MODEL_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}
# Gemini 3.8 Live, USD per 1M tokens by modality (same source and date as above).
LIVE_PRICES = {
    "in": {"TEXT": 0.75, "AUDIO": 3.00, "IMAGE": 1.00, "VIDEO": 1.00},
    "out": {"TEXT": 4.50, "AUDIO": 12.00},
}

# Where to look when the policyholder's country has no marketplace of its own.
FALLBACK_COUNTRY = "US"


@dataclass(frozen=True)
class Locale:
    country: str
    currency: str

    @classmethod
    def parse(cls, country: str, currency: str) -> Locale:
        country, currency = country.strip().upper(), currency.strip().upper()
        if len(country) != 2 or not country.isalpha():
            raise ValueError(f"country must be an ISO 3166 alpha-2 code, got {country!r}")
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError(f"currency must be an ISO 4217 code, got {currency!r}")
        return cls(country, currency)

    @property
    def ebay_market(self) -> str | None:
        market = EBAY_MARKETS.get(self.country)
        return market[0] if market else None


@dataclass(frozen=True)
class Settings:
    locale: Locale
    google_api_key: str = ""
    google_books_key: str = ""
    ebay_client_id: str = ""
    ebay_client_secret: str = ""
    serpapi_key: str = ""
    live_model: str = "gemini-3.8-live"
    vision_model: str = "gemini-3.8-flash"
    vision_escalation_model: str = "gemini-3.1-pro-preview"
    appraisal_threshold: float = 150.0
    marker_size_cm: float = 15.0

    def price_token(self) -> str:
        """eBay's own API when its keys are set, else eBay through SerpApi, else none (lines stay unpriced)."""
        if self.ebay_client_id and self.ebay_client_secret:
            from .pricing import ebay_token
            return ebay_token(self.ebay_client_id, self.ebay_client_secret)
        if self.serpapi_key:
            from .pricing import SerpApi
            return SerpApi(self.serpapi_key)
        return ""

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(REPO_ROOT / ".env")
        env = os.getenv
        return cls(
            locale=Locale.parse(env("COUNTRY", "US"), env("CURRENCY", "USD")),
            google_api_key=env("GOOGLE_API_KEY", ""),
            google_books_key=env("GOOGLE_BOOKS_API_KEY", ""),
            ebay_client_id=env("EBAY_CLIENT_ID", ""),
            ebay_client_secret=env("EBAY_CLIENT_SECRET", ""),
            serpapi_key=env("SERPAPI_API_KEY", ""),
            live_model=env("LIVE_MODEL", "gemini-3.8-live"),
            vision_model=env("VISION_MODEL", "gemini-3.8-flash"),
            vision_escalation_model=env("VISION_ESCALATION_MODEL", "gemini-3.1-pro-preview"),
            appraisal_threshold=float(env("APPRAISAL_THRESHOLD", "150")),
            marker_size_cm=float(env("MARKER_SIZE_CM", "15")),
        )
