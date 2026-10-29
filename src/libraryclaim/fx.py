"""Currency conversion at the published ECB reference rate (via Frankfurter), with the rate's date."""

from __future__ import annotations

from dataclasses import dataclass

from .fetch import FetchError, GetJson

FRANKFURTER = "https://api.frankfurter.app/latest"


@dataclass
class Rate:
    rate: float
    date: str
    url: str


def get_rate(get_json: GetJson, src: str, dst: str) -> Rate | None:
    src, dst = src.upper(), dst.upper()
    if src == dst:
        return Rate(1.0, "", "")
    try:
        r = get_json(FRANKFURTER, {"from": src, "to": dst})
    except FetchError:
        return None
    data = r.data if isinstance(r.data, dict) else {}
    rate = (data.get("rates") or {}).get(dst) if isinstance(data.get("rates"), dict) else None
    if not isinstance(rate, (int, float)) or rate <= 0:
        return None
    return Rate(float(rate), str(data.get("date", "")), r.url)
