"""Is everything ready for a live sweep? One command, a clear GO or NO-GO.

    uv run python -m libraryclaim.preflight

Checks what has actually broken or would break a session: Gemini credit (one tiny call, about
$0.0001), the catalogue key, price credentials, the depth model, the phone certificate matching this
computer's current network address, and the markers file. Each line says what to do about it.
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
import sys
from dataclasses import dataclass

from .config import REPO_ROOT, Settings

OK, WARN, FAIL = "ok", "warn", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def gemini(s: Settings) -> Check:
    if not s.google_api_key:
        return Check("Gemini", FAIL, "GOOGLE_API_KEY missing in .env")
    try:
        from google import genai
        client = genai.Client(api_key=s.google_api_key)  # keep a reference: a discarded client closes mid-call
        r = client.models.generate_content(model=s.vision_model, contents="Reply: ok")
        return Check("Gemini", OK, f"{s.vision_model} answered ({(r.text or '').strip()[:10]})")
    except Exception as exc:  # noqa: BLE001 - the message is what the person needs
        msg = str(exc)
        if "402" in msg or "credits are depleted" in msg:
            return Check("Gemini", FAIL, "prepaid credit is used up: top up in Google AI Studio")
        if "401" in msg or "403" in msg or "API key" in msg:
            return Check("Gemini", FAIL, "key rejected: check GOOGLE_API_KEY")
        return Check("Gemini", FAIL, f"call failed: {msg[:120]}")


def books(s: Settings) -> Check:
    if not s.google_books_key:
        return Check("Google Books", WARN, "no GOOGLE_BOOKS_API_KEY: matching falls back to Open Library only")
    import httpx
    try:
        r = httpx.get("https://www.googleapis.com/books/v1/volumes", params={"q": "the hobbit tolkien", "maxResults": 1,
                                                                             "key": s.google_books_key}, timeout=10)
        if r.status_code == 200 and r.json().get("items"):
            return Check("Google Books", OK, "key works")
        return Check("Google Books", FAIL, f"HTTP {r.status_code}: {r.text[:100]}")
    except Exception as exc:  # noqa: BLE001
        return Check("Google Books", FAIL, f"unreachable: {exc}")


def prices(s: Settings) -> Check:
    if not (s.ebay_client_id and s.ebay_client_secret):
        if s.serpapi_key:
            return serpapi(s)
        return Check("Prices", WARN, "no eBay or SerpApi key: every line will be left unpriced (allowed, but costs the pricing score)")
    try:
        from .pricing import ebay_token
        ebay_token(s.ebay_client_id, s.ebay_client_secret)
        return Check("Prices (eBay)", OK, "token issued")
    except Exception as exc:  # noqa: BLE001
        return Check("Prices (eBay)", FAIL, f"token request failed: {str(exc)[:120]}")


def serpapi(s: Settings) -> Check:
    """SerpApi's account endpoint is free (it doesn't use a search) and says how many searches are left."""
    import httpx
    try:
        j = httpx.get("https://serpapi.com/account", params={"api_key": s.serpapi_key}, timeout=10).json()
    except Exception as exc:  # noqa: BLE001
        return Check("Prices (SerpApi)", FAIL, f"unreachable: {str(exc)[:100]}")
    if "error" in j:
        return Check("Prices (SerpApi)", FAIL, f"key rejected: {j['error'][:100]}")
    left = j.get("plan_searches_left", j.get("total_searches_left"))
    status = OK if isinstance(left, int) and left >= 150 else WARN
    return Check("Prices (SerpApi)", status, f"{left} searches left this month (a 60-book sweep needs about 150; replays are free)")


def depth() -> Check:
    try:
        import torch
        from huggingface_hub import try_to_load_from_cache
        from .depth import MODEL_ID
        cached = try_to_load_from_cache(MODEL_ID, "config.json")
        where = "downloaded" if isinstance(cached, str) else "NOT downloaded yet (~1.9 GB on first use: do it before leaving)"
        accel = "Apple GPU" if torch.backends.mps.is_available() else "CPU only (slow)"
        status = OK if isinstance(cached, str) and torch.backends.mps.is_available() else WARN
        return Check("Depth model", status, f"Depth Pro {where}; runs on {accel}")
    except Exception as exc:  # noqa: BLE001
        return Check("Depth model", FAIL, f"not usable: {exc}")


def certificate() -> Check:
    from .live.__main__ import lan_ip
    cert = REPO_ROOT / "data" / "certs" / "cert.pem"
    ip = lan_ip()
    if not cert.is_file():
        return Check("Phone certificate", FAIL, "none yet: run ./tools/make_cert.sh")
    try:
        info = ssl._ssl._test_decode_cert(str(cert))  # standard library, reads the certificate's addresses
        names = {v for k, v in info.get("subjectAltName", ())}
    except Exception as exc:  # noqa: BLE001
        return Check("Phone certificate", FAIL, f"unreadable: {exc}")
    if ip not in names:
        return Check("Phone certificate", FAIL,
                     f"made for {', '.join(sorted(n for n in names if n[0].isdigit()))}, but this network gave the Mac {ip}: "
                     f"run ./tools/make_cert.sh again (the phone keeps trusting it, nothing to reinstall)")
    return Check("Phone certificate", OK, f"matches this network: open https://{ip}:8443 on the phone")


def markers() -> Check:
    pdf = REPO_ROOT / "ground_truth" / "markers.pdf"
    if not pdf.is_file():
        return Check("Markers", FAIL, "ground_truth/markers.pdf missing: uv run python tools/make_markers.py")
    return Check("Markers", OK, "markers.pdf ready: print at 100%, one per bookcase, measure the black square")


def disk() -> Check:
    free = shutil.disk_usage(REPO_ROOT).free / 1e9
    # a 3-minute sweep at 3 frames/s of ~300 KB is about 0.5 GB; leave room for several takes
    return Check("Disk space", OK if free > 5 else WARN, f"{free:.0f} GB free")


def run(checks=None) -> list[Check]:
    s = Settings.from_env()
    return [c() for c in (checks or [lambda: gemini(s), lambda: books(s), lambda: prices(s), depth, certificate, markers, disk])]


def main() -> int:
    results = run()
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}
    for c in results:
        print(f"[{mark[c.status]}] {c.name:18} {c.detail}")
    fails = [c for c in results if c.status == FAIL]
    print("\nGO" if not fails else f"\nNO-GO: fix {len(fails)} item(s) above before a live sweep")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
