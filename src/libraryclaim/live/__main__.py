"""Start the live sweep server.

    uv run python -m libraryclaim.live            # laptop: http://localhost:8765 (browsers allow camera on localhost)
    uv run python -m libraryclaim.live --phone    # phone on the same Wi-Fi: https://<this Mac's LAN IP>:8443

--phone needs a certificate from tools/make_cert.sh, whose CA you install once on the phone.
"""

from __future__ import annotations

import argparse
import socket
import sys

import uvicorn

from ..config import REPO_ROOT

CERTS = REPO_ROOT / "data" / "certs"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no packet is sent; this just picks the outbound interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(prog="libraryclaim.live")
    ap.add_argument("--phone", action="store_true", help="serve HTTPS on the LAN for a phone")
    args = ap.parse_args()
    if not args.phone:
        print("open http://localhost:8765 in a browser on this computer")
        uvicorn.run("libraryclaim.live.server:app", host="127.0.0.1", port=8765)
        return 0
    cert, key = CERTS / "cert.pem", CERTS / "key.pem"
    if not (cert.is_file() and key.is_file()):
        print("no certificate yet: run tools/make_cert.sh first", file=sys.stderr)
        return 2
    print(f"open https://{lan_ip()}:8443 on the phone (same Wi-Fi)")
    uvicorn.run("libraryclaim.live.server:app", host="0.0.0.0", port=8443,
                ssl_certfile=str(cert), ssl_keyfile=str(key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
