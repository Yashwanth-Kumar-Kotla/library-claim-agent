#!/usr/bin/env bash
# HTTPS certificate so a phone on the same Wi-Fi can use its camera and microphone on the sweep page.
#
# Uses mkcert to make a private certificate authority and a certificate for this Mac's LAN IP.
# It does NOT install the authority on this Mac (the laptop uses http://localhost, which needs none).
# Install it on the phone once:
#   iPhone: AirDrop rootCA.pem to the phone -> Settings > Profile Downloaded > Install,
#           then Settings > General > About > Certificate Trust Settings > enable it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/certs"
mkdir -p "$OUT"
command -v mkcert >/dev/null || { echo "mkcert not found: brew install mkcert"; exit 1; }

IP="$(cd "$ROOT" && uv run python -c 'from libraryclaim.live.__main__ import lan_ip; print(lan_ip())')"
export CAROOT="$OUT/ca"
mkcert -cert-file "$OUT/cert.pem" -key-file "$OUT/key.pem" "$IP" localhost 127.0.0.1
cp "$CAROOT/rootCA.pem" "$OUT/rootCA.pem"
echo
echo "certificate for $IP written to $OUT"
echo "send $OUT/rootCA.pem to the phone and trust it (see the top of this script)"
echo "the private CA key stays in $CAROOT; never share that folder"
