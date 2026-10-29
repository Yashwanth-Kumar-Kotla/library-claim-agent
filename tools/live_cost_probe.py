"""Measure what a live session costs as it gets longer, with the app's real agent settings.

    uv run python tools/live_cost_probe.py --minutes 6 --every 30

Streams a saved sweep's frames at 1 fps (as the phone does), continuous silence as microphone audio,
and every N seconds asks the agent to say one sentence, then records the usage the Live API reports
for each reply. Writes data/cost_probe/<time>.json and prints cost by elapsed minute.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libraryclaim.config import REPO_ROOT, Settings  # noqa: E402
from libraryclaim.live.agent import live_config  # noqa: E402
from libraryclaim.live.usage import usage_entry  # noqa: E402


def frames(sweep: Path) -> list[bytes]:
    out = []
    for f in sorted((sweep / "frames").glob("*.jpg")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        s = 768 / img.shape[1]
        img = cv2.resize(img, (768, round(img.shape[0] * s)))
        out.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes())
    return out


FRAME_EVERY_S = 2.0  # matches the phone page's LIVE_MS


async def probe(minutes: float, every: float, sweep: Path) -> list[dict]:
    s = Settings.from_env()
    client = genai.Client(api_key=s.google_api_key)
    pics = frames(sweep)
    silence = np.zeros(1600, np.int16).tobytes()  # 100 ms of 16 kHz PCM
    log, t0 = [], time.monotonic()
    async with client.aio.live.connect(model=s.live_model, config=live_config()) as session:
        async def send_media():
            i = 0
            while True:
                await session.send_realtime_input(video=types.Blob(data=pics[i % len(pics)], mime_type="image/jpeg"))
                for _ in range(int(FRAME_EVERY_S * 10)):
                    await session.send_realtime_input(audio=types.Blob(data=silence, mime_type="audio/pcm;rate=16000"))
                    await asyncio.sleep(0.1)
                i += 1

        async def prompt():
            n = 0
            while True:
                text = ("[app] The policyholder has opened the app. Greet them in one sentence." if n == 0 else
                        "[app] Say one short sentence about what you can see now.")
                await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]),
                                                  turn_complete=True)
                n += 1
                await asyncio.sleep(every)

        async def receive():
            while True:
                async for m in session.receive():
                    if m.tool_call and m.tool_call.function_calls:
                        await session.send_tool_response(function_responses=[
                            types.FunctionResponse(id=fc.id, name=fc.name, response={"ok": True})
                            for fc in m.tool_call.function_calls])
                    if m.usage_metadata:
                        e = usage_entry(m.usage_metadata)
                        e["t"] = round(time.monotonic() - t0, 1)
                        log.append(e)
                        print(f"t={e['t']:6.1f}s  in={sum(e['prompt'].values()):7d} {e['prompt']}  "
                              f"out={sum(e['response'].values()):5d}  ${e['usd']:.4f}", flush=True)

        tasks = [asyncio.create_task(c()) for c in (send_media, prompt, receive)]
        await asyncio.sleep(minutes * 60)
        for t in tasks:
            t.cancel()
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--sweep", type=Path, default=REPO_ROOT / "data/sweeps/live-20261028-193109")
    args = ap.parse_args()
    log = asyncio.run(probe(args.minutes, args.every, args.sweep))
    out = REPO_ROOT / "data" / "cost_probe" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=1))
    total = sum(e["usd"] for e in log)
    print(f"\n{len(log)} usage reports, total ${total:.4f} over {args.minutes} min -> {out}")
    for minute in range(1, int(args.minutes) + 1):
        upto = sum(e["usd"] for e in log if e["t"] <= minute * 60)
        print(f"  by {minute} min: ${upto:.4f}")


if __name__ == "__main__":
    main()
