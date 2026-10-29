"""Phone <-> server <-> Gemini Live, plus the background pipeline.

    uv run python -m libraryclaim.live          # see __main__.py for HTTPS on the phone

One WebSocket per sweep. From the phone, binary messages start with a one-byte tag:
    A  microphone audio, PCM16 mono 16 kHz       -> Gemini Live
    V  low-resolution camera frame (JPEG, ~1/s)  -> Gemini Live, so the agent can see
    F  full-resolution frame (JPEG, ~3/s)        -> saved, scored, used for counting and reading
and JSON text messages for controls. To the phone: binary agent audio (PCM16 24 kHz, tag A) and
JSON events: transcripts, capture remarks, rows, books and items as they are identified, the end.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.genai import types

from ..archive import archive_run
from ..config import MODEL_PRICES, REPO_ROOT, Settings
from ..depth import DepthPro
from ..fetch import Fetcher
from ..pipeline import Deps
from ..vision import Meter, gemini_model
from .agent import dispatch, live_config
from .usage import usage_entry
from .session import LiveSweep

log = logging.getLogger("libraryclaim.live")
STATIC = Path(__file__).parent / "static"
SWEEPS = REPO_ROOT / "data" / "sweeps"
SWEEPS.mkdir(parents=True, exist_ok=True)


DEPTH = DepthPro()  # one model for the whole server; loading it takes a few seconds


def real_deps(settings: Settings) -> Deps:
    meter = Meter(MODEL_PRICES)
    token = settings.price_token()
    return Deps(vision=gemini_model(settings.google_api_key, settings.vision_model, meter=meter),
                escalate=gemini_model(settings.google_api_key, settings.vision_escalation_model, meter=meter, timeout_s=15),
                get_json=Fetcher(cache_dir=REPO_ROOT / "data/cache/http"), ebay_token=token,
                google_books_key=settings.google_books_key, meter=meter, depth=DEPTH)


def real_connect(settings: Settings):
    from google import genai
    client = genai.Client(api_key=settings.google_api_key)
    return lambda config: client.aio.live.connect(model=settings.live_model, config=config)


app = FastAPI(on_startup=[lambda: app.state.deps is real_deps and DEPTH.warm_up()])
app.state.settings = None
app.state.deps = real_deps        # replaced in tests
app.state.connect = real_connect  # replaced in tests


def settings() -> Settings:
    if app.state.settings is None:
        app.state.settings = Settings.from_env()
    return app.state.settings


@app.middleware("http")
async def revalidate(request, call_next):
    """Phones keep yesterday's page otherwise: make the browser re-check the page files on every load."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(("/static/", "/sweeps/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/check")
def check():
    """Open this on the phone first: proves it reaches the Mac and can use camera, mic and motion."""
    return FileResponse(STATIC / "check.html")


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/sweeps", StaticFiles(directory=SWEEPS), name="sweeps")


@app.websocket("/ws")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    s = settings()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()

    sweep_id = datetime.now().strftime("live-%Y%m%d-%H%M%S")
    log_path = SWEEPS / sweep_id / "out" / "session.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()
    t0 = time.monotonic()

    def record(entry: dict) -> None:
        """Everything said and done in the session, for debugging and the failure log."""
        with log_lock, log_path.open("a") as f:
            f.write(json.dumps({"t": round(time.monotonic() - t0, 2), **entry}, default=str) + "\n")

    def emit(kind: str, data: dict) -> None:  # called from pipeline worker threads too
        msg = {"type": kind, **data}
        record(msg)
        loop.call_soon_threadsafe(outbox.put_nowait, msg)

    async def say(session, text: str) -> None:
        record({"type": "app_to_agent", "text": text})
        await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]),
                                          turn_complete=True)

    sweep = LiveSweep(SWEEPS / sweep_id, app.state.deps(s), s, on_event=emit)
    speaking = {"now": False}   # is the agent's voice currently streaming?
    spend = {"live": 0.0}       # what the voice agent has cost so far, from the Live API's own usage reports
    shown = {"hint": None}
    emit("session", {"sweep": sweep_id})
    tasks: set[asyncio.Task] = set()

    async def to_phone():
        while True:
            msg = await outbox.get()
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_text(json.dumps(msg, default=str))

    async def run_tool(fc, session):
        name, args = str(fc.name or ""), dict(fc.args or {})
        emit("tool", {"name": name, "args": args, "phase": "running"})
        try:
            result, urgent = await asyncio.to_thread(dispatch, sweep, name, args)
        except Exception as exc:  # noqa: BLE001 - the agent is told, the call goes on
            log.exception("tool %s failed", name)
            result, urgent = {"ok": False, "error": f"{name} failed: {exc}"}, False
        emit("tool", {"name": name, "args": args, "phase": "done", "result": result})
        if name == "end_sweep" and result.get("ok"):
            emit("done", {"report": result["report"], "summary": result["summary"]})
        scheduling = types.FunctionResponseScheduling.INTERRUPT if urgent else types.FunctionResponseScheduling.WHEN_IDLE
        await session.send_tool_response(function_responses=[
            types.FunctionResponse(id=fc.id, name=name, response=result, scheduling=scheduling)])

    async def from_phone(session):
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            data = msg.get("bytes")
            if data:
                tag, body = data[:1], data[1:]
                if tag == b"A":
                    await session.send_realtime_input(audio=types.Blob(data=body, mime_type="audio/pcm;rate=16000"))
                elif tag == b"V":
                    await session.send_realtime_input(video=types.Blob(data=body, mime_type="image/jpeg"))
                elif tag in (b"F", b"G"):
                    orientation = None
                    if tag == b"G" and len(body) > 12:  # G: alpha, beta, gamma as 3 float32, then the JPEG
                        orientation, body = struct.unpack("<3f", body[:12]), body[12:]
                    spoken, screen = await asyncio.to_thread(sweep.add_frame, body, orientation)
                    if screen and screen != shown["hint"]:
                        emit("coach", {"text": screen})  # banner only; the agent is not told
                    shown["hint"] = screen
                    # Never talk over the agent: a remark that arrives while it speaks is dropped.
                    if speaking["now"]:
                        continue
                    if spoken:
                        await say(session, f"[capture] {spoken}")
                    elif (hint := sweep.nudge()):
                        await say(session, f"[app] {hint}")
                continue
            text = msg.get("text")
            if text:
                ctl = json.loads(text)
                if ctl.get("type") == "end":
                    await say(session, "[app] The policyholder pressed End. Call end_sweep now.")

    async def from_gemini(session):
        while True:
            async for m in session.receive():
                if m.usage_metadata:
                    entry = usage_entry(m.usage_metadata)
                    spend["live"] = round(spend["live"] + entry["usd"], 6)
                    record({"type": "live_usage", **entry, "live_usd_so_far": spend["live"]})
                if m.tool_call and m.tool_call.function_calls:
                    for fc in m.tool_call.function_calls:
                        t = asyncio.create_task(run_tool(fc, session))
                        tasks.add(t)
                        t.add_done_callback(tasks.discard)
                c = m.server_content
                if not c:
                    continue
                if c.interrupted:
                    speaking["now"] = False
                    emit("interrupted", {})
                if c.turn_complete:
                    speaking["now"] = False
                for who, chunk in (("you", c.input_transcription), ("agent", c.output_transcription)):
                    if chunk and chunk.text:
                        emit("transcript", {"who": who, "text": chunk.text})
                if c.model_turn and not c.interrupted:
                    for part in c.model_turn.parts or []:
                        if part.inline_data and (part.inline_data.mime_type or "").startswith("audio/pcm"):
                            speaking["now"] = True
                            outbox.put_nowait(b"A" + part.inline_data.data)

    sender = asyncio.create_task(to_phone())
    try:
        async with app.state.connect(s)(live_config()) as session:
            await say(session, "[app] The policyholder has opened the app and the camera is on. Greet them.")
            pair = [asyncio.create_task(from_phone(session)), asyncio.create_task(from_gemini(session))]
            done, _ = await asyncio.wait(pair, return_when=asyncio.FIRST_COMPLETED)
            for t in pair:
                t.cancel()
            for t in done:
                if not isinstance(t.exception(), WebSocketDisconnect):
                    t.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.exception("live session failed")
        with contextlib.suppress(Exception):
            await ws.send_text(json.dumps({"type": "error", "message": f"live session ended: {exc}"}))
    finally:
        record({"type": "live_cost", "usd": spend["live"]})
        # The browser closed or the connection dropped before end_sweep: keep what was captured.
        # Started before any other await and shielded: the handler is often being cancelled here,
        # and the worker thread finishes the packet even if this coroutine does not resume.
        # Archived last, so the copy has the session log with its final cost line.
        def close_out():
            if not sweep.ended and sweep.n:
                record({"type": "closed_early", **sweep.end(True, True)})
            if sweep.ended:
                archive_run(sweep.root, "live")
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(asyncio.to_thread(close_out))
        for t in list(tasks):
            t.cancel()
        sender.cancel()
        with contextlib.suppress(Exception):
            await ws.close()
