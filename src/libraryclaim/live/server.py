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
import re
import threading
from dataclasses import dataclass, field
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
SESSION_WARN_S = 13 * 60    # a spoken two-minute warning ...
SESSION_CAP_S = 15 * 60     # ... then the agent is told to end ...
SESSION_GRACE_S = 60        # ... and if it hasn't, the app ends the session itself
RECONNECT_GRACE_S = 90      # a dropped phone has this long to come back before the packet is built
HISTORY_TURNS = 40          # conversation replayed into a new Gemini session on reconnect
MAX_RUNNING_TOOLS = 4
MARKUP = re.compile(r"<!--.*?-->|<[^>]{1,40}>")  # stray markup the model sometimes puts in its transcript
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


@dataclass
class Sweep:
    """Everything that outlives one WebSocket connection, so a dropped phone can come back to the same
    sweep: the sweep itself, its log, what the policyholder has seen, and the conversation so far."""
    id: str
    t0: float
    log_path: Path
    sweep: LiveSweep | None = None
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    outbox: asyncio.Queue | None = None          # the current connection's; swapped on reconnect
    loop: asyncio.AbstractEventLoop | None = None
    attached: bool = False
    detached_at: float = 0.0
    grace: asyncio.Task | None = None
    spend: float = 0.0
    shown: dict = field(default_factory=lambda: {"hint": None})
    lines: dict = field(default_factory=dict)    # latest book/item message per id, resent on reconnect
    history: list = field(default_factory=list)  # (role, text) turns, replayed into a new Gemini session
    pending: dict = field(default_factory=lambda: {"user": "", "model": ""})
    tasks: set = field(default_factory=set)
    running: dict = field(default_factory=dict)  # tool call id -> task
    cancelled: set = field(default_factory=set)  # calls Gemini withdrew: their work finishes, no answer is sent
    seen_calls: set = field(default_factory=set)
    warned: bool = False
    capped: bool = False

    def record(self, entry: dict) -> None:
        """Everything said and done in the session, for debugging and the failure log."""
        with self.log_lock, self.log_path.open("a") as f:
            f.write(json.dumps({"t": round(time.monotonic() - self.t0, 2), **entry}, default=str) + "\n")

    def emit(self, kind: str, data: dict) -> None:  # called from pipeline worker threads too
        msg = {"type": kind, **data}
        self.record(msg)
        if kind in ("book", "item"):
            self.lines[data.get("id")] = msg
        if kind == "done":
            self.shown["done_msg"] = msg
        if self.attached and self.loop:
            self.loop.call_soon_threadsafe(self.outbox.put_nowait, msg)

    def heard(self, role: str, text: str) -> None:
        other = "model" if role == "user" else "user"
        if self.pending[other].strip():
            self.finish_turn(other)
        self.pending[role] += text

    def finish_turn(self, *roles: str) -> None:
        for role in roles or ("user", "model"):
            if self.pending[role].strip():
                self.history = (self.history + [(role, self.pending[role].strip())])[-HISTORY_TURNS:]
            self.pending[role] = ""

    def where(self) -> str:
        """Where the sweep stands, for an agent picking up after a reconnect."""
        sw = self.sweep
        done = ", ".join(sw.rows_done) or "none yet"
        if sw.row:
            now = f"shelf {sw.row.shelf} is still open; ask them to carry on panning it from where they were"
        elif sw.scanning:
            now = "the room turn is in progress; ask them to carry on following the ring"
        else:
            now = "no shelf is open; carry on with the next step"
        return f"Shelves done: {done}. Now: {now}."


ACTIVE: dict[str, Sweep] = {}


@app.websocket("/ws")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    s = settings()
    rid = ws.query_params.get("resume", "")
    st = ACTIVE.get(rid)
    if rid and st is None:
        # Closed already (time limit, or the phone was away too long): say so rather than start a new sweep.
        report = SWEEPS / Path(rid).name / "report.html"
        await ws.send_text(json.dumps({"type": "gone", "report": f"/sweeps/{Path(rid).name}/report.html"
                                       if report.is_file() else None}))
        await ws.close()
        return
    if st and not st.attached and st.sweep.ended:
        # It finished while the phone was away: hand over the report instead of starting a new sweep.
        if st.shown.get("done_msg"):
            await ws.send_text(json.dumps(st.shown["done_msg"], default=str))
        await ws.close()
        return
    resumed = bool(st and not st.attached)
    if resumed:
        if st.grace:
            st.grace.cancel()
        away = time.monotonic() - st.detached_at
    else:
        sweep_id = datetime.now().strftime("live-%Y%m%d-%H%M%S")
        n = 1
        while (SWEEPS / sweep_id).exists() or sweep_id in ACTIVE:  # two starts in one second
            n += 1
            sweep_id = datetime.now().strftime("live-%Y%m%d-%H%M%S") + f"-{n}"
        log_path = SWEEPS / sweep_id / "out" / "session.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        st = Sweep(sweep_id, time.monotonic(), log_path)
        st.sweep = LiveSweep(SWEEPS / sweep_id, app.state.deps(s), s, on_event=st.emit)
        ACTIVE[sweep_id] = st
    st.loop, st.outbox, st.attached = asyncio.get_running_loop(), asyncio.Queue(), True
    sweep, outbox = st.sweep, st.outbox
    speaking = {"now": False}   # is the agent's voice currently streaming?
    st.emit("session", {"sweep": st.id, "resumed": resumed})
    if resumed:
        st.record({"type": "reconnected", "away_s": round(away, 1)})
        for msg in st.lines.values():  # whatever the phone missed while it was away, after "session"
            st.loop.call_soon_threadsafe(outbox.put_nowait, msg)

    async def say(session, text: str, reply: bool = True) -> None:
        """reply=False adds context without asking for a reply: what the screen shows, what is being
        built. Forcing a reply to every app message made the agent talk over the policyholder."""
        st.record({"type": "app_to_agent", "text": text, "reply": reply})
        await session.send_client_content(turns=types.Content(role="user", parts=[types.Part(text=text)]),
                                          turn_complete=reply)

    async def to_phone():
        while True:
            msg = await outbox.get()
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_text(json.dumps(msg, default=str))

    async def run_tool(fc, session):
        name, args = str(fc.name or ""), dict(fc.args or {})
        st.emit("tool", {"name": name, "args": args, "phase": "running"})
        try:
            result, urgent = await asyncio.to_thread(dispatch, sweep, name, args)
        except Exception as exc:  # noqa: BLE001 - the agent is told, the call goes on
            log.exception("tool %s failed", name)
            result, urgent = {"ok": False, "error": f"{name} failed: {exc}"}, False
        st.emit("tool", {"name": name, "args": args, "phase": "done", "result": result})
        if name == "end_sweep" and result.get("report") and not st.shown.get("done"):
            st.shown["done"] = True
            st.emit("done", {"report": result["report"], "summary": result["summary"], "figures": result.get("figures", {})})
        if str(fc.id) in st.cancelled:
            return
        scheduling = types.FunctionResponseScheduling.INTERRUPT if urgent else types.FunctionResponseScheduling.WHEN_IDLE
        with contextlib.suppress(Exception):  # the connection this call came from may be gone
            await session.send_tool_response(function_responses=[
                types.FunctionResponse(id=fc.id, name=name, response=result, scheduling=scheduling)])

    async def launch_tool(fc, session):
        key = str(fc.id)
        if key in st.running or key in st.seen_calls:
            return  # the same call delivered twice
        st.seen_calls.add(key)
        if len(st.running) >= MAX_RUNNING_TOOLS:
            await session.send_tool_response(function_responses=[types.FunctionResponse(
                id=fc.id, name=fc.name, response={"ok": False, "error": "busy", "next_step": "Wait a moment, then try again."})])
            return
        t = asyncio.create_task(run_tool(fc, session))
        st.running[key] = t
        st.tasks.add(t)
        t.add_done_callback(lambda _, k=key: (st.running.pop(k, None), st.tasks.discard(_)))

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
                    if screen and screen != st.shown["hint"]:
                        st.emit("coach", {"text": screen})
                        # The agent knows what the screen says, so it doesn't repeat it or contradict it.
                        await say(session, f"[app state] The screen is showing them: {screen}. Do not repeat it.", reply=False)
                    st.shown["hint"] = screen
                    # Never talk over the agent: a remark that arrives while it speaks is dropped.
                    if speaking["now"]:
                        continue
                    if spoken:
                        await say(session, f"[capture] {spoken}")
                    elif (hint := sweep.scan_prompt()):
                        await say(session, f"[app] {hint}")
                    elif (hint := sweep.nudge()):
                        await say(session, f"[app] {hint}")
                continue
            text = msg.get("text")
            if text:
                ctl = json.loads(text)
                if ctl.get("type") == "end":
                    if sweep.ended:  # already being built: asking again made the agent call end_sweep twice
                        await say(session, "[app state] The policyholder pressed End; the packet is already being built.", reply=False)
                    else:
                        await say(session, "[app] The policyholder pressed End. Call end_sweep now.")

    async def from_gemini(session):
        while True:
            async for m in session.receive():
                if m.usage_metadata:
                    entry = usage_entry(m.usage_metadata)
                    st.spend = round(st.spend + entry["usd"], 6)
                    st.record({"type": "live_usage", **entry, "live_usd_so_far": st.spend})
                if m.tool_call and m.tool_call.function_calls:
                    for fc in m.tool_call.function_calls:
                        await launch_tool(fc, session)
                if m.tool_call_cancellation:
                    # Our tools change the sweep in a worker thread, so the work is not undone; the
                    # result is just not sent back. end_sweep still finishes and shows the report.
                    for call_id in m.tool_call_cancellation.ids or []:
                        st.cancelled.add(str(call_id))
                        st.record({"type": "tool_cancelled", "id": str(call_id)})
                c = m.server_content
                if not c:
                    continue
                if c.interrupted:
                    speaking["now"] = False
                    st.emit("interrupted", {})
                for who, chunk in (("you", c.input_transcription), ("agent", c.output_transcription)):
                    if who == "agent" and c.interrupted:
                        continue  # cut-off speech: never shown or logged as said
                    text = MARKUP.sub("", chunk.text) if chunk and chunk.text else ""
                    if text:
                        st.emit("transcript", {"who": who, "text": text})
                        st.heard("user" if who == "you" else "model", text)
                if c.turn_complete:
                    speaking["now"] = False
                    st.finish_turn()
                    st.emit("turn_end", {})  # separate turns stay separate in captions and the log
                if c.model_turn and not c.interrupted:
                    for part in c.model_turn.parts or []:
                        if part.inline_data and (part.inline_data.mime_type or "").startswith("audio/pcm"):
                            speaking["now"] = True
                            outbox.put_nowait(b"A" + part.inline_data.data)

    async def time_cap(session):
        """A backstop on length and cost, counted from the start of the sweep, not the connection: warn,
        ask the agent to end, then end it ourselves (the packet is built from what was captured)."""
        async def until(t):
            await asyncio.sleep(max(0.0, st.t0 + t - time.monotonic()))
        await until(SESSION_WARN_S)
        if not st.warned:
            st.warned = True
            await say(session, "[app] About two minutes are left in this session. Tell them briefly and help them finish.")
        await until(SESSION_CAP_S)
        if not sweep.ended:
            await say(session, "[app] The session time limit is reached. Call end_sweep with confirmed true now.")
        await until(SESSION_CAP_S + SESSION_GRACE_S)
        if not sweep.ended:
            st.capped = True
            st.record({"type": "time_cap"})

    sender = asyncio.create_task(to_phone())
    try:
        # On a reconnect the new Gemini session starts with the conversation so far (sent as history,
        # not as a request to speak), then hears where the sweep stands.
        history = [types.Content(role=r, parts=[types.Part(text=t)]) for r, t in st.history] if resumed else []
        async with app.state.connect(s)(live_config(seed_history=bool(history))) as session:
            if history:
                await session.send_client_content(turns=history, turn_complete=True)
            if resumed:
                await say(session, f"[app] The connection dropped for {away:.0f} seconds and is back. {st.where()} "
                                   "Say in a few words that you are back, then carry on.")
            else:
                await say(session, "[app] The policyholder has opened the app and the camera is on. Greet them.")
            pair = [asyncio.create_task(from_phone(session)), asyncio.create_task(from_gemini(session)),
                    asyncio.create_task(time_cap(session))]
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
        st.attached, st.detached_at = False, time.monotonic()
        st.finish_turn()
        sender.cancel()
        if sweep.ended or st.capped or not sweep.n or RECONNECT_GRACE_S <= 0:
            # Finished, out of time, or nothing captured: close it out now.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(asyncio.to_thread(close_out, st))
        else:
            # The phone dropped mid-sweep: give it time to come back before building the packet.
            st.record({"type": "disconnected"})
            st.grace = asyncio.get_running_loop().create_task(close_after_grace(st))
        with contextlib.suppress(Exception):
            await ws.close()


def close_out(st: Sweep) -> None:
    """Build the packet from what was captured (if the sweep wasn't ended), then archive. Runs in a worker
    thread, shielded: the handler is often being cancelled here and the packet must still be written.
    Archived last, so the copy has the session log with its final cost line."""
    if ACTIVE.pop(st.id, None) is None:
        return  # already closed out
    st.record({"type": "live_cost", "usd": st.spend})
    sweep = st.sweep
    if not sweep.ended and sweep.n:
        st.record({"type": "closed_early", **sweep.end(True, True)})
    if sweep.ended:
        archive_run(sweep.root, "live")
    for t in list(st.tasks):
        st.loop.call_soon_threadsafe(t.cancel)  # this runs in a worker thread


async def close_after_grace(st: Sweep) -> None:
    await asyncio.sleep(RECONNECT_GRACE_S)
    if not st.attached:
        await asyncio.to_thread(close_out, st)
