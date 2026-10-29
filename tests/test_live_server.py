"""The live path end to end, with a scripted stand-in for Gemini Live (no network, no cost)."""

import asyncio
import json
import time

import cv2
import pytest
from fastapi.testclient import TestClient
from google.genai import types

from libraryclaim.config import Locale, Settings
from libraryclaim.live import server
from libraryclaim.pipeline import Deps
from helpers import FakeWeb, textured
from synthetic import ROWS, FakeVision, draw_row, fake_catalogue


class FakeLive:
    """Records what the server sends; the test pushes what 'Gemini' says."""

    def __init__(self):
        self.audio, self.video, self.texts, self.tool_responses = 0, 0, [], []
        self.loop = None
        self.inbox = None

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        self.inbox = asyncio.Queue()
        return self

    async def __aexit__(self, *exc):
        return False

    async def send_realtime_input(self, audio=None, video=None):
        self.audio += audio is not None
        self.video += video is not None

    async def send_client_content(self, turns, turn_complete):
        self.texts.append(turns.parts[0].text)

    async def send_tool_response(self, function_responses):
        self.tool_responses += function_responses

    async def receive(self):
        while True:
            yield await self.inbox.get()

    def push(self, msg):
        while self.loop is None:
            time.sleep(0.01)
        self.loop.call_soon_threadsafe(self.inbox.put_nowait, msg)


def call(name, **args):
    return types.LiveServerMessage(tool_call=types.LiveServerToolCall(
        function_calls=[types.FunctionCall(id=f"c-{name}-{time.monotonic_ns()}", name=name, args=args)]))


def jpeg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


@pytest.fixture
def live(tmp_path, monkeypatch):
    fake = FakeLive()
    monkeypatch.setattr(server, "SWEEPS", tmp_path)
    server.app.state.settings = Settings(locale=Locale("US", "USD"))
    server.app.state.deps = lambda s: Deps(vision=FakeVision(), get_json=FakeWeb(fake_catalogue()), ebay_token="t")
    server.app.state.connect = lambda s: (lambda config: fake)
    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        yield ws, fake, tmp_path


def receive(ws, timeout=10.0):
    """ws.receive() with a deadline, so a missing event fails the test instead of hanging it."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        try:
            return pool.submit(ws.receive).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise AssertionError("no message from the server within 10 s") from None


def frames_saved(root, n, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if sum(len(list((d / "frames").glob("*.jpg"))) for d in root.glob("live-*")) >= n:
            return
        time.sleep(0.02)
    raise AssertionError(f"{n} frames were not saved in time")


def wait_for(ws, kind, pred=lambda m: True, limit=400, fail_on=("row_rejected", "error")):
    seen = []
    for _ in range(limit):
        m = receive(ws)
        if m.get("bytes") is not None:
            seen.append(("audio", len(m["bytes"])))
            continue
        msg = json.loads(m["text"])
        seen.append(msg["type"])
        if msg["type"] == kind and pred(msg):
            return msg, seen
        if msg["type"] in fail_on and kind not in fail_on:
            raise AssertionError(f"got {msg['type']} while waiting for {kind}: {msg}")
    raise AssertionError(f"no {kind} in {seen[-20:]}")


def test_page_served():
    with TestClient(server.app) as c:
        assert "Start" in c.get("/").text and c.get("/static/app.js").status_code == 200


def test_full_sweep_over_the_socket(live):
    ws, fake, root = live
    wait_for(ws, "session")
    assert "Greet them" in fake.texts[0]

    fake.push(call("set_locale", country="US", currency="USD"))
    msg, _ = wait_for(ws, "tool", lambda m: m["name"] == "set_locale" and m["phase"] == "done")
    assert msg["result"]["ok"]

    fake.push(call("start_row", row="A1"))
    wait_for(ws, "row_open")
    row = jpeg(draw_row(ROWS["A1"])[0])
    for _ in range(3):
        ws.send_bytes(b"F" + row)
    ws.send_bytes(b"A" + b"\x00\x00" * 160)
    ws.send_bytes(b"V" + jpeg(textured(240, 320)))
    frames_saved(root, 3)  # frames and tool calls travel on different tasks; close the row after they land
    fake.push(call("finish_row"))
    closed, _ = wait_for(ws, "row_closed")
    assert closed["ok"] and closed["rows_done"] == ["A1"]
    book, _ = wait_for(ws, "book", lambda m: m.get("status") == "identified")  # inventory fills during the sweep
    assert book["id"].startswith("A1-")

    # The agent speaks: its audio reaches the phone as tagged binary.
    fake.push(types.LiveServerMessage(server_content=types.LiveServerContent(model_turn=types.Content(
        role="model", parts=[types.Part(inline_data=types.Blob(data=b"\x01\x02" * 50, mime_type="audio/pcm;rate=24000"))]))))
    got_audio = False
    for _ in range(200):
        m = receive(ws)
        if m.get("bytes") is not None:
            assert m["bytes"][:1] == b"A" and len(m["bytes"]) == 101
            got_audio = True
            break
    assert got_audio and fake.audio >= 1 and fake.video >= 1

    fake.push(call("end_sweep"))
    done, _ = wait_for(ws, "done")
    assert "books counted" in done["summary"] and done["report"].endswith("report.html")
    sweep_dir = next(p for p in root.iterdir() if p.name.startswith("live-"))
    packet = json.loads((sweep_dir / "claim_packet.json").read_text())
    assert len(packet["books"]) == len(ROWS["A1"])
    names = [r.name for r in fake.tool_responses]
    assert names == ["set_locale", "start_row", "finish_row", "end_sweep"]
    assert all(r.scheduling == types.FunctionResponseScheduling.WHEN_IDLE for r in fake.tool_responses)


def test_bad_row_interrupts_the_agent_and_blur_is_coached(live):
    ws, fake, root = live
    wait_for(ws, "session")
    fake.push(call("start_row", row="B1"))
    wait_for(ws, "row_open")
    blurry = jpeg(cv2.GaussianBlur(textured(), (31, 31), 12))
    for _ in range(4):
        ws.send_bytes(b"F" + blurry)
    coach, _ = wait_for(ws, "coach")  # the banner shows it; the agent is not interrupted for 1 s of blur
    frames_saved(root, 4)
    assert "blurry" in coach["text"]
    fake.push(call("finish_row"))
    wait_for(ws, "row_rejected")
    for _ in range(100):
        if fake.tool_responses:
            break
        time.sleep(0.02)
    r = fake.tool_responses[-1]
    assert r.name == "finish_row" and r.scheduling == types.FunctionResponseScheduling.INTERRUPT
    assert not any(t.startswith("[capture]") for t in fake.texts)


def test_end_button_asks_the_agent_to_finish(live):
    ws, fake, _ = live
    wait_for(ws, "session")
    ws.send_text(json.dumps({"type": "end"}))
    for _ in range(100):
        if any("pressed End" in t for t in fake.texts):
            break
        time.sleep(0.02)
    assert any("pressed End" in t for t in fake.texts)


def test_unknown_tool_and_garbage_frames_do_not_break_the_session(live):
    ws, fake, _ = live
    wait_for(ws, "session")
    ws.send_bytes(b"F" + b"not a jpeg")
    ws.send_bytes(b"Z" + b"unknown tag")
    fake.push(call("delete_everything"))
    msg, _ = wait_for(ws, "tool", lambda m: m["phase"] == "done")
    assert msg["result"]["ok"] is False
    fake.push(call("start_row", row="A1"))
    wait_for(ws, "row_open")  # still alive


def test_session_log_and_packet_when_browser_closes(tmp_path, monkeypatch):
    fake = FakeLive()
    monkeypatch.setattr(server, "SWEEPS", tmp_path)
    server.app.state.settings = Settings(locale=Locale("US", "USD"))
    server.app.state.deps = lambda s: Deps(vision=FakeVision(), get_json=FakeWeb(fake_catalogue()), ebay_token="t")
    server.app.state.connect = lambda s: (lambda config: fake)
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            wait_for(ws, "session")
            ws.send_bytes(b"F" + jpeg(textured()))
            frames_saved(tmp_path, 1)
        # browser gone without end_sweep
        sweep = next(tmp_path.glob("live-*"))
        for _ in range(200):
            if (sweep / "claim_packet.json").is_file():
                break
            time.sleep(0.02)
    assert (sweep / "claim_packet.json").is_file()
    log = [json.loads(l) for l in (sweep / "out" / "session.jsonl").read_text().splitlines()]
    kinds = [e["type"] for e in log]
    assert kinds[0] == "session" and "app_to_agent" in kinds and "closed_early" in kinds


def test_orientation_tagged_frames_reach_the_scan(live):
    import struct
    ws, fake, root = live
    wait_for(ws, "session")
    fake.push(call("start_room_scan"))
    wait_for(ws, "scan_open")
    img = jpeg(textured())
    for a in range(0, 360, 20):
        ws.send_bytes(b"G" + struct.pack("<3f", float(a), 90.0, 0.0) + img)
    frames_saved(root, 18)
    fake.push(call("finish_room_scan"))
    msg, _ = wait_for(ws, "tool", lambda m: m["name"] == "finish_room_scan" and m["phase"] == "done")
    assert msg["result"]["covered_deg"] >= 300 and "tilt" in msg["result"]["advice"]
    for _ in range(100):
        if any(r.name == "finish_room_scan" for r in fake.tool_responses):
            break
        time.sleep(0.02)
    r = next(r for r in fake.tool_responses if r.name == "finish_room_scan")
    assert r.scheduling == types.FunctionResponseScheduling.INTERRUPT  # redo needed while they are still there
