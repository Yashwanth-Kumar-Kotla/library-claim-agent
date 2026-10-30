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
        self.replies = []  # turn_complete flag of each text, in order
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
        if isinstance(turns, list):  # conversation history replayed after a reconnect
            self.history = [(t.role, t.parts[0].text) for t in turns]
            return
        self.texts.append(turns.parts[0].text)
        self.replies.append(turn_complete)

    async def send_tool_response(self, function_responses):
        self.tool_responses += function_responses

    async def receive(self):
        while True:
            yield await self.inbox.get()

    def push(self, msg):
        while self.loop is None:
            time.sleep(0.01)
        self.loop.call_soon_threadsafe(self.inbox.put_nowait, msg)


def call(name, call_id=None, **args):
    return types.LiveServerMessage(tool_call=types.LiveServerToolCall(
        function_calls=[types.FunctionCall(id=call_id or f"c-{name}-{time.monotonic_ns()}", name=name, args=args)]))


def jpeg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


@pytest.fixture
def live(tmp_path, monkeypatch):
    fake = FakeLive()
    monkeypatch.setattr(server, "SWEEPS", tmp_path)
    monkeypatch.setattr(server, "RECONNECT_GRACE_S", 0)  # closing the test client ends the sweep at once
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
    monkeypatch.setattr(server, "RECONNECT_GRACE_S", 0.2)  # it never comes back
    archived = []
    monkeypatch.setattr(server, "archive_run", lambda root, kind: archived.append((root, kind, (root / "out" / "session.jsonl").read_text())))
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
            if archived:
                break
            time.sleep(0.02)
    assert (sweep / "claim_packet.json").is_file()
    log = [json.loads(l) for l in (sweep / "out" / "session.jsonl").read_text().splitlines()]
    kinds = [e["type"] for e in log]
    assert kinds[0] == "session" and "app_to_agent" in kinds and "closed_early" in kinds
    # archived once, after the packet and with the session's final cost line in the log
    [(root, kind, logged)] = archived
    assert root == sweep and kind == "live" and '"live_cost"' in logged and '"closed_early"' in logged


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


def responses(fake, name, n, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = [r for r in fake.tool_responses if r.name == name]
        if len(got) >= n:
            return got
        time.sleep(0.02)
    return [r for r in fake.tool_responses if r.name == name]


def test_greeting_asks_for_a_reply(live):
    ws, fake, _ = live
    wait_for(ws, "session")
    assert "Greet" in fake.texts[0] and fake.replies[0] is True


def test_every_tool_result_says_what_to_do_next(live):
    ws, fake, _ = live
    wait_for(ws, "session")
    fake.push(call("start_row", row="A1"))
    fake.push(call("finish_row"))       # too few frames: advice
    fake.push(call("record_remark", kind="signed", said="that one is signed"))  # no title: error
    for name in ("start_row", "finish_row", "record_remark"):
        [r] = responses(fake, name, 1)
        assert r.response["next_step"]


def test_duplicate_tool_call_runs_once(live):
    ws, fake, _ = live
    wait_for(ws, "session")
    fake.push(call("start_row", call_id="same", row="A1"))
    fake.push(call("start_row", call_id="same", row="A1"))
    wait_for(ws, "tool", lambda m: m["phase"] == "done")
    time.sleep(0.3)
    assert len(responses(fake, "start_row", 1)) == 1


def test_cancelled_end_gets_no_answer_but_the_report_still_appears(live):
    ws, fake, root = live
    wait_for(ws, "session")
    ws.send_bytes(b"F" + jpeg(textured()))
    frames_saved(root, 1)
    msg = call("end_sweep", call_id="end-1", confirmed=True)
    msg.tool_call_cancellation = types.LiveServerToolCallCancellation(ids=["end-1"])
    fake.push(msg)
    wait_for(ws, "done")
    time.sleep(0.3)
    assert responses(fake, "end_sweep", 1, timeout=0.5) == []


def test_end_pressed_while_the_packet_builds_is_context_only(live):
    ws, fake, root = live
    wait_for(ws, "session")
    ws.send_bytes(b"F" + jpeg(textured()))
    frames_saved(root, 1)
    fake.push(call("end_sweep", confirmed=True))
    wait_for(ws, "done")
    ws.send_text(json.dumps({"type": "end"}))
    for _ in range(100):
        if "pressed End" in fake.texts[-1]:
            break
        time.sleep(0.02)
    assert "already being built" in fake.texts[-1] and fake.replies[-1] is False


def test_transcripts_drop_cut_off_speech_and_markup_and_mark_turn_ends(live):
    ws, fake, _ = live
    wait_for(ws, "session")
    T = types.Transcription
    fake.push(types.LiveServerMessage(server_content=types.LiveServerContent(
        output_transcription=T(text="Got it.<!-- Silent wait for user -->"))))
    msg, _ = wait_for(ws, "transcript")
    assert msg["text"] == "Got it."
    fake.push(types.LiveServerMessage(server_content=types.LiveServerContent(
        interrupted=True, output_transcription=T(text="and now the next"))))
    fake.push(types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True)))
    _, seen = wait_for(ws, "turn_end")
    assert "transcript" not in seen  # the cut-off words never arrive


def test_time_cap_warns_then_asks_to_end_then_ends_and_keeps_the_packet(tmp_path, monkeypatch):
    fake = FakeLive()
    monkeypatch.setattr(server, "SWEEPS", tmp_path)
    monkeypatch.setattr(server, "SESSION_WARN_S", 0.3)
    monkeypatch.setattr(server, "SESSION_CAP_S", 0.6)
    monkeypatch.setattr(server, "SESSION_GRACE_S", 0.3)
    monkeypatch.setattr(server, "archive_run", lambda *a: None)
    server.app.state.settings = Settings(locale=Locale("US", "USD"))
    server.app.state.deps = lambda s: Deps(vision=FakeVision(), get_json=FakeWeb(fake_catalogue()), ebay_token="t")
    server.app.state.connect = lambda s: (lambda config: fake)
    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        wait_for(ws, "session")
        ws.send_bytes(b"F" + jpeg(textured()))
        frames_saved(tmp_path, 1)
        kinds = []
        for _ in range(400):  # the server closes the socket at the cap
            m = receive(ws, timeout=5)
            kinds.append(m.get("type"))
            if m.get("type") == "websocket.close":
                break
        assert kinds[-1] == "websocket.close"
    sweep = next(tmp_path.glob("live-*"))
    for _ in range(200):
        if (sweep / "claim_packet.json").is_file():
            break
        time.sleep(0.02)
    assert any("two minutes" in t for t in fake.texts) and any("time limit" in t for t in fake.texts)
    assert (sweep / "claim_packet.json").is_file()
    assert '"time_cap"' in (sweep / "out" / "session.jsonl").read_text()


@pytest.fixture
def phone(tmp_path, monkeypatch):
    """A test client whose phone can drop and come back; configs records each Gemini session's setup."""
    fake, configs = FakeLive(), []
    monkeypatch.setattr(server, "SWEEPS", tmp_path)
    monkeypatch.setattr(server, "RECONNECT_GRACE_S", 30)
    monkeypatch.setattr(server, "archive_run", lambda *a: None)
    server.app.state.settings = Settings(locale=Locale("US", "USD"))
    server.app.state.deps = lambda s: Deps(vision=FakeVision(), get_json=FakeWeb(fake_catalogue()), ebay_token="t")
    server.app.state.connect = lambda s: (lambda config: (configs.append(config), fake)[1])
    with TestClient(server.app) as client:
        yield client, fake, configs, tmp_path
        for st in list(server.ACTIVE.values()):  # don't leave a sweep waiting for its phone
            server.close_out(st)


def test_dropped_phone_resumes_the_same_sweep_and_conversation(phone):
    client, fake, configs, root = phone
    T = types.Transcription
    with client.websocket_connect("/ws") as ws:
        sid = wait_for(ws, "session")[0]["sweep"]
        fake.push(types.LiveServerMessage(server_content=types.LiveServerContent(input_transcription=T(text="Hi there"))))
        fake.push(types.LiveServerMessage(server_content=types.LiveServerContent(output_transcription=T(text="Welcome."))))
        fake.push(types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True)))
        wait_for(ws, "turn_end")
        fake.push(call("start_row", row="A1"))
        wait_for(ws, "row_open")
        ws.send_bytes(b"F" + jpeg(textured()))
        frames_saved(root, 1)
    # Wi-Fi blip: the phone is gone, the sweep is not
    for _ in range(100):
        if not server.ACTIVE[sid].attached:
            break
        time.sleep(0.02)
    assert not (root / sid / "claim_packet.json").exists()
    server.ACTIVE[sid].lines["b1"] = {"type": "book", "id": "b1", "title": "Dune"}  # shown before the drop
    with client.websocket_connect(f"/ws?resume={sid}") as ws:
        msg, seen = wait_for(ws, "book")
        assert "session" in seen
        assert msg["title"] == "Dune"
        assert configs[-1].history_config.initial_history_in_client_content
        assert fake.history == [("user", "Hi there"), ("model", "Welcome.")]
        back = fake.texts[-1]
        assert "dropped" in back and "A1 is still open" in back and fake.replies[-1] is True
        assert server.ACTIVE[sid].sweep.row.shelf == "A1"  # the open shelf survived


def test_resume_of_a_closed_sweep_says_so_and_links_its_report(phone):
    client, fake, configs, root = phone
    with client.websocket_connect("/ws") as ws:
        sid = wait_for(ws, "session")[0]["sweep"]
        ws.send_bytes(b"F" + jpeg(textured()))
        frames_saved(root, 1)
        fake.push(call("end_sweep", confirmed=True))
        wait_for(ws, "done")
    assert sid not in server.ACTIVE  # closed out at disconnect: nothing left to resume
    with client.websocket_connect(f"/ws?resume={sid}") as ws:
        msg = json.loads(receive(ws)["text"])
        assert msg["type"] == "gone" and msg["report"] == f"/sweeps/{sid}/report.html"  # not a new sweep


def test_finished_while_away_hands_over_the_report_not_a_new_sweep(phone):
    client, fake, configs, root = phone
    with client.websocket_connect("/ws") as ws:
        sid = wait_for(ws, "session")[0]["sweep"]
        ws.send_bytes(b"F" + jpeg(textured()))
        frames_saved(root, 1)
    st = server.ACTIVE[sid]
    for _ in range(100):
        if not st.attached:
            break
        time.sleep(0.02)
    st.sweep.end(True)  # e.g. end_sweep finished just as the phone dropped
    st.emit("done", {"report": "/sweeps/x/report.html", "summary": "s"})
    with client.websocket_connect(f"/ws?resume={sid}") as ws:
        msg = json.loads(receive(ws)["text"])
        assert msg["type"] == "done" and msg["report"].endswith("report.html")
