// Phone side of a sweep: mic and camera up, agent voice down, inventory as it arrives.
const $ = (id) => document.getElementById(id);
const HI_RES_MS = 333, HI_RES_W = 1920, HI_Q = 0.85;   // evidence frames: counting, reading, measuring
const LIVE_MS = 2000, LIVE_W = 768, LIVE_Q = 0.6;      // what the agent sees: frames are most of its re-billed context
let ws, stream, ctx, micNode, playAt = 0, speaking = [], timers = [], lastAudioAt = 0, finished = false;
let sweepId = null, capturing = false, wakeLock = null;
const RECONNECT_FOR_MS = 75000;   // the server holds a dropped sweep for 90 s
let orientation = null;  // latest {alpha, beta, gamma} from the phone's motion sensors, for room size
const lines = new Map(), state = new Map();   // id -> <li>, id -> {kind, status}

$("start").onclick = start;
$("end").onclick = () => { send(JSON.stringify({ type: "end" })); $("end").disabled = true; status("Finishing the claim packet…"); };
$("handle").onclick = () => $("sheet").classList.toggle("open");
let started = 0;

function status(t) { $("status").textContent = t; }
function send(data) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(data); }
function tagged(tag, buf) { const out = new Uint8Array(buf.byteLength + 1); out[0] = tag.charCodeAt(0); out.set(new Uint8Array(buf), 1); return out; }

function listenForOrientation(permission) {
  const on = () => window.addEventListener("deviceorientation", (e) => {
    if (e.alpha === null) return;
    orientation = { alpha: e.alpha, beta: e.beta, gamma: e.gamma };
    if (scan.on) scanMotion(orientation, e.timeStamp);
  });
  if (!permission) { on(); return; }                 // Android and desktop: no prompt needed
  permission.then((state) => (state === "granted" ? on() : status("Motion access denied: room size won't be measured.")))
            .catch(() => status("Motion access unavailable: room size won't be measured."));
}

async function start() {
  // iOS only grants motion sensors when asked directly inside the tap, before anything is awaited.
  const motion = typeof DeviceOrientationEvent !== "undefined" && DeviceOrientationEvent.requestPermission
    ? DeviceOrientationEvent.requestPermission() : null;
  listenForOrientation(motion);
  $("start").disabled = true;
  try {
    ctx = new AudioContext();             // created inside the tap so iOS lets it play
    await ctx.resume();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
  } catch (e) {
    status(`Camera or microphone unavailable: ${e.message}. On a phone this page must be opened over https.`);
    $("start").disabled = false;
    return;
  }
  $("video").srcObject = stream;
  showSession();
  keepAwake();
  connect(sessionGet("sweep"));   // a reload mid-sweep picks the same sweep back up
}

// One sweep can span several connections: a Wi-Fi blip or a backgrounded tab reconnects to the same
// sweep (shelves, frames and conversation kept) instead of ending it.
function connect(resume, since = Date.now()) {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws${resume ? `?resume=${encodeURIComponent(resume)}` : ""}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    status(resume ? "Reconnected" : "Connected");
    if (!resume) $("agent").textContent = "Listening…";
    $("end").disabled = false;
    if (!capturing) { capturing = true; startMic(); startFrames(); }
  };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onEvent(JSON.parse(ev.data)) : onAudio(ev.data));
  ws.onclose = () => {
    if (finished || !sweepId) { stopAll(); if (!finished) status("Disconnected."); return; }
    if (Date.now() - since > RECONNECT_FOR_MS) { stopAll(); status("Connection lost. The sweep so far has been saved."); return; }
    status("Connection lost · reconnecting…");
    setTimeout(() => connect(sweepId, since), 1500);
  };
}

function sessionGet(k) { try { return sessionStorage.getItem(k); } catch { return null; } }
function sessionSet(k, v) { try { v ? sessionStorage.setItem(k, v) : sessionStorage.removeItem(k); } catch {} }

// iPhone auto-lock suspends the page mid-sweep; a screen wake lock stops that (iOS 16.4+), and it
// has to be asked for again whenever the page comes back to the front.
async function keepAwake() {
  try { if (!finished && navigator.wakeLock) wakeLock = await navigator.wakeLock.request("screen"); } catch {}
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && started) keepAwake(); });

function startMic() {
  const src = ctx.createMediaStreamSource(stream);
  micNode = ctx.createScriptProcessor(4096, 1, 1);
  micNode.onaudioprocess = (e) => {
    e.outputBuffer.getChannelData(0).fill(0);
    send(tagged("A", toPcm16(e.inputBuffer.getChannelData(0), ctx.sampleRate, 16000).buffer));
  };
  src.connect(micNode);
  micNode.connect(ctx.destination);
}

function toPcm16(input, from, to) {
  const ratio = from / to, n = Math.floor(input.length / ratio), out = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function startFrames() {
  const video = $("video"), canvas = document.createElement("canvas");
  let busy = false;
  const grab = (tag, width, quality) => {
    if (busy || !video.videoWidth) return;
    busy = true;
    const scale = Math.min(1, width / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const pose = tag === "F" && orientation ? orientation : null;   // orientation at capture time, not at send time
    canvas.toBlob(async (blob) => {
      busy = false;
      if (!blob) return;
      const jpeg = await blob.arrayBuffer();
      if (pose) {                                    // G: three float32 angles, then the JPEG
        const head = new Float32Array([pose.alpha, pose.beta, pose.gamma]).buffer;
        const both = new Uint8Array(12 + jpeg.byteLength);
        both.set(new Uint8Array(head), 0);
        both.set(new Uint8Array(jpeg), 12);
        send(tagged("G", both.buffer));
        if (scan.on) scanAdd(pose);                  // the ring counts exactly the frames the server measures from
      } else send(tagged(tag, jpeg));
    }, "image/jpeg", quality);
  };
  timers.push(setInterval(() => grab("F", HI_RES_W, HI_Q), HI_RES_MS));
  timers.push(setInterval(() => grab("V", LIVE_W, LIVE_Q), LIVE_MS));
}

function onAudio(buf) {                  // tag byte + PCM16 mono 24 kHz from the agent
  lastAudioAt = Date.now();
  const pcm = new Int16Array(buf.slice(1));
  const audio = ctx.createBuffer(1, pcm.length, 24000);
  const ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 0x8000;
  const node = ctx.createBufferSource();
  node.buffer = audio;
  node.connect(voiceOut());
  playAt = Math.max(playAt, ctx.currentTime);
  node.start(playAt);
  playAt += audio.duration;
  speaking.push(node);
  $("wave").classList.add("on");
  node.onended = () => {
    speaking = speaking.filter((n) => n !== node);
    if (!speaking.length) $("wave").classList.remove("on");
  };
}

function showSession() {
  $("intro").hidden = true;
  for (const id of ["hud", "sheet"]) $(id).hidden = false;
  started = Date.now();
  timers.push(setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    $("clock").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 1000));
}

function place(text, scanning = false) {
  const el = $("place");
  if (!text) { el.hidden = true; $("frame").hidden = true; return; }
  el.textContent = text;
  el.classList.toggle("scan", scanning);
  el.hidden = false;
  $("frame").hidden = scanning;               // the turn has its own guide
}

const shelfName = (s) => `Bookcase ${s[0]} · shelf ${s.slice(1)}`;
let coachTimer;

function onEvent(m) {
  if (m.type === "session") { sweepId = m.sweep; sessionSet("sweep", m.sweep); }
  else if (m.type === "gone") {           // tried to resume a sweep the server has already closed
    finished = true;
    sessionSet("sweep", null);
    stopCapture();
    status("This sweep has ended and what was captured has been saved.");
    if (m.report) { $("report").href = m.report; $("report").hidden = false; $("end").hidden = true; }
    showFinish();
    finishReady(m.report, null);
  }
  else if (m.type === "interrupted") { speaking.forEach((n) => n.stop()); speaking = []; playAt = 0; $("wave").classList.remove("on"); }
  else if (m.type === "transcript") caption(m.who, m.text);
  else if (m.type === "turn_end") lastWho = "";             // the next words start a fresh caption
  else if (m.type === "coach") {
    $("coach").textContent = m.text[0].toUpperCase() + m.text.slice(1);
    $("coach").hidden = false;
    clearTimeout(coachTimer);
    coachTimer = setTimeout(() => ($("coach").hidden = true), 3200);
  }
  else if (m.type === "row_open") place(shelfName(m.shelf));
  else if (m.type === "row_closed" || m.type === "row_rejected") place(null);
  else if (m.type === "scan_open") { place("Measuring the room", true); scanShow(); }
  else if (m.type === "scan_closed" || m.type === "scan_rejected") {
    place(null); scanHide();
    if (m.type === "scan_closed") roomScanned = true;
  }
  else if (m.type === "book") upsert(m.id, m, "book");
  else if (m.type === "item") upsert(m.id, m, "item");
  else if (m.type === "finishing") { stopCapture(); status("Finishing the claim packet…"); showFinish(); }
  else if (m.type === "done") {
    finished = true;
    sessionSet("sweep", null);
    wakeLock?.release().catch(() => {});
    status("Claim packet ready");
    place(null);
    scanHide();
    stopCapture();
    finishReady(m.report, m.figures || {});
    hangUpAfterSummary();
    $("report").href = m.report; $("report").hidden = false;
    $("end").hidden = true;
    $("sheet").classList.add("open");
  } else if (m.type === "error") status(m.message);
}

let lastWho = "", buffers = { agent: "", you: "" };
function caption(who, text) {
  if (who !== lastWho) buffers[who] = "";
  lastWho = who;
  buffers[who] = (buffers[who] + text).slice(-220);
  $(who === "agent" ? "agent" : "you").textContent = buffers[who];
  if (who === "agent") $("finishAgent").textContent = buffers[who];
}

const LABEL = { identified: "named", unidentified: "unread", needs_appraisal: "appraise" };

function describe(m, kind) {
  if (kind === "item") return [m.description || m.category, m.brand_model || m.category];
  if (m.title) return [m.title, m.author || ""];
  return ["Unidentified book", m.spine_height_cm ? `${m.spine_height_cm} cm tall` : "spine not readable"];
}

function upsert(id, m, kind) {
  const isNew = !lines.has(id);
  let li = lines.get(id);
  if (!li) { li = document.createElement("li"); lines.set(id, li); $("lines").prepend(li); }
  const [main, sub] = describe(m, kind);
  li.innerHTML = '<span class="id"></span><span class="t"><span></span><small></small></span><span class="tag"></span>';
  li.querySelector(".id").textContent = id;                 // textContent: titles come from outside, never HTML
  li.querySelector(".t span").textContent = main;
  li.querySelector(".t small").textContent = sub;
  const tag = li.querySelector(".tag");
  tag.textContent = kind === "item" ? "item" : LABEL[m.status] || m.status;
  tag.className = `tag ${kind === "item" ? "item" : m.status}`;

  const was = state.get(id);
  state.set(id, { kind, status: m.status });
  if (isNew || (was && was.status !== m.status && m.status !== "unidentified")) addCard(main, kind === "item" ? m.category : id);
  const all = [...state.values()];
  count("nBooks", all.filter((x) => x.kind === "book").length);
  count("nIds", all.filter((x) => x.kind === "book" && x.status !== "unidentified").length);
  count("nItems", all.filter((x) => x.kind === "item").length);
}

function count(id, n) {
  const el = $(id);
  if (el.textContent === String(n)) return;
  el.textContent = n;
  el.classList.remove("bump"); void el.offsetWidth; el.classList.add("bump");
}

function addCard(main, sub) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = "<b></b><small></small>";
  card.querySelector("b").textContent = main;
  card.querySelector("small").textContent = sub;
  const row = $("latest");
  row.prepend(card);
  while (row.children.length > 6) row.lastChild.remove();
}

// The sweep is over once the packet is being built: no more frames, and the clock stops. The mic stays
// on until the packet is done, in case the agent needs to ask something.
function stopCapture() {
  timers.forEach(clearInterval); timers = [];
  stream?.getVideoTracks().forEach((t) => t.stop());
  scanHide();
  $("end").disabled = true;
}

// Mic off, let the agent read the summary, then hang up once it has been quiet for a few seconds
// (or after 45 s at most), so nothing keeps recording or billing after the report exists.
function hangUpAfterSummary() {
  finished = true;
  micNode?.disconnect();
  stream?.getAudioTracks().forEach((t) => t.stop());
  const doneAt = Date.now();
  let quietSince = null;
  const check = setInterval(() => {
    const now = Date.now();
    quietSince = speaking.length ? null : quietSince ?? now;
    const spoke = lastAudioAt > doneAt;
    if ((spoke && quietSince && now - quietSince > 4000) || now - doneAt > 45000) {
      clearInterval(check);
      ws?.close();
      status("Sweep finished. The claim report is ready.");
      $("finishNote").textContent = "Session closed. The report stays available on the computer.";
    }
  }, 500);
}

function stopAll() {
  timers.forEach(clearInterval); timers = [];
  micNode?.disconnect();
  stream?.getTracks().forEach((t) => t.stop());
  $("end").disabled = true;
}

// ---- room turn guide ----
// Drawn from the phone's own orientation, so it reacts at once; the agent only hears when the turn
// is complete or has stalled. Sector, level and tilt rules match session.py (scan_coverage).
const BINS = 16, SECTOR = 360 / BINS, LEVEL = 20, TILT = 25, NEED = 300, FAST = 45;   // degrees, deg/s
const scan = { on: false };
const wrap = (d) => ((d + 540) % 360) - 180;                  // signed difference in (-180, 180]
const PHONE = '<svg viewBox="0 0 26 44"><rect x="2" y="2" width="22" height="40" rx="5" fill="none" stroke="currentColor" stroke-width="2.5"/><line x1="10" y1="37" x2="16" y2="37" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/></svg>';

function arc(i) {                // sector i drawn anticlockwise from the top: turning left raises alpha
  const pt = (deg) => { const r = (deg - 90) * Math.PI / 180; return `${(60 + 50 * Math.cos(r)).toFixed(2)} ${(60 + 50 * Math.sin(r)).toFixed(2)}`; };
  const a = -(i + 1) * SECTOR + 1.5, b = -i * SECTOR - 1.5;
  return `M${pt(a)} A50 50 0 0 1 ${pt(b)}`;
}

function scanShow() {
  if (!orientation) return;      // no motion sensors: the agent explains, the server says what happened
  Object.assign(scan, { on: true, bins: new Set(), up: false, down: false, done: false, dir: 0,
                        hx: 0, hy: 1, speed: 0, last: null, intro: Date.now() + 4000 });
  $("ringSegs").innerHTML = Array.from({ length: BINS }, (_, i) => `<path class="seg" d="${arc(i)}"/>`).join("");
  $("scanGuide").className = "scan-guide";
  $("tilt").hidden = true;
  for (const a of $("tilt").querySelectorAll(".aim")) a.classList.remove("done");
  $("scanGuide").hidden = false;
  requestAnimationFrame(scanDraw);
}

function scanHide() { scan.on = false; $("scanGuide").hidden = true; }

function scanAdd(p) {
  if (Math.abs(p.beta - 90) <= LEVEL) scan.bins.add(Math.floor((((p.alpha % 360) + 360) % 360) / SECTOR));
  scan.up ||= p.beta - 90 >= TILT;
  scan.down ||= 90 - p.beta >= TILT;
  if (!scan.done && scan.bins.size * SECTOR >= NEED && scan.up && scan.down) { scan.done = true; chime(); }
}

function scanMotion(o, t) {      // smoothed heading (no flicker on a wobble) and turning speed
  const r = o.alpha * Math.PI / 180;
  scan.hx += 0.25 * (Math.sin(r) - scan.hx);
  scan.hy += 0.25 * (Math.cos(r) - scan.hy);
  if (scan.last) {
    const dt = (t - scan.last.t) / 1000;
    if (dt > 0 && dt < 0.5) scan.speed += 0.2 * (Math.abs(wrap(o.alpha - scan.last.a)) / dt - scan.speed);
  }
  scan.last = { a: o.alpha, t };
}

function scanDraw() {
  if (!scan.on) return;
  const heading = (Math.atan2(scan.hx, scan.hy) * 180 / Math.PI + 360) % 360;
  const here = Math.floor(heading / SECTOR);
  $("ringSegs").setAttribute("transform", `rotate(${heading.toFixed(1)} 60 60)`);
  $("ringSegs").querySelectorAll(".seg").forEach((el, i) => {
    el.classList.toggle("on", scan.bins.has(i));
    el.classList.toggle("here", i === here);
  });
  const g = $("scanGuide"), covered = scan.bins.size * SECTOR;
  g.classList.toggle("done", scan.done);
  g.classList.toggle("fast", !scan.done && scan.speed > FAST);

  // Point to the gaps, keeping one direction while there are gaps that way so it doesn't flip-flop.
  const gaps = covered >= NEED ? [] : [...Array(BINS).keys()].filter((i) => !scan.bins.has(i)).map((i) => wrap((i + 0.5) * SECTOR - heading));
  if (!gaps.some((d) => Math.sign(d) === scan.dir)) {
    scan.dir = gaps.length ? Math.sign(gaps.reduce((a, b) => (Math.abs(b) < Math.abs(a) ? b : a))) : 0;
  }
  $("turnL").hidden = scan.dir <= 0 || scan.done;   // alpha grows as the phone turns left
  $("turnR").hidden = scan.dir >= 0 || scan.done;

  // Tilt comes second: shown once most of the turn is done, so there is one thing to do at a time.
  $("tilt").hidden = scan.done || !(covered >= 0.8 * NEED || scan.up || scan.down);
  if (orientation) $("tiltDot").style.top = `${50 - Math.max(-1, Math.min(1, (orientation.beta - 90) / 45)) * 45}%`;
  const [up, down] = $("tilt").querySelectorAll(".aim");
  up.classList.toggle("done", scan.up);
  down.classList.toggle("done", scan.down);

  // Circle done but not the tilt: the middle says which way to tilt, so the next step is unmissable.
  const tiltNext = !scan.done && covered >= NEED ? (!scan.up ? "up" : "down") : "";
  const mid = scan.done ? "✓" : tiltNext ? `<span class="tilt-arrow ${tiltNext}">${tiltNext === "up" ? "↑" : "↓"}</span>`
    : Date.now() < scan.intro && !scan.bins.size ? PHONE : `${Math.round(covered)}°`;
  if ($("ringMid").dataset.v !== mid) { $("ringMid").innerHTML = mid; $("ringMid").dataset.v = mid; }
  requestAnimationFrame(scanDraw);
}

function chime() {               // iPhone Safari has no vibration, so a short soft tone; vibrate where supported
  navigator.vibrate?.(40);
  if (!ctx) return;
  const osc = ctx.createOscillator(), gain = ctx.createGain();
  osc.frequency.value = 880;
  gain.gain.setValueAtTime(0.0001, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.12, ctx.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);
  osc.connect(gain).connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + 0.3);
}

// ---- after End: the finishing screen ----
// A line follows the agent's voice (an analyser on its playback), so the screen visibly "talks" while
// the packet is built; the steps show what is being finished, and the report button appears when ready.
let analyser = null, voiceBus = null, finishTimer = null, finishAt = 0, roomScanned = false;

function voiceOut() {
  if (!voiceBus) {
    voiceBus = ctx.createGain();
    analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    voiceBus.connect(analyser);
    analyser.connect(ctx.destination);
  }
  return voiceBus;
}

function showFinish() {
  if (!$("finish").hidden) return;
  finishAt = Date.now();
  $("finish").hidden = false;
  $("finishAgent").textContent = buffers.agent || "";
  const books = [...state.values()].filter((x) => x.kind === "book");
  setStep("stepBooks", "fBooks", books.length ? `${books.length} found` : "–");
  setStep("stepItems", "fItems", `${[...state.values()].filter((x) => x.kind === "item").length} found`);
  setStep("stepRoom", "fRoom", roomScanned ? "measuring" : "not scanned", !roomScanned);
  finishTimer = setInterval(() => ($("fClock").textContent = `${Math.round((Date.now() - finishAt) / 1000)} s`), 500);
  requestAnimationFrame(drawVoice);
}

function setStep(row, cell, text, skipped = false, done = false) {
  $(cell).textContent = text;
  $(row).classList.toggle("skip", skipped);
  $(row).classList.toggle("done", done);
}

function finishReady(report, f) {
  clearInterval(finishTimer);
  const secs = Math.round((Date.now() - finishAt) / 1000);
  lineReady = true;
  $("finishEyebrow").textContent = finishAt ? `Done in ${secs} s` : "Sweep ended";
  $("finishTitle").textContent = report ? "Your claim report is ready" : "This sweep has ended";
  if (f) {
    setStep("stepBooks", "fBooks", `${f.books ?? 0} · ${f.identified ?? 0} named`, false, true);
    setStep("stepItems", "fItems", `${f.items ?? 0}`, false, true);
    setStep("stepRoom", "fRoom", f.floor_m2 ? `${f.floor_m2} m² floor` : "not measured", !f.floor_m2, !!f.floor_m2);
    setStep("stepPacket", "fClock", `${secs} s`, false, true);
  } else {
    for (const row of ["stepBooks", "stepItems", "stepRoom", "stepPacket"]) $(row).classList.add("done");
  }
  if (report) { $("finishReport").href = report; $("finishReport").hidden = false; }
  $("finishNote").textContent = report ? "The agent is reading the summary; the session closes by itself." : "";
}

let lineReady = false, lineLevel = 0;
const ORB = {   // top, middle and bottom tones, and the light band that drifts through the middle
  building: ["#8a5a1c", "#d9a95a", "#f3e2bd", "255,246,226"],
  ready: ["#3f6b4a", "#8fbf98", "#e3f1e2", "244,255,245"],
};

function drawVoice(now) {
  if ($("finish").hidden) return;
  const c = $("voiceOrb"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  // Loudness of the agent's voice, eased so the sphere swells and settles rather than flickers.
  let level = 0;
  if (analyser) {
    const buf = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (const v of buf) sum += ((v - 128) / 128) ** 2;
    level = Math.min(1, Math.sqrt(sum / buf.length) * 5);
  }
  lineLevel += (level - lineLevel) * (level > lineLevel ? 0.3 : 0.06);

  const [top, middle, bottom, light] = lineReady ? ORB.ready : ORB.building;
  const t = now / 1000, cx = w / 2, cy = h / 2;
  const r = w * 0.44 * (1 + lineLevel * 0.05);             // breathes a little with the voice
  const speed = 0.35 + lineLevel * 1.4, amp = r * (0.12 + lineLevel * 0.22);

  g.save();
  g.beginPath();
  g.arc(cx, cy, r, 0, Math.PI * 2);
  g.clip();
  const sky = g.createLinearGradient(0, cy - r, 0, cy + r);
  sky.addColorStop(0, top);
  sky.addColorStop(0.5, middle);
  sky.addColorStop(1, bottom);
  g.fillStyle = sky;
  g.fillRect(cx - r, cy - r, 2 * r, 2 * r);

  // Soft-edged shapes: the canvas blur where the browser has it, layered transparency otherwise.
  const soft = "filter" in g;
  const wave = (offset, phase, k, a) => {
    g.beginPath();
    for (let i = 0; i <= 60; i++) {
      const x = cx - r + (2 * r * i) / 60;
      const y = cy + offset + Math.sin(i / 60 * Math.PI * k + t * speed + phase) * a
                            + Math.sin(i / 60 * Math.PI * (k + 1.7) - t * speed * 0.7 + phase) * a * 0.4;
      g[i ? "lineTo" : "moveTo"](x, y);
    }
  };
  // a drifting dark cloud in the upper half, then two light bands that flow across the middle
  if (soft) g.filter = `blur(${r * 0.16}px)`;
  const cloud = g.createRadialGradient(cx + Math.sin(t * speed * 0.6) * r * 0.35, cy - r * 0.45, 0,
                                       cx, cy - r * 0.45, r * 0.8);
  cloud.addColorStop(0, top);
  cloud.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = cloud;
  g.fillRect(cx - r, cy - r, 2 * r, 2 * r);
  for (const [offset, phase, k, alpha, width] of [[r * 0.05, 0, 1.6, 0.85, r * 0.34], [r * 0.38, 2.1, 1.2, 0.45, r * 0.5]]) {
    wave(offset, phase, k, amp);
    g.strokeStyle = `rgba(${light},${alpha})`;
    g.lineWidth = width;
    g.lineCap = "round";
    g.stroke();
  }
  g.restore();
  requestAnimationFrame(drawVoice);
}
