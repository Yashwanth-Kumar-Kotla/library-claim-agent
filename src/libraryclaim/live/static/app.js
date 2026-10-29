// Phone side of a sweep: mic and camera up, agent voice down, inventory as it arrives.
const $ = (id) => document.getElementById(id);
const HI_RES_MS = 333, HI_RES_W = 1920, HI_Q = 0.85;   // evidence frames: counting, reading, measuring
const LIVE_MS = 1000, LIVE_W = 768, LIVE_Q = 0.6;      // what the agent sees
let ws, stream, ctx, micNode, playAt = 0, speaking = [], timers = [];
let orientation = null;  // latest {alpha, beta, gamma} from the phone's motion sensors, for room size
const lines = new Map(), state = new Map();   // id -> <li>, id -> {kind, status}

$("start").onclick = start;
$("end").onclick = () => { send(JSON.stringify({ type: "end" })); $("end").disabled = true; status("Finishing the claim packet…"); };

function status(t) { $("status").textContent = t; }
function send(data) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(data); }
function tagged(tag, buf) { const out = new Uint8Array(buf.byteLength + 1); out[0] = tag.charCodeAt(0); out.set(new Uint8Array(buf), 1); return out; }

function listenForOrientation(permission) {
  const on = () => window.addEventListener("deviceorientation", (e) => {
    if (e.alpha !== null) orientation = { alpha: e.alpha, beta: e.beta, gamma: e.gamma };
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
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { status("Connected. The assistant will talk you through it."); $("end").disabled = false; startMic(); startFrames(); };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onEvent(JSON.parse(ev.data)) : onAudio(ev.data));
  ws.onclose = () => { stopAll(); status("Disconnected."); };
}

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
      } else send(tagged(tag, jpeg));
    }, "image/jpeg", quality);
  };
  timers.push(setInterval(() => grab("F", HI_RES_W, HI_Q), HI_RES_MS));
  timers.push(setInterval(() => grab("V", LIVE_W, LIVE_Q), LIVE_MS));
}

function onAudio(buf) {                  // tag byte + PCM16 mono 24 kHz from the agent
  const pcm = new Int16Array(buf.slice(1));
  const audio = ctx.createBuffer(1, pcm.length, 24000);
  const ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 0x8000;
  const node = ctx.createBufferSource();
  node.buffer = audio;
  node.connect(ctx.destination);
  playAt = Math.max(playAt, ctx.currentTime);
  node.start(playAt);
  playAt += audio.duration;
  speaking.push(node);
  node.onended = () => { speaking = speaking.filter((n) => n !== node); };
}

function onEvent(m) {
  if (m.type === "interrupted") { speaking.forEach((n) => n.stop()); speaking = []; playAt = 0; }
  else if (m.type === "transcript") caption(m.who, m.text);
  else if (m.type === "coach") { $("coach").textContent = m.text; $("coach").hidden = false; setTimeout(() => ($("coach").hidden = true), 3000); }
  else if (m.type === "row_open") { $("row").textContent = `Shelf ${m.shelf}`; $("row").hidden = false; }
  else if (m.type === "row_closed" || m.type === "row_rejected") $("row").hidden = true;
  else if (m.type === "scan_open") { $("row").textContent = "Measuring the room: turn slowly"; $("row").hidden = false; }
  else if (m.type === "scan_closed") $("row").hidden = true;
  else if (m.type === "book") upsert(m.id, m, "book");
  else if (m.type === "item") upsert(m.id, m, "item");
  else if (m.type === "done") {
    status("Claim packet ready.");
    $("report").href = m.report; $("report").hidden = false;
  } else if (m.type === "error") status(m.message);
}

let lastWho = "", buffers = { agent: "", you: "" };
function caption(who, text) {
  if (who !== lastWho) buffers[who] = "";
  lastWho = who;
  buffers[who] = (buffers[who] + text).slice(-220);
  $(who === "agent" ? "agent" : "you").textContent = buffers[who];
}

function upsert(id, m, kind) {
  let li = lines.get(id);
  if (!li) { li = document.createElement("li"); lines.set(id, li); $("lines").prepend(li); }
  const flag = m.status === "needs_appraisal" ? '<span class="flag">needs appraisal</span>' : "";
  const label = kind === "book" ? (m.title ? `${m.title}${m.author ? " — " + m.author : ""}` : "Unidentified book")
                                : `${m.category}: ${m.description}`;
  li.className = m.status === "unidentified" ? "unidentified" : "";
  li.innerHTML = `<span class="id"></span><span class="t"></span>${flag}`;
  li.querySelector(".id").textContent = id;
  li.querySelector(".t").textContent = label;   // textContent: titles come from outside, never as HTML
  state.set(id, { kind, status: m.status });
  const all = [...state.values()];
  $("nBooks").textContent = all.filter((x) => x.kind === "book").length;
  $("nIds").textContent = all.filter((x) => x.kind === "book" && x.status !== "unidentified").length;
  $("nItems").textContent = all.filter((x) => x.kind === "item").length;
}

function stopAll() {
  timers.forEach(clearInterval); timers = [];
  micNode?.disconnect();
  stream?.getTracks().forEach((t) => t.stop());
  $("end").disabled = true;
}
