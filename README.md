# Library contents claim agent

A live voice-and-vision agent that inventories a home library for an insurance contents claim.

The policyholder walks the room once with a phone, talking to the agent. The agent directs the
sweep shelf by shelf. While the walk is still going, background workers count, read, measure and
price everything. A few seconds after the sweep ends there is a `claim_packet.json` and an HTML
report. Every figure in them points back to a saved frame, and every price to a dated source.

One rule runs through the design: **the language models read and talk; code measures, counts,
prices and adds up.** No model produces a price, a size or a total. Anything the system isn't sure
of goes to the review queue instead of being guessed.

## Contents

- [Quick start](#quick-start-about-10-minutes): offline demo and tests, no keys
- [Keys](#keys)
- [Live sweep](#live-sweep): laptop or phone
- [Re-run, score and compare sweeps](#re-run-score-and-compare-sweeps)
- [What happens during a sweep](#what-happens-during-a-sweep)
- [How each number is made](#how-each-number-is-made)
- [Models, services and libraries](#models-services-and-libraries)
- [Cost and latency](#cost-and-latency)
- [Assumptions](#assumptions) and [known limitations](#known-limitations)
- [ARCHITECTURE.md](ARCHITECTURE.md): how the pieces fit, where scale and prices come from

## Quick start (about 10 minutes)

You need macOS or Linux and [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 itself.

```bash
git clone https://github.com/Yashwanth-Kumar-Kotla/library-claim-agent.git
cd library-claim-agent
uv sync                                   # everything, including PyTorch for the depth model
uv run python tools/demo_offline.py       # the whole pipeline with no API keys; opens a report
uv run pytest -q                          # 328 tests, about 10 s, no network
```

The offline demo runs on a synthetic shelf with fake model replies. Only the model calls are fake.
Everything after them is the real code: tiling, merging, scale, sizes, matching, pricing rules,
review queue, totals and the report.

## Keys

```bash
cp .env.example .env    # then fill in the keys below
```

| Key | Needed for | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini: the voice agent, reading spines, finding objects | Google AI Studio |
| `GOOGLE_BOOKS_API_KEY` | Matching spine text to a specific book | Google Cloud console, Books API |
| `SERPAPI_API_KEY` | Prices: eBay searches through SerpApi | serpapi.com (free tier: 250 searches a month) |
| `EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET` | Prices from eBay's own API, instead of SerpApi | developer.ebay.com |

With no price source everything still runs. Each line is marked "price source not configured";
nothing is guessed. `COUNTRY` and `CURRENCY` in `.env` set the policyholder's market (default US, USD).

Check everything before a real sweep:

```bash
uv run python -m libraryclaim.preflight   # GO / NO-GO: Gemini credit, keys, SerpApi quota, depth model, certificate
```

## Live sweep

**Print the scale markers** (`ground_truth/markers.pdf`, at 100%) and tape one to each bookcase,
in view of the camera. They are what turns pixels into centimetres.

On a laptop:

```bash
uv run python -m libraryclaim.live        # then open http://localhost:8765 in Chrome
```

On a phone on the same Wi-Fi. Phone browsers only allow the camera, microphone and motion sensors
on HTTPS pages, so this needs a local certificate:

```bash
brew install mkcert                       # once
./tools/make_cert.sh                      # certificate for this computer's LAN address
uv run python -m libraryclaim.live --phone
```

1. Install `data/certs/rootCA.pem` on the phone once. On an iPhone: AirDrop it, then Settings >
   Profile Downloaded > Install, then Settings > General > About > Certificate Trust Settings and
   turn it on.
2. Open `https://<lan-ip>:8443/check`. It confirms the phone reaches the computer and can use the
   camera, microphone and motion sensors.
3. Open the printed `https://<lan-ip>:8443` address, tap Start, and allow camera, microphone and
   motion access.

No live app at hand? Photos work too. Photograph each shelf with the marker in the shot, then turn
them into a sweep:

```bash
uv run python tools/make_sweep.py data/sweeps/home-1 --row A1 IMG_0101.HEIC --row A2 IMG_0102.HEIC IMG_0103.HEIC
uv run python -m libraryclaim run data/sweeps/home-1
```

## Re-run, score and compare sweeps

```bash
uv run python -m libraryclaim run data/sweeps/<sweep>                  # rebuild packet + report
uv run python -m libraryclaim reprice data/sweeps/<sweep> --country GB --currency GBP
uv run python -m libraryclaim.evaluate data/sweeps/<sweep> ground_truth/ground_truth.xlsx
```

- **`run`** puts a saved sweep through the current pipeline. A live sweep saves everything needed
  for this. Model replies are cached on disk, so a re-run costs nothing after the first time
  (`--no-cache` to measure). The packet the policyholder saw stays in `out/claim_packet.live.json`.
- **`reprice`** prices up to 10 of the claim's books in a second country. They are shown in the
  report next to the local prices and are not added to the totals.
- **`evaluate`** scores a packet against a hand-counted ground truth sheet and writes the system's
  answers next to it in `ground_truth_results.xlsx`.

Each sweep folder under `data/sweeps/` keeps its frames, the session log (`out/session.jsonl`:
everything said and done), each stage's output, timings and cost.

Every live session, `run`, `reprice` and `evaluate` also copies its outputs to
`results/<sweep>/<run>/` and adds one line to [`results/index.md`](results/index.md). The copy holds
the packet, report, scores, timings, cost and session log, but never frames. The index line gives book
count, identified, appraisal, values, floor area, model cost and time to packet, so runs can be
compared side by side.

## What happens during a sweep

1. The agent greets the policyholder, confirms country and currency, and explains the sweep.
2. **Shelves.** The agent names each shelf (A1 = bookcase A, top shelf) and asks for a slow pan
   left to right. When a shelf is done, its frames are stitched and sent off at once. Spines are
   found and read, and each book becomes its own job: catalogue match, appraisal rules, prices.
   Lines appear on the phone while the policyholder moves to the next shelf.
3. **Capture quality is measured in code** on every full-resolution frame: blur, glare, darkness
   and speed. Problems show as a banner. The agent only speaks up when a problem lasts on a shelf,
   and asks for a re-capture when a shelf had too few clear frames. Skipped shelves are caught
   before the end.
4. **Room size.** The policyholder stands mid-room, turns once and tilts the phone up and down.
   Phone orientation plus a metric depth model give the floor outline and ceiling height. The
   printed marker corrects the depth model's scale.
5. **Everything else in the room** is found in views taken between shelves and during the turn.
   Objects are de-duplicated across views, sized from the depth maps and priced, or given a
   sourced range. Art defaults to "needs appraisal", and the agent asks whether it is an original
   or a print.
6. **Corrections by voice** ("that one is signed", "skip that shelf, not mine") become recorded
   remarks. At the end each one is tied to a specific line, or reported if it can't be.
7. The packet is built from the finished jobs, totals are summed in code, and the agent reads the
   summary back.

## How each number is made

| Figure | Made by | Guardrail |
|---|---|---|
| Book count | Gemini Flash finds and reads spines in overlapping tiles, two independent passes per tile | Passes merged by filling gaps only; overlapping boxes that resolve to the same book count once |
| Title, author, edition | Spine text matched to Google Books, then Open Library | Short title-only matches, same title by another author, and close ties stay unidentified; edition only when the spine shows the publisher |
| Spine height, thickness, shelf run | Homography from the printed ArUco marker to centimetres on the shelf plane | Implausible sizes dropped; without a marker, sizes come from the depth model and the shelf gets a review note saying so |
| Prices | eBay listings in the policyholder's marketplace: median of new (replacement) and used (market value) fixed-price listings | Lots, study guides and sequels filtered; under 3 listings means no price; other markets converted at the ECB rate and labelled |
| Rare, signed, antiquarian, expensive | Spine text, how the spine looks, the edition year, the price, or what the policyholder said | Not priced; sent to appraisal |
| Room size | Turn-around scan: Depth Pro metric depth + phone orientation, marker-corrected | Incomplete turns or no tilt are refused on the spot; unmeasured height leaves walls blank |
| Item size | Box size times depth, front face only | Marked approximate; depth not given |
| Totals | Python, from the lines | Lines without price evidence are removed before totals |

## Models, services and libraries

- **Gemini 3.8 Live**: the voice agent (audio in and out, plus a camera frame every 2 s).
- **Gemini 3.8 Flash**, thinking set to low: finding and reading spines, finding room objects.
- **Gemini 3.1 Pro**: a second look at hard-to-read spines.
- **Depth Pro** (Apple, pretrained, run locally): metric depth for room size and item sizes. The
  weights (about 1.9 GB) download from Hugging Face the first time a sweep needs them.
- **Google Books API** and **Open Library**: catalogue matching.
- **eBay listings**, through **SerpApi** or eBay's Browse API: prices.
- **Frankfurter** (ECB reference rates): currency conversion.
- **OpenCV** (ArUco markers, stitching, image quality), **rapidfuzz**, **FastAPI**, **PyTorch** and
  **transformers**, **openpyxl**, **Jinja2**.

No model was trained or fine-tuned.

### Reference app: what was kept, changed and dropped

The starting point was the Insurance Claim Live Agent Team example.

- **Kept:** the Gemini Live session held on the server; non-blocking tools whose results arrive
  while the agent keeps talking, interrupting only when the person must act; and the prompting
  rule "report only what you can see".
- **Changed:**
  - The camera pipeline. The reference sends one 512 px frame per second and keeps only the
    latest. Here every full-resolution frame is saved, scored and used for counting and reading.
  - The tools: sweep direction, shelves, room scan and remarks, instead of policy lookup and
    sketches.
  - The background work: a streaming pipeline instead of one LLM graph.
  - Phone access over HTTPS. The reference only accepts localhost.
- **Dropped:** the policy directory, fraud and routing rules, incident sketches and the avatar.

## Cost and latency

Measured on real runs at Gemini paid-tier prices, October 2026:

| | |
|---|---|
| Voice agent | $0.14 for a 6-minute session (`tools/live_cost_probe.py`). It was $0.29 before its context was capped at 20k tokens and frames were cut to one every 2 s |
| Vision calls, short live sweep | about $0.02 to $0.04 |
| Vision calls, 6-shelf batch run | about $0.22 (43 Flash calls, 22 Pro calls) |
| Packet ready after the sweep ends | 4 to 11 s on the recorded sweeps |
| Spine detection and reading | about 10 s per shelf, in the background during the sweep |
| Depth model | about 2.3 s per frame on an M5 MacBook, in the background |

The voice agent's cost grows faster than the session's length. The Live API bills the whole
conversation again on every reply, and the cap keeps that in check. Each session's measured cost
is in its session log and in [`results/index.md`](results/index.md).

## Assumptions

- **Confidently wrong** = books marked identified whose title is not on the shelf, divided by books
  marked identified.
- **Dimension and price bars:** pass when the mean absolute percentage error is within the bar and
  every sample line was produced.
- **Wall area** is gross: perimeter times height. Net of known doors and windows is computed too.
- **Used value** is the median asking price of active used listings, because eBay's public data does
  not include sold prices.
- **Appraisal threshold:** 150 in the local currency (`APPRAISAL_THRESHOLD`).
- **Shelf run** is the length of shelf occupied by books, gaps excluded.
- **Room scan:** the phone is held about 25 cm in front of the body's turning axis.

## Known limitations

- Thin spines with no readable text are the most common miss on densely packed shelves.
- Without a marker in view, spine sizes come from the depth model. Its scale is set shelf by shelf
  from identified books whose published edition heights agree, or a typical book height when too
  few do. The room scan then relies on the depth model's own scale. On real phone shelves that
  scale was off by −25% to +40% depending on the image, which is why the marker is the primary
  method.
- A bookcase that reaches the ceiling hides the wall behind it, so that wall reads about one
  bookcase depth too close.
- Item sizes are the front face only.
- The voice agent decides when to speak. Its patience and pacing are set by instructions and voice
  activity settings, so they aren't guaranteed.

## Layout

```
src/libraryclaim/        pipeline: quality, scale, boxes, vision, resolve, pricing, review, packet, report,
                         room, room_scan, depth, pipeline (streaming), evaluate, archive
src/libraryclaim/live/   live sweep: session state, agent tools, usage and cost, WebSocket server, phone page
tests/                   unit and end-to-end tests with synthetic shelves and rooms
tools/                   markers, ground truth template, offline demo, certificate, photos to sweep, live cost probe
ground_truth/            printable markers and the ground truth sheet
results/                 outputs of every processed sweep, with index.md
```
