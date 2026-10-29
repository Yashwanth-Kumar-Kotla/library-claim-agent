# Library contents claim agent

A live voice-and-vision agent that inventories a home library for an insurance contents claim.
The policyholder walks the room once with a phone while talking to the agent. The agent directs
the sweep shelf by shelf, and background workers count, read, measure and price everything while
the walk is still going. A few seconds after the sweep ends there is a `claim_packet.json` and an
HTML report in which every figure points back to a saved frame and every price to a dated source.

The rule the whole design follows: **the language models read and talk; code measures, counts,
prices and adds up.** No model ever produces a price, a size or a total, and anything the system
is not sure of goes to the review queue instead of being guessed.

## Quick start (about 10 minutes)

You need macOS or Linux, [uv](https://docs.astral.sh/uv/), and Python 3.12 (uv installs it).

```bash
git clone https://github.com/Yashwanth-Kumar-Kotla/library-claim-agent.git
cd library-claim-agent
uv sync                                   # installs everything, including PyTorch for the depth model
uv run python tools/demo_offline.py       # whole pipeline, no API keys, opens a report
uv run pytest -q                          # ~290 tests, ~10 s, no network
```

The offline demo uses a synthetic shelf and fake model replies, so everything after the model
calls (tiling, merging, scale, sizes, matching, pricing rules, review queue, totals, report) is the
real code. To use the real models:

```bash
cp .env.example .env    # then fill in the keys below
```

| Key | Needed for | Where |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini: the voice agent, reading spines, finding objects | Google AI Studio |
| `GOOGLE_BOOKS_API_KEY` | Matching spine text to a specific book | Google Cloud console, Books API |
| `EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET` | Prices | developer.ebay.com |

Without eBay keys everything runs and every line is marked "price source not configured"; nothing
is guessed.

### Live sweep on a laptop

```bash
uv run python -m libraryclaim.live        # then open http://localhost:8765 in Chrome
```

### Live sweep on a phone (same Wi-Fi)

Phone browsers only allow camera, microphone and motion sensors on HTTPS pages.

```bash
brew install mkcert                       # once
./tools/make_cert.sh                      # certificate for this computer's LAN address
uv run python -m libraryclaim.live --phone
```

Install `data/certs/rootCA.pem` on the phone once (iPhone: AirDrop it, Settings > Profile
Downloaded > Install, then Settings > General > About > Certificate Trust Settings > enable), then
open the printed `https://<lan-ip>:8443` address. Tap Start and allow camera, microphone and
motion access.

**Print the scale markers** (`ground_truth/markers.pdf`, at 100%) and tape one to each bookcase,
in view of the camera. They are what turns pixels into centimetres.

### Re-run a saved sweep, score it, price it for another country

```bash
uv run python -m libraryclaim run data/sweeps/<sweep>                  # rebuild packet + report
uv run python -m libraryclaim reprice data/sweeps/<sweep> --country GB --currency GBP
uv run python -m libraryclaim.evaluate data/sweeps/<sweep> ground_truth/ground_truth.xlsx
```

Every sweep folder keeps its frames, the session log (`out/session.jsonl`: everything said and
done), each stage's output, timings and cost.

## What happens during a sweep

1. The agent greets the policyholder, confirms country and currency, and explains the sweep.
2. **Shelves.** The agent names each shelf (A1 = bookcase A, top shelf) and asks for a slow pan
   left to right. When a shelf is done, its frames are stitched and sent off at once: spines are
   found and read, each book becomes its own job (catalogue match, appraisal rules, prices), and
   lines appear on the phone while the policyholder moves to the next shelf.
3. **Capture quality is measured in code** on every full-resolution frame: blur, glare, darkness,
   speed. Problems show as a banner; the agent only speaks up when a problem lasts on a shelf, and
   asks for a re-capture when a shelf had too few clear frames. Skipped shelves are caught before
   the end.
4. **Room size.** The policyholder stands mid-room, turns once and tilts up and down. Phone
   orientation plus a metric depth model give the floor outline and ceiling height; the printed
   marker corrects the depth model's scale.
5. **Everything else in the room** is found in views taken between shelves and during the turn,
   de-duplicated across views, sized from the depth maps and priced (or given a sourced range).
   Art defaults to "needs appraisal" and the agent asks whether it is an original or a print.
6. **Corrections by voice** ("that one is signed", "skip that shelf, not mine") become recorded
   remarks that are tied to a specific line at the end, or reported if they can't be.
7. The packet is built from the finished jobs, totals are summed in code, and the agent reads the
   summary back.

## How each number is made

| Figure | Made by | Guardrail |
|---|---|---|
| Book count | Gemini Flash finds and reads spines in overlapping tiles, two independent passes per tile | Passes merged by filling gaps only; overlapping boxes that resolve to the same book count once |
| Title, author, edition | Spine text matched to Google Books, then Open Library | Short title-only matches, same title by another author, and close ties stay unidentified; edition only when the spine shows the publisher |
| Spine height, thickness, shelf run | Homography from the printed ArUco marker to centimetres on the shelf plane | Implausible sizes dropped; rows without a marker get no sizes |
| Prices | eBay listings in the policyholder's marketplace: median of new (replacement) and used (market value) fixed-price listings | Lots, study guides and sequels filtered; under 3 listings means no price; other markets converted at the ECB rate and labelled |
| Rare, signed, antiquarian, expensive | Spine text, how the spine looks, the edition year, the price, or what the policyholder said | Not priced; sent to appraisal |
| Room size | Turn-around scan: Depth Pro metric depth + phone orientation, marker-corrected | Incomplete turns or no tilt are refused on the spot; unmeasured height leaves walls blank |
| Item size | Box size times depth, front face only | Marked approximate; depth not given |
| Totals | Python, from the lines | Lines without price evidence are removed before totals |

## Models, services and libraries

- **Gemini 3.8 Live**: the voice agent (audio in and out, low-rate video).
- **Gemini 3.8 Flash**: finding and reading spines, finding room objects. Thinking set to low.
- **Gemini 3.1 Pro**: a second look at hard-to-read spines.
- **Depth Pro** (Apple, pretrained, run locally): metric depth for room size and item sizes. The
  weights (about 1.9 GB) download from Hugging Face the first time a sweep needs them.
- **Google Books API, Open Library**: catalogue matching.
- **eBay Browse API**: prices. **Frankfurter** (ECB reference rates): currency conversion.
- **OpenCV** (ArUco markers, stitching, image quality), **rapidfuzz**, **FastAPI**, **PyTorch** /
  **transformers**, **openpyxl**, **Jinja2**.

No model was trained or fine-tuned.

## Reference app: what was kept, changed and dropped

The starting point was the Insurance Claim Live Agent Team example.

- **Kept:** the Gemini Live session held on the server, non-blocking tools whose results arrive
  while the agent keeps talking (interrupting only when the person must act), and the prompting
  rule "report only what you can see".
- **Changed:** the camera pipeline (the reference sends one 512 px frame per second and keeps only
  the latest; here every full-resolution frame is saved, scored and used for counting and reading),
  the tools (sweep direction, shelves, room scan, remarks instead of policy lookup and sketches),
  the background work (a streaming pipeline instead of one LLM graph), and phone access over HTTPS
  (the reference only accepts localhost).
- **Dropped:** the policy directory, fraud and routing rules, incident sketches and the avatar.

## Cost and latency

Measured on real runs (Gemini paid tier, October 2026 prices):

| | |
|---|---|
| Vision calls for a 6-shelf batch run | about $0.22 (43 Flash calls, 22 Pro calls) |
| Vision calls for a short live sweep | about $0.02 to $0.04 |
| Voice agent | about $0.025 per minute (audio in and out, low-rate video) |
| Packet ready after the sweep ends | about 5 s |
| Spine detection and reading | about 10 s per shelf, in the background during the sweep |
| Depth model | about 2.3 s per frame on an M5 MacBook, in the background |

Model replies are cached on disk, so re-running a saved sweep while tuning costs nothing
(`--no-cache` to measure).

## Assumptions

- **Confidently wrong** = books marked identified whose title is not on the shelf, divided by books
  marked identified.
- **Dimension and price bars:** pass when the mean absolute percentage error is within the bar and
  every sample line was produced.
- **Wall area** is gross (perimeter times height); net of known doors and windows is computed too.
- **Used value** is the median asking price of active used listings: eBay's public API does not
  expose sold prices.
- **Appraisal threshold:** 150 in the local currency (configurable).
- **Shelf run** is the length of shelf occupied by books, gaps excluded.
- **Room scan:** the phone is held about 25 cm in front of the body's turning axis.

## Known limitations

- Thin spines with no readable text are the most common miss on densely packed shelves.
- Without a marker in view, spine sizes are not reported and the room scan relies on the depth
  model's own scale (it can be off by 10 to 25 percent; the review queue says so).
- A bookcase that reaches the ceiling hides the wall behind it; that wall reads about one bookcase
  depth too close.
- Item sizes are the front face only.
- The voice agent decides when to speak; its patience and pacing are set by instructions and voice
  activity settings, not guaranteed.

## Layout

```
src/libraryclaim/        pipeline: quality, scale, boxes, vision, resolve, pricing, review, packet, report,
                         room, room_scan, depth, pipeline (streaming), evaluate
src/libraryclaim/live/   live sweep: session state, agent tools, WebSocket server, phone page
tests/                   unit and end-to-end tests with synthetic shelves and rooms
tools/                   markers, ground truth template, offline demo, certificate, photo-to-sweep
```
