# Library contents claim agent

A live voice and camera agent for documenting a home library for an insurance contents claim.

The policyholder opens a web page on their phone and walks the room once while talking to the
agent. The agent guides them shelf by shelf, asks for a re-capture when spines are unreadable, asks
whether a framed picture is an original or a print, and takes corrections by voice ("that one is
signed", "skip that shelf, it isn't mine"). While they walk, the books are counted, read, matched
to catalogue records, measured and priced in the background. About 15 seconds after they press
End there is a `claim_packet.json` and a readable HTML report.

The design rule: the models read and talk, the code measures, counts, prices and adds up. No model
writes a price, a size or a total. Every line in the packet points to the saved frame it was seen
in, and every price to an eBay listing search with its date. Anything uncertain goes to a review
queue for the adjuster instead of being guessed.

Demo (one unedited take, 3:49): https://youtu.be/I9TRdhfWdEc

## Results

Scored against ground truth I collected by hand for my own room: 40 books on two bookcases,
15 other items, a tape-measured room, 20 spines measured with a ruler and 15 prices checked on eBay.

| Measure | Demo sweep | Pass bar |
|---|---|---|
| Titles identified | 75% of legible spines | at least 70% |
| Confidently wrong titles | 2.7% | at most 3% |
| Book prices | 23% mean error | within 25% |
| Non-book items found | 93% | at least 80% |
| Packet after the sweep ends | 15 s | under 5 min |
| Book count | 49 vs 40 | within 5% (missed) |
| Spine sizes | 24% mean error | within 15% (missed) |
| Floor and wall area | 85% and 31% too large | within 10% and 15% (missed) |

The misses, their causes and a measured fix are in [FAILURE_LOG.md](FAILURE_LOG.md). The demo
sweep's packet, report, frames and scores are in [`submission/`](submission/).

## Run it in 10 minutes, no keys needed

You need macOS or Linux and [uv](https://docs.astral.sh/uv/getting-started/installation/), which
installs Python 3.12 by itself.

```bash
git clone https://github.com/Yashwanth-Kumar-Kotla/library-claim-agent.git
cd library-claim-agent
uv sync
uv run python tools/demo_offline.py
uv run pytest -q
```

`uv sync` takes a few minutes the first time because it downloads PyTorch. The offline demo runs the
whole pipeline on a synthetic shelf and opens the report. Only the model replies are faked; tiling,
scale, sizes, catalogue matching rules, pricing rules, the review queue and the totals are the real
code. The tests (394, about 20 seconds) need no network.

## Live sweep

Copy the keys file and fill it in:

```bash
cp .env.example .env
```

| Key | Used for | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini: the voice agent, reading spines, finding objects | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GOOGLE_BOOKS_API_KEY` | Matching spine text to a book | Google Cloud console, Books API |
| `SERPAPI_API_KEY` | Prices (eBay searches through SerpApi) | [serpapi.com](https://serpapi.com), free tier 250 searches a month |

`COUNTRY` and `CURRENCY` in `.env` set the policyholder's market (default US, USD). Without a price
key everything still runs and each line says the price source isn't configured.

Check the setup:

```bash
uv run python -m libraryclaim.preflight
```

It reports GO or NO-GO for Gemini, the keys, the SerpApi quota, the depth model and the phone
certificate.

Print `ground_truth/markers.pdf` at 100% and tape a marker to each bookcase and one to a wall at
chest height. The printed square has a known size, which is how pixels become centimetres. If your
print isn't 15 cm, set `MARKER_SIZE_CM` in `.env`.

**On a laptop** (Chrome allows the camera on localhost):

```bash
uv run python -m libraryclaim.live
```

Then open http://localhost:8765.

**On a phone** on the same Wi-Fi. Phone browsers only give the camera, microphone and motion sensors
to HTTPS pages, so this needs a local certificate:

```bash
brew install mkcert
./tools/make_cert.sh
uv run python -m libraryclaim.live --phone
```

1. Send `data/certs/rootCA.pem` to the phone and trust it once. On an iPhone: AirDrop it, install it
   under Settings > Profile Downloaded, then turn it on under Settings > General > About >
   Certificate Trust Settings.
2. Open `https://<computer's LAN address>:8443/check` to confirm the phone can reach the computer
   and use the camera, microphone and motion sensors.
3. Open `https://<computer's LAN address>:8443`, tap Start and follow the agent.

Some guest Wi-Fi networks block devices from reaching each other. If `/check` never loads, use the
phone's own hotspot for both devices and run `./tools/make_cert.sh` again.

## Re-run and score a sweep

Every live sweep is saved under `data/sweeps/<sweep>/`: frames, phone orientation, the session log
and each stage's output.

```bash
uv run python -m libraryclaim run data/sweeps/<sweep>
uv run python -m libraryclaim reprice data/sweeps/<sweep> --country GB --currency GBP
uv run python -m libraryclaim.evaluate data/sweeps/<sweep> ground_truth/ground_truth.xlsx
```

- `run` rebuilds the packet and report with the current code. Model replies and web lookups are
  cached, so a re-run is close to free.
- `reprice` prices up to 10 of the books for a second country. They appear in the report beside the
  local prices and are not added to the totals.
- `evaluate` scores the packet against the ground truth sheet and writes the system's numbers next
  to the hand-collected ones in `ground_truth_results.xlsx`.

Each run is also copied to `results/<sweep>/` with a line in [results/index.md](results/index.md)
(counts, values, area, model cost, time to packet), so runs can be compared.

You can also make a sweep from photos, one or more per shelf with the marker in view:

```bash
uv run python tools/make_sweep.py data/sweeps/home-1 --row A1 IMG_0101.HEIC --row A2 IMG_0102.HEIC
uv run python -m libraryclaim run data/sweeps/home-1
```

## How it works

[ARCHITECTURE.md](ARCHITECTURE.md) has the diagram and details. In short:

- **Conversation.** Gemini Live runs on the server and talks to the phone over a WebSocket. Its tools
  open and close shelves, run the room scan, record remarks and end the sweep. Each tool result tells
  it what to say next, so it doesn't improvise the flow.
- **Capture quality.** Blur, glare, darkness and speed are measured in code on every frame. Problems
  show on screen; the agent only speaks up when a problem lasts or a shelf had too few clear frames.
- **Books.** When a shelf is finished its frames are stitched, and Gemini Flash finds and reads every
  spine in overlapping tiles, two passes per tile. Each book then becomes its own job: match the
  spine text in Google Books and Open Library, check for rare or signed books, price it.
- **Identification rules.** A title is only accepted when the evidence supports it. Short title-only
  matches, a spine that shows only a series name, a year that disagrees with the record and close
  ties are all left unidentified.
- **Sizes.** The marker gives a centimetre scale on the shelf. Without one, each shelf is scaled from
  the catalogue heights of the books identified on it.
- **Prices.** The median of new fixed-price eBay listings is the replacement cost, the median of used
  listings the market value. Fewer than 3 matching listings means no price. Other markets are
  converted at the ECB rate and labelled. Rare, signed, antiquarian and high-value books, and
  original art, go to appraisal and are not priced.
- **Room.** The policyholder turns once in the middle of the room and tilts up and down. Phone
  orientation plus Depth Pro metric depth give the floor outline and ceiling height, and the marker
  corrects the scale.
- **Packet.** Totals are summed in Python from the lines. Lines without price evidence are left out
  of the totals and listed.

## Models and services

- Gemini 3.8 Live: the voice agent
- Gemini 3.8 Flash (low thinking): finding and reading spines, finding room objects
- Gemini 3.1 Pro: a second look at hard-to-read spines
- Depth Pro (Apple, pretrained, runs locally): metric depth for the room and item sizes. The weights
  (about 1.9 GB) download the first time a sweep needs them.
- Google Books and Open Library: catalogue matching
- eBay listings through SerpApi: prices
- Frankfurter (ECB reference rates): currency conversion
- OpenCV, rapidfuzz, FastAPI, PyTorch, transformers, openpyxl, Jinja2

No model was trained or fine-tuned. `pricing.py` can also use eBay's own Browse API
(`EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET`); that path is tested but wasn't used for these runs.

## What I took from the reference app

I started from the Insurance Claim Live Agent Team example.

- **Kept:** the Gemini Live session held on the server, non-blocking tools that return while the
  agent keeps talking, and the rule that the agent only reports what it can see.
- **Changed:** every full-resolution frame is saved, scored and used for counting (the reference keeps
  one small frame a second); the tools are about shelves, the room scan and remarks; the background
  work is a streaming pipeline rather than one LLM graph; and the page runs on a phone over HTTPS.
- **Dropped:** the policy directory, fraud and routing rules, incident sketches and the avatar.

## Cost and speed

| | |
|---|---|
| Demo sweep, all in | $0.42 (voice agent $0.32 for 3.5 minutes, vision $0.10) |
| Packet ready after the sweep ends | 15 s; nearly all the work runs during the walk |
| Spine detection and reading | about 10 s per shelf, in the background |
| Depth model | about 2.3 s per frame on an M5 MacBook |

The Live API re-reads the whole conversation on every reply, so its cost grows faster than the
session's length. The agent's context is capped at 20k tokens and it sees a frame every 2 seconds,
which halved its cost (measured with `tools/live_cost_probe.py`).

## Assumptions

- **Dimension and price bars:** pass when the mean absolute percentage error is within the bar and
  every sample line was produced.
- **Wall area** is gross: perimeter times height. Net of known doors and windows is computed too.
- **Used value** is the median asking price of active used listings, because eBay's public data does
  not include sold prices.
- **Appraisal threshold:** 150 in the local currency (`APPRAISAL_THRESHOLD`).
- **Shelf run** is the length of shelf occupied by books, gaps excluded.
- **Room scan:** the phone is held about 25 cm in front of the body's turning axis.

## Known limitations

- Room size is the weakest part. On the demo the room came out 85% too large even with the marker
  in view: the marker fixes the scale of the frame it's in, but Depth Pro's scale varies between
  frames. A web page can't use the phone's LiDAR.
- When one frame shows two shelves, books get counted under both. Identified books are now merged;
  unreadable ones can't be.
- Thin spines with no readable text are the most common miss on packed shelves.
- Leaning books measure short, and item sizes are the front face only.
- iPhone screen recording leaves out a web page's call audio, so the agent's voice doesn't appear in
  a screen recording. The demo was filmed with a second phone.

## Layout

```
src/libraryclaim/        pipeline: quality, scale, vision, resolve, pricing, review, room scan, packet, report, evaluate
src/libraryclaim/live/   live sweep: session state, agent and tools, WebSocket server, phone page
tests/                   unit and end-to-end tests on synthetic shelves and rooms
tools/                   offline demo, markers, certificate, photos to sweep, cost probe, packaging
ground_truth/            printable markers and the ground truth sheet
submission/              the demo sweep's packet, report, frames and scores
results/                 runs cited in the failure log, with index.md
```
