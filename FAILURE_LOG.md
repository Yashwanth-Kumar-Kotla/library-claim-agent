# Failure log

Every number here comes from a real run archived in [`results/`](results/index.md) (packet, timings and
session log per run). Home rehearsals: `live-20261029-192400` (before the fixes) and `live-20261029-230838`
(after), same room, same 6 books; `replay-home2` replays one set of frames through each version of the code.

## The three worst errors

**1. Non-book items over-counted 3-4×, inflating the contents value.** The 19:24 sweep listed 62 items
(USD 3,359–7,205) in one small room: the same mirror, keyboard and bin twice each, a fridge from the next
room, packaging, a curtain rod. *Root cause:* detections were merged only on box overlap within one frame, so
each new angle made a new item, and the prompt accepted fittings, packaging and objects cut off by the frame.
*Change:* the prompt asks for insurable contents (about $25+) with a plain short name and excludes those; a
detection is an earlier item when name and brand match within 3 views or the colours agree. Measured below.

**2. Packet 111 s after End, and the phone kept recording.** *Root cause:* the item, depth and pricing work
started only at End, and End did not stop the camera or the agent (the session stayed open 78 s past the packet).
*Change:* items are detected and priced during the walk; End stops capture, the agent reads the summary and hangs
up. *Result:* 111.0 s → 40.6 s. The rest is the room scan (Depth Pro on the turn frames); next, run it during the turn.

**3. Room read as 6.36 × 4.36 m and 4.34 m high for a small bedroom.** *Root causes, from replaying the turn:*
with the phone upright the browser's angles are in gimbal lock (gamma -55° to +84° on level frames), so frames
were filed under the wrong heading and the outline had 20 corners; Depth Pro's scale differs per image (the one
frame tilted up read about 2× deeper than the one tilted down) and the height came from that one frame; no marker
was in view. *Changes:* heading and pitch from the full rotation (ring, coverage, measurement); each tilted frame
rescaled to the level frames' walls, ceiling as the median over frames; a Manhattan rectangle unless one large notch
shows an L; the marker searched in every turn frame, with "marker ✓" on the ring and the agent asking once.
*Measured on two recorded turns of the room:* clean rectangles of 22.8 and 25.0 m² (within 10% of each other), heights
3.46 and 3.11 m. Floor and height are off by the same factor, about 1.45 if the ceiling is a standard 2.44 m:
Depth Pro's own scale, which the wall marker removes.

## Measured fix (error 1), same frames replayed

| Version | Items | Contents value | After End |
|---|---|---|---|
| Before (live 19:24) | 62 | USD 3,359–7,205 | 111.0 s |
| Replay of home2, old merge | 34 | — | 250.6 s |
| Replay of home2, new prompt + merge | **16** | — | **30.2 s** |
| After (live 23:08, new run) | 18 | USD 1,687–2,510 | 40.6 s |

Review-queue lines: 113 → 26 on the replay, 6 on the 23:08 live run.

## Cost per sweep and latency per stage

The 23:08 live sweep took 2.5 min of walking, with 6 books and 18 items. Times come from `out/timings.json`.

| Stage | Total s | Slowest job s |
|---|---|---|
| Book detect + read (Gemini 3.8 Flash) | 6.0 | 6.0 |
| Title match (Google Books, Open Library) | 0.8 | 0.8 |
| Book prices (eBay) | 0.1 (cached) | 0.02 |
| Item detect (Flash, during the sweep) | 80.4 | 7.0 |
| Item prices (eBay via SerpApi) | 83.6 | 10.1 |
| Depth (Depth Pro, local GPU) | 49.8 | 8.0 |
| Room scan | 59.7 | 59.7 |
| **Packet ready after End** | **40.6** | |

**Cost:** $0.186 for the sweep. That is $0.142 for the voice agent (Gemini Live audio and frames) and $0.0445
for 22 Flash vision calls. Depth Pro runs locally at no cost. Prices use about one SerpApi search per book or item.

## Also found and fixed earlier (measured on real runs)

- **Book reading too slow:** 620 s per sweep → 23 s (one detect-and-read call per tile, streamed per shelf).
- **Venue Wi-Fi blocked phone-to-laptop traffic** (a store guest network on the day). The phone's own
  hotspot avoids it; the setup checklist says so.
