# Failure log

Every number here comes from a real run archived in [`results/`](results/index.md) (packet, timings and
session log per run). Home rehearsals: `live-20261029-192400` (before the fixes) and `live-20261029-230838`
(after), same room, same 6 books; `replay-home2` replays one set of frames through each version of the code.

## The three worst errors

**1. Non-book items over-counted 3-4x, which inflated the contents value.**
The 19:24 home sweep listed 62 items worth USD 3,359–7,205 in one small room. The list had the same mirror, keyboard and trash
can twice each, a fridge and dishwasher from the next room, packaging boxes, a curtain rod and a wall hook.
*Root cause:* items were detected per view and merged only when their boxes overlapped in the same frame,
so every new angle made a new item. The prompt also accepted fittings, packaging, other rooms and objects cut off by the frame.
*Change:* the prompt asks for insurable contents (about $25 and up) with a plain one- or two-word name, and
excludes fittings, packaging, other rooms and partial views. A detection is the same item as an earlier one when the name and
brand match within 3 views, or the colours agree. Measured below.

**2. The packet took 111 s after the user pressed End, while the phone kept recording.**
*Root cause:* the item detection, depth and pricing work for the room ran only after End, and End did not stop the
camera or the agent session. In the 19:24 log the agent was still talking 33 s after the packet was done,
and the session stayed open 78 s past it.
*Change:* items are detected and priced while the user is still walking, on a worker pool. End stops capture,
waits for the agent's summary, then hangs up. The phone shows a finishing screen with live progress.
*Result:* 111.0 s → 40.6 s (23:08 run). What remains is the room scan (59.7 s of Depth Pro on the turn frames,
partly overlapped with the sweep). Next step: run depth on each turn frame as it arrives, during the turn.

**3. Room height read as 4.34 m in a normal house.**
The 23:08 sweep measured the floor at 6.36 × 4.36 m (20.62 m²), but the height at 4.34 m. A typical house
ceiling is about 2.4–2.7 m.
*Root cause (likely):* floor and ceiling are taken as the extreme horizontal layers. One mis-scaled depth frame
looking up during the turn pushes the ceiling layer too high, and the marker that corrects the depth scale was
not in view during the turn.
*What I would change:* take a robust percentile of the ceiling points rather than the extreme, and require
the wall marker to be seen in the turn before reporting a height. Until then, the height is flagged in the report.

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

- **Book reading too slow:** 620 s per sweep → 23 s. One model call per spine became one detect-and-read
  call per tile, with low thinking, streamed per shelf.
- **Venue Wi-Fi blocked phone-to-laptop traffic** (a store guest network on the day). The phone's own
  hotspot avoids it; the setup checklist says so.
