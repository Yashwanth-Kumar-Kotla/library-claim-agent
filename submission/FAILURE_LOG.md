# Failure log

Scored run: the demo sweep `live-20261031-031807`. That is 40 books on two bookcases, 15 items, and one bedroom
measured with a tape. Its packet, report, frames and scores are in [`submission/`](submission/eval.md). A replay
runs the same frames through newer code and is listed as `replay-demo` in [`results/`](results/index.md).

| Measure | Demo sweep | Bar |
|---|---|---|
| Book count | 49 vs 40 (22.5% off) | fail |
| Titles correct | 30/40 legible (75%) | pass |
| Confidently wrong | 1/37 (2.7%): "Theodore Boone: The Activist" | pass |
| Spine size | 24% mean error, 10 of 36 samples not produced | fail |
| Items found | 14/15 (93%) | pass |
| Floor / wall area | 19.8 vs 10.7 m² / 57.5 vs 43.8 m² | fail |
| Packet after End | 15 s | pass |

## The three worst errors

**1. Room measured at 19.8 m² for a 10.7 m² room (+85%).** The printed marker was seen during the turn and
corrected the scale by ×1.067. The walls still came out about 30% too far away. *Root cause:* the marker fixes
only the depth scale of the frame it appears in. Depth Pro's scale differs from image to image, and on this room
it read the far walls long. Tilted frames are already rescaled to match the level frames' walls, but the level
frames are not matched to each other. *What I would change:* chain the level frames to the marker frame through
the walls they share, so the marker's scale reaches every direction. On LiDAR phones, use a native capture
(RoomPlan), which a web page cannot reach.

**2. Book count 49 for 40 (+22.5%).** Seven bottom-shelf books were counted again under the top shelf. *Root
cause:* the frame for one shelf also held the next one, and duplicates were only merged within a single shelf.
*Change:* the same identified book on two adjacent shelves of one bookcase is now counted once, and the dropped
line stays in the review queue. *Measured on the same frames:* **49 → 44**. The remaining 4 are unidentified
spines: there is no title to compare, so nothing is merged. *Next:* cut each frame down to the named shelf,
between its two boards.

**3. Spine sizes off 24% on average, 10 of 36 samples missing.** *Root causes:* there was no marker on the
bookcases. Each shelf is scaled from catalogue heights of the books identified on it, and bookcase B had too few
of those, so it fell back to a typical height and read 12–23% tall. Leaning books look 12–18% short. The missing
samples are books that were not identified. *Change:* thickness now comes from boxes split where neighbours
overlap; before, it was the loose box width, which made a 1.6 cm paperback 11 cm. *Measured on one earlier sweep:*
mean error 115% → 37%. Standing the books upright brought the next sweep to 9%. *Next:* a marker on every bookcase.

## Also fixed tonight, measured on real sweeps

- **Wrong titles from series names.** A spine showing only "THEODORE BOONE" was matched to one book of the series.
  The Missal was matched to the 2026 edition while its spine says 2019. A bare series name is now logged as
  unidentified, and a year printed on the spine must agree with the record. On the 02:43 sweep: **2 wrong → 0**.
- **Agent repeated itself and finished early.** It repeated the pan instruction after every item, read an internal
  screen notice aloud, and started the room turn after three shelves. These are now fixed by the tool results'
  next steps. In the demo: 1 interruption in 3.5 minutes.
- **Echo.** iPhone Screen Recording drops a web page's call audio, so the demo was filmed with a second phone.
  Turning echo cancellation off to get the voice recorded made the agent hear itself, so that was rolled back.

## Cost per sweep and latency per stage (demo sweep)

| Stage | Total s | Slowest job s |
|---|---|---|
| Book detect + read (Gemini 3.8 Flash) | 34.7 | 21.7 |
| Title match (Google Books, Open Library) | 17.7 | 3.4 |
| Book prices (eBay via SerpApi) | 0.5 | 0.0 (cached) |
| Item detect (Flash, during the sweep) | 84.9 | 4.7 |
| Item prices | 4.3 | 4.2 |
| Depth (Depth Pro, local GPU) | 52.3 | 21.5 |
| Room scan | 53.8 | 53.8 |
| **Packet ready after End** | **15.4** | |

Almost all of the work runs while the policyholder is still walking, so only 15 s is left after End.

**Cost: $0.42 per sweep.** That is $0.32 for the voice agent (Gemini Live audio and frames for 3.5 minutes) and
$0.10 for 35 vision calls (34 Flash, 1 Pro). Depth Pro runs locally. Prices use about one SerpApi search per book
and item, cached between runs.
