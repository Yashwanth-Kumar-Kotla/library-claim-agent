# Submission checklist (against the brief)

✅ done and checked · ⏳ needs the demo sweep · 👤 only you can do it

## Scenario

| Brief | Status | Evidence |
|---|---|---|
| Agent greets, confirms country and currency, explains the sweep in 1–2 sentences | ✅ | `live/agent.py` prompt "Moments", `set_locale` tool; 23:08 rehearsal log |
| One continuous pass, target under 3 min | ✅ | 23:08 rehearsal: 83 s from start to End |
| Agent says what it logs, asks to slow down or step closer, asks questions it can't see ("original or print?") | ✅ | quality banner + spoken remarks; art question asked in the 23:08 run (I-12) |
| Inventory fills in live on screen | ✅ | `book` / `item` events → phone tally and list |
| Background agents finish ID, pricing, measurement | ✅ | streaming `SweepRun` pipeline |
| Summary read back and packet within 5 min | ✅ | 40.6 s after End (23:08); worst seen 111 s |
| Never asks to pull books, scan barcodes, photograph items one by one | ✅ | prompt rules; no such tool exists |

## Six capabilities

| Brief | Status | Evidence |
|---|---|---|
| **1. Live sweep:** real-time voice + camera in a browser | ✅ | phone page over LAN HTTPS |
| Agent catches blur, glare, occlusion, missed shelves while the user is there | ✅ | `quality.py` (blur, glare, dark, speed); prompt covers occlusion and skipped shelves |
| Inventory updates during the sweep | ✅ | |
| User interrupts and corrects by voice (first edition, skip shelf, not mine) | ✅ | `record_remark` tool; `test_live_session.py` |
| **2. Books:** detect and count from spines, incl. flat and stacked | ✅ | `vision.py` detect prompt, `lying_flat` |
| Read title, author, publisher; resolve the work and edition where evidence supports it | ✅ | publisher now kept on each book; edition only when the spine's publisher matches the record |
| Spine height and thickness in cm; shelf run in m | ✅ | marker homography; without a marker: per-shelf scale, shelf run = sum of thicknesses (fixed tonight; it was empty before) |
| Explain metric scale | ✅ | ARCHITECTURE.md "Where metric scale comes from" |
| Unreadable spine → unidentified with dimensions, no guessed title | ✅ | `test_honesty_rules_hold` |
| **3. Book prices:** local country and currency | ✅ | eBay marketplace per country |
| Replacement and used value, each with source, date, condition | ✅ | contract fields filled in every real packet |
| Converted prices labelled; no price → excluded from total | ✅ | `converted` flag; `excluded_from_totals` |
| Rare, signed, antiquarian, over-threshold → appraisal, not priced | ✅ | `review.py`; tests |
| Same 10 books priced for a second country | ✅ | `second_country` in packet + report table |
| **4. Non-book items:** detect and classify | ✅ | items prompt; 18 items in the 23:08 run |
| Category, material, approx. dimensions, brand/model if legible | ✅ | description holds material; Depth Pro size |
| Local replacement price with source, or sourced range | ✅ | bare brand ("LG") now searched with the object and priced as a range (fixed tonight) |
| Art defaults to appraisal unless the user says print; agent asks | ✅ | I-12 in the 23:08 run |
| **5. Room:** floor and wall area in m² and ft² | ✅ | report shows both |
| Length, width, height; non-rectangle handled | ✅ | `room.py` shape note in report |
| Shelved wall area separately | ✅ | `shelved_wall_area_m2` |
| Same sweep, no separate measuring pass | ✅ | turn-around during the sweep |
| **6. Packet:** JSON + readable report per sweep | ✅ | all contract fields checked against a real packet |
| Review queue with reasons | ✅ | |
| Totals add up from lines; any figure traceable to frame + source | ✅ | `compute_totals` in code; frame_ref + URL + date on every line |

## Pass bars (need the demo sweep's ground truth)

| Measure | Bar | Status |
|---|---|---|
| Book count | within 5% | ⏳ |
| Title identification | ≥70% of legible, ≤3% confidently wrong | ⏳ (6/6 at home) |
| Spine dimensions | within 15% on 20 | ⏳ |
| Book prices | within 25% on 15 | ⏳ |
| Non-book items | ≥80% found, right category | ⏳ |
| Floor area | within 10% | ⏳ — room height looked wrong at home (4.34 m); being reworked |
| Wall area | within 15% | ⏳ |
| Time to packet | under 5 min | ✅ 40.6 s |

## Do / do not

| Brief | Status |
|---|---|
| Ran the reference first; README says what was kept, changed, dropped | ✅ README "Reference app" |
| Real space: 60+ books, 2+ shelving units, 8+ items, one enclosed room | 👤 ⏳ tonight's sweep |
| Own ground truth before tuning | 👤 ⏳ `ground_truth/ground_truth.xlsx` |
| Pretrained models / APIs listed | ✅ README "Models, services and libraries" |
| **"Use AI coding tools freely. Tell us which."** | 👤 **your call** — the README does not name any, per your rule |
| Separate, testable stages | ✅ 361 tests |
| Every number traceable | ✅ |
| "I do not know" → review queue | ✅ |
| Cost per sweep and latency per stage | ✅ FAILURE_LOG.md, README |
| What broke, what was tried, two more weeks | ✅ FAILURE_LOG.md |
| Assumptions stated | ✅ README "Assumptions" |
| No prices from model memory; no hardcoded catalogue/prices/room | ✅ |
| No stitched or edited video | 👤 one take |
| No people or personal documents in frame | 👤 |

## Deliverables

| # | Item | Status |
|---|---|---|
| 1 | Private repo, runs from README in < 15 min, `.env.example`, no secrets | ✅ fresh clone: install + offline demo + 361 tests in under a minute (cached); 👤 share the private repo with the reviewers |
| 2 | Demo video, one take, ≤ 6 min, agent voice audible | 👤 ⏳ film over the shoulder (iOS screen recording drops the agent's voice) |
| 3 | Packet + report + frames from the demo sweep | ⏳ `tools/package_submission.py data/sweeps/<demo sweep>` |
| 4 | Ground truth + results side by side | ⏳ `uv run python -m libraryclaim.evaluate data/sweeps/<demo sweep> ground_truth/ground_truth.xlsx` |
| 5 | Architecture note, 1 page | ✅ ARCHITECTURE.md |
| 6 | Failure log, 1 page | ✅ FAILURE_LOG.md (home runs); refresh the three errors from the demo sweep |
| — | Reply to the brief's email with repo link, video link, four documents | 👤 |

## After the demo sweep (about 20 minutes)

```bash
uv run python -m libraryclaim.evaluate data/sweeps/<demo sweep> ground_truth/ground_truth.xlsx
uv run python tools/package_submission.py data/sweeps/<demo sweep>
```

Then the pass-bar rows above and the failure log's three errors get the demo numbers.
