# Ground-truth dataset for detection regression

Hand-labelled violations per video, used to measure the algorithm's
recall / precision over time. **Source of truth** for what the detector
SHOULD find in each test video.

## Why this exists

Before this directory, every algorithm change was a blind tuning pass —
fix one FP, quietly lose a real detection elsewhere, no way to know.
Ground-truth JSONs + a scoring script turn each change into a
measurable delta: `+2 phone recall, −0 writing precision`.

## File format

One JSON per video, named after the source filename (without `.mp4`).
Schema:

```json
{
  "video_filename": "all_activities.mp4",
  "video_url": "https://gpu.mindcoinapps.com:9000/cvss/all_activities.mp4",
  "duration_sec": 2288.76,
  "resolution": "1280x720",
  "reviewed_date": "2026-04-20",
  "reviewer": "manual+subagent dense 0.5fps sampling",
  "notes": "Free-form context — stitched compilation, camera angle, time of day, etc.",
  "violations": [
    {
      "id": 1,
      "type": "cell_phone",
      "ocr_start": "10:46:33",
      "ocr_end":   "10:46:49",
      "video_start_sec": 276.0,
      "video_end_sec":   288.0,
      "duration_sec":     12.0,
      "motion_state":    "RUNNING",
      "person_role":     "ALP",
      "confidence":      "high",
      "evidence":        "ALP smartphone clearly held to right ear",
      "detected_by_automated": false
    }
  ]
}
```

## Field conventions

| Field | Values | Purpose |
|---|---|---|
| `type` | `cell_phone`, `microsleep`, `sleep`, `writing`, `packing_bags`, `eating_drinking`, `hand_gesture`, `mind_diversion`, `group_presence`, `no_person` | Matches activity_registry keys |
| `motion_state` | `RUNNING`, `STOPPED`, `UNCERTAIN` | Your visual judgement; crucial for "while running" violations |
| `person_role` | `LP`, `ALP`, `both`, `unknown` | If ambiguous, `unknown` |
| `confidence` | `high`, `medium`, `low` | How sure are you? Low-confidence events can be filtered out of strict-recall metrics |
| `video_start_sec` / `end_sec` | Absolute seconds from video start | Canonical time reference. Use this, not OCR, in the scoring script |
| `ocr_start` / `ocr_end` | `HH:MM:SS` | Human-readable cross-reference, matches what shows in annotated frames |
| `detected_by_automated` | bool | Did the current pipeline catch it? Update after each major algorithm change |

## Adding a new video

1. Upload to MinIO `cvss` bucket
2. Run the detector once — note its output activities
3. Manually (or with the dense-review subagent) scan the video end-to-end
4. Fill out the JSON — be **strict about edge cases**:
   - If a behaviour is visible for < 3 seconds, mark `confidence: "low"` (below detector temporal-filter floor, won't be scored harshly)
   - If train state is ambiguous, mark `motion_state: "UNCERTAIN"`
5. Commit the JSON alongside a short note in git describing what you saw
6. Re-run the scoring script; verify the baseline numbers update sanely

## Scoring (future)

A `scripts/eval_recall.py` will read each ground-truth JSON, call
`/api/video/analyze` on the video, and compute per-type recall / precision
by matching detections to ground-truth events (overlap on time window).
Exit non-zero if recall drops below a configured threshold — suitable
for CI.

Until the scoring script lands, `detected_by_automated` is the
manually-maintained ground truth about current pipeline state.

## Current dataset

| Video | Duration | Violations labelled | Should-fire (non-exempt) | Last reviewed |
|---|---|---|---|---|
| `all_activities.mp4` | 38:08 | 9 | 6 | 2026-04-20 |
| `ch04_20260402025737_.mp4` | 22:06 | 4 | 0 (all at station stop, exempt) | 2026-04-20 |
| `ch04_20260402034641_.mp4` | 27:49 | 2 | 0 (ambiguous / operational) | 2026-04-20 |

**Aggregate baseline @ commit 24197e2 (2026-04-20)**:
- 15 labelled violations across 3 videos
- 6 should fire per spec
- 2 detected correctly (microsleep @ 1630s, packing @ 11:15 on `all_activities.mp4`)
- 4 missed — 3 cell_phone events + 1 sleep event
- Overall should-fire recall: 2/6 = **33%**
