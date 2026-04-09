# Architecture Review 2026-04 — Tasks

Source: static architecture review of the Locopilot Monitoring System
conducted on 2026-04-09 against branch `feature/cropping-image-applying-yolo`.

This file is the index. One task file per primary recommendation lives under
`tasks/`. The review report identified 8 prioritized recommendations, so there
are exactly 8 primary tasks.

## Primary tasks

| # | Task | Impact | Effort | Axis |
|---|---|---|---|---|
| 0001 | Consolidate activity metadata into a single `ActivityConfig` dataclass | H | M | Duplication / Extensibility |
| 0002 | Extract `_process_frames_core` into an ordered `FramePipeline` with typed Stage objects | H | H | Module boundaries / Rule layering |
| 0003 | Widen `overlap_seconds` default to cover baseline + coordination windows | H | L | Determinism / Chunk-boundary state |
| 0004 | Replace per-call `cv2.VideoCapture` in voting with a long-lived per-worker `VideoReader` | M | L | Performance |
| 0005 | Add `model_validator` to `Settings` and auto-generate `.env.example` | M | M | Config surface / Feature flags |
| 0006 | Add unit + integration test scaffolding (SleepDetector, GestureDetector, FramePipeline snapshot) | H | M | Testability |
| 0007 | Move detector construction into `worker_initializer` and cache on `_worker_models` | M | M | Performance |
| 0008 | Deduplicate `activity_key_map` and apply the train-STOPPED gate to `persons_data` | L | L | Rule layering / Duplication |

## Top 3 "if you do one thing"

1. **0001** — one-afternoon change that immediately kills the `eating_drinking` / `alp_not_standing` drift between the monitor and the mock service.
2. **0003** — two-line config change that closes the chunk-boundary baseline gap silently suppressing pose-based sleep for ~10s of every 15s chunk.
3. **0006** — unblocks every future refactor. Detectors are already isolated-constructible; the only thing missing is a fixture scaffold.
