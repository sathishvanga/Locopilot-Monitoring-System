# Executor brief — Locopilot monitor refactor

You are a kiro-executor working on ONE extraction task from the plan at:

`/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/docs/specs/locopilot-refactor/PLAN.md`

## Your task

You will be given a single task ID (e.g. `T1`, `T3`, `T5`). Open `PLAN.md`,
find your section under "Section 2 — Independent Extraction Tasks", and
execute it precisely.

## Worktree

You operate in your own git worktree based on
`feature/may-2nd-enhancements`. The base repo path is:

`/Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System`

Your branch is named after your task ID (e.g. `refactor/T1`).

## Hard rules — non-negotiable

1. **Do NOT edit `locopilot_monitor.py`.** Not a single line. The rewire
   that updates the monolith is reserved for the final sequential task `TR`
   and will be done by a different executor. Touching the monolith creates a
   merge conflict between parallel workers.
2. **Do NOT edit `app/controllers/video_controller.py` or
   `app/services/video_processing_service.py`.** Those files have
   uncommitted changes from another developer; stay clear of them.
3. **Create new files only.** Your task lists exactly which new file(s) to
   create. Two parallel executors will never share a target file — if you
   find yourself wanting to edit a file already created by another task,
   stop and re-read the plan; you have the wrong scope.
4. **Behavior must be byte-identical.** This is a deploy-by-rsync codebase
   with no git on the production server. Preserve every log message string
   verbatim, every numeric default, every condition. If your task spec
   says "lift lines 963-1081", do exactly that with the only allowed delta
   being: `self.foo` reads become `foo` arguments per the State Contract
   in your task.
5. **No new dependencies.** stdlib + numpy + cv2 + the existing
   `app.core.*` modules + `app.services.*` modules already used by the
   monolith. No emojis in code.
6. **Singletons / `__init__` patterns:** the existing extracted modules
   (e.g. `SleepDetector`, `EvidenceManager`) are constructed by the
   monolith and held as `self.*`. Match that idiom — your new class (if
   any) takes a logger and config parameters in `__init__`; do NOT
   instantiate yourself at module level.

## What "done" looks like for your task

- The new files in your task spec exist and import cleanly.
- The verification command in your task spec runs to "ok" with the env
  `/Users/satishvanga/miniconda3/envs/vanga/bin/python3.11`.
- The unit test file under `tests/refactor/` exists for your task and
  passes with that same env: `pytest tests/refactor/test_<task>.py -q`.
- `locopilot_monitor.py` is byte-identical to its state at the start of
  your task (`git diff feature/may-2nd-enhancements -- locopilot_monitor.py`
  shows nothing).
- Commit message format: `refactor(<area>): extract <thing> from monolith
  (T<N>)`. No Co-Authored-By unless the user already requested it.

## When you are unsure

- Ask the planner before you invent. The plan is the contract; if your
  task seems to require touching a file outside its declared scope, that's
  a planning bug — flag it and stop.
- Do not write a SUMMARY.md or REPORT.md. Final output is your code +
  one-line PR description.
