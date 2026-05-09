# Task 0008 — Extract `process_all_persons_activities` from the monolith

**Severity:** HIGH (god-method, top of monolith refactor backlog)
**Source:** Architecture review 2026-05-09, finding #2.
**Estimated effort:** 2–3 days. THIS IS THE RISKIEST TASK IN THE CLEANUP.

---

## Problem

`LocopilotActivityMonitor.process_all_persons_activities` lives at `locopilot_monitor.py:1777–2964` — **~1,200 lines inside one method on one god-class**. It is the per-frame multi-person dispatcher: it runs YOLO-Pose, matches poses to roles, then for each person iterates all activity detectors (sleep, microsleep, cell phone, writing, packing bags, hand gestures, mind diversion, eating-drinking) and produces aggregated flags. It calls into `self.sleep_detector`, `self.activity_detector`, `self.gesture_detector`, `self.mind_diversion_detector`, `self.object_detector` — those detector extractions are already done. What remains is the orchestrator.

Extracting this into its own class:
- Cuts the monolith file by ~1,200 LOC (from 4,407 → ~3,200).
- Lets new contributors reason about per-frame multi-person dispatch in isolation.
- Lets us write unit tests against the orchestrator with stub detectors.

---

## Behavior contract — the strictest in the cleanup

This is a MOVE, not a redesign. The output of `process_all_persons_activities` for a given input must remain **byte-identical**. The following e2e tests are the contract:
- `pytest tests/regression/` — real-video fixture suite.
- `pytest tests/detectors/` — detector reset/coordination contract.
- `pytest tests/test_train_stopped_resume.py`, `pytest tests/test_determinism.py` — behavior contracts.

If any of these fail, you have changed behavior. STOP and revert.

---

## Files to change

**Create:**
- `app/core/multi_person_runner.py` (~1,250 LOC) — `MultiPersonActivityRunner` class containing the extracted method body. Constructor takes detector instances + tracking state by reference (NOT by copy — it must mutate the monolith's existing state).

**Modify:**
- `locopilot_monitor.py` — replace the `process_all_persons_activities` method body with a delegation:
  ```python
  def process_all_persons_activities(self, *args, **kwargs):
      return self._multi_person_runner.run(self, *args, **kwargs)
  ```
  In `__init__`, wire `self._multi_person_runner = MultiPersonActivityRunner()`.
- The runner's `run(monitor, frame, detections, person_roles, ...)` method is the relocated body. Every `self.foo` reference inside the original method becomes `monitor.foo`.

**DO NOT touch:**
- The detector classes (`SleepDetector`, `ActivityDetector`, `GestureDetector`, `MindDiversionDetector`, `ObjectDetector`) — they're already extracted.
- Other monolith methods (`_process_frames_core`, `start_activity`, `end_activity`, `process_video`, etc.).
- The pipeline scaffolding — already deleted in Wave 1.

---

## Extraction recipe

1. **Snapshot first.** Run the full regression suite *before* any change. Save the `activities.json` output of one representative video; that's your golden snapshot.
2. **Copy the entire method body verbatim** into `MultiPersonActivityRunner.run(self, monitor, frame, detections, person_roles, timestamp_sec, face_results=None, frame_number=None, precomputed_pose_results=None, precomputed_sleep_pose_results=None, is_dark_frame=None)`.
3. **Mechanically replace `self.X` with `monitor.X`** everywhere the original method referenced the monitor's attributes — `self.sleep_detector`, `self.activity_detector`, `self.logger`, `self.consecutive_detections`, `self.activity_thresholds`, `self.per_person_sleep_tracking`, etc. Use `sed` or a careful manual pass; nothing else should change.
4. **Replace any helper-method calls** inside the original method body (`self._helper(...)`) with `monitor._helper(...)`. If a helper is purely about per-frame multi-person logic and only used here, consider moving it to the runner — but only if there's clearly zero risk. Default: leave helpers on the monitor.
5. **In `LocopilotActivityMonitor.__init__`**, instantiate `self._multi_person_runner = MultiPersonActivityRunner()` AFTER all detectors are initialized.
6. **Replace the original method body** with the one-line delegation shown above.
7. **Re-run the regression suite.** Output must be byte-identical (modulo wall-clock fields). If it isn't, revert and find the variable you mis-replaced.

---

## What you must NOT do

- Do not rename any variable.
- Do not "improve" any docstring or comment.
- Do not split the runner's `run()` method into smaller methods. That's a future task.
- Do not change the signature of `process_all_persons_activities` — it is called from elsewhere in the monolith and via `monitor.process_all_persons_activities(...)` in `app/core/pipeline/stages/per_person_activities_stage.py`... wait, that file was deleted in Wave 1. So today, the only caller is inside `_process_frames_core`. Verify with `grep -rn "process_all_persons_activities" --include="*.py" .` before editing.
- Do not introduce dependency injection beyond what's required (one new instance attribute).
- Do not touch the detector instances' state.

---

## Acceptance criteria

1. `pytest tests/` is **fully green**, identical pass count to pre-refactor.
2. `pytest tests/regression/` is green; output JSONs match the pre-refactor golden snapshots byte-for-byte (excluding wall-clock fields).
3. `pytest tests/test_determinism.py` is green.
4. `wc -l locopilot_monitor.py` shows reduction of at least 1,100 LOC (from ~4,407 to ~3,300 or less).
5. `wc -l app/core/multi_person_runner.py` shows roughly that same number of lines added.
6. `python -c "from locopilot_monitor import LocopilotActivityMonitor; m = LocopilotActivityMonitor.__init__; print('ok')"` — at minimum the class imports cleanly. (If running `__init__` requires GPU/video, skip the instantiation.)
7. `grep -rn "process_all_persons_activities" --include="*.py" /Users/satishvanga/Documents/Experiment/Locopilot-Monitoring-System/` shows the method exists on `LocopilotActivityMonitor` (one-line delegate) AND `MultiPersonActivityRunner.run` exists. The full body is in the runner only.

---

## If anything goes wrong

- **First failed regression test:** stop. Don't try to "fix it forward." Revert to last green and re-do the extraction more carefully. The bug is almost always a `self.X` you forgot to convert to `monitor.X`.
- **Behavior diverges by one violation:** likely a tracking-dict reference. Check `monitor.per_person_sleep_tracking`, `monitor.consecutive_detections`, `monitor.last_activity_state` — these are mutated and any divergence propagates.
- **Method needed an attribute that's set later in `__init__`:** the runner instantiation must happen after that attribute. Move the wiring line in `__init__` if needed.

---

## Out of scope

- Splitting the runner's `run()` method into smaller methods.
- Extracting per-detector dispatch into a registry.
- Removing the monolith's `process_all_persons_activities` entry point — keep the one-line delegate. Other callers may bind to it.
- Refactoring `_process_frames_core` itself. That's a separate (not currently scheduled) task.
- Cleaning up `set_trip_schedule` plumbing left from Wave 2 — out of scope.
