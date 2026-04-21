# Locopilot Monitoring — Business Requirements Document

## 1. Purpose

Automated CCTV-based safety compliance monitoring for Indian Railways locomotive crews. The system reviews cabin video footage from the Loco Pilot (LP) and Assistant Loco Pilot (ALP) cameras and flags safety-critical behaviors that violate operating discipline while the train is in service.

## 2. Scope

- Inputs: Recorded CCTV footage from overhead cabin cameras (LP side and ALP side).
- Outputs: Time-stamped activity reports, evidence frames/clips, and summary dashboards for safety auditors.
- Users: Railway safety officers, divisional supervisors, compliance teams.

## 3. Crew Roles

- **LP (Loco Pilot)**: Primary driver, seated at controls, must remain alert and forward-facing.
- **ALP (Assistant Loco Pilot)**: Assists with signals, logbook, and coordination gestures.

## 4. Monitored Activities

### 4.1 Safety-Critical (monitored at all times — running or stopped)

| Activity | Business Rule |
|----------|---------------|
| **Microsleep** | Eyes closed continuously for 5 seconds or more. Indicates dozing at controls. |
| **Cell Phone Use** | Mobile phone visible in crew member's hand or near face. Prohibited in cabin at all times per railway rules. |

### 4.2 In-Motion Violations (monitored only when train is running)

| Activity | Business Rule |
|----------|---------------|
| **Sleep** | Sustained head drop, reclined posture, or closed eyes for 30+ seconds. Severe fatigue violation. |
| **Writing Logbook While Running** | Crew writing in logbook while train is in motion. Logbook entries must be made at stations, not while the train is running. |
| **Packing Bags** | Crew handling personal bags/backpacks while train is running. Indicates premature shift handover or distraction. |
| **Coordination Gesture Failure** | One crew member performs the required coordination hand gesture (raise-hold-lower) and the counterpart fails to reciprocate within the expected response window. Reported as **LP did not reciprocate** or **ALP did not reciprocate** depending on who failed to respond. A missing gesture is only a violation when a gesture was initiated — solo absence of gestures is not flagged. |
| **Mind Diversion** | Crew attention diverted from controls — looking sideways, looking down for extended periods, or looking away from forward view. |
| **Eating / Drinking** | Cup or bottle brought to face while train is running. |

### 4.3 Structural / Presence Checks (always monitored)

| Activity | Business Rule |
|----------|---------------|
| **No Person Detected** | Cabin empty (no crew visible) for 10+ seconds. Indicates abandoned controls. |
| **Group Detected** | More than 5 people in the cabin. Indicates unauthorized persons riding in the loco beyond normal crew and occasional authorised staff. |

## 5. Train Motion Awareness

The system must distinguish whether the train is **RUNNING** or **STOPPED**, because crew behavior rules differ by state.

### 5.1 Why Motion State Matters

Many activities are legitimate when the train is stopped at a station (logbook entry, bag handling, eating/drinking, no need for coordination gestures). Flagging them as violations would produce excessive false alerts and undermine auditor trust.

### 5.2 Motion Detection Approach

Train motion is inferred from the cabin video itself — no external GPS or speedometer feed is required. Two visual cues are fused:

1. **Cabin vibration** — train vibrations on static interior surfaces (excluding persons and windows) indicate the train is moving on tracks.
2. **Interior stability changes** — block-level variance shifts in the cabin confirm sustained motion vs. a still frame.

A combined confidence score over a rolling window produces one of three states: **RUNNING**, **STOPPED**, or **UNCERTAIN** (treated conservatively as RUNNING to avoid missing violations).

### 5.3 Activity Gating Rules by Motion State

| Activity | When Train is RUNNING | When Train is STOPPED |
|---|---|---|
| Microsleep | Flag | Flag |
| Cell Phone | Flag | Flag |
| Sleep | Flag | Suppress |
| Writing Logbook | Flag | Suppress |
| Packing Bags | Flag | Suppress |
| Coordination Gesture Failure | Flag | Suppress |
| Mind Diversion | Flag | Suppress |
| Eating / Drinking | Flag | Suppress |
| No Person Detected | Flag | Flag |
| Group Detected (> 5 persons) | Flag | Flag |

## 6. Coordination Gesture Logic

The coordination gesture is a two-party signal: one crew member raises a hand in acknowledgement of a signal/event and the other reciprocates.

**Detection rule:**
1. Detect that **Person A** (LP or ALP) initiated a valid gesture (raise → hold → lower trajectory).
2. Start a response window (configurable, default: a few seconds).
3. Within the window, check if **Person B** (the counterpart) performs a matching gesture.
4. If Person B does **not** reciprocate → raise violation against **Person B** ("ALP did not reciprocate" or "LP did not reciprocate").
5. If neither person initiates a gesture → **no violation**. The system does not flag absence of conversation, only failure to respond to an initiated gesture.

**Rationale:** Many railway sections don't require continuous gesturing. Only the exchange — one signalling, the other acknowledging — is the safety-critical behaviour.

## 7. Reporting Requirements

- **Evidence**: Each flagged activity produces a snapshot frame and a short video clip with the crew member clearly visible.
- **Timestamps**: Each activity record carries the wall-clock time (OCR'd from on-screen overlay or derived from video).
- **Per-role attribution**: Violations are attributed to LP or ALP (not "someone in the cabin"). For gesture-reciprocation violations, the record names who initiated and who failed to respond.
- **Duration**: Start and end time of each sustained activity.
- **Motion context**: Whether the train was running or stopped when the activity occurred.

## 8. False Positive Controls

- Temporal filtering: activity must persist for a minimum duration and consecutive frames before reporting.
- Baseline calibration: first 10–30 seconds used to learn crew's normal posture before flagging deviations.
- Confounder handling: operating radio handset, pressing overhead controls, or looking at instrument panel must not be misreported as phone use or mind diversion.
- Gesture reciprocation: a gesture-failure event is only raised after a valid initiator gesture is confirmed, preventing solo-idle crew from being flagged.

## 9. Configurability

Safety officers / ops teams must be able to tune per-site:
- Detection sensitivity (strict vs. lenient).
- Motion detection on/off (disable for shunting yards or stabled rakes).
- Motion-based suppression on/off (force all activities always evaluated, for diagnostic audits).
- Camera side (LP cabin vs. ALP cabin).
- Group-size threshold (default 5; site-adjustable).
- Gesture reciprocation response window duration.

## 10. Success Criteria

- High recall on the safety-critical activities (microsleep, cell phone, sleep).
- Low false-positive rate on station-dwell footage (≤ 5% of flagged activities traceable to legitimate station behavior).
- Gesture-failure flags correlate with a confirmed initiating gesture in ≥ 95% of cases (no solo-absence false alarms).
- End-to-end per-trip report generated without manual review of raw footage.

## 11. Out of Scope

- Live / real-time alerting (system processes recorded trips post-hoc).
- Automatic disciplinary actions — the system reports; human supervisors decide.
- Audio analysis — video-only.
- Track-side or external-view cameras.

## 12. Evidence API Access Control

Evidence clips and the per-run processing status contain PII (cabin video,
crew IDs). Internal clients fetching these resources must authenticate each
request with a shared API key.

### 12.1 Protected endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/jobs/{run_id}/media/{filename}` | Serve evidence clips and images for a run. |
| `GET /api/status/{run_id}` | Read per-run processing status. |

The health endpoint (`GET /api/health`) and the upload / processing POST
endpoints are unaffected by this requirement and remain reachable without
the header.

### 12.2 Required header

Clients must send the header `X-API-Key: <key>` on every request to the
endpoints in 12.1. Missing or mismatched keys receive a `401
invalid_or_missing_api_key` response. Values are compared with
`hmac.compare_digest` to avoid timing-side-channel leakage.

### 12.3 Configuration and rotation

The expected key is read from the `MEDIA_API_KEY` environment variable
(loaded via `app/utils/config.py :: Settings.media_api_key`). The
server-side source of truth in production is `.env.production` on the GPU
server.

Rollout behaviour: when `MEDIA_API_KEY` is unset or empty, the auth
dependency logs a one-shot warning per process and allows the request
through, so existing clients keep working during the rollout window. Set
the variable once every client has been updated to send the header.

Key rotation procedure:

1. Ops generates a new random key (e.g. `python -c "import secrets;
   print(secrets.token_urlsafe(32))"`).
2. Update `MEDIA_API_KEY` in `.env.production` on the GPU server.
3. Redeploy / restart the `locopilot` systemd service so the new value is
   picked up (`sudo systemctl restart locopilot`).
4. Update the key in every internal client's secret store.

Keys must never be committed in plaintext to the repository history.
