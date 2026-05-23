#!/usr/bin/env python3
"""Compare two VLM models against the same activity set + keyframes.

Designed for head-to-head evaluation of a candidate VLM (e.g.
InternVL3-14B-AWQ) against the production baseline (Qwen2.5-VL-7B-AWQ).
Reads an existing ``activities.json`` from a Locopilot run directory,
re-runs the VLM call against the supplied endpoint using the SAME
production prompts and SAME keyframe stitching, and writes a structured
JSON of per-activity verdicts plus a markdown comparison report.

Usage:
    # Step 1: capture baseline (current vLLM endpoint, default :8001)
    python scripts/vlm_benchmark.py run \\
        --run-dir /opt/poc2/locopilot_evidence/run_20260520_153244 \\
        --vlm-url http://localhost:8001/v1 \\
        --vlm-model "Qwen/Qwen2.5-VL-7B-Instruct-AWQ" \\
        --label qwen2.5-vl-7b-awq \\
        --out /tmp/bench_qwen.json

    # Step 2: capture candidate after swapping the vLLM service
    python scripts/vlm_benchmark.py run \\
        --run-dir /opt/poc2/locopilot_evidence/run_20260520_153244 \\
        --vlm-url http://localhost:8001/v1 \\
        --vlm-model "OpenGVLab/InternVL3-14B-AWQ" \\
        --label internvl3-14b-awq \\
        --out /tmp/bench_internvl.json

    # Step 3: diff + markdown report
    python scripts/vlm_benchmark.py compare \\
        --a /tmp/bench_qwen.json \\
        --b /tmp/bench_internvl.json \\
        --out /tmp/bench_report.md
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set a writable LOG_DIR before importing app.* — the production default
# is /opt/poc2/logs which only exists on the server. This lets the script
# run from a dev Mac too. No-op on the server where LOG_DIR is set
# correctly via .env.
os.environ.setdefault("LOG_DIR", os.path.join(tempfile.gettempdir(), "locopilot_bench_logs"))

import httpx

# Ensure the project root is on sys.path so we can import the production
# code's prompts + keyframe processor (which is what we want — the
# benchmark MUST use the same prompts as production to be apples-to-apples).
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from app.services.vlm.verdict_parser import (  # noqa: E402
    _PROMPTS_BY_OBJECT_TYPE,
    _parse_verdict,
)
from app.services.vlm.keyframe_processor import (  # noqa: E402
    _FULL_FRAME_OBJECT_TYPES,
)

# Known Qwen confabulation tics — used to flag hallucinated motion/person
# observations regardless of which model emitted them. If the same exact
# phrase appears across many verdicts, the model is generating boilerplate
# rather than describing the image.
_QWEN_STOCK_MOTION_PATTERNS = [
    r"FRAME\s*\d+:\s*motion\s+blur\s+in\s+(right|left)\s+window",
    r"motion\s+blur\s+visible\s+in\s+the\s+window",
    r"scenery\s+is\s+streaking",
]
_STOCK_REGEX = re.compile("|".join(_QWEN_STOCK_MOTION_PATTERNS), re.IGNORECASE)

# Sometimes the VLM claims "6 persons visible" on group_detected frames
# that have only 3 — flag that too.
_PERSON_COUNT_REGEX = re.compile(r"(\d+)\s+(?:distinct\s+)?(?:individuals|persons|people)", re.IGNORECASE)


def _stitch_keyframes(
    paths: List[str], crop_to_roi: bool, stack: str = "horizontal"
) -> Optional[bytes]:
    """Minimal in-script stitcher mirroring the production behavior.

    Loads each JPG, optionally crops to a centered ROI (placeholder — the
    production crop uses the activity's bbox metadata; for the benchmark we
    skip crop and let the VLM see the whole frame). Concatenates horizontally
    or vertically to a single JPG byte buffer.

    Returns None if any image can't be loaded.
    """
    import cv2
    import numpy as np

    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [warn] could not read {p}", file=sys.stderr)
            return None
        imgs.append(img)
    if not imgs:
        return None

    # Normalize heights (horizontal) or widths (vertical) before concat.
    if stack == "horizontal":
        target_h = max(i.shape[0] for i in imgs)
        resized = [
            cv2.resize(i, (int(i.shape[1] * target_h / i.shape[0]), target_h))
            for i in imgs
        ]
        strip = np.hstack(resized)
    else:
        target_w = max(i.shape[1] for i in imgs)
        resized = [
            cv2.resize(i, (target_w, int(i.shape[0] * target_w / i.shape[1])))
            for i in imgs
        ]
        strip = np.vstack(resized)

    # Downscale to keep image-token count well under vLLM max_model_len.
    # Qwen2.5-VL produces ~256 visual tokens per 28x28 patch group; a
    # 1280x720 frame is ~1280 visual tokens which combined with the writing
    # prompt's ~1900 text tokens overflows the 3072 budget. 800px wide is
    # ~620 visual tokens with room to spare.
    max_w = 800
    if strip.shape[1] > max_w:
        scale = max_w / strip.shape[1]
        strip = cv2.resize(
            strip, (max_w, int(strip.shape[0] * scale)),
        )

    # Annotate FRAME N labels on each tile so the prompt's "frame N" rules apply.
    n = len(imgs)
    if n > 1 and stack == "horizontal":
        tile_w = strip.shape[1] // n
        for i in range(n):
            x = i * tile_w + 12
            cv2.rectangle(strip, (x - 6, 8), (x + 90, 36), (0, 0, 0), -1)
            cv2.putText(
                strip, f"FRAME {i+1}", (x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

    ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else None


def _resolve_keyframes_for_activity(act: Dict[str, Any]) -> List[str]:
    """Find on-disk keyframe paths for an activity.

    Pipeline-1 writes per-frame keyframes alongside the activity clip with
    a predictable name pattern. We look up to 5 consecutive frames around
    the recorded activityImage (the strongest-evidence frame).
    """
    primary = act.get("activityImage") or ""
    if not primary or not os.path.exists(primary):
        return []

    # Activity image filenames look like:
    # <prefix>_<type>_frame<NNNNNNNN>_<idx>_activity.jpg
    # Sibling burst keyframes share the same <prefix>_<type>_frame<...> pattern;
    # for the benchmark we send just the primary image as a single-frame strip.
    # That matches what the production VLM call does for many short activities.
    return [primary]


def _call_vlm(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout_sec: float = 60.0,
) -> Tuple[Dict[str, Any], float]:
    """POST one chat-completion request and return (parsed_body, latency_sec).

    Raises on transport errors; the caller decides whether to record as
    a parse_error or a timeout / failure.
    """
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 400,
        "temperature": 0.0,
        # NOTE: response_format={"type":"json_object"} triggers vLLM's
        # xgrammar structured-output path which crashed InternVL3-8B
        # mid-run (nanobind "leaked function" errors). The production
        # service uses it (and Qwen handles it fine), but for the
        # benchmark we rely on the prompt's "STRICT JSON ONLY"
        # instruction + _parse_verdict's tolerant fallback. Apples-
        # to-apples is preserved because both models see the same
        # prompt; only the post-parse path differs (the tolerant
        # parser already handles both formatted and code-fenced JSON).
    }
    t0 = time.time()
    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
    latency = round(time.time() - t0, 3)
    resp.raise_for_status()
    return resp.json(), latency


def _extract_text(body: Dict[str, Any]) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def _detect_stock_phrase(text: str) -> bool:
    return bool(_STOCK_REGEX.search(text or ""))


def _detect_person_count(text: str) -> Optional[int]:
    m = _PERSON_COUNT_REGEX.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _synthesize_activities_from_clips(run_dir: Path) -> List[Dict[str, Any]]:
    """Reconstruct an activity list from clips/<...>_<type>_frame<N>_..._activity.jpg.

    Used when activities.json is empty (the post-VLM rewrite dropped
    everything, but the pre-VLM keyframes are still on disk). Extracts:
      - objectType from the filename's <type> token
      - activityStartTime from the frame number, assuming the source video
        is 25 fps (worst case the timestamp is for display only — the
        benchmark uses object_type, not the timestamp, for routing).
    """
    clips_dir = run_dir / "clips"
    if not clips_dir.is_dir():
        return []
    pattern = re.compile(
        r"_(writing|eating_drinking|packing_bags|cell_phone|sleep|microsleep|"
        r"mind_diversion|lp_hand_gesture|alp_hand_gesture|no_person_detected|"
        r"group_detected|solo_person)_frame(\d+)_(\d+)_activity\.jpg$"
    )
    out: List[Dict[str, Any]] = []
    for p in sorted(clips_dir.glob("*_activity.jpg")):
        m = pattern.search(p.name)
        if not m:
            continue
        object_type = m.group(1)
        frame_n = int(m.group(2))
        # Approximate source-video timestamp at 25 fps. The benchmark uses
        # this for display only; route selection keys on object_type.
        t_sec = round(frame_n / 25.0, 2)
        out.append({
            "_synthetic": True,
            "activityType": -1,
            "objectType": object_type,
            "activityStartTime": f"{t_sec:.2f}",
            "activityImage": str(p),
        })
    return out


def cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    activities_path = run_dir / "activities.json"
    if not activities_path.exists():
        print(f"[error] activities.json not found at {activities_path}", file=sys.stderr)
        return 2

    with open(activities_path, "r", encoding="utf-8") as f:
        activities = json.load(f)

    if not activities:
        synth = _synthesize_activities_from_clips(run_dir)
        if synth:
            print(f"[bench] activities.json is empty; synthesized {len(synth)} "
                  f"activities from clips/ filenames")
            activities = synth
        else:
            print(f"[error] activities.json empty and no clips/ keyframes found",
                  file=sys.stderr)
            return 2

    if args.max_activities and len(activities) > args.max_activities:
        activities = activities[: args.max_activities]

    print(f"[bench] run_dir={run_dir} activities={len(activities)} model={args.model}")
    print(f"[bench] endpoint={args.vlm_url}")

    results: List[Dict[str, Any]] = []
    stats = {
        "verdict_counts": {"TRUE_POSITIVE": 0, "FALSE_POSITIVE": 0, "UNCERTAIN": 0, "ERROR": 0},
        "stock_phrase_hits": 0,
        "parse_errors": 0,
        "timeouts": 0,
        "latencies": [],
        "by_object_type": {},
    }

    for i, act in enumerate(activities):
        object_type = (act.get("objectType") or "").strip().lower().replace(" ", "_")
        prompt = _PROMPTS_BY_OBJECT_TYPE.get(object_type)
        if prompt is None:
            print(f"  [{i}] skip: no prompt for object_type={object_type!r}")
            continue

        keyframes = _resolve_keyframes_for_activity(act)
        if not keyframes:
            print(f"  [{i}] skip: no keyframes for activity at t={act.get('activityStartTime')}")
            continue

        crop_to_roi = object_type not in _FULL_FRAME_OBJECT_TYPES
        stack = "vertical" if not crop_to_roi else "horizontal"
        strip = _stitch_keyframes(keyframes, crop_to_roi=crop_to_roi, stack=stack)
        if strip is None:
            print(f"  [{i}] skip: stitch failed")
            continue
        b64 = base64.b64encode(strip).decode("ascii")

        try:
            body, latency = _call_vlm(
                endpoint=args.vlm_url,
                model=args.model,
                prompt=prompt,
                image_b64=b64,
                timeout_sec=args.timeout,
            )
        except httpx.TimeoutException as e:
            stats["timeouts"] += 1
            stats["verdict_counts"]["ERROR"] += 1
            results.append({
                "activity_index": i,
                "activity_type": act.get("activityType"),
                "activity_start_time": act.get("activityStartTime"),
                "object_type": object_type,
                "verdict": "ERROR_TIMEOUT",
                "error": str(e)[:200],
            })
            print(f"  [{i}] TIMEOUT after {args.timeout}s")
            continue
        except Exception as e:
            stats["verdict_counts"]["ERROR"] += 1
            results.append({
                "activity_index": i,
                "activity_type": act.get("activityType"),
                "activity_start_time": act.get("activityStartTime"),
                "object_type": object_type,
                "verdict": "ERROR_HTTP",
                "error": str(e)[:200],
            })
            print(f"  [{i}] HTTP ERROR: {e}")
            continue

        text = _extract_text(body)
        parsed = _parse_verdict(text)
        if not parsed or parsed.get("parse_error"):
            stats["parse_errors"] += 1
            verdict = "PARSE_ERROR"
        else:
            verdict = parsed.get("verdict") or "PARSE_ERROR"

        if verdict in stats["verdict_counts"]:
            stats["verdict_counts"][verdict] += 1
        else:
            stats["verdict_counts"][verdict] = stats["verdict_counts"].get(verdict, 0) + 1

        stock = _detect_stock_phrase(text)
        if stock:
            stats["stock_phrase_hits"] += 1
        person_count = _detect_person_count(text)
        stats["latencies"].append(latency)
        bo = stats["by_object_type"].setdefault(
            object_type, {"n": 0, "verdicts": {}, "stock_hits": 0}
        )
        bo["n"] += 1
        bo["verdicts"][verdict] = bo["verdicts"].get(verdict, 0) + 1
        if stock:
            bo["stock_hits"] += 1

        results.append({
            "activity_index": i,
            "activity_type": act.get("activityType"),
            "activity_start_time": act.get("activityStartTime"),
            "object_type": object_type,
            "verdict": verdict,
            "confidence": (parsed or {}).get("confidence"),
            "primary_object_in_hand": (parsed or {}).get("primary_object_in_hand"),
            "train_appears_to_be": (parsed or {}).get("train_appears_to_be"),
            "motion_evidence": (parsed or {}).get("motion_evidence", "")[:200],
            "reasoning": (parsed or {}).get("reasoning", "")[:300],
            "structured_fields": {
                k: parsed.get(k) for k in (
                    "pen_in_hand", "hand_actually_on_book", "book_visible_on_desk",
                    "actively_handling_papers", "head_oriented_to_book",
                    "eyes_closed", "object_in_hand", "object_at_mouth",
                ) if parsed and k in parsed
            },
            "stock_phrase_detected": stock,
            "person_count_in_text": person_count,
            "latency_sec": latency,
            "raw_text": text[:600],
        })
        v_short = verdict[:2]
        stock_tag = " STOCK" if stock else ""
        print(f"  [{i}] {object_type:18s} t={act.get('activityStartTime'):>8s} "
              f"{v_short} conf={(parsed or {}).get('confidence', 0.0):.2f} "
              f"lat={latency:.2f}s{stock_tag}")

    # Latency stats
    if stats["latencies"]:
        lats = sorted(stats["latencies"])
        stats["avg_latency_sec"] = round(sum(lats) / len(lats), 3)
        stats["p50_latency_sec"] = lats[len(lats) // 2]
        stats["p95_latency_sec"] = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
    del stats["latencies"]

    out = {
        "run_dir": str(run_dir),
        "model_label": args.label,
        "model_name": args.model,
        "endpoint": args.vlm_url,
        "n_activities_total": len(activities),
        "n_activities_evaluated": len(results),
        "stats": stats,
        "per_activity": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[bench] wrote {args.out}")
    print(f"[bench] verdicts: {stats['verdict_counts']}")
    print(f"[bench] avg latency: {stats.get('avg_latency_sec', 0)}s p95={stats.get('p95_latency_sec', 0)}s")
    print(f"[bench] stock_phrase_hits={stats['stock_phrase_hits']} parse_errors={stats['parse_errors']} timeouts={stats['timeouts']}")
    return 0


def _index_by_key(per_activity: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        f"{r['activity_index']}_{r['object_type']}_{r['activity_start_time']}": r
        for r in per_activity
    }


def cmd_compare(args: argparse.Namespace) -> int:
    with open(args.a, "r", encoding="utf-8") as f:
        A = json.load(f)
    with open(args.b, "r", encoding="utf-8") as f:
        B = json.load(f)

    a_idx = _index_by_key(A.get("per_activity", []))
    b_idx = _index_by_key(B.get("per_activity", []))
    common = sorted(set(a_idx.keys()) & set(b_idx.keys()))

    lines: List[str] = []
    L = lines.append
    L(f"# VLM Benchmark: {A['model_label']} vs {B['model_label']}")
    L("")
    L(f"- run_dir: `{A.get('run_dir')}`")
    L(f"- A: **{A['model_label']}** ({A['model_name']})")
    L(f"- B: **{B['model_label']}** ({B['model_name']})")
    L(f"- A activities evaluated: {A['n_activities_evaluated']}")
    L(f"- B activities evaluated: {B['n_activities_evaluated']}")
    L(f"- common activities (key match): {len(common)}")
    L("")
    L("## Headline numbers")
    L("")
    L("| Metric | A | B |")
    L("|---|---|---|")
    for verdict in ("TRUE_POSITIVE", "FALSE_POSITIVE", "UNCERTAIN", "ERROR"):
        a_n = A["stats"]["verdict_counts"].get(verdict, 0)
        b_n = B["stats"]["verdict_counts"].get(verdict, 0)
        L(f"| Verdict: {verdict} | {a_n} | {b_n} |")
    L(f"| Stock-phrase hits | {A['stats']['stock_phrase_hits']} | {B['stats']['stock_phrase_hits']} |")
    L(f"| Parse errors | {A['stats']['parse_errors']} | {B['stats']['parse_errors']} |")
    L(f"| Timeouts | {A['stats']['timeouts']} | {B['stats']['timeouts']} |")
    L(f"| Avg latency (s) | {A['stats'].get('avg_latency_sec', '—')} | {B['stats'].get('avg_latency_sec', '—')} |")
    L(f"| p95 latency (s) | {A['stats'].get('p95_latency_sec', '—')} | {B['stats'].get('p95_latency_sec', '—')} |")
    L("")

    # Per-activity diff table for the cases where verdicts disagree.
    L("## Disagreements")
    L("")
    L("Cases where A and B return different verdicts on the same activity.")
    L("")
    L("| t | type | A verdict (conf) | A reasoning | B verdict (conf) | B reasoning |")
    L("|---|---|---|---|---|---|")
    n_disagree = 0
    for k in common:
        a = a_idx[k]
        b = b_idx[k]
        if a.get("verdict") == b.get("verdict"):
            continue
        n_disagree += 1
        L("| {t} | {ot} | {av} ({ac}) | {ar} | {bv} ({bc}) | {br} |".format(
            t=a.get("activity_start_time"),
            ot=a.get("object_type"),
            av=a.get("verdict"),
            ac=f"{(a.get('confidence') or 0.0):.2f}",
            ar=(a.get("reasoning") or "")[:120].replace("|", "\\|"),
            bv=b.get("verdict"),
            bc=f"{(b.get('confidence') or 0.0):.2f}",
            br=(b.get("reasoning") or "")[:120].replace("|", "\\|"),
        ))
    L("")
    L(f"**Disagreement count: {n_disagree} / {len(common)} common activities ({100 * n_disagree / max(1, len(common)):.1f}%)**")
    L("")

    # Per-object-type breakdown
    L("## Per-object-type")
    L("")
    L("| Type | A: n / TP / FP / Stock | B: n / TP / FP / Stock |")
    L("|---|---|---|")
    types = sorted(set(A["stats"]["by_object_type"].keys()) | set(B["stats"]["by_object_type"].keys()))
    for t in types:
        a = A["stats"]["by_object_type"].get(t, {"n": 0, "verdicts": {}, "stock_hits": 0})
        b = B["stats"]["by_object_type"].get(t, {"n": 0, "verdicts": {}, "stock_hits": 0})
        L("| {t} | {an}/{atp}/{afp}/{ash} | {bn}/{btp}/{bfp}/{bsh} |".format(
            t=t,
            an=a["n"], atp=a["verdicts"].get("TRUE_POSITIVE", 0),
            afp=a["verdicts"].get("FALSE_POSITIVE", 0), ash=a["stock_hits"],
            bn=b["n"], btp=b["verdicts"].get("TRUE_POSITIVE", 0),
            bfp=b["verdicts"].get("FALSE_POSITIVE", 0), bsh=b["stock_hits"],
        ))
    L("")

    out = "\n".join(lines)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[bench] wrote {args.out}")
    else:
        print(out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run benchmark against a VLM endpoint")
    p_run.add_argument("--run-dir", required=True,
                       help="Locopilot run directory containing activities.json + clips/")
    p_run.add_argument("--vlm-url", default="http://localhost:8001/v1",
                       help="vLLM OpenAI-compatible base URL")
    p_run.add_argument("--vlm-model", dest="model", required=True,
                       help="vLLM model name (must match the model the endpoint is serving)")
    p_run.add_argument("--label", required=True,
                       help="Short label for this run (e.g. qwen2.5-vl-7b-awq)")
    p_run.add_argument("--out", required=True, help="Output JSON path")
    p_run.add_argument("--timeout", type=float, default=60.0,
                       help="Per-call HTTP timeout in seconds (default 60)")
    p_run.add_argument("--max-activities", type=int, default=0,
                       help="Cap number of activities (0=no cap)")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="Diff two benchmark runs into a markdown report")
    p_cmp.add_argument("--a", required=True, help="JSON output from `run` (model A)")
    p_cmp.add_argument("--b", required=True, help="JSON output from `run` (model B)")
    p_cmp.add_argument("--out", required=False, help="Markdown output path (omit to stdout)")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
