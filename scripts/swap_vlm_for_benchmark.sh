#!/usr/bin/env bash
# Swap the production vLLM service to a candidate model for benchmarking,
# then restore Qwen at the end. RUNS ON THE GPU SERVER ONLY (103.116.80.162).
#
# Production impact: while this script runs the locopilot service's VLM
# verifier will be calling a DIFFERENT model (InternVL3 by default). Any
# /api/video/analyze requests during the benchmark window will use the
# candidate model, not Qwen. Expected duration: ~10 min model download +
# ~5 min benchmark + restart. Plan a quiet window.
#
# Usage (on the GPU server as admin1, with sudo password handy):
#   bash swap_vlm_for_benchmark.sh <CANDIDATE_MODEL_ID> <RUN_DIR>
#
# Examples:
#   bash swap_vlm_for_benchmark.sh OpenGVLab/InternVL3-14B-AWQ \
#        /opt/poc2/locopilot_evidence/run_20260521_120000
#
# Output:
#   /tmp/bench_qwen.json     -- Qwen verdicts (read from existing activities.json's vlm_review)
#   /tmp/bench_internvl.json -- candidate verdicts (from this script's vLLM run)
#   /tmp/bench_report.md     -- side-by-side comparison

set -euo pipefail

CANDIDATE_MODEL="${1:-OpenGVLab/InternVL3-14B-AWQ}"
RUN_DIR="${2:?usage: $0 <CANDIDATE_MODEL_ID> <RUN_DIR>}"
ACTIVITIES_JSON="${RUN_DIR%/}/activities.json"

APP_DIR=/opt/poc2
PY="$APP_DIR/venv/bin/python3"
BENCH="$APP_DIR/scripts/vlm_benchmark.py"

# Production Qwen settings from the systemd unit. Update if the unit changes.
QWEN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
VLM_PORT=8001
VLM_HOST=0.0.0.0
QWEN_MAX_MODEL_LEN=3072
QWEN_GPU_MEM=0.55  # leave room for the detector + 3 GB margin

if [[ ! -f "$ACTIVITIES_JSON" ]]; then
  echo "[error] $ACTIVITIES_JSON not found"
  exit 2
fi

echo "============================================================"
echo "[1/6] Saving Qwen baseline verdicts from activities.json"
echo "============================================================"
# Qwen verdicts are already in activities.json's vlm_review block. Extract
# into the same JSON shape that vlm_benchmark.py produces so the compare
# step can diff against them.
"$PY" - <<'PYEOF' "$ACTIVITIES_JSON" /tmp/bench_qwen.json
import json, sys, statistics
path, out = sys.argv[1], sys.argv[2]
with open(path) as f:
    acts = json.load(f)
per = []
vc = {"TRUE_POSITIVE": 0, "FALSE_POSITIVE": 0, "UNCERTAIN": 0, "ERROR": 0}
by_ot = {}
lats = []
stock_hits = 0
import re
STOCK = re.compile(r"FRAME\s*\d+:\s*motion\s+blur|motion\s+blur\s+visible\s+in\s+the\s+window", re.I)
for i, a in enumerate(acts):
    review = a.get("vlm_review") or {}
    verdict = (review.get("verdict") or {}).get("verdict") or "MISSING"
    vc[verdict] = vc.get(verdict, 0) + 1
    ot = (a.get("objectType") or "").strip().lower().replace(" ", "_")
    bo = by_ot.setdefault(ot, {"n": 0, "verdicts": {}, "stock_hits": 0})
    bo["n"] += 1
    bo["verdicts"][verdict] = bo["verdicts"].get(verdict, 0) + 1
    lat = review.get("latency_sec")
    if isinstance(lat, (int, float)):
        lats.append(float(lat))
    me = (review.get("verdict") or {}).get("motion_evidence") or ""
    stock = bool(STOCK.search(me))
    if stock:
        stock_hits += 1
        bo["stock_hits"] += 1
    per.append({
        "activity_index": i,
        "activity_type": a.get("activityType"),
        "activity_start_time": str(a.get("activityStartTime") or ""),
        "object_type": ot,
        "verdict": verdict,
        "confidence": (review.get("verdict") or {}).get("confidence"),
        "primary_object_in_hand": (review.get("verdict") or {}).get("primary_object_in_hand"),
        "train_appears_to_be": (review.get("verdict") or {}).get("train_appears_to_be"),
        "motion_evidence": me[:200],
        "reasoning": ((review.get("verdict") or {}).get("reasoning") or "")[:300],
        "stock_phrase_detected": stock,
        "latency_sec": lat,
    })
stats = {
    "verdict_counts": vc,
    "stock_phrase_hits": stock_hits,
    "parse_errors": 0,
    "timeouts": 0,
    "by_object_type": by_ot,
}
if lats:
    lats.sort()
    stats["avg_latency_sec"] = round(sum(lats)/len(lats), 3)
    stats["p50_latency_sec"] = lats[len(lats)//2]
    stats["p95_latency_sec"] = lats[min(len(lats)-1, int(len(lats)*0.95))]
out_obj = {
    "run_dir": path.rsplit("/", 1)[0],
    "model_label": "qwen2.5-vl-7b-awq (cached)",
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "endpoint": "n/a (read from vlm_review in activities.json)",
    "n_activities_total": len(acts),
    "n_activities_evaluated": len(per),
    "stats": stats,
    "per_activity": per,
}
with open(out, "w") as g:
    json.dump(out_obj, g, indent=2)
print(f"wrote {out} ({len(per)} activities)")
PYEOF

echo
echo "============================================================"
echo "[2/6] Stopping Qwen vLLM service"
echo "============================================================"
if [[ -f /tmp/.sp ]]; then
    cat /tmp/.sp | sudo -S systemctl stop locopilot-vlm.service
else
    sudo systemctl stop locopilot-vlm.service
fi
sleep 3
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

echo
echo "============================================================"
echo "[3/6] Launching candidate vLLM: $CANDIDATE_MODEL"
echo "============================================================"
# Launch in the background under nohup so we can run the benchmark
# against it without holding a session open. Logs go to /tmp.
nohup "$APP_DIR/venv/bin/vllm" serve "$CANDIDATE_MODEL" \
    --host "$VLM_HOST" \
    --port "$VLM_PORT" \
    --max-model-len "$QWEN_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$QWEN_GPU_MEM" \
    --trust-remote-code \
    --enforce-eager \
    > /tmp/vllm_candidate.log 2>&1 &
CAND_PID=$!
echo "candidate vLLM PID=$CAND_PID  log=/tmp/vllm_candidate.log"

echo "Waiting for candidate to become healthy (up to 5 min)..."
for i in {1..60}; do
    sleep 5
    if curl -sf "http://localhost:$VLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "candidate vLLM healthy after ${i}*5s"
        break
    fi
    if ! kill -0 "$CAND_PID" 2>/dev/null; then
        echo "[error] candidate vLLM crashed during startup"
        echo "[tail of /tmp/vllm_candidate.log]"
        tail -50 /tmp/vllm_candidate.log
        echo
        echo "Restarting production Qwen..."
        cat /tmp/.sp | sudo -S systemctl start locopilot-vlm.service
        exit 3
    fi
done

echo
echo "============================================================"
echo "[4/6] Running benchmark against candidate"
echo "============================================================"
"$PY" "$BENCH" run \
    --run-dir "$RUN_DIR" \
    --vlm-url "http://localhost:$VLM_PORT/v1" \
    --vlm-model "$CANDIDATE_MODEL" \
    --label "$(echo "$CANDIDATE_MODEL" | sed 's|.*/||')" \
    --out /tmp/bench_internvl.json

echo
echo "============================================================"
echo "[5/6] Shutting down candidate, restoring Qwen"
echo "============================================================"
kill "$CAND_PID" 2>/dev/null || true
sleep 5
# Force-kill any leftover vLLM workers
pkill -9 -f "vllm.*$CANDIDATE_MODEL" || true
sleep 2
cat /tmp/.sp | sudo -S systemctl start locopilot-vlm.service
sleep 5
echo "Production Qwen status:"
sudo systemctl is-active locopilot-vlm.service
curl -sf "http://localhost:$VLM_PORT/v1/models" | head -c 200 || true
echo

echo
echo "============================================================"
echo "[6/6] Generating comparison report"
echo "============================================================"
"$PY" "$BENCH" compare \
    --a /tmp/bench_qwen.json \
    --b /tmp/bench_internvl.json \
    --out /tmp/bench_report.md

echo
echo "DONE. Report at /tmp/bench_report.md"
echo "Pull locally with:"
echo "  scp -P 3781 admin1@103.116.80.162:/tmp/bench_report.md ./"
