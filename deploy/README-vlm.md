# VLM Verifier — deployment

Pipeline-2 false-positive filter. Runs Qwen2.5-VL-7B-Instruct-AWQ on a separate
vLLM process so it can be restarted independently of the main locopilot
service. The main service calls this endpoint over HTTP at
`http://localhost:8001/v1` and is fail-open if it's down.

## Install on the GPU server

```bash
# 1. Make sure the AWQ model is in the HF cache (~6.5 GB).
#    Already present after the 2026-04-26 spike, but if missing:
hf download Qwen/Qwen2.5-VL-7B-Instruct-AWQ

# 2. Copy the unit file and enable
sudo cp deploy/locopilot-vlm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable locopilot-vlm.service
sudo systemctl start locopilot-vlm.service

# 3. Verify
sudo systemctl status locopilot-vlm.service
curl -sf http://localhost:8001/v1/models | jq .
```

Startup takes ~60 s (model load + AWQ Marlin kernel warmup).

## GPU sharing

| Service           | VRAM peak | Notes |
|-------------------|-----------|-------|
| locopilot.service | ~7-8 GB   | YOLO v8 + pose + face mesh, lazy-loaded per request |
| locopilot-vlm     | ~10 GB    | vLLM, capped at `--gpu-memory-utilization 0.50`     |
| Free margin       | ~3 GB     | Headroom on the 20 GB RTX 4000 Ada                  |

If memory ever turns red on `nvidia-smi`, drop `--gpu-memory-utilization` to
0.45 (~9 GB) and `--max-model-len` to 1536 in the unit file.

## Enable the verifier in the main service

In `/opt/poc2/.env` (or `.env.production`):

```bash
VLM_VERIFICATION_ENABLED=1
VLM_VERIFY_ACTIVITIES=writing,eating_drinking
VLM_DROP_THRESHOLD=0.80                   # FP @ conf >= 0.80 are dropped
```

Restart the main service: `sudo systemctl restart locopilot.service`.

Activities written to `activities.json` will carry a `vlm_review` field with
the verdict, reasoning, and per-call latency. To run in observe-only mode
(record verdicts without dropping anything), raise `VLM_DROP_THRESHOLD`
above 1.0.

## Logs

```bash
# vLLM server logs
journalctl -u locopilot-vlm.service -f

# Verifier output in main service logs
grep '\[vlm\]\|\[VLM\]' /opt/poc2/logs/LocopilotMonitoring.log | tail -50
```

## Rollback

```bash
# Disable verifier without stopping vLLM
echo 'VLM_VERIFICATION_ENABLED=0' >> /opt/poc2/.env
sudo systemctl restart locopilot.service

# Or stop vLLM entirely (verifier becomes fail-open immediately)
sudo systemctl stop locopilot-vlm.service
```
