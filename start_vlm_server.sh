#!/bin/bash
# ============================================================
# VLM Verification Server - Qwen3-VL-8B-Instruct-AWQ via vLLM
# ============================================================
# Serves the vision-language model for secondary verification
# of YOLO detections. Runs alongside the main Locopilot app.
#
# Usage:
#   ./start_vlm_server.sh          # foreground
#   ./start_vlm_server.sh &        # background
#   nohup ./start_vlm_server.sh &  # persist after logout
# ============================================================

set -e

# Configuration
VENV_PATH="/opt/poc2/venv"
MODEL_NAME="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
PORT=8001
HOST="0.0.0.0"

# GPU settings - Reserve ~10GB for VLM, leave ~10GB for YOLO
# Tested: AWQ model takes ~6.6GB, KV cache needs headroom for inference
GPU_MEMORY_UTILIZATION=0.60  # 60% of 20GB = ~12GB for model + KV cache
MAX_MODEL_LEN=2048           # Sufficient for single-image prompts
MAX_NUM_SEQS=4               # Low concurrency (sequential frame verification)

# Logging
LOG_DIR="/opt/poc2/logs"
LOG_FILE="${LOG_DIR}/vlm_server.log"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Starting VLM Verification Server"
echo "Model: ${MODEL_NAME}"
echo "Port:  ${PORT}"
echo "GPU Memory: ${GPU_MEMORY_UTILIZATION} ($(echo "${GPU_MEMORY_UTILIZATION} * 20" | bc)GB)"
echo "Log:   ${LOG_FILE}"
echo "============================================================"

# Activate venv
source "${VENV_PATH}/bin/activate"

# Prevent CPU contention with YOLO workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Launch vLLM server
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --dtype auto \
    --port "${PORT}" \
    --host "${HOST}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enable-chunked-prefill \
    --disable-log-requests \
    2>&1 | tee -a "${LOG_FILE}"
