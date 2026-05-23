#!/usr/bin/env bash
# Compile requirements.in -> requirements.lock with hashes.
#
# WHERE TO RUN
#   On a host whose Python is the same minor version as production
#   (Python 3.12 on the GPU box at /opt/poc2/venv). Running on a
#   different minor will produce a lock that pip rejects on the server.
#
# WHAT IT DOES
#   1. Installs pip-tools into the active venv if missing.
#   2. Resolves requirements.in using the PyPI index *plus* the PyTorch
#      CUDA 12.1 wheel index — required so torch/torchvision pin to the
#      GPU wheels, not the default CPU wheels.
#   3. Emits requirements.lock with --hash=sha256:... entries for every
#      package and transitive dep. Production installs use
#      `pip install --require-hashes -r requirements.lock`, which fails
#      hard on any tampering or unpinned transitive.
#
# AFTER RUNNING
#   - Review the diff in requirements.lock.
#   - Run `pip-audit --requirement requirements.lock --strict` and
#     resolve any HIGH/CRITICAL CVEs (or document approved waivers).
#   - Commit BOTH requirements.in and requirements.lock together.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v pip-compile >/dev/null 2>&1; then
    echo "[lock-deps] pip-compile not found. Installing pip-tools into the active venv..."
    python -m pip install --upgrade 'pip-tools>=7.4,<8'
fi

echo "[lock-deps] Compiling requirements.in -> requirements.lock with hashes"
pip-compile \
    --generate-hashes \
    --resolver=backtracking \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --output-file requirements.lock \
    requirements.in

echo "[lock-deps] Done. requirements.lock is up to date."
echo ""
echo "Next steps:"
echo "  1. git diff requirements.lock"
echo "  2. pip-audit --requirement requirements.lock --strict"
echo "  3. (in a clean venv on the GPU box):"
echo "       pip install --require-hashes --no-deps -r requirements.lock"
echo "       python -c 'import torch; print(torch.cuda.is_available())'  # must print True"
