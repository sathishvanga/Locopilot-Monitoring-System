#!/usr/bin/env bash

# Clean Deployment script for Locopilot Monitoring System
# This script completely removes the old deployment and does a fresh install

set -euo pipefail

# Server configuration
SERVER="103.195.244.67"
USER="root"
PASS="Login@123@@@"
APP_DIR="/opt/poc2"

echo "=========================================="
echo "Locopilot Monitoring System - CLEAN DEPLOYMENT"
echo "Server: ${SERVER}"
echo "App Directory: ${APP_DIR}"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This will:"
echo "   1. Stop the service"
echo "   2. COMPLETELY REMOVE ${APP_DIR}"
echo "   3. Do a fresh deployment"
echo ""
read -p "Continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Deployment cancelled."
    exit 1
fi

echo ""
echo "🚀 Starting clean deployment..."
echo ""

# Step 1: Stop service and clean server
echo "Step 1/3: Stopping service and cleaning server..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} bash -s <<'CLEANUP_EOF'

set -euo pipefail

APP_DIR="/opt/poc2"

echo "🛑 Stopping service..."
systemctl stop poc2 2>/dev/null || true
systemctl disable poc2 2>/dev/null || true
echo "✅ Service stopped"

echo "🧹 Removing old deployment..."
if [ -d "$APP_DIR" ]; then
    # Remove everything
    rm -rf "$APP_DIR"
    echo "✅ Old deployment removed"
else
    echo "ℹ️  Directory already clean"
fi

echo "📁 Creating fresh directory structure..."
mkdir -p "$APP_DIR"
mkdir -p "$APP_DIR/output"
mkdir -p "$APP_DIR/logs"
mkdir -p /tmp/locopilot_uploads
chmod 755 /tmp/locopilot_uploads
echo "✅ Fresh directory structure created"

CLEANUP_EOF

echo "✅ Step 1 completed - Server cleaned"
echo ""

# Step 2: Deploy fresh files
echo "Step 2/3: Deploying fresh application files..."
if [ -n "$PASS" ]; then
  COPYFILE_DISABLE=1 tar --format=ustar --no-xattrs --no-acls --no-fflags --no-mac-metadata -czf - \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'output' \
    --exclude 'outputs' \
    --exclude 'uploads' \
    --exclude 'locopilot_evidence' \
    --exclude 'evidence' \
    --exclude '.DS_Store' \
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude 'example_data' \
    --exclude '*.log' \
    --exclude '.env' \
    --exclude '*.md' \
    --exclude 'clean_deploy_to_server.sh' \
    -C "$(pwd)" . \
    | sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} "tar --no-same-owner -xzf - -C ${APP_DIR}"
else
  COPYFILE_DISABLE=1 tar --format=ustar --no-xattrs --no-acls --no-fflags --no-mac-metadata -czf - \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'output' \
    --exclude 'outputs' \
    --exclude 'uploads' \
    --exclude 'locopilot_evidence' \
    --exclude 'evidence' \
    --exclude '.DS_Store' \
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude 'example_data' \
    --exclude '*.log' \
    --exclude '.env' \
    --exclude '*.md' \
    --exclude 'clean_deploy_to_server.sh' \
    -C "$(pwd)" . \
    | ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} "tar --no-same-owner -xzf - -C ${APP_DIR}"
fi

echo "✅ Step 2 completed - Files deployed"
echo ""

# Step 3: Install and start service
echo "Step 3/3: Installing dependencies and starting service..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} bash -s <<'INSTALL_EOF'

set -euo pipefail

APP_DIR="/opt/poc2"

echo "=========================================="
echo "Fresh Installation Starting"
echo "=========================================="

# Install system dependencies
echo "📦 Installing system dependencies..."
if command -v apt-get >/dev/null 2>&1; then
  apt-get update || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.12 python3.12-venv python3.12-dev build-essential \
    libgl1-mesa-glx libglib2.0-0 libjpeg-dev zlib1g-dev \
    tesseract-ocr ffmpeg
elif command -v yum >/dev/null 2>&1; then
  yum install -y python3 python3-venv python3-devel gcc gcc-c++ \
    glib2 glib2-devel tesseract ffmpeg
fi

echo "✅ System dependencies installed"

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3.12 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip setuptools wheel

echo "✅ Virtual environment created"

# Clean up any existing OpenCV installations
echo "🧹 Cleaning up existing OpenCV installations..."
"$APP_DIR/venv/bin/pip" uninstall -y opencv-python opencv-contrib-python opencv-python-headless 2>/dev/null || true

# Split requirements into base vs ML/OpenCV
echo "📝 Processing requirements..."
"$APP_DIR/venv/bin/python" - "$APP_DIR" <<'PY'
import sys, re, codecs

src = f"{sys.argv[1]}/requirements.txt"
dst = f"{sys.argv[1]}/requirements.base.txt"
skip_re = re.compile(r'^(torch|torchvision|tensorflow|jax|jaxlib|keras|opencv.*)\b', re.IGNORECASE)

raw = open(src, 'rb').read()

def decode_bytes(b: bytes) -> str:
    # Detect UTF-16 by BOM or presence of many NUL bytes
    if b.startswith(codecs.BOM_UTF16_LE) or b.startswith(codecs.BOM_UTF16_BE) or b.count(b'\x00') > 0:
        try:
            return b.decode('utf-16')
        except Exception:
            pass
    try:
        return b.decode('utf-8')
    except Exception:
        return b.decode('latin-1', 'ignore')

text = decode_bytes(raw)
lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

with open(dst, 'w', encoding='utf-8', newline='\n') as out:
    for line in lines:
        s = line.strip()
        if not s or s.startswith('#') or skip_re.match(s):
            continue
        out.write(s + '\n')

print(f"✅ Created {dst}")
PY

# Install base requirements
echo "📦 Installing base requirements..."
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.base.txt"

# Install headless OpenCV (no X11/GL dependencies)
echo "📦 Installing OpenCV (headless)..."
"$APP_DIR/venv/bin/pip" install opencv-contrib-python-headless==4.11.0.86

# Install CPU-only PyTorch/torchvision
echo "📦 Installing PyTorch (CPU-only)..."
eval $("$APP_DIR/venv/bin/python" - "$APP_DIR/requirements.txt" <<'PY'
import sys, codecs

raw = open(sys.argv[1], 'rb').read()

def decode_bytes(b: bytes) -> str:
    if b.startswith(codecs.BOM_UTF16_LE) or b.startswith(codecs.BOM_UTF16_BE) or b.count(b'\x00') > 0:
        try:
            return b.decode('utf-16')
        except Exception:
            pass
    try:
        return b.decode('utf-8')
    except Exception:
        return b.decode('latin-1', 'ignore')

text = decode_bytes(raw).replace('\r\n','\n').replace('\r','\n').split('\n')

def find(name):
    name = name.lower()
    for line in text:
        s = line.strip()
        if s.lower().startswith(name + '=='):
            return s.split('==',1)[1].strip()
    return ''

torch_ver = find('torch')
tv_ver = find('torchvision')
print(f"TORCH_VER={torch_ver}")
print(f"TV_VER={tv_ver}")
PY
)

CPU_IDX="--index-url https://download.pytorch.org/whl/cpu"
if [ -n "${TORCH_VER:-}" ] && [ -n "${TV_VER:-}" ]; then
  "$APP_DIR/venv/bin/pip" install $CPU_IDX torch=="$TORCH_VER" torchvision=="$TV_VER"
else
  "$APP_DIR/venv/bin/pip" install $CPU_IDX torch torchvision
fi

echo "✅ All Python packages installed"

# Verify installation with smoke test
echo "🔍 Running installation verification..."
cat > "$APP_DIR/_import_check.py" <<'PY'
import sys
import numpy as np

try:
    import cv2
    print("✅ cv2:", getattr(cv2, "__version__", "unknown"), "file:", getattr(cv2, "__file__", "n/a"))
except Exception as e:
    print("❌ cv2 import failed:", repr(e))
    raise

import torch, torchvision
print("✅ torch:", torch.__version__, "cuda?", torch.cuda.is_available())

try:
    import mediapipe as mp
    print("✅ mediapipe:", mp.__version__)
except Exception as e:
    print("⚠️ mediapipe optional import failed:", repr(e))

print("\n✅ All critical imports successful!")
PY

"$APP_DIR/venv/bin/python" "$APP_DIR/_import_check.py"

# Determine CPU count for worker configuration
POOL_PROCS=$( (command -v nproc >/dev/null 2>&1 && nproc) || (getconf _NPROCESSORS_ONLN) || echo 4 )
echo "🖥️ Detected ${POOL_PROCS} CPU cores"

# Create systemd service with production configuration
echo "⚙️ Creating systemd service..."
cat >/etc/systemd/system/poc2.service <<UNIT
[Unit]
Description=Locopilot Monitoring System - POC2 FastAPI Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/poc2

# Python configuration
Environment=PYTHONUNBUFFERED=1

# Application environment
Environment=ENVIRONMENT=production
Environment=DEBUG=false

# Directory configuration (IMPORTANT: Uses /tmp for uploads in production)
Environment=UPLOAD_DIR=/tmp/locopilot_uploads
Environment=OUTPUT_DIR=/opt/poc2/output
Environment=LOG_DIR=/opt/poc2/logs

# CORS configuration
Environment=CORS_ORIGINS=*

# Video processing configuration
Environment=SAMPLE_FPS=0.5
Environment=ENABLE_MULTIPROCESSING=true
Environment=MP_CHUNK_DURATION=6
Environment=MP_MAX_WORKERS=${POOL_PROCS}

# Model configuration
Environment=YOLO_WEIGHTS=yolo11s.pt
Environment=PRELOAD_OCR=1

# Performance tuning (CPU-only)
Environment=CUDA_VISIBLE_DEVICES=
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1
Environment=OPENCV_NUM_THREADS=1
Environment=TORCH_NUM_THREADS=1
Environment=TORCH_NUM_INTEROP_THREADS=1

# Start service
ExecStart=/opt/poc2/venv/bin/gunicorn -c gunicorn_config.py app.main:app -k uvicorn.workers.UvicornWorker

# Restart policy
Restart=always
RestartSec=5

# Security (optional hardening)
NoNewPrivileges=true
PrivateTmp=false

[Install]
WantedBy=multi-user.target
UNIT

# Reload systemd and enable service
echo "🔄 Reloading systemd..."
systemctl daemon-reload

echo "🚀 Starting service..."
systemctl enable --now poc2

# Wait a moment for service to start
sleep 3

# Check if service is running
echo "🔍 Checking service status..."
if systemctl is-active --quiet poc2; then
    echo "✅ Service is running"
else
    echo "❌ Service failed to start"
    journalctl -u poc2 --no-pager -n 50
    exit 1
fi

# Verify port binding
echo "🔍 Verifying port binding on :8000..."
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | grep ':8000' || (echo "❌ Port 8000 not bound"; journalctl -u poc2 --no-pager -n 100; exit 1)
else
  netstat -ltnp | grep ':8000' || (echo "❌ Port 8000 not bound"; journalctl -u poc2 --no-pager -n 100; exit 1)
fi

echo ""
echo "=========================================="
echo "✅ Fresh Installation Completed!"
echo "=========================================="

# Show final directory contents
echo ""
echo "📁 Final directory contents:"
ls -la "$APP_DIR" | grep -v "^d.*\.$"

echo ""
echo "=========================================="
echo "Service Information"
echo "=========================================="
echo "Service: poc2"
echo "Status: systemctl status poc2"
echo "Logs: journalctl -u poc2 -f"
echo "API: http://$(hostname -I | awk '{print $1}'):8000"
echo "Docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "=========================================="

INSTALL_EOF

echo "✅ Step 3 completed - Service running"
echo ""
echo "=========================================="
echo "🎉 CLEAN DEPLOYMENT COMPLETED!"
echo "=========================================="
echo ""
echo "Summary:"
echo "✅ Old deployment removed"
echo "✅ Fresh files deployed (only necessary files)"
echo "✅ Dependencies installed"
echo "✅ Service started and verified"
echo ""
echo "Next steps:"
echo "1. Test API: curl http://${SERVER}:8000/health"
echo "2. View docs: http://${SERVER}:8000/docs"
echo "3. Check logs: ssh ${USER}@${SERVER} 'journalctl -u poc2 -f'"
echo ""
echo "Note: No documentation or test files were deployed to keep the server clean."
echo "=========================================="

