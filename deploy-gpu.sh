#!/bin/bash
# Locopilot Monitoring System - GPU Server Deployment Script
# Usage: ./deploy-gpu.sh
# Target: GPU Server (103.116.80.162)

set -e

# Server Configuration
SERVER_IP="103.116.80.162"
SERVER_PORT="3781"
SERVER_USER="admin1"
# SECURITY: the deploy password is read from the LOCOPILOT_DEPLOY_PASS env
# var. The bash ``${VAR:?msg}`` expansion exits the script with a clear
# error if the variable is unset or empty. Set it in your shell before
# running this script, e.g.:
#     export LOCOPILOT_DEPLOY_PASS='...'
#     ./deploy-gpu.sh
# Never commit the literal password to source.
SERVER_PASS="${LOCOPILOT_DEPLOY_PASS:?set this env var (export LOCOPILOT_DEPLOY_PASS=...)}"
# Base64-encode password to safely pass through SSH command strings
# (password may contain shell metacharacters that break double-quoted
# expansions).
PASS_B64=$(printf '%s' "$SERVER_PASS" | base64)
REMOTE_PATH="/opt/poc2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Locopilot - GPU Server Deployer"
echo "  Target: $SERVER_IP (New GPU Server)"
echo "=========================================="
echo ""

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo -e "${RED}Error: sshpass is not installed${NC}"
    echo "Install it with: brew install sshpass"
    exit 1
fi

# Get the script's directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}[1/6] Syncing files to GPU server...${NC}"
sshpass -p "$SERVER_PASS" rsync -avz --progress \
    `# Python / venv / build artifacts` \
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude 'build' \
    --exclude 'dist' \
    `# Git / editor / CI` \
    --exclude '.git' \
    --exclude '.github' \
    --exclude '.claude' \
    --exclude '.DS_Store' \
    `# Local runtime artifacts (server has its own)` \
    --exclude 'locopilot_evidence' \
    --exclude 'locopilot_evidence_1' \
    --exclude 'logs' \
    --exclude '*.log' \
    --exclude 'uploads' \
    --exclude 'output' \
    --exclude 'evidence' \
    --exclude 'activities.json' \
    --exclude 'n_5_violations_frames' \
    --exclude 'example_data' \
    `# Training pipeline (2.4GB, not needed at runtime)` \
    --exclude 'auto_label' \
    --exclude 'auto_labeling' \
    `# Tests / docs / specs (not needed at runtime)` \
    --exclude 'tests' \
    --exclude 'docs' \
    --exclude 'doc' \
    --exclude 'tasks' \
    --exclude 'BUSINESS_REQUIREMENTS.md' \
    --exclude 'CLAUDE.md' \
    `# Experimental / one-off scripts` \
    --exclude 'test_train_motion.py' \
    --exclude 'compare_pose.py' \
    --exclude 'simple_violations.py' \
    --exclude 'pose_comparison' \
    --exclude 'pose_landmarker_*.task' \
    `# Local-only env + secrets (server uses .env.production → .env copy below)` \
    --exclude '.env' \
    --exclude '.env.example' \
    --exclude 'server details.txt' \
    `# Model weights (already on server, avoid re-uploading ~200MB)` \
    --exclude 'yolo*.pt' \
    --delete \
    -e "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no" \
    "$SCRIPT_DIR/" \
    "$SERVER_USER@$SERVER_IP:$REMOTE_PATH/"

echo ""
echo -e "${YELLOW}[2/6] Updating environment configuration...${NC}"
sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "cp $REMOTE_PATH/.env.production $REMOTE_PATH/.env"

echo ""
echo -e "${YELLOW}[3/6] Installing system dependencies (ffmpeg)...${NC}"
sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "which ffmpeg > /dev/null 2>&1 || (echo '$PASS_B64' | base64 -d | sudo -S apt-get update -qq && echo '$PASS_B64' | base64 -d | sudo -S apt-get install -y ffmpeg -qq)"

echo ""
echo -e "${YELLOW}[4/6] Installing Python dependencies...${NC}"
sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "cd $REMOTE_PATH && source venv/bin/activate && pip install -r requirements.txt -q"

echo ""
echo -e "${YELLOW}[5/6] Updating systemd service and restarting...${NC}"
# Update systemd service with correct environment variables
# Write the service file locally, then copy it over
SERVICE_FILE=$(mktemp)
cat > "$SERVICE_FILE" << 'SERVICE'
[Unit]
Description=Locopilot Monitoring System
After=network.target

[Service]
Type=simple
User=admin1
WorkingDirectory=/opt/poc2
EnvironmentFile=/opt/poc2/.env
Environment=PATH=/opt/poc2/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=YOLO_DEVICE=0
Environment=MP_MAX_WORKERS_CAP=6
Environment=TORCH_THREADS=1
Environment=OPENCV_THREADS=2
ExecStart=/opt/poc2/venv/bin/gunicorn -c gunicorn_config.py app.main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE

# Copy service file to remote server
sshpass -p "$SERVER_PASS" scp -P "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVICE_FILE" "$SERVER_USER@$SERVER_IP:/tmp/locopilot.service"
rm -f "$SERVICE_FILE"

# Install service and restart
sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "echo '$PASS_B64' | base64 -d | sudo -S cp /tmp/locopilot.service /etc/systemd/system/locopilot.service && \
     echo '$PASS_B64' | base64 -d | sudo -S systemctl daemon-reload && \
     echo '$PASS_B64' | base64 -d | sudo -S systemctl restart locopilot && \
     rm -f /tmp/locopilot.service"

echo ""
echo -e "${YELLOW}[6/6] Verifying deployment...${NC}"
sleep 5

# Check service status
STATUS=$(sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "echo '$PASS_B64' | base64 -d | sudo -S systemctl is-active locopilot 2>/dev/null")

if [ "$STATUS" = "active" ]; then
    echo -e "${GREEN}Service is running!${NC}"

    # Check GPU status
    echo ""
    echo -e "${YELLOW}GPU Status:${NC}"
    sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
        "$SERVER_USER@$SERVER_IP" \
        "nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader"

    # Test health endpoint
    HEALTH=$(curl -s --connect-timeout 10 "http://$SERVER_IP:8000/health" 2>/dev/null || echo "failed")

    if [[ "$HEALTH" == *"healthy"* ]]; then
        echo ""
        echo -e "${GREEN}Health check passed!${NC}"
        echo ""
        echo "=========================================="
        echo -e "${GREEN}  GPU Server Deployment Successful!${NC}"
        echo "=========================================="
        echo ""
        echo "Application URL: http://$SERVER_IP:8000"
        echo "Health Check:    http://$SERVER_IP:8000/health"
        echo "GPU Workers:     6 (MP_MAX_WORKERS_CAP)"
    else
        echo -e "${YELLOW}Warning: Health check did not return expected response${NC}"
        echo "Response: $HEALTH"
    fi
else
    echo -e "${RED}Error: Service is not running!${NC}"
    echo "Check logs with: ssh $SERVER_USER@$SERVER_IP 'journalctl -u locopilot -n 50'"
    exit 1
fi
