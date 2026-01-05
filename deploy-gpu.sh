#!/bin/bash
# Locopilot Monitoring System - GPU Server Deployment Script
# Usage: ./deploy-gpu.sh
# Target: GPU Server (95.216.66.168) - GTX 1080 8GB

set -e

# Server Configuration
SERVER_IP="95.216.66.168"
SERVER_PORT="22"
SERVER_USER="root"
SERVER_PASS="Login@123@@@"
REMOTE_PATH="/opt/poc2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Locopilot - GPU Server Deployer"
echo "  Target: $SERVER_IP (GTX 1080)"
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
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'locopilot_evidence' \
    --exclude 'locopilot_evidence_1' \
    --exclude 'voting_debug_frames' \
    --exclude 'logs' \
    --exclude '.DS_Store' \
    --exclude '*.log' \
    --exclude 'uploads' \
    --exclude 'output' \
    --exclude 'example_data' \
    --exclude '.claude' \
    --exclude 'build' \
    --exclude 'dist' \
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
    "which ffmpeg > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y ffmpeg -qq)"

echo ""
echo -e "${YELLOW}[4/6] Installing Python dependencies...${NC}"
sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "cd $REMOTE_PATH && source venv/bin/activate && pip install -r requirements.txt -q"

echo ""
echo -e "${YELLOW}[5/6] Updating systemd service and restarting...${NC}"
# Update systemd service with correct environment variables
sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" << 'EOF'
cat > /etc/systemd/system/locopilot.service << 'SERVICE'
[Unit]
Description=Locopilot Monitoring System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/poc2
EnvironmentFile=/opt/poc2/.env
Environment=PATH=/opt/poc2/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=YOLO_DEVICE=0
Environment=MP_MAX_WORKERS_CAP=2
Environment=TORCH_THREADS=1
Environment=OPENCV_THREADS=2
ExecStart=/opt/poc2/venv/bin/gunicorn -c gunicorn_config.py app.main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl restart locopilot
EOF

echo ""
echo -e "${YELLOW}[6/6] Verifying deployment...${NC}"
sleep 5

# Check service status
STATUS=$(sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "systemctl is-active locopilot")

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
        echo "GPU Workers:     2 (MP_MAX_WORKERS_CAP)"
    else
        echo -e "${YELLOW}Warning: Health check did not return expected response${NC}"
        echo "Response: $HEALTH"
    fi
else
    echo -e "${RED}Error: Service is not running!${NC}"
    echo "Check logs with: ssh $SERVER_USER@$SERVER_IP 'journalctl -u locopilot -n 50'"
    exit 1
fi
