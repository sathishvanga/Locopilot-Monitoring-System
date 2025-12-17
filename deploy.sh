#!/bin/bash
# Locopilot Monitoring System - Deployment Script
# Usage: ./deploy.sh

set -e

# Server Configuration
SERVER_IP="103.195.244.66"
SERVER_PORT="7291"
SERVER_USER="root"
SERVER_PASS="Login@123@@@"
REMOTE_PATH="/opt/poc2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Locopilot Monitoring System Deployer"
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

echo -e "${YELLOW}[1/6] Syncing files to server...${NC}"
sshpass -p "$SERVER_PASS" rsync -avz --progress \
    --exclude 'venv' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'locopilot_evidence' \
    --exclude 'logs' \
    --exclude '.DS_Store' \
    --exclude '*.log' \
    --exclude 'uploads' \
    --exclude '.claude' \
    --exclude 'build' \
    --exclude 'dist' \
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
echo -e "${YELLOW}[5/6] Restarting service...${NC}"
sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "systemctl restart locopilot"

echo ""
echo -e "${YELLOW}[6/6] Verifying deployment...${NC}"
sleep 5

# Check service status
STATUS=$(sshpass -p "$SERVER_PASS" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no \
    "$SERVER_USER@$SERVER_IP" \
    "systemctl is-active locopilot")

if [ "$STATUS" = "active" ]; then
    echo -e "${GREEN}Service is running!${NC}"

    # Test health endpoint
    HEALTH=$(curl -s --connect-timeout 10 "http://$SERVER_IP:8000/health" 2>/dev/null || echo "failed")

    if [[ "$HEALTH" == *"healthy"* ]]; then
        echo -e "${GREEN}Health check passed!${NC}"
        echo ""
        echo "=========================================="
        echo -e "${GREEN}  Deployment Successful!${NC}"
        echo "=========================================="
        echo ""
        echo "Application URL: http://$SERVER_IP:8000"
        echo "Health Check:    http://$SERVER_IP:8000/health"
    else
        echo -e "${YELLOW}Warning: Health check did not return expected response${NC}"
        echo "Response: $HEALTH"
    fi
else
    echo -e "${RED}Error: Service is not running!${NC}"
    echo "Check logs with: ssh -p $SERVER_PORT $SERVER_USER@$SERVER_IP 'journalctl -u locopilot -n 50'"
    exit 1
fi
