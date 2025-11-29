#!/bin/bash
# Quick verification script for deployment

SERVER="103.195.244.67"
USER="root"
PASS="Login@123@@@"

echo "=========================================="
echo "Verifying Deployment Status"
echo "=========================================="

sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} <<'VERIFY_EOF'

echo "1. Checking service status..."
systemctl status poc2 --no-pager -l | head -20

echo ""
echo "2. Checking chunk directory..."
ls -la /tmp/locopilot_uploads_chunks 2>/dev/null || echo "Directory not found"

echo ""
echo "3. Checking cleanup script..."
ls -la /opt/poc2/scripts/cleanup_old_uploads.sh 2>/dev/null || echo "Script not found"

echo ""
echo "4. Checking cron job..."
crontab -l 2>/dev/null | grep cleanup || echo "Cron job not found"

echo ""
echo "5. Checking aiofiles installation..."
/opt/poc2/venv/bin/pip list | grep aiofiles || echo "aiofiles not installed"

echo ""
echo "6. Checking v2 controller exists..."
ls -la /opt/poc2/app/controllers/v2_video_controller.py 2>/dev/null || echo "v2 controller not found"

echo ""
echo "7. Checking port binding..."
ss -ltnp | grep ':8000' || netstat -ltnp 2>/dev/null | grep ':8000' || echo "Port 8000 not bound"

VERIFY_EOF

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="

