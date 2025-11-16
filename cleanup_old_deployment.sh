#!/usr/bin/env bash

# Script to clean up old deployment before deploying new version
# This will stop and completely remove the existing poc2 service

set -euo pipefail

SERVER="103.195.244.67"
USER="root"
PASS="Login@123@@@"
APP_DIR="/opt/poc2"

echo "=========================================="
echo "Cleaning Up Old Deployment"
echo "Server: ${SERVER}"
echo "App Directory: ${APP_DIR}"
echo "=========================================="

sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no ${USER}@${SERVER} bash -s <<'REMOTE_EOF'

set -euo pipefail

APP_DIR="/opt/poc2"

echo ""
echo "🛑 Step 1: Stopping existing service..."
if systemctl is-active --quiet poc2 2>/dev/null; then
    systemctl stop poc2
    echo "✅ Service stopped"
else
    echo "⚠️  Service not running or doesn't exist"
fi

echo ""
echo "🔧 Step 2: Disabling service..."
if systemctl is-enabled --quiet poc2 2>/dev/null; then
    systemctl disable poc2
    echo "✅ Service disabled"
else
    echo "⚠️  Service not enabled or doesn't exist"
fi

echo ""
echo "🗑️  Step 3: Removing old application directory..."
if [ -d "$APP_DIR" ]; then
    rm -rf "$APP_DIR"
    echo "✅ Old application directory removed"
else
    echo "⚠️  Application directory doesn't exist"
fi

echo ""
echo "🧹 Step 4: Cleaning up old uploads..."
if [ -d "/tmp/locopilot_uploads" ]; then
    rm -rf /tmp/locopilot_uploads
    echo "✅ Old uploads cleaned"
fi

echo ""
echo "🧹 Step 5: Removing systemd service file..."
if [ -f "/etc/systemd/system/poc2.service" ]; then
    rm -f /etc/systemd/system/poc2.service
    systemctl daemon-reload
    echo "✅ Systemd service file removed"
else
    echo "⚠️  Service file doesn't exist"
fi

echo ""
echo "=========================================="
echo "✅ Cleanup Completed Successfully!"
echo "=========================================="
echo ""
echo "You can now run: ./deploy_to_server.sh"
echo "=========================================="

REMOTE_EOF

echo ""
echo "=========================================="
echo "🎉 Remote Cleanup Complete!"
echo "=========================================="
echo ""
echo "✅ Old service stopped and removed"
echo "✅ Application directory cleaned"
echo "✅ Ready for fresh deployment"
echo ""
echo "Next step: Run ./deploy_to_server.sh"
echo "=========================================="

