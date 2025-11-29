#!/bin/bash
# Cleanup chunked upload sessions older than 24 hours
# This script is run daily via cron to prevent disk space issues

CHUNK_DIR="/tmp/locopilot_uploads_chunks"
MAX_AGE_HOURS=24
LOG_FILE="/opt/poc2/logs/cleanup.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting cleanup of uploads older than ${MAX_AGE_HOURS}h"

# Check if directory exists
if [ ! -d "$CHUNK_DIR" ]; then
    log "Chunk directory does not exist: $CHUNK_DIR"
    exit 0
fi

# Find and remove expired sessions
REMOVED=0
find "$CHUNK_DIR" -mindepth 1 -maxdepth 1 -type d -mmin +$((MAX_AGE_HOURS * 60)) | while read dir; do
    if [ -d "$dir" ]; then
        log "Removing expired session: $(basename "$dir")"
        rm -rf "$dir"
        REMOVED=$((REMOVED + 1))
    fi
done

log "Cleanup complete. Removed $REMOVED expired session(s)"

