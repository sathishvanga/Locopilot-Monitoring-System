# Deployment Guide - Locopilot Monitoring System

## 🚀 Quick Deployment

### Prerequisites
- `sshpass` installed on your local machine
- SSH access to the server (root@103.195.244.67)
- Server running Ubuntu/Debian or RHEL/CentOS

### Deploy to Server

```bash
# Make the script executable
chmod +x deploy_to_server.sh

# Run deployment
./deploy_to_server.sh
```

## 📋 What Gets Deployed

### ✅ Included in Deployment
- Application code (`app/`, `locopilot_monitor.py`)
- Configuration files (`gunicorn_config.py`, etc.)
- Requirements file (`requirements.txt`)
- YOLO model weights (`.pt` files)
- Documentation files (`.md`)

### ❌ Excluded from Deployment
- `.git` directory
- `__pycache__` and `.pyc` files
- `output/` and `outputs/` directories
- `uploads/` directory
- `locopilot_evidence/` directory
- `venv/` and `.venv/` directories
- `example_data/` (large video files)
- `.DS_Store` (macOS metadata)
- Log files (`.log`)
- Environment files (`.env`)

## 🔧 Server Configuration

### Directory Structure
```
/opt/poc2/
├── app/                    # Application code
├── venv/                   # Python virtual environment
├── output/                 # Evidence output (activities.json, clips)
├── logs/                   # Application logs
├── gunicorn_config.py      # Gunicorn configuration
├── requirements.txt        # Python dependencies
└── yolo11s.pt             # YOLO model weights

/tmp/locopilot_uploads/     # Temporary uploads (auto-cleaned)
```

### Environment Variables

The systemd service is configured with:

```bash
# Application
ENVIRONMENT=production
DEBUG=false

# Directories (IMPORTANT!)
UPLOAD_DIR=/tmp/locopilot_uploads    # Temp uploads (auto-cleanup)
OUTPUT_DIR=/opt/poc2/output           # Evidence output
LOG_DIR=/opt/poc2/logs                # Logs

# Processing
SAMPLE_FPS=0.5                        # Sample at 0.5 FPS
ENABLE_MULTIPROCESSING=true           # Enable parallel processing
MP_CHUNK_DURATION=6                   # 6-second chunks
MP_MAX_WORKERS=<CPU_COUNT>            # Auto-detected

# Models
YOLO_WEIGHTS=yolo11s.pt
PRELOAD_OCR=1

# Performance (CPU-only)
CUDA_VISIBLE_DEVICES=                 # Disable GPU
OMP_NUM_THREADS=1                     # CPU threading limits
```

## 🎯 API Endpoints

### Base URL
```
http://103.195.244.67:8000
```

### Key Endpoints

#### 1. Health Check
```bash
GET /health
GET /api/health

# Test
curl http://103.195.244.67:8000/health
```

#### 2. Process Video
```bash
POST /api/jobs

# Example with curl
curl -X POST http://103.195.244.67:8000/api/jobs \
  -F "video=@/path/to/video.mp4" \
  -F "tripId=TRIP_12345" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP001" \
  -F "alpCrewName=Jane Smith" \
  -F "alpCrewId=ALP001" \
  -F "saveClips=false"
```

**Parameters:**
- `video` (required): Video file to process
- `tripId` (required): Unique trip identifier
- `lpCrewName` (required): Loco Pilot crew member name
- `lpCrewId` (required): Loco Pilot crew member ID
- `alpCrewName` (optional): Assistant Loco Pilot name
- `alpCrewId` (optional): Assistant Loco Pilot ID
- `useMockDetection` (optional): Use mock data (default: false)
- `useMultiprocessing` (optional): Enable parallel processing (default: true)
- `saveClips` (optional): Save video clips and images (default: **false** for production)

#### 3. API Documentation
```
http://103.195.244.67:8000/docs         # Swagger UI
http://103.195.244.67:8000/redoc        # ReDoc
```

## 🔍 Service Management

### Systemd Commands

```bash
# Check service status
sudo systemctl status poc2

# View logs (real-time)
sudo journalctl -u poc2 -f

# View last 100 lines
sudo journalctl -u poc2 -n 100

# Restart service
sudo systemctl restart poc2

# Stop service
sudo systemctl stop poc2

# Start service
sudo systemctl start poc2

# Disable auto-start
sudo systemctl disable poc2

# Enable auto-start
sudo systemctl enable poc2
```

### Quick Health Check

```bash
# SSH into server and check
ssh root@103.195.244.67

# Check if service is running
systemctl is-active poc2

# Check port binding
ss -ltnp | grep ':8000'

# Test API
curl http://localhost:8000/health
```

## 🗂️ File Storage Behavior

### Production Mode (Current Setup)

✅ **Uploads** → `/tmp/locopilot_uploads/` → **Auto-deleted after processing**
✅ **Activities JSON** → `/opt/poc2/output/run_YYYYMMDD_HHMMSS/activities.json` → **Saved**
❌ **Video Clips** → **NOT saved by default** (set `saveClips=true` to enable)
❌ **Frame Images** → **NOT saved by default**

### Storage Optimization

```bash
# Check output directory size
du -sh /opt/poc2/output/

# List all run directories
ls -lah /opt/poc2/output/

# Clean old runs (older than 7 days)
find /opt/poc2/output/ -type d -name "run_*" -mtime +7 -exec rm -rf {} \;

# Clean temp uploads (should be auto-cleaned, but just in case)
rm -rf /tmp/locopilot_uploads/*
```

## 🔐 Security Considerations

### ⚠️ Important Security Notes

1. **SSH Password in Script**: The deployment script contains hardcoded credentials
   - **Recommended**: Use SSH keys instead
   - Add your public key to server: `ssh-copy-id root@103.195.244.67`
   - Remove password from script

2. **Root User**: Currently running as root
   - **Recommended**: Create dedicated user for the application
   - Use `User=locopilot` in systemd service

3. **CORS**: Currently allows all origins (`*`)
   - **Recommended**: Restrict to specific origins in production
   - Set `CORS_ORIGINS=https://yourdomain.com` in environment

4. **Firewall**: Ensure port 8000 is accessible
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 8000/tcp
   
   # RHEL/CentOS
   sudo firewall-cmd --add-port=8000/tcp --permanent
   sudo firewall-cmd --reload
   ```

## 🐛 Troubleshooting

### Service Won't Start

```bash
# Check service status
systemctl status poc2

# View detailed logs
journalctl -u poc2 -n 200 --no-pager

# Check Python environment
/opt/poc2/venv/bin/python --version

# Test imports manually
/opt/poc2/venv/bin/python -c "import cv2, torch, mediapipe"
```

### Port Already in Use

```bash
# Find process using port 8000
sudo lsof -i :8000
sudo ss -tlnp | grep :8000

# Kill process if needed
sudo kill -9 <PID>
```

### Out of Memory

```bash
# Check memory usage
free -h

# Check service memory
systemctl status poc2

# Reduce workers in /etc/systemd/system/poc2.service
# Edit: Environment=MP_MAX_WORKERS=4
sudo systemctl daemon-reload
sudo systemctl restart poc2
```

### Disk Space Issues

```bash
# Check disk space
df -h

# Clean old outputs
find /opt/poc2/output/ -type d -mtime +7 -exec rm -rf {} \;

# Clean temp files
rm -rf /tmp/locopilot_uploads/*
```

## 📊 Performance Tuning

### Adjust Worker Count

Edit `/etc/systemd/system/poc2.service`:

```bash
# For 12-core server, recommended: 6-8 workers
Environment=MP_MAX_WORKERS=8

# Apply changes
sudo systemctl daemon-reload
sudo systemctl restart poc2
```

### Adjust Chunk Duration

```bash
# Smaller chunks = more parallel processing, more overhead
Environment=MP_CHUNK_DURATION=4

# Larger chunks = less parallelism, less overhead
Environment=MP_CHUNK_DURATION=10
```

### Monitor Performance

```bash
# CPU usage
top -bn1 | head -20

# Service resource usage
systemctl status poc2

# Real-time logs
journalctl -u poc2 -f | grep "processing\|completed"
```

## 🔄 Update Deployment

To update the application after code changes:

```bash
# Simply run the deployment script again
./deploy_to_server.sh

# The script will:
# 1. Transfer new code
# 2. Reinstall dependencies (if requirements.txt changed)
# 3. Restart the service
```

## 📞 Quick Reference

```bash
# Deploy
./deploy_to_server.sh

# Check status
ssh root@103.195.244.67 'systemctl status poc2'

# View logs
ssh root@103.195.244.67 'journalctl -u poc2 -f'

# Test API
curl http://103.195.244.67:8000/health

# Restart service
ssh root@103.195.244.67 'systemctl restart poc2'
```

## 📝 Important Notes

### ⚠️ Production Configuration
- **Uploads are temporary**: Files in `/tmp/locopilot_uploads/` are automatically deleted after processing
- **Clips disabled by default**: Set `saveClips=true` in API request to generate video clips
- **JSON always saved**: `activities.json` is always generated in `/opt/poc2/output/`

### 🎯 Recommended Settings
- **saveClips=false**: For production (JSON only, minimal storage)
- **saveClips=true**: For debugging (includes video clips and frame images)
- **useMultiprocessing=true**: Always enabled for faster processing

---

**Need Help?** Check the API docs at `http://103.195.244.67:8000/docs`

