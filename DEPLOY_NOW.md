# 🚀 Deployment Steps - Quick Guide

## Step 1: Clean Up Old Deployment

```bash
# Make cleanup script executable
chmod +x cleanup_old_deployment.sh

# Run cleanup (stops service and removes everything)
./cleanup_old_deployment.sh
```

**What this does:**
- ✅ Stops the existing `poc2` service
- ✅ Disables the service from auto-starting
- ✅ Removes `/opt/poc2` directory completely
- ✅ Cleans up `/tmp/locopilot_uploads`
- ✅ Removes systemd service file
- ✅ No backup created (clean slate)

---

## Step 2: Deploy New Version

```bash
# Make deployment script executable  
chmod +x deploy_to_server.sh

# Deploy fresh installation
./deploy_to_server.sh
```

**What this does:**
- ✅ Transfers application code to server
- ✅ Installs all dependencies (Python, PyTorch, OpenCV, etc.)
- ✅ Creates production systemd service
- ✅ Starts the service
- ✅ Verifies deployment

---

## Step 3: Verify Deployment

```bash
# Test the API health endpoint
curl http://103.195.244.67:8000/health

# Should return:
# {"status":"healthy","application":"Locopilot Monitoring System","version":"1.0.0"}
```

---

## 📋 One-Liner Deployment

```bash
chmod +x cleanup_old_deployment.sh deploy_to_server.sh && ./cleanup_old_deployment.sh && ./deploy_to_server.sh
```

---

## 🔍 Post-Deployment Checks

```bash
# Check service status
ssh root@103.195.244.67 'systemctl status poc2'

# View real-time logs
ssh root@103.195.244.67 'journalctl -u poc2 -f'

# Check port binding
ssh root@103.195.244.67 'ss -ltnp | grep :8000'

# Test API endpoints
curl http://103.195.244.67:8000/
curl http://103.195.244.67:8000/health
curl http://103.195.244.67:8000/api/health
```

---

## 📝 Production Configuration Summary

### ✅ What's Different in Production:
- **Uploads:** Go to `/tmp/locopilot_uploads/` → Auto-deleted after processing
- **Clips:** NOT saved by default (saves ~99% storage)
- **JSON:** Always saved to `/opt/poc2/output/run_*/activities.json`
- **Multiprocessing:** Enabled by default (uses all CPU cores)
- **Auto-restart:** Service restarts automatically on failure

### 🎯 Default Behavior:
```python
# When client sends a video:
1. Video uploaded to /tmp/locopilot_uploads/trip_123_timestamp.mp4
2. Video processed with ML models
3. activities.json generated in /opt/poc2/output/run_YYYYMMDD_HHMMSS/
4. Video automatically deleted from /tmp
5. Only JSON remains (minimal storage)
```

### 📊 Storage Comparison:
| Item | Saved? | Size | 
|------|--------|------|
| Uploaded Video | ❌ Auto-deleted | ~0 MB |
| activities.json | ✅ Saved | ~100 KB |
| Video Clips | ❌ Not generated* | ~0 MB |
| Frame Images | ❌ Not generated* | ~0 MB |
| **Total** | | **~100 KB per request** |

*Can enable with `saveClips=true` parameter

---

## 🧪 Test the Deployment

```bash
# Simple test (JSON only - production mode)
curl -X POST http://103.195.244.67:8000/api/jobs \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST_001" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP001"

# With clips (debugging mode)
curl -X POST http://103.195.244.67:8000/api/jobs \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST_002" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP001" \
  -F "saveClips=true"
```

---

## 🐛 Troubleshooting

### Cleanup fails?
```bash
# Manually stop and clean
ssh root@103.195.244.67 'systemctl stop poc2; rm -rf /opt/poc2; rm -f /etc/systemd/system/poc2.service; systemctl daemon-reload'
```

### Deployment fails?
```bash
# Check logs on server
ssh root@103.195.244.67 'journalctl -u poc2 -n 100'
```

### Port already in use?
```bash
# Find and kill process using port 8000
ssh root@103.195.244.67 'sudo lsof -i :8000'
ssh root@103.195.244.67 'sudo kill -9 <PID>'
```

---

## 📚 Documentation

- **DEPLOYMENT_GUIDE.md** - Comprehensive deployment guide
- **PRODUCTION_CHANGES.md** - What changed for production
- **QUICKSTART_DEPLOYMENT.md** - Quick reference
- **API_USAGE_GUIDE.md** - API examples

---

## ✅ Ready to Deploy!

**Execute these commands:**

```bash
# 1. Clean up old deployment
chmod +x cleanup_old_deployment.sh
./cleanup_old_deployment.sh

# 2. Deploy new version
chmod +x deploy_to_server.sh
./deploy_to_server.sh

# 3. Verify it's working
curl http://103.195.244.67:8000/health
```

**That's it! Your production-ready API is deployed! 🎉**

