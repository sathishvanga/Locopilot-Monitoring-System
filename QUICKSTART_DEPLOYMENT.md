# 🚀 Quick Start - Production Deployment

## Ready to Deploy? Follow These Steps:

### Step 1: Make Script Executable
```bash
chmod +x deploy_to_server.sh
```

### Step 2: Deploy to Server
```bash
./deploy_to_server.sh
```

That's it! The script will:
- ✅ Package and transfer your code
- ✅ Install all dependencies
- ✅ Configure the systemd service
- ✅ Start the application
- ✅ Verify it's running

### Step 3: Verify Deployment
```bash
# Test the API
curl http://103.195.244.67:8000/health

# Should return:
# {"status":"healthy","application":"Locopilot Monitoring System","version":"1.0.0"}
```

---

## 🔧 What Changed for Production?

### 1. **Automatic Upload Cleanup** ✅
- Uploaded videos are **automatically deleted** after processing
- No manual cleanup needed
- Saves disk space

### 2. **Temporary Upload Directory** ✅
- Uploads go to `/tmp/locopilot_uploads/`
- OS automatically cleans this directory
- No permanent storage of uploaded videos

### 3. **Clips Disabled by Default** ✅
- Only `activities.json` is generated (default)
- Video clips and images NOT saved (saves ~99% storage)
- Can enable clips with `saveClips=true` if needed

---

## 📝 Test the API

### Simple Test (JSON only)
```bash
curl -X POST http://103.195.244.67:8000/api/jobs \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST_001" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP001"
```

### With Clips (for debugging)
```bash
curl -X POST http://103.195.244.67:8000/api/jobs \
  -F "video=@example_data/latest.mp4" \
  -F "tripId=TEST_002" \
  -F "lpCrewName=John Doe" \
  -F "lpCrewId=LP001" \
  -F "saveClips=true"
```

---

## 📊 Storage Comparison

| Mode | Uploaded Video | activities.json | Video Clips | Storage per Request |
|------|---------------|-----------------|-------------|---------------------|
| **Before** | ❌ Kept | ✅ Saved | ✅ Always saved | ~650 MB |
| **After (Production)** | ✅ Auto-deleted | ✅ Saved | ❌ Not saved* | ~100 KB |

*Can enable with `saveClips=true`

**Savings: 99.98% less storage per request!** 🎉

---

## 🎯 API Endpoints

- **Health Check:** `GET http://103.195.244.67:8000/health`
- **Process Video:** `POST http://103.195.244.67:8000/api/jobs`
- **API Docs:** `http://103.195.244.67:8000/docs`
- **ReDoc:** `http://103.195.244.67:8000/redoc`

---

## 🔍 Service Management

```bash
# Check status
ssh root@103.195.244.67 'systemctl status poc2'

# View logs (real-time)
ssh root@103.195.244.67 'journalctl -u poc2 -f'

# Restart service
ssh root@103.195.244.67 'systemctl restart poc2'

# Stop service
ssh root@103.195.244.67 'systemctl stop poc2'

# Start service
ssh root@103.195.244.67 'systemctl start poc2'
```

---

## 📚 Full Documentation

- **DEPLOYMENT_GUIDE.md** - Comprehensive deployment guide
- **PRODUCTION_CHANGES.md** - Detailed changes and rationale
- **API_USAGE_GUIDE.md** - API usage examples

---

## ⚠️ Important Notes

### Upload Behavior
- ✅ Videos uploaded to `/tmp/locopilot_uploads/`
- ✅ Automatically deleted after processing
- ✅ No manual cleanup needed

### Output Behavior
- ✅ `activities.json` always saved to `/opt/poc2/output/run_*/`
- ❌ Video clips NOT saved by default (use `saveClips=true` to enable)
- ❌ Frame images NOT saved by default

### Configuration
- All settings configurable via environment variables in systemd service
- See `/etc/systemd/system/poc2.service` on server
- Edit and restart: `systemctl daemon-reload && systemctl restart poc2`

---

## 🐛 Troubleshooting

### Service won't start?
```bash
ssh root@103.195.244.67 'journalctl -u poc2 -n 100'
```

### Port not accessible?
```bash
ssh root@103.195.244.67 'ss -ltnp | grep :8000'
```

### Need to redeploy?
```bash
./deploy_to_server.sh
# That's it! The script handles everything.
```

---

## ✅ Deployment Checklist

- [ ] Run `chmod +x deploy_to_server.sh`
- [ ] Run `./deploy_to_server.sh`
- [ ] Verify service is running: `curl http://103.195.244.67:8000/health`
- [ ] Test video processing with sample video
- [ ] Verify uploaded video is deleted after processing
- [ ] Check logs for any errors: `journalctl -u poc2 -n 100`
- [ ] Monitor disk space: `df -h`

---

**🎉 Your production-ready API is now deployed!**

**Questions?** Check DEPLOYMENT_GUIDE.md for detailed documentation.

