# Deployment Checklist for 1GB Upload Feature

## Pre-Deployment Checklist

- [x] All code changes committed
- [x] New files created (disk_utils.py, chunked_upload_models.py, etc.)
- [x] Configuration updated (config.py, gunicorn_config.py)
- [x] Requirements.txt updated with aiofiles
- [x] Deployment script updated

## Deployment Steps

### Step 1: Run Deployment Script
```bash
./deploy_to_server.sh
```

This will:
- Transfer all code to server
- Install aiofiles dependency
- Create chunk upload directory
- Deploy cleanup script
- Setup cron job
- Restart service

### Step 2: Update Nginx Configuration (REQUIRED)

**IMPORTANT:** This must be done manually on the server after deployment.

SSH to server:
```bash
ssh root@103.195.244.67
```

Edit Nginx config:
```bash
sudo nano /etc/nginx/sites-available/poc2
```

Add/modify these settings in the `server` block:
```nginx
server {
    listen 80;
    server_name 103.195.244.67;

    # File upload limits (ADD THESE)
    client_max_body_size 1G;             # Allow 1GB uploads
    client_body_timeout 1800s;           # 30 minute timeout
    client_header_timeout 60s;           # Header timeout

    # Proxy timeouts (ADD/MODIFY THESE)
    proxy_connect_timeout 60s;
    proxy_send_timeout 1800s;            # 30 minutes
    proxy_read_timeout 1800s;              # 30 minutes

    # Buffering (ADD THESE for memory efficiency)
    proxy_request_buffering off;          # Stream directly to backend
    proxy_buffering off;                   # Don't buffer responses

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Override timeouts for upload endpoints
        proxy_read_timeout 1800s;
        proxy_send_timeout 1800s;
    }
}
```

Test Nginx config:
```bash
sudo nginx -t
```

Reload Nginx:
```bash
sudo systemctl reload nginx
```

### Step 3: Verify Deployment

Check service status:
```bash
ssh root@103.195.244.67 "systemctl status poc2"
```

Check chunk directory:
```bash
ssh root@103.195.244.67 "ls -la /tmp/locopilot_uploads_chunks"
```

Check cleanup script:
```bash
ssh root@103.195.244.67 "ls -la /opt/poc2/scripts/cleanup_old_uploads.sh"
```

Check cron job:
```bash
ssh root@103.195.244.67 "crontab -l | grep cleanup"
```

Check logs:
```bash
ssh root@103.195.244.67 "tail -f /opt/poc2/logs/LocopilotMonitoring.log"
```

### Step 4: Test Endpoints

Test health endpoint:
```bash
curl http://103.195.244.67/api/health
```

Test v2 endpoints in browser:
- http://103.195.244.67/docs
- Look for `/api/v2/jobs/streaming` and `/api/v2/upload/*` endpoints

## Post-Deployment Verification

- [ ] Service is running (`systemctl status poc2`)
- [ ] Chunk directory exists (`/tmp/locopilot_uploads_chunks`)
- [ ] Cleanup script is executable
- [ ] Cron job is configured
- [ ] Nginx config updated and reloaded
- [ ] V2 endpoints visible in `/docs`
- [ ] No errors in logs

## Rollback Plan (If Needed)

If deployment fails:

1. SSH to server: `ssh root@103.195.244.67`
2. Revert code: `cd /opt/poc2 && git reset --hard HEAD~1` (if using git)
3. Revert Nginx config (restore old values)
4. Reload Nginx: `sudo systemctl reload nginx`
5. Restart service: `sudo systemctl restart poc2`

