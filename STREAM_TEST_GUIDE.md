# Stream Capture Test Guide

This guide helps you test the YouTube stream capture functionality step-by-step.

## What Was Added

### 1. **Simplified Web UI** ([services/web-ui/app_simple.py](services/web-ui/app_simple.py))
- Shows live video stream from YouTube
- Auto-refreshing display
- Raw and preprocessed views side-by-side
- Perfect for testing stream capture

### 2. **Image Serving Endpoints** ([services/preprocessing/main.py](services/preprocessing/main.py))
- `GET /image/latest` - Serve raw captured frame
- `GET /image/processed` - Serve preprocessed frame
- `POST /preprocess` - Process the current frame

### 3. **Proxy Support** for yt-dlp
- `PROXY_URL` environment variable
- Example: `http://10.0.0.100:8888`

### 4. **Deno Installation** in stream-capture container
- Required by yt-dlp for some streams

## Quick Test (Automated)

Run the automated test script:

```bash
./scripts/test-stream-step-by-step.sh
```

This will:
1. Check connection
2. Verify configuration
3. Deploy code
4. Start services
5. Show logs
6. Open SSH tunnel to Web UI

## Manual Test Steps

### Step 1: Configure Environment

```bash
ssh dockerserver 'cd flowium && nano .env'
```

Add your configuration:
```bash
# YouTube Stream URL (use the Baja camera or any live stream)
YOUTUBE_URL=https://www.youtube.com/watch?v=OTjFtQIg0O0

# Proxy for yt-dlp (if needed)
PROXY_URL=http://10.0.0.100:8888

# Weather API (optional for now)
WEATHER_API_KEY=your_key_here
```

### Step 2: Start Services

Start only the services needed for stream testing:

```bash
# Build and start
./scripts/remote-docker.sh up -d --build stream-capture preprocessing web-ui
```

This will:
- Build Docker images
- Install Deno
- Start stream capture
- Start preprocessing API
- Start web UI

### Step 3: Check Logs

Watch the stream-capture logs:

```bash
./scripts/remote-docker.sh logs -f stream-capture
```

Look for:
- ✅ `Got stream URL: ...` = Successfully connected to YouTube
- ✅ `Captured frame X -> ...` = Frames are being saved
- ✅ `Using proxy: ...` = Proxy is configured
- ❌ `ERROR: ...` = Something went wrong

### Step 4: Verify Frames

Check if frames are being saved:

```bash
ssh dockerserver 'ls -lh flowium/volumes/frames/'
```

You should see:
- `latest.jpg` - Most recent frame (constantly updated)
- `frame_*.jpg` - Timestamped frames

### Step 5: Access Web UI

**Option A: SSH Tunnel (Recommended)**

```bash
# Open tunnel
ssh -L 8501:localhost:8501 dockerserver

# Open in browser
http://localhost:8501
```

**Option B: Direct Access**

```bash
# Get server IP
ssh dockerserver 'hostname -I'

# Open in browser
http://<server-ip>:8501
```

### Step 6: View Stream

In the web UI you'll see:
- **Left column**: Raw captured frames from YouTube
- **Right column**: Preprocessed/cropped view
- **Auto-refresh**: Frames update every 2 seconds
- **Status indicators**: Service health and frame availability

## Troubleshooting

### No frames appearing

**Check if stream-capture is running:**
```bash
./scripts/remote-docker.sh ps
```

**Check logs for errors:**
```bash
./scripts/remote-docker.sh logs stream-capture
```

**Common issues:**
- Invalid YouTube URL → Update YOUTUBE_URL in .env
- Proxy not working → Check PROXY_URL or remove if not needed
- Stream requires Deno → Already installed in latest version

### Web UI not loading

**Check if web-ui is running:**
```bash
./scripts/remote-docker.sh ps
```

**Check web-ui logs:**
```bash
./scripts/remote-docker.sh logs web-ui
```

**Check preprocessing service:**
```bash
curl http://localhost:8001/health
```

### Frames captured but not showing in UI

**Check preprocessing service:**
```bash
./scripts/remote-docker.sh logs preprocessing
```

**Test image endpoint:**
```bash
ssh dockerserver 'curl -I http://localhost:8001/image/latest'
```

### Proxy not working

**Test proxy:**
```bash
ssh dockerserver 'curl --proxy http://10.0.0.100:8888 https://www.youtube.com'
```

**Check yt-dlp with proxy:**
```bash
./scripts/remote-docker.sh exec stream-capture yt-dlp --proxy http://10.0.0.100:8888 -F YOUR_YOUTUBE_URL
```

## Next Steps After Testing

Once stream capture is working:

1. **Test YOLO Detection**
   ```bash
   ./scripts/remote-docker.sh up -d yolo-detector data-manager
   ```

2. **Test Weather Service**
   ```bash
   ./scripts/remote-docker.sh up -d weather-service
   ```

3. **Test Online Learning**
   ```bash
   ./scripts/remote-docker.sh up -d online-learner
   ```

4. **Switch to Full UI**
   ```bash
   # Edit .env to add:
   STREAMLIT_APP=app.py

   # Restart web-ui
   ./scripts/remote-docker.sh restart web-ui
   ```

## Useful Commands

### Restart a service
```bash
./scripts/remote-docker.sh restart stream-capture
```

### Rebuild after code changes
```bash
./scripts/deploy-remote.sh
./scripts/remote-docker.sh build stream-capture
./scripts/remote-docker.sh restart stream-capture
```

### Stop all services
```bash
./scripts/remote-docker.sh down
```

### Check resource usage
```bash
ssh dockerserver 'docker stats --no-stream'
```

### Clean old frames
```bash
ssh dockerserver 'find flowium/volumes/frames/ -name "frame_*.jpg" -delete'
```

## Architecture (Stream Test)

```
┌─────────────────┐
│   YouTube       │
│   Live Stream   │
└────────┬────────┘
         │
         │ yt-dlp (with proxy)
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ stream-capture  │────>│  preprocessing   │────>│    web-ui       │
│  (captures)     │     │  (serves images) │     │  (displays)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         │
    volumes/frames/         HTTP APIs                      │
    - latest.jpg        /image/latest                      │
    - frame_*.jpg       /image/processed                   │
                                                            ▼
                                                    http://localhost:8501
```

## Configuration Reference

### Environment Variables (.env)

```bash
# Required for stream capture
YOUTUBE_URL=https://www.youtube.com/watch?v=VIDEO_ID

# Optional: Proxy for yt-dlp
PROXY_URL=http://10.0.0.100:8888

# Optional: Stream processing
FRAME_RATE=5

# Optional: UI selection
STREAMLIT_APP=app_simple.py  # or app.py for full UI
```

### Services for Stream Test

Minimal setup:
- `stream-capture` - Captures frames from YouTube
- `preprocessing` - Serves images via API
- `web-ui` - Displays stream in browser

## Support

If you encounter issues:

1. Check logs: `./scripts/remote-docker.sh logs <service-name>`
2. Check service health: `curl http://localhost:<port>/health`
3. Verify configuration: `ssh dockerserver 'cd flowium && cat .env'`
4. Restart services: `./scripts/remote-docker.sh restart <service-name>`

---

Happy testing! Once the stream is working, we can move to the next phase (YOLO detection).
