# Remote Deployment Guide

This guide explains how to deploy and manage Flowium on your remote Docker server (`dockerserver`).

## Initial Deployment

The project has been deployed to: `dockerserver:/root/flowium`

## Quick Start

### 1. Configure Environment Variables

Run the interactive setup script:
```bash
./scripts/remote-setup-env.sh
```

Or manually:
```bash
ssh dockerserver 'cd flowium && nano .env'
```

Add your:
- `YOUTUBE_URL` - Your YouTube live stream URL
- `WEATHER_API_KEY` - Your OpenWeatherMap API key

### 2. Build and Start Services

```bash
# Build and start all services
./scripts/remote-docker.sh up -d --build

# Or use docker-compose directly on remote
ssh dockerserver 'cd flowium && docker-compose up -d --build'
```

### 3. Access the Web UI

The web UI will be available at:
```
http://dockerserver:8501
```

Or if you know the server's IP:
```
http://<server-ip>:8501
```

## Management Commands

All commands use the `remote-docker.sh` wrapper script:

### Start/Stop Services
```bash
# Start all services
./scripts/remote-docker.sh up -d

# Stop all services
./scripts/remote-docker.sh down

# Restart all services
./scripts/remote-docker.sh restart

# Restart specific service
./scripts/remote-docker.sh restart yolo-detector
```

### View Logs
```bash
# All services
./scripts/remote-docker.sh logs -f

# Specific service
./scripts/remote-docker.sh logs -f yolo-detector

# Last 100 lines
./scripts/remote-docker.sh logs --tail=100
```

### Check Status
```bash
# List all services
./scripts/remote-docker.sh ps

# Service health
ssh dockerserver 'curl -s http://localhost:8003/health | jq'
```

### Rebuild Services
```bash
# Rebuild all
./scripts/remote-docker.sh build

# Rebuild specific service
./scripts/remote-docker.sh build yolo-detector

# Rebuild and restart
./scripts/remote-docker.sh up -d --build yolo-detector
```

### Shell Access
```bash
# Get shell in container
./scripts/remote-docker.sh exec yolo-detector /bin/bash

# Run command in container
./scripts/remote-docker.sh exec data-manager sqlite3 /data/flowium.db
```

## Updating the Deployment

When you make changes locally, redeploy:

```bash
# 1. Deploy updated files
./scripts/deploy-remote.sh

# 2. Rebuild affected services
./scripts/remote-docker.sh build <service-name>
./scripts/remote-docker.sh restart <service-name>

# Or rebuild everything
./scripts/remote-docker.sh up -d --build
```

## Accessing Services

From your local machine, you can access services via SSH tunnels:

### Web UI
```bash
# Create SSH tunnel
ssh -L 8501:localhost:8501 dockerserver

# Then access http://localhost:8501 in your browser
```

### All Services
```bash
# Create tunnels for all ports
ssh -L 8501:localhost:8501 \
    -L 8003:localhost:8003 \
    -L 8002:localhost:8002 \
    dockerserver
```

Then access:
- Web UI: http://localhost:8501
- Data Manager API: http://localhost:8003
- YOLO Detector API: http://localhost:8002

## Monitoring

### Check Container Status
```bash
./scripts/remote-docker.sh ps
```

### Resource Usage
```bash
ssh dockerserver 'docker stats --no-stream'
```

### Disk Usage
```bash
ssh dockerserver 'cd flowium && du -sh volumes/*'
```

### View Errors
```bash
# Recent errors in logs
./scripts/remote-docker.sh logs --tail=50 | grep -i error
```

## Data Management

### Backup Database
```bash
ssh dockerserver 'cp flowium/volumes/db-data/flowium.db flowium/volumes/db-data/flowium_backup_$(date +%Y%m%d).db'
```

### Download Database Locally
```bash
scp dockerserver:flowium/volumes/db-data/flowium.db ./flowium_backup.db
```

### Clear Old Frames
```bash
ssh dockerserver 'find flowium/volumes/frames/ -name "frame_*.jpg" -mtime +7 -delete'
```

## Troubleshooting

### Service Won't Start

1. Check logs:
```bash
./scripts/remote-docker.sh logs <service-name>
```

2. Check if port is in use:
```bash
ssh dockerserver 'netstat -tulpn | grep 8501'
```

3. Restart service:
```bash
./scripts/remote-docker.sh restart <service-name>
```

### Out of Disk Space

Check usage:
```bash
ssh dockerserver 'df -h'
```

Clean up:
```bash
# Remove old containers and images
ssh dockerserver 'docker system prune -a'

# Remove old frames
ssh dockerserver 'find flowium/volumes/frames/ -name "frame_*.jpg" -delete'
```

### Connection Issues

Check Docker network:
```bash
./scripts/remote-docker.sh exec web-ui ping data-manager
```

### Performance Issues

Check resource usage:
```bash
ssh dockerserver 'docker stats'
```

Adjust frame rate in `.env`:
```bash
ssh dockerserver 'cd flowium && nano .env'
# Set FRAME_RATE=3 (lower for less CPU usage)
```

## Directory Structure on Remote

```
/root/flowium/
├── docker-compose.yml
├── .env                      # Your environment variables
├── .env.example             # Template
├── README.md
├── services/                # All 7 microservices
│   ├── stream-capture/
│   ├── preprocessing/
│   ├── yolo-detector/
│   ├── data-manager/
│   ├── online-learner/
│   ├── weather-service/
│   └── web-ui/
├── volumes/                 # Persistent data
│   ├── frames/             # Video frames
│   ├── models/             # ML models
│   └── db-data/            # SQLite database
└── scripts/                # Helper scripts
```

## Security Notes

1. **Firewall**: Make sure only necessary ports are exposed
2. **Environment Variables**: Never commit `.env` to git
3. **API Keys**: Keep your OpenWeatherMap API key secure
4. **SSH Keys**: Use key-based authentication only

## Maintenance Tasks

### Weekly
- Check disk usage: `ssh dockerserver 'df -h'`
- Backup database: `./scripts/remote-docker.sh exec data-manager sqlite3 /data/flowium.db '.backup /data/backup.db'`

### Monthly
- Clean old frames: `ssh dockerserver 'find flowium/volumes/frames/ -name "frame_*.jpg" -mtime +30 -delete'`
- Update Docker images: `./scripts/remote-docker.sh pull`

### As Needed
- Check logs for errors: `./scripts/remote-docker.sh logs | grep -i error`
- Monitor resource usage: `ssh dockerserver 'docker stats'`

## Support

For issues or questions, refer to:
- Main README: [README.md](README.md)
- Docker Compose logs: `./scripts/remote-docker.sh logs -f`
- Service health endpoints: `curl http://<service>:8000/health`
