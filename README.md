# Flowium - Real-Time Traffic Monitoring System

Real-time vehicle detection and traffic prediction using YOLOv8, online machine learning with River, and Streamlit dashboard.

## Project Overview

Flowium is a microservices-based traffic monitoring system that:
- Captures live video streams from YouTube
- Detects vehicles using YOLOv8
- Stores detection data with weather conditions
- Uses online learning (River) to predict traffic patterns
- Provides a web-based dashboard for visualization

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Stream Capture │────>│  Preprocessing   │────>│ YOLO Detector   │
│   (yt-dlp)      │     │  (OpenCV)        │     │  (YOLOv8)       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                            │
                        ┌───────────────────────────────────┘
                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Weather API    │────>│  Data Manager    │────>│ Online Learner  │
│ (OpenWeatherMap)│     │  (SQLite/API)    │     │  (River ML)     │
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   Web UI         │
                        │  (Streamlit)     │
                        └──────────────────┘
```

## Services

1. **stream-capture**: Captures frames from YouTube live stream
2. **preprocessing**: Crops and enhances images
3. **yolo-detector**: Detects vehicles using YOLOv8
4. **data-manager**: Manages SQLite database (detections, weather)
5. **online-learner**: Online ML for traffic prediction (River)
6. **weather-service**: Fetches weather data from OpenWeatherMap
7. **web-ui**: Streamlit dashboard for visualization

## Technology Stack

- **Python 3.11**: All services
- **FastAPI**: REST APIs
- **Streamlit**: Web UI
- **YOLOv8**: Vehicle detection
- **OpenCV**: Image processing
- **yt-dlp**: YouTube stream capture
- **SQLite**: Database
- **River**: Online machine learning
- **Docker**: Containerization

## Prerequisites

- Docker and Docker Compose
- OpenWeatherMap API key (free tier: https://openweathermap.org/api)
- YouTube live stream URL (e.g., traffic camera)

## Quick Start

### 1. Clone and Setup

```bash
cd /home/multibox/source/repos/flowium

# Copy environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

### 2. Configure Environment Variables

Edit `.env` file:

```bash
YOUTUBE_URL=https://www.youtube.com/watch?v=YOUR_STREAM_ID
WEATHER_API_KEY=your_openweathermap_api_key
```

### 3. Build and Run

**Local Docker:**
```bash
docker-compose up --build
```

**Remote Docker:**
```bash
# Set remote Docker host
export DOCKER_HOST=ssh://user@remote-host

# Or specify in command
docker -H ssh://user@remote-host compose up --build
```

### 4. Access Services

- **Web UI**: http://localhost:8501
- **Data Manager API**: http://localhost:8003
- **YOLO Detector API**: http://localhost:8002
- **Preprocessing API**: http://localhost:8001

## Development Setup

### Local Development with VSCode

1. **Install VSCode Extensions:**
   - Remote - SSH
   - Docker
   - Python

2. **Local Development:**
```bash
# Start services locally
docker-compose up

# Develop in VSCode, changes reflect in real-time
```

3. **Remote Docker Development:**
```bash
# Connect VSCode to remote Docker
export DOCKER_HOST=ssh://user@remote-host

# Or use VSCode Remote-SSH extension
```

### Service Structure

Each service follows this structure:
```
services/<service-name>/
├── Dockerfile
├── requirements.txt
├── main.py (or app.py)
└── ...
```

## Usage

### Starting the System

1. **Start all services:**
```bash
docker-compose up -d
```

2. **Check service health:**
```bash
curl http://localhost:8003/health  # Data Manager
curl http://localhost:8002/health  # YOLO Detector
```

3. **Access Web UI:**
Open http://localhost:8501 in your browser

### Web UI Features

- **Dashboard**: System status, statistics, hourly traffic
- **Live Detection**: Run detection, view recent detections
- **Analytics**: Predict traffic based on time and weather
- **Settings**: Start/stop continuous detection and weather monitoring

### API Usage

#### Detect Vehicles
```bash
curl -X POST http://localhost:8002/detect
```

#### Get Recent Detections
```bash
curl http://localhost:8003/detections?limit=10
```

#### Get Current Weather
```bash
curl http://localhost:8005/weather
```

#### Predict Traffic
```bash
curl -X POST http://localhost:8004/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 14,
    "day_of_week": 3,
    "temperature": 22.5,
    "humidity": 60.0
  }'
```

#### Train Model
```bash
curl -X POST http://localhost:8004/auto-train
```

## Configuration

### Preprocessing Configuration

Edit crop region in `docker-compose.yml`:
```yaml
environment:
  - CROP_X=560
  - CROP_Y=340
  - CROP_WIDTH=800
  - CROP_HEIGHT=400
```

### YOLO Configuration

Change confidence threshold:
```yaml
environment:
  - CONFIDENCE_THRESHOLD=0.5  # 0.0 - 1.0
```

### Frame Rate

Adjust capture frame rate:
```yaml
environment:
  - FRAME_RATE=5  # frames per second
```

## Project Phases

### Phase 1: Core Pipeline (Current)
- ✅ Stream capture
- ✅ Preprocessing
- ✅ YOLO detection
- ✅ Data storage

### Phase 2: Intelligence
- ✅ Online learning with River
- ✅ Weather integration
- ✅ Traffic prediction

### Phase 3: Visualization
- ✅ Streamlit web UI
- ✅ Real-time dashboard
- ✅ Statistics and charts

### Future Enhancements
- [ ] Video feed display in UI
- [ ] Historical data export
- [ ] Email/SMS alerts for traffic anomalies
- [ ] Multi-camera support
- [ ] Advanced ML models
- [ ] Mobile app

## Troubleshooting

### Stream Capture Issues

**Problem**: "Failed to get stream URL"
```bash
# Test yt-dlp manually
docker-compose run stream-capture yt-dlp -F YOUR_YOUTUBE_URL
```

### YOLO Model Download

**Problem**: Model not downloading
```bash
# Manually download model
docker-compose run yolo-detector python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Permission Issues

**Problem**: Cannot write to volumes
```bash
chmod -R 777 volumes/
```

### Service Not Starting

**Problem**: Container crashes
```bash
# View logs
docker-compose logs <service-name>

# Example:
docker-compose logs yolo-detector
```

## Performance Optimization

### CPU-Only Environment

The system is optimized for CPU-only environments:

1. **YOLOv8n (nano)**: Smallest, fastest model
2. **Frame skipping**: Process 4-8 FPS instead of 30 FPS
3. **Image cropping**: Process only relevant region
4. **Async processing**: FastAPI async endpoints

### Expected Performance

**Hardware**: 4 CPU cores @ 2.2 GHz (no GPU)
- Stream capture: 5 FPS
- YOLO detection: ~2-4 FPS
- Memory: ~2-4 GB total

## Data Management

### Database Location

SQLite database: `volumes/db-data/flowium.db`

### Backup Database

```bash
cp volumes/db-data/flowium.db backups/flowium_$(date +%Y%m%d).db
```

### Export Data

```bash
# Connect to data-manager container
docker exec -it flowium-data-manager sqlite3 /data/flowium.db

# Export to CSV
.mode csv
.output /data/detections.csv
SELECT * FROM detections;
.quit
```

## Monitoring

### Container Status

```bash
docker-compose ps
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f yolo-detector
```

### Resource Usage

```bash
docker stats
```

## Contributing

This is an academic project. Contributions welcome:

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## License

Academic project - see your institution's guidelines.

## Acknowledgments

- YOLOv8 by Ultralytics
- River online learning library
- Streamlit for rapid UI development
- OpenCV for computer vision
- yt-dlp for YouTube stream capture

## Contact

For questions about this project, please refer to the project documentation or contact the maintainer.

---

**Note**: This system is designed for educational purposes and traffic monitoring research. Ensure you comply with YouTube's Terms of Service and local regulations regarding video surveillance and data collection.
