# Flowium Performance Optimizations

This document describes the performance optimizations implemented in Flowium to maximize speed and stability on CPU-only systems.

## Implemented Optimizations

### 1. ONNX Model Export (2-3x Speed Improvement)

**What it does**: Converts YOLOv8n from PyTorch (.pt) to ONNX format with optimizations for CPU inference.

**Impact**: 2-3x faster inference on CPU
- Before: ~300-400ms per frame
- After: ~100-150ms per frame (estimated)

**How it works**:
- `export_model.py` converts the model during Docker build
- ONNX runtime is optimized for CPU with graph simplification
- Fixed input size (384x384) for maximum optimization

**Files modified**:
- `/services/yolo-detector/export_model.py` - Export script
- `/services/yolo-detector/main.py` - Auto-loads ONNX if available
- `/services/yolo-detector/Dockerfile` - Exports during build

### 2. Reduced Inference Size (30-50% Speed Improvement)

**What it does**: Reduces YOLO inference image size from 480px to 384px.

**Impact**: 30-50% faster inference
- Smaller images = faster processing
- With object tracking, accuracy remains high despite smaller size

**Configuration**: `imgsz=384` in YOLO inference call

### 3. Adaptive FPS (Prevents Queue Buildup)

**What it does**: Skips new frames if previous frame is still being processed.

**Impact**:
- Prevents processing queue from building up
- Ensures system always processes latest available frames
- Reduces lag and improves real-time responsiveness

**How it works**:
- Tracks `is_processing` flag
- Skips frames when CPU is busy
- Automatically adapts to system load

### 4. Smart Frame Dropping (Avoids Duplicate Processing)

**What it does**: Tracks last processed frame's modification time to avoid reprocessing the same frame.

**Impact**:
- Eliminates wasted CPU cycles on duplicate frames
- Improves actual throughput
- Reduces unnecessary detections

**How it works**:
- Stores `last_processed_mtime`
- Compares with current frame's mtime
- Skips if mtime unchanged

### 5. CPU Threading Optimization (80-100% CPU Utilization)

**What it does**: Configures all libraries to use all 8 CPU cores.

**Impact**: Increases CPU utilization from 40-50% to 80-90%

**Configuration**:
- PyTorch: `torch.set_num_threads(8)`
- OpenCV: `cv2.setNumThreads(8)`
- OpenMP: `OMP_NUM_THREADS=8`
- Intel MKL: `MKL_NUM_THREADS=8`

### 6. Enhanced Performance Monitoring

**What it does**: Tracks and displays detailed performance metrics.

**Metrics shown**:
- Last/average processing time for preprocessing and detection
- Overall FPS
- Frame drop rate (adaptive FPS indicator)
- Total frames dropped vs processed

**UI Display**:
- Real-time performance stats in sidebar
- Drop rate indicator with color coding
- Warnings if drop rate is high

## Performance Comparison

### Before Optimizations
```
Preprocessing: 34.8ms
YOLO Detection: 307.1ms
Overall: 341.9ms (2.9 FPS)
CPU Usage: 40-50%
```

### After Optimizations (Expected)
```
Preprocessing: 30-35ms (minimal change)
YOLO Detection: 100-150ms (2-3x faster with ONNX + smaller imgsz)
Overall: 130-185ms (5-7 FPS)
CPU Usage: 80-90%
Frame Drops: Adaptive (0-20% depending on load)
```

## Advanced Optimizations (Not Yet Implemented)

The following optimizations were considered but not yet implemented:

### 7. Batch Processing
- Process 2-3 frames simultaneously
- More efficient GPU/CPU utilization
- Requires queue management

### 8. Multi-Process Detection
- Use ProcessPoolExecutor with 2-3 workers
- Parallel processing of frames
- Requires inter-process communication

### 9. Message Queue (Redis/RabbitMQ)
- Replace HTTP calls with message queue
- Better async communication
- Reduces overhead between services
- Requires architecture change

## How to Verify Optimizations

1. **Check ONNX model is loaded**:
   ```bash
   docker logs flowium-yolo-detector | grep ONNX
   ```
   Should see: "✅ Loading optimized ONNX model (2-3x faster on CPU)..."

2. **Monitor performance in Web UI**:
   - Open http://your-server:8501
   - Check "⚡ Performance" section in sidebar
   - Look for improved ms times and higher FPS

3. **Check CPU usage**:
   ```bash
   docker stats flowium-yolo-detector
   ```
   Should see 80-90% CPU usage (up from 40-50%)

4. **Monitor frame drop rate**:
   - In Web UI sidebar, check "⏭️ Frame Drop Rate"
   - 0-20% is optimal (system adapting to load)
   - >20% means system is under heavy load but still processing latest frames

## Troubleshooting

**ONNX model not found**:
- Rebuild the yolo-detector service: `docker compose build --no-cache yolo-detector`
- Check logs for export errors

**High frame drop rate (>30%)**:
- Expected if stream resolution is very high
- Reduce crop region size in `.env`
- Consider reducing PROCESSING_FPS from 5 to 3-4

**Low CPU usage (<70%)**:
- Check that thread optimizations are applied (check logs)
- Verify ONNX model is loaded (not .pt model)
- Monitor with `htop` to see per-core usage

## Configuration

All optimizations are automatic. Key environment variables:

```bash
# Processing speed
PROCESSING_FPS=5           # Target FPS (adaptive FPS will adjust)

# Crop region (smaller = faster)
CROP_WIDTH=700
CROP_HEIGHT=430

# Detection confidence
CONFIDENCE_THRESHOLD=0.5   # Lower = more detections but slower
```

## Future Improvements

1. **INT8 Quantization**: Reduce model precision for 2-4x additional speedup
2. **OpenVINO**: Intel's optimized inference engine for even faster CPU inference
3. **Dynamic resolution**: Reduce resolution further at night or when no vehicles detected
4. **GPU support**: Add CUDA support for 10-20x speedup if GPU available
