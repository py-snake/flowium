# Accuracy Tuning Guide

## Current Issue: Speed vs Accuracy Trade-off

After initial optimizations, you achieved **11.8 FPS** (4x faster!), but accuracy decreased for:
- Parked/stationary vehicles (intermittent detection)
- Passing vehicles (some missed)

## ✅ Applied Fixes

### 1. Increased Image Size: 384 → 416
- **Before**: imgsz=384 (maximum speed, lower accuracy)
- **After**: imgsz=416 (balanced speed + accuracy)
- **Impact**: ~30% slower but much better detection
- **Expected FPS**: 7-9 FPS (still 3x faster than original!)

### 2. Lowered Confidence Threshold: 0.5 → 0.4
- **Before**: CONFIDENCE_THRESHOLD=0.5 (default)
- **After**: CONFIDENCE_THRESHOLD=0.4 (more detections)
- **Impact**: Catches more vehicles, especially stationary ones
- **Trade-off**: May have slightly more false positives (filtered by tracking)

## 📊 Expected Performance After Tuning

```
🎨 Preprocessing:    20-25 ms   (unchanged)
🚗 YOLO Detection:   80-100 ms  (slightly slower for accuracy)
⏱️ Overall:         100-125 ms  (8-10 FPS)
💻 CPU Usage:       80-90%      (still excellent)
🎯 Accuracy:        Much better! (parked + passing cars)
```

## 🎛️ Fine-Tuning Options

If you still have accuracy issues, try these adjustments:

### Option 1: Further Lower Confidence (More Detections)
```bash
# In .env file:
CONFIDENCE_THRESHOLD=0.35  # Even more sensitive
```
**Pro**: Catches almost everything
**Con**: More false positives (but tracking filters these out)

### Option 2: Increase Image Size More (Better Accuracy)
```bash
# In services/yolo-detector/main.py and export_model.py:
imgsz=448  # Even better accuracy
```
**Pro**: Best accuracy for small/distant vehicles
**Con**: Slower (~6-7 FPS)

### Option 3: Use Larger YOLO Model (Best Accuracy)
```bash
# In services/yolo-detector/main.py:
model = YOLO('yolov8s.pt')  # Small model (larger than nano)
# Then export to ONNX
```
**Pro**: Much better accuracy
**Con**: Significantly slower (maybe 4-5 FPS)

### Option 4: Adjust Tracker Settings (For Stationary Vehicles)
```python
# In services/yolo-detector/main.py:
tracker = VehicleTracker(
    iou_threshold=0.25,  # Lower = more lenient (was 0.3)
    max_age=45,          # Longer memory (was 30)
    min_hits=2           # Faster confirmation (was 3)
)
```
**Pro**: Better tracking of slow-moving/stationary vehicles
**Con**: May create duplicate tracks if too lenient

## 🔄 How to Deploy Changes

The changes I made (imgsz=416 + CONFIDENCE_THRESHOLD=0.4) are ready to deploy:

```bash
cd /home/multibox/source/repos/flowium

# Rebuild with new settings (ONNX will be re-exported with imgsz=416)
docker compose build --no-cache yolo-detector
docker compose up -d
```

## 📈 Testing Accuracy

After deploying, test with:

1. **Parked vehicles**: Should be detected consistently
2. **Passing vehicles**: Should catch all or most
3. **FPS**: Should still be 7-10 FPS (much better than original 2.9!)
4. **CPU usage**: Should remain 80-90%

## 🎯 Recommended Settings by Scenario

### High-Speed Road (Fast Moving Vehicles)
```bash
CONFIDENCE_THRESHOLD=0.4
imgsz=416
min_hits=2  # Faster confirmation
```

### Urban/Parking (Slow/Stationary Vehicles)
```bash
CONFIDENCE_THRESHOLD=0.35  # More sensitive
imgsz=448  # Better accuracy for distant vehicles
min_hits=2
max_age=50  # Longer tracking memory
```

### Balanced (Default - Works for Most Cases)
```bash
CONFIDENCE_THRESHOLD=0.4
imgsz=416
min_hits=3
max_age=30
```

## 🔍 Debugging Detection Issues

If specific vehicles still aren't detected:

1. **Check the raw frame** - Is the vehicle visible/clear?
2. **Check crop region** - Is the vehicle in the cropped area?
3. **Check lighting** - Night scenes may need preprocessing adjustments
4. **Check size** - Very small/distant vehicles may need larger imgsz
5. **Check logs** - Look for detection counts in container logs

### View Detection Logs
```bash
docker logs -f flowium-yolo-detector | grep "Frame:"
```

You should see:
```
Frame: 3 detections, 2 active tracks, 1 NEW vehicles | ⚡ 95ms
```

## 📊 Performance vs Accuracy Matrix

| imgsz | Conf | FPS  | Accuracy | Use Case |
|-------|------|------|----------|----------|
| 384   | 0.5  | 11-13| Low      | Maximum speed (not recommended) |
| 384   | 0.4  | 11-13| Medium   | High speed, ok accuracy |
| 416   | 0.5  | 8-10 | Medium   | Balanced |
| **416** | **0.4** | **7-9** | **High** | **Recommended** ⭐ |
| 448   | 0.4  | 6-8  | Very High| Best accuracy |
| 480   | 0.4  | 5-7  | Excellent| Maximum accuracy |
| 640   | 0.4  | 3-4  | Best     | Original (slow) |

## 💡 Quick Fixes

**Problem**: Parked cars still missed
- **Solution**: Lower CONFIDENCE_THRESHOLD to 0.35
- **Or**: Increase imgsz to 448

**Problem**: Too many false positives
- **Solution**: Raise CONFIDENCE_THRESHOLD to 0.45
- **Or**: Increase min_hits to 4 in tracker

**Problem**: FPS too low
- **Solution**: Reduce imgsz to 384 (but check accuracy)
- **Or**: Increase PROCESSING_FPS from 5 to 3

**Problem**: Vehicles lost during tracking
- **Solution**: Increase max_age to 40-50
- **Or**: Lower iou_threshold to 0.25

## ✅ Current Settings (After This Update)

```bash
# .env
CONFIDENCE_THRESHOLD=0.4

# main.py
imgsz=416

# tracker settings (default)
iou_threshold=0.3
max_age=30
min_hits=3
```

**These should give you ~8 FPS with much better accuracy!**
