"""
YOLO Detector Service
Detects vehicles using YOLOv8 and sends results to Data Manager
"""
import os
import time
import cv2
import requests
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from ultralytics import YOLO
from typing import List, Dict
from datetime import datetime
from tracker import VehicleTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimize CPU usage for maximum performance
import torch
torch.set_num_threads(8)  # Use all 8 CPU cores for PyTorch
cv2.setNumThreads(8)  # Use all 8 cores for OpenCV operations
os.environ['OMP_NUM_THREADS'] = '8'  # OpenMP threads
os.environ['MKL_NUM_THREADS'] = '8'  # Intel MKL threads

app = FastAPI(title="YOLO Detector Service")

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', '/models/yolov8n.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
DATA_MANAGER_URL = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')
PROCESSING_FPS = int(os.getenv('PROCESSING_FPS', '5'))
INPUT_DIR = Path('/shared/frames')

# Initialize YOLO model
print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO('yolov8n.pt')  # Will download on first run
print("YOLO model loaded successfully")

# Vehicle class IDs in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Performance tracking
performance_stats = {
    "last_detection_time_ms": 0,
    "avg_detection_time_ms": 0,
    "total_frames": 0
}

def atomic_write_image(image, output_path: Path):
    """Write image atomically to prevent race conditions"""
    import tempfile
    import numpy as np
    # Write to temp file first
    temp_fd, temp_path_str = tempfile.mkstemp(suffix='.jpg', dir=output_path.parent)
    temp_path = Path(temp_path_str)
    try:
        # Write image to temp file
        cv2.imwrite(str(temp_path), image)
        # Atomically rename to final path (atomic on Linux)
        temp_path.rename(output_path)
    finally:
        # Clean up temp file descriptor
        os.close(temp_fd)
        # Clean up temp file if rename failed
        if temp_path.exists():
            temp_path.unlink()

# Initialize tracker
tracker = VehicleTracker(
    iou_threshold=0.3,  # Lower = stricter matching
    max_age=30,         # Keep tracks for 30 frames without detection (6 seconds at 5 FPS)
    min_hits=3          # Require 3 detections before confirming (0.6 seconds at 5 FPS)
)

detection_running = False

def detect_and_annotate_vehicles(image_path: Path, save_annotated: bool = True, tracked_detections: List[Dict] = None) -> tuple[List[Dict], str]:
    """Detect vehicles in image and optionally save annotated version with track IDs"""

    if not image_path.exists():
        return [], None

    # Read image for annotation
    image = cv2.imread(str(image_path))
    if image is None:
        return [], None

    # Run inference with optimized image size for faster CPU processing
    # imgsz=480 will resize the longest edge to 480px for inference (faster)
    results = model(str(image_path), conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=480)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # Only keep vehicle classes
            if cls in VEHICLE_CLASSES:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)

                detections.append({
                    "class_id": cls,
                    "class_name": result.names[cls],
                    "confidence": conf,
                    "bbox": xyxy,
                    "timestamp": datetime.utcnow().isoformat()
                })

    # Draw bounding boxes with track IDs if available
    if save_annotated and tracked_detections:
        for det in tracked_detections:
            cls = det['class_id']
            conf = det['confidence']
            x1, y1, x2, y2 = map(int, det['bbox'])
            track_id = det.get('track_id', '?')
            hits = det.get('hits', 0)

            # Choose color based on vehicle type
            colors = {
                2: (0, 255, 0),      # car - green
                3: (255, 0, 0),      # motorcycle - blue
                5: (0, 255, 255),    # bus - yellow
                7: (0, 165, 255)     # truck - orange
            }
            color = colors.get(cls, (255, 255, 255))

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label with track ID and confidence
            label = f"ID:{track_id} {det['class_name']} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1, label_size[1] + 10)

            # Draw label background
            cv2.rectangle(image, (x1, label_y - label_size[1] - 10),
                        (x1 + label_size[0], label_y), color, -1)

            # Draw label text
            cv2.putText(image, label, (x1, label_y - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw hit counter (small indicator showing how many times detected)
            if hits >= 3:  # Only show for confirmed tracks
                hit_label = f"✓{hits}"
                cv2.putText(image, hit_label, (x2 - 40, y1 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    # Save annotated image atomically (prevents race conditions)
    annotated_path = None
    if save_annotated:
        annotated_path = INPUT_DIR / 'detected_latest.jpg'
        atomic_write_image(image, annotated_path)

    return detections, str(annotated_path) if annotated_path else None

def send_to_data_manager(detections: List[Dict]):
    """Send detections to Data Manager service"""
    try:
        response = requests.post(
            f"{DATA_MANAGER_URL}/detections",
            json={"detections": detections},
            timeout=5
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send detections to data manager: {e}")
        return False

@app.get("/")
def root():
    return {
        "service": "YOLO Detector Service",
        "status": "running",
        "model": "YOLOv8n",
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "detection_running": detection_running,
        "active_tracks": tracker.get_active_track_count(),
        "confirmed_tracks": tracker.get_confirmed_track_count()
    }

@app.post("/detect")
def detect():
    """Detect vehicles in the latest processed frame"""

    # Try processed frame first, fall back to raw frame
    processed_frame = INPUT_DIR / 'processed_latest.jpg'
    raw_frame = INPUT_DIR / 'latest.jpg'

    image_path = processed_frame if processed_frame.exists() else raw_frame

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="No frame available")

    detections, annotated_path = detect_and_annotate_vehicles(image_path, save_annotated=True)

    # Send to data manager
    if detections:
        send_to_data_manager(detections)

    return {
        "status": "success",
        "detections_count": len(detections),
        "detections": detections,
        "annotated_image": annotated_path
    }

@app.get("/image/detected")
def get_detected_image():
    """Serve the latest annotated frame with detection boxes"""
    detected_frame = INPUT_DIR / 'detected_latest.jpg'

    if not detected_frame.exists():
        raise HTTPException(status_code=404, detail="No detected frame available")

    return FileResponse(detected_frame, media_type="image/jpeg")

@app.get("/performance")
def get_performance_stats():
    """Get detection performance statistics"""
    return {
        "last_detection_time_ms": round(performance_stats["last_detection_time_ms"], 2),
        "avg_detection_time_ms": round(performance_stats["avg_detection_time_ms"], 2),
        "total_frames_processed": performance_stats["total_frames"]
    }

@app.get("/stats")
def get_detection_stats():
    """Get latest detection statistics"""
    detected_frame = INPUT_DIR / 'detected_latest.jpg'

    if not detected_frame.exists():
        return {
            "detections_available": False,
            "vehicle_count": 0
        }

    # Re-run detection to get current stats (without saving)
    processed_frame = INPUT_DIR / 'processed_latest.jpg'
    raw_frame = INPUT_DIR / 'latest.jpg'
    image_path = processed_frame if processed_frame.exists() else raw_frame

    if not image_path.exists():
        return {
            "detections_available": False,
            "vehicle_count": 0
        }

    detections, _ = detect_and_annotate_vehicles(image_path, save_annotated=False)

    # Count by vehicle type
    vehicle_counts = {}
    for det in detections:
        class_name = det['class_name']
        vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

    return {
        "detections_available": True,
        "vehicle_count": len(detections),
        "vehicle_types": vehicle_counts,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

@app.post("/start-continuous")
def start_continuous_detection(background_tasks: BackgroundTasks):
    """Start continuous detection (runs in background)"""
    global detection_running

    if detection_running:
        return {"status": "already_running"}

    detection_running = True
    background_tasks.add_task(continuous_detection_loop)

    return {"status": "started"}

@app.post("/stop-continuous")
def stop_continuous_detection():
    """Stop continuous detection"""
    global detection_running
    detection_running = False
    return {"status": "stopped"}

async def auto_detect_task():
    """Background task that automatically detects vehicles at specified FPS with tracking"""
    global detection_running

    logger.info(f"Starting automatic detection task at {PROCESSING_FPS} FPS with object tracking...")
    logger.info(f"Tracker config: IoU threshold=0.3, max_age=30 frames, min_hits=3")

    # Calculate sleep time based on desired FPS
    sleep_time = 1.0 / PROCESSING_FPS

    while True:
        try:
            processed_frame = INPUT_DIR / 'processed_latest.jpg'
            raw_frame = INPUT_DIR / 'latest.jpg'

            image_path = processed_frame if processed_frame.exists() else raw_frame

            if image_path.exists():
                # Start timing
                import time as time_module
                detect_start = time_module.time()

                # Detect all vehicles in current frame (without annotation yet)
                detections, _ = detect_and_annotate_vehicles(image_path, save_annotated=False)

                if detections:
                    # Update tracker with detections
                    tracked_detections = tracker.update(detections)

                    # Now annotate with track IDs
                    _, annotated_path = detect_and_annotate_vehicles(
                        image_path,
                        save_annotated=True,
                        tracked_detections=tracked_detections
                    )

                    # Update performance stats
                    detect_time_ms = (time_module.time() - detect_start) * 1000
                    performance_stats["last_detection_time_ms"] = detect_time_ms
                    performance_stats["total_frames"] += 1

                    # Calculate rolling average
                    alpha = 0.01
                    if performance_stats["total_frames"] == 1:
                        performance_stats["avg_detection_time_ms"] = detect_time_ms
                    else:
                        performance_stats["avg_detection_time_ms"] = (
                            alpha * detect_time_ms +
                            (1 - alpha) * performance_stats["avg_detection_time_ms"]
                        )

                    # Get only NEW confirmed tracks (haven't been stored yet)
                    new_confirmed_tracks = tracker.get_confirmed_new_tracks(tracked_detections)

                    # Only send new confirmed tracks to database
                    if new_confirmed_tracks:
                        send_to_data_manager(new_confirmed_tracks)
                        logger.info(
                            f"Frame: {len(detections)} detections, "
                            f"{tracker.get_active_track_count()} active tracks, "
                            f"{len(new_confirmed_tracks)} NEW vehicles counted"
                        )
                    else:
                        # Log occasionally even when no new vehicles
                        if tracker.frame_count % 25 == 0:  # Every 5 seconds at 5 FPS
                            logger.info(
                                f"Tracking: {tracker.get_active_track_count()} active tracks, "
                                f"{tracker.get_confirmed_track_count()} confirmed (no new vehicles)"
                            )

        except Exception as e:
            logger.error(f"Error in auto-detection: {e}")

        # Sleep for the calculated interval to maintain desired FPS
        await asyncio.sleep(sleep_time)

async def continuous_detection_loop():
    """Continuously detect vehicles (deprecated - use auto_detect_task)"""
    global detection_running

    logger.info("Starting continuous detection loop...")

    while detection_running:
        try:
            processed_frame = INPUT_DIR / 'processed_latest.jpg'
            raw_frame = INPUT_DIR / 'latest.jpg'

            image_path = processed_frame if processed_frame.exists() else raw_frame

            if image_path.exists():
                detections, annotated_path = detect_and_annotate_vehicles(image_path, save_annotated=True)

                if detections:
                    send_to_data_manager(detections)
                    logger.info(f"Detected {len(detections)} vehicles")

            # Wait before next detection
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
            time.sleep(5)

    logger.info("Continuous detection loop stopped")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(auto_detect_task())
