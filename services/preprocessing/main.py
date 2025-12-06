"""
Preprocessing Service
Provides a robust, adaptive image preprocessing pipeline optimized for 
Computer Vision (YOLOv8) detection accuracy, focusing on signal-to-noise ratio
and consistent illumination across day and night scenes.
"""
import os
import cv2
import numpy as np
import asyncio
import time
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimize CPU usage for maximum performance
cv2.setNumThreads(8)  # Use all 8 cores for OpenCV operations
os.environ['OMP_NUM_THREADS'] = '8'  # OpenMP threads
os.environ['MKL_NUM_THREADS'] = '8'  # Intel MKL threads

app = FastAPI(title="Preprocessing Service")

# Configuration
INPUT_DIR = Path(os.getenv('INPUT_DIR', '/shared/frames'))
CROP_X = int(os.getenv('CROP_X', '560'))
CROP_Y = int(os.getenv('CROP_Y', '340'))
CROP_WIDTH = int(os.getenv('CROP_WIDTH', '800'))
CROP_HEIGHT = int(os.getenv('CROP_HEIGHT', '400'))
PROCESSING_FPS = int(os.getenv('PROCESSING_FPS', '5'))

# New configuration for adaptive processing
# Mean V-channel value below this is considered "dark" (0-255)
BRIGHTNESS_THRESHOLD = int(os.getenv('BRIGHTNESS_THRESHOLD', '60')) 

# Create a thread pool for heavy image processing
# This prevents OpenCV from blocking the async event loop
# Using all 8 CPU cores for maximum performance
thread_pool = ThreadPoolExecutor(max_workers=8)

# Performance tracking
performance_stats = {
    "last_processing_time_ms": 0,
    "avg_processing_time_ms": 0,
    "total_frames": 0
}

# Background Subtraction (MOG2)
# Global background subtractor that adapts to lighting changes over time
background_subtractor = None
background_subtraction_enabled = False

def init_background_subtractor():
    """Initialize MOG2 background subtractor with optimized parameters"""
    global background_subtractor
    # MOG2 parameters:
    # - history: number of frames to build background model (500 = ~100 seconds at 5 FPS)
    # - varThreshold: threshold for foreground detection (lower = more sensitive)
    # - detectShadows: detect and mark shadows (useful for vehicles)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )
    logger.info("Background subtractor initialized (MOG2)")

class ProcessingConfig(BaseModel):
    crop: bool = True
    adaptive_enhancement: bool = True
    denoising_level: int = 5 # 0=off, 1-20
    sharpness_amount: float = 0.5 # 0.0=off, 0.5=moderate, 1.0=strong
    background_subtraction: bool = False # Enable background subtraction

def atomic_write_image_sync(image: np.ndarray, output_path: Path):
    """
    Synchronous version of atomic write.
    Uses tempfile + rename to ensure downstream services never read partial files.
    """
    # Create temp file in the same directory (ensures atomic rename works)
    temp_fd, temp_path_str = tempfile.mkstemp(suffix='.jpg', dir=output_path.parent)
    temp_path = Path(temp_path_str)
    
    try:
        # Write image content (JPEG quality 90 is a good trade-off for CV models)
        cv2.imwrite(str(temp_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]) 
        
        # Change permissions to 644 (Read/Write Owner, Read Group/Others)
        os.chmod(temp_path, 0o644)
        
        # Atomically rename
        temp_path.rename(output_path)
    finally:
        os.close(temp_fd)
        # Clean up temp file if rename failed
        if temp_path.exists():
            temp_path.unlink()

def detect_brightness(image: np.ndarray) -> bool:
    """Checks if the image is in low light based on mean V-channel value (HSV)."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # The V (Value/Brightness) channel is the third channel (index 2)
    v_channel = hsv[:, :, 2] 
    mean_v = np.mean(v_channel)
    return mean_v < BRIGHTNESS_THRESHOLD

def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Applies gamma correction to globally adjust lightness."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_image_sync(image: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Apply YOLO-optimized preprocessing steps (CPU bound) with adaptive logic.)"""

    # 1. CROP (Always first for performance)
    if config.crop:
        image = image[CROP_Y:CROP_Y+CROP_HEIGHT, CROP_X:CROP_X+CROP_WIDTH]

    processed_image = image.copy()

    # 2. BACKGROUND SUBTRACTION (Before other enhancements)
    # This removes static obstacles like trees and poles
    if config.background_subtraction and background_subtractor is not None:
        # Apply background subtraction to get foreground mask
        fg_mask = background_subtractor.apply(processed_image)

        # Post-process the mask to reduce noise
        # Remove shadows (they're marked as 127, we want only foreground=255)
        fg_mask[fg_mask == 127] = 0

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Apply mask to image (keep only foreground objects)
        # Convert mask to 3-channel for multiplication
        fg_mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        processed_image = cv2.bitwise_and(processed_image, fg_mask_3ch)
    
    # 3. ADAPTIVE ENHANCEMENT FOR YOLO
    if config.adaptive_enhancement:
        is_dark = detect_brightness(processed_image)

        # 3a. Low-Light Uplift (Gamma Correction)
        if is_dark:
            logger.debug(f"Applying Gamma Correction (Night) - Mean V: {np.mean(cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)[:,:,2]):.1f}")
            # Toned down Gamma from 1.8 to 1.5 for less light/aggressive uplift
            processed_image = apply_gamma_correction(processed_image, gamma=1.5)
            clahe_clip = 3.0
        else:
            logger.debug(f"Applying mild CLAHE (Day) - Mean V: {np.mean(cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)[:,:,2]):.1f}")
            clahe_clip = 2.0 # Mild CLAHE clip limit for natural contrast

        # 3b. CLAHE for Local Contrast Normalization (L*a*b*)
        # CLAHE is highly resource-effective and powerful filter for CV models
        lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        processed_image = cv2.merge([l, a, b])
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_LAB2BGR)

    # 4. DENOISE (Extreme Optimization for old CPU)
    denoising_level = config.denoising_level
    if denoising_level > 0:
        # Median Blur (3x3 kernel) is extremely fast and effective for suppressing noise
        # before the final sharpness step, making it ideal for old CPUs.
        processed_image = cv2.medianBlur(processed_image, 3)

    # 5. SHARPNESS (Enhances edges for feature maps)
    sharpness = config.sharpness_amount
    if sharpness > 0.0:
        # Convolution with a kernel is one of the fastest ways to sharpen
        kernel = np.array([
            [0, -1, 0],
            [-1, 5 + sharpness, -1],
            [0, -1, 0]
        ])
        
        # Normalize the central value based on the sharpness amount
        kernel[1, 1] = 5 + sharpness * 4.0

        # Apply the kernel (convolution)
        processed_image = cv2.filter2D(processed_image, -1, kernel)

    return processed_image

@app.get("/")
def root():
    return {
        "service": "Preprocessing Service",
        "status": "running",
        "crop_region": f"{CROP_X},{CROP_Y} - {CROP_WIDTH}x{CROP_HEIGHT}"
    }

@app.get("/health")
def health():
    latest_frame = INPUT_DIR / 'latest.jpg'
    return {
        "status": "healthy",
        "latest_frame_exists": latest_frame.exists(),
        "background_subtraction_enabled": background_subtraction_enabled,
        "background_subtractor_initialized": background_subtractor is not None
    }

@app.post("/background-subtraction/enable")
def enable_background_subtraction():
    """Enable background subtraction (MOG2)"""
    global background_subtraction_enabled

    if background_subtractor is None:
        init_background_subtractor()

    background_subtraction_enabled = True
    logger.info("Background subtraction ENABLED")

    return {
        "status": "enabled",
        "message": "Background subtraction is now active. It will adapt to the scene over the next ~100 seconds."
    }

@app.post("/background-subtraction/disable")
def disable_background_subtraction():
    """Disable background subtraction"""
    global background_subtraction_enabled
    background_subtraction_enabled = False
    logger.info("Background subtraction DISABLED")

    return {
        "status": "disabled",
        "message": "Background subtraction is now inactive"
    }

@app.post("/background-subtraction/reset")
def reset_background_subtraction():
    """Reset the background model (useful if scene changes significantly)"""
    global background_subtractor

    if background_subtractor is not None:
        init_background_subtractor()
        logger.info("Background model RESET")
        return {
            "status": "reset",
            "message": "Background model has been reset. It will re-learn the scene over the next ~100 seconds."
        }
    else:
        return {
            "status": "not_initialized",
            "message": "Background subtractor is not initialized"
        }

@app.get("/background-subtraction/status")
def get_background_subtraction_status():
    """Get background subtraction status"""
    return {
        "enabled": background_subtraction_enabled,
        "initialized": background_subtractor is not None
    }

@app.get("/performance")
def get_performance_stats():
    """Get preprocessing performance statistics"""
    return {
        "last_processing_time_ms": round(performance_stats["last_processing_time_ms"], 2),
        "avg_processing_time_ms": round(performance_stats["avg_processing_time_ms"], 2),
        "total_frames_processed": performance_stats["total_frames"]
    }

@app.post("/preprocess")
async def preprocess(config: ProcessingConfig = ProcessingConfig()):
    """Preprocess the latest frame (Non-blocking)"""
    latest_frame = INPUT_DIR / 'latest.jpg'

    if not latest_frame.exists():
        raise HTTPException(status_code=404, detail="No frame available")

    loop = asyncio.get_running_loop()

    # Offload reading and processing to thread pool
    try:
        image = await loop.run_in_executor(thread_pool, cv2.imread, str(latest_frame))
        if image is None:
             raise HTTPException(status_code=500, detail="Failed to read image")
        
        processed = await loop.run_in_executor(
            thread_pool, 
            partial(preprocess_image_sync, image, config)
        )
        
        output_path = INPUT_DIR / 'processed_latest.jpg'
        await loop.run_in_executor(
            thread_pool,
            partial(atomic_write_image_sync, processed, output_path)
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "original_shape": image.shape,
        "processed_shape": processed.shape,
        "output_path": str(output_path)
    }

# --- Background Task (Using the optimized settings) ---

@app.get("/image/latest")
def get_latest_image():
    latest_frame = INPUT_DIR / 'latest.jpg'
    if not latest_frame.exists():
        raise HTTPException(status_code=404, detail="No frame available")
    return FileResponse(latest_frame, media_type="image/jpeg")

@app.get("/image/processed")
def get_processed_image():
    processed_frame = INPUT_DIR / 'processed_latest.jpg'
    if not processed_frame.exists():
        raise HTTPException(status_code=404, detail="No processed frame available")
    return FileResponse(processed_frame, media_type="image/jpeg")

async def auto_preprocess_task():
    """Background task with true FPS targeting and non-blocking execution"""
    logger.info(f"Starting automatic preprocessing task at {PROCESSING_FPS} FPS...")

    target_interval = 1.0 / PROCESSING_FPS
    loop = asyncio.get_running_loop()

    while True:
        start_time = time.time()

        # Default config optimized for detection stability and resource efficiency
        # Sharpness and Denoising are disabled here for maximum speed and least aggression
        # on low-quality input. Only essential adaptive contrast remains.
        # Background subtraction uses global flag
        default_config = ProcessingConfig(
            crop=True,
            adaptive_enhancement=True,
            denoising_level=0,      # DISABLED: Preserve features in low quality images.
            sharpness_amount=0.0,   # DISABLED: Prevent over-sharpening noise.
            background_subtraction=background_subtraction_enabled  # Use global flag
        )

        try:
            latest_frame = INPUT_DIR / 'latest.jpg'

            if latest_frame.exists():
                # Start timing
                process_start = time.time()

                # Read image
                image = await loop.run_in_executor(thread_pool, cv2.imread, str(latest_frame))

                if image is not None:
                    # Process image
                    processed = await loop.run_in_executor(
                        thread_pool,
                        partial(preprocess_image_sync, image, default_config)
                    )

                    # Save image atomically
                    output_path = INPUT_DIR / 'processed_latest.jpg'
                    await loop.run_in_executor(
                        thread_pool,
                        partial(atomic_write_image_sync, processed, output_path)
                    )

                    # Update performance stats
                    process_time_ms = (time.time() - process_start) * 1000
                    performance_stats["last_processing_time_ms"] = process_time_ms
                    performance_stats["total_frames"] += 1

                    # Calculate rolling average (last 100 frames)
                    alpha = 0.01  # Weight for exponential moving average
                    if performance_stats["total_frames"] == 1:
                        performance_stats["avg_processing_time_ms"] = process_time_ms
                    else:
                        performance_stats["avg_processing_time_ms"] = (
                            alpha * process_time_ms +
                            (1 - alpha) * performance_stats["avg_processing_time_ms"]
                        )

        except Exception as e:
            logger.error(f"Error in auto-preprocessing: {e}")

        # Calculate how long processing took and subtract from sleep
        elapsed = time.time() - start_time
        sleep_time = max(0.01, target_interval - elapsed)
        
        await asyncio.sleep(sleep_time)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_preprocess_task())

@app.on_event("shutdown")
def shutdown_event():
    thread_pool.shutdown()