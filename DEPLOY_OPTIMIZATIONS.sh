#!/bin/bash

echo "Deploying Flowium Performance Optimizations"
echo "=============================================="
echo ""

echo "Optimizations included:"
echo "- ONNX model export (2-3x faster inference)"
echo "- Reduced image size (384px for speed)"
echo "- Adaptive FPS (smart frame skipping)"
echo "- Smart frame dropping (avoid duplicates)"
echo "- CPU threading optimization (use all 8 cores)"
echo "- Enhanced performance monitoring"
echo ""

echo "Building optimized services..."
docker compose build --no-cache yolo-detector preprocessing web-ui

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed! Check the error messages above."
    exit 1
fi

echo ""
echo "Build complete!"
echo ""

echo "Starting services..."
docker compose up -d

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start services!"
    exit 1
fi

echo ""
echo "Services started!"
echo ""

echo "Waiting for services to initialize (10 seconds)..."
sleep 10

echo ""
echo "Checking ONNX model..."
if docker logs flowium-yolo-detector 2>&1 | grep -q "Loading optimized ONNX model"; then
    echo "ONNX model loaded successfully!"
else
    echo "WARNING: ONNX model not detected. Check logs:"
    echo "    docker logs flowium-yolo-detector | grep -i onnx"
fi

echo ""
echo "Deployment Complete!"
echo ""
echo "Expected improvements:"
echo "  - Detection speed: 2-3x faster (300ms → 100-150ms)"
echo "  - Overall FPS: 2.9 → 5-7 FPS"
echo "  - CPU usage: 40-50% → 80-90%"
echo ""
echo "Open Web UI to see performance:"
echo "  http://your-server:8501"
echo ""
echo "Monitor performance:"
echo "  docker stats flowium-yolo-detector flowium-preprocessing"
echo ""
echo "Check logs:"
echo "  docker logs -f flowium-yolo-detector"
echo ""
