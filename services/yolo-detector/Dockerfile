FROM python:3.11-slim

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download YOLOv8n model and export to optimized ONNX format (2-3x faster on CPU)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    python export_model.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
