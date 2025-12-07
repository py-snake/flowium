"""
Export YOLOv8n model to ONNX format for faster CPU inference
"""
from ultralytics import YOLO

print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

print("Exporting to ONNX format (optimized for CPU)...")
# Export with optimizations:
# - format='onnx': ONNX format
# - imgsz=416: Balanced size
# - simplify=True: Simplify the ONNX graph
# - opset=12: ONNX opset version
# - dynamic=False: Fixed input size for better optimization
model.export(
    format='onnx',
    imgsz=416,
    simplify=True,
    opset=12,
    dynamic=False
)

print("Model exported successfully to: yolov8n.onnx")
print("This ONNX model is optimized for:")
print("  - CPU inference (2-3x faster than .pt)")
print("  - Image size: 416x416 (balanced speed + accuracy)")
print("  - Fixed input dimensions")
