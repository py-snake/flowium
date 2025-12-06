"""
Online Learner Service
Uses River library for online/incremental learning to predict traffic patterns
"""
import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from river import linear_model, preprocessing, compose, metrics

app = FastAPI(title="Online Learner Service")

# Configuration
DATA_MANAGER_URL = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')
MODEL_PATH = Path(os.getenv('MODEL_PATH', '/models/river_model.pkl'))

# Initialize River model (online linear regression)
# Predicts traffic volume based on time features and weather
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LinearRegression()
)

# Metric to track model performance
mae = metrics.MAE()

# Load existing model if available
if MODEL_PATH.exists():
    try:
        with open(MODEL_PATH, 'rb') as f:
            saved_state = pickle.load(f)
            model = saved_state['model']
            mae = saved_state['mae']
        print(f"Loaded existing model from {MODEL_PATH}")
    except Exception as e:
        print(f"Could not load model: {e}")

def save_model():
    """Save model to disk"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'mae': mae}, f)

class TrainingData(BaseModel):
    hour: int
    day_of_week: int
    temperature: Optional[float] = 20.0
    humidity: Optional[float] = 50.0
    precipitation: Optional[float] = 0.0
    vehicle_count: int

class PredictionRequest(BaseModel):
    hour: int
    day_of_week: int
    temperature: Optional[float] = 20.0
    humidity: Optional[float] = 50.0
    precipitation: Optional[float] = 0.0

def extract_features(data: Dict) -> Dict:
    """Extract features for model"""
    return {
        'hour': float(data['hour']),
        'day_of_week': float(data['day_of_week']),
        'temperature': float(data.get('temperature', 20.0)),
        'humidity': float(data.get('humidity', 50.0)),
        'precipitation': float(data.get('precipitation', 0.0)),
    }

@app.get("/")
def root():
    return {
        "service": "Online Learner Service",
        "status": "running",
        "model": "River Linear Regression",
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_trained": True,
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0
    }

@app.post("/train")
def train(data: TrainingData):
    """Train model with new data point (online learning)"""

    features = extract_features(data.dict())
    target = float(data.vehicle_count)

    # Make prediction first (for metric calculation)
    try:
        y_pred = model.predict_one(features)
        mae.update(target, y_pred)
    except:
        pass  # First prediction might fail

    # Learn from this sample
    model.learn_one(features, target)

    # Save model periodically
    save_model()

    return {
        "status": "success",
        "features": features,
        "actual": target,
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0
    }

@app.post("/predict")
def predict(data: PredictionRequest):
    """Predict traffic volume"""

    features = extract_features(data.dict())

    try:
        prediction = model.predict_one(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "prediction": float(prediction),
        "features": features,
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0
    }

@app.post("/auto-train")
def auto_train():
    """Automatically fetch data from data-manager and train"""

    try:
        # Get recent detections
        response = requests.get(f"{DATA_MANAGER_URL}/stats/hourly", timeout=5)
        response.raise_for_status()
        stats = response.json()['stats']

        # Get weather data
        weather_response = requests.get(f"{DATA_MANAGER_URL}/weather/latest", timeout=5)
        weather = weather_response.json() if weather_response.ok else {}

        # Train on each hourly stat
        trained_count = 0
        for stat in stats:
            training_data = TrainingData(
                hour=int(stat['hour']),
                day_of_week=datetime.now().weekday(),
                temperature=weather.get('temperature', 20.0),
                humidity=weather.get('humidity', 50.0),
                precipitation=weather.get('precipitation', 0.0),
                vehicle_count=stat['count']
            )

            train(training_data)
            trained_count += 1

        return {
            "status": "success",
            "trained_samples": trained_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
def model_info():
    """Get model information"""
    return {
        "model_type": str(type(model)),
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }
