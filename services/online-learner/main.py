"""
Online Learner Service
Uses River library for online/incremental learning to predict traffic patterns
"""
import os
import pickle
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import deque

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from river import linear_model, preprocessing, compose, metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Training history and statistics
total_samples_trained = 0
training_history = deque(maxlen=100)  # Keep last 100 training samples
prediction_history = deque(maxlen=50)  # Keep last 50 predictions

# Load existing model if available
if MODEL_PATH.exists():
    try:
        with open(MODEL_PATH, 'rb') as f:
            saved_state = pickle.load(f)
            model = saved_state['model']
            mae = saved_state['mae']
            total_samples_trained = saved_state.get('total_samples', 0)
            training_history = deque(saved_state.get('training_history', []), maxlen=100)
            prediction_history = deque(saved_state.get('prediction_history', []), maxlen=50)
        logger.info(f"Loaded existing model from {MODEL_PATH} with {total_samples_trained} samples")
    except Exception as e:
        logger.error(f"Could not load model: {e}")

def save_model():
    """Save model to disk"""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'mae': mae,
            'total_samples': total_samples_trained,
            'training_history': list(training_history),
            'prediction_history': list(prediction_history)
        }, f)

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
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0,
        "total_samples_trained": total_samples_trained
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_trained": total_samples_trained > 0,
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0,
        "total_samples": total_samples_trained
    }

@app.post("/train")
def train(data: TrainingData):
    """Train model with new data point (online learning)"""
    global total_samples_trained

    features = extract_features(data.dict())
    target = float(data.vehicle_count)

    # Make prediction first (for metric calculation)
    y_pred = None
    try:
        y_pred = model.predict_one(features)
        mae.update(target, y_pred)
    except:
        y_pred = 0.0  # First prediction might fail

    # Learn from this sample
    model.learn_one(features, target)
    total_samples_trained += 1

    # Store in training history
    training_history.append({
        'timestamp': datetime.now().isoformat(),
        'hour': data.hour,
        'day_of_week': data.day_of_week,
        'actual': target,
        'predicted': float(y_pred) if y_pred is not None else None,
        'error': abs(target - y_pred) if y_pred is not None else None,
        'temperature': data.temperature,
        'mae': float(mae.get()) if hasattr(mae, 'get') else 0.0
    })

    # Save model every 10 samples
    if total_samples_trained % 10 == 0:
        save_model()

    return {
        "status": "success",
        "features": features,
        "actual": target,
        "predicted": float(y_pred) if y_pred is not None else None,
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0,
        "total_samples": total_samples_trained
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
        "model_exists": MODEL_PATH.exists(),
        "total_samples_trained": total_samples_trained,
        "training_history_size": len(training_history),
        "prediction_history_size": len(prediction_history)
    }

@app.get("/training/history")
def get_training_history():
    """Get recent training history"""
    return {
        "history": list(training_history),
        "total_samples": total_samples_trained,
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0
    }

@app.get("/predictions/next-hours")
async def predict_next_hours(hours: int = 24):
    """Predict traffic for next N hours"""

    # Get current weather or use defaults
    try:
        weather_response = requests.get(f"{DATA_MANAGER_URL}/weather/latest", timeout=2)
        weather = weather_response.json() if weather_response.ok else {}
    except:
        weather = {}

    temperature = weather.get('temperature', 20.0)
    humidity = weather.get('humidity', 50.0)
    precipitation = weather.get('precipitation', 0.0)

    now = datetime.now()
    predictions = []

    for i in range(hours):
        future_time = now + timedelta(hours=i)

        prediction_data = PredictionRequest(
            hour=future_time.hour,
            day_of_week=future_time.weekday(),
            temperature=temperature,
            humidity=humidity,
            precipitation=precipitation
        )

        features = extract_features(prediction_data.dict())

        try:
            prediction = model.predict_one(features)
        except:
            prediction = 0.0

        predictions.append({
            'hour': future_time.hour,
            'day': future_time.strftime('%Y-%m-%d'),
            'day_of_week': future_time.weekday(),
            'day_name': future_time.strftime('%A'),
            'time': future_time.strftime('%H:%M'),
            'predicted_vehicles': max(0, float(prediction)),
            'temperature': temperature,
            'humidity': humidity
        })

    return {
        "predictions": predictions,
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0,
        "total_samples_trained": total_samples_trained
    }

@app.get("/stats/performance")
def get_performance_stats():
    """Get model performance statistics"""

    if len(training_history) == 0:
        return {
            "status": "no_data",
            "message": "No training data available yet"
        }

    # Calculate statistics from training history
    recent_errors = [h['error'] for h in training_history if h.get('error') is not None]
    recent_actuals = [h['actual'] for h in training_history]
    recent_predictions = [h['predicted'] for h in training_history if h.get('predicted') is not None]

    avg_error = sum(recent_errors) / len(recent_errors) if recent_errors else 0
    avg_actual = sum(recent_actuals) / len(recent_actuals) if recent_actuals else 0
    avg_predicted = sum(recent_predictions) / len(recent_predictions) if recent_predictions else 0

    return {
        "mae": float(mae.get()) if hasattr(mae, 'get') else 0.0,
        "average_error": avg_error,
        "average_actual_traffic": avg_actual,
        "average_predicted_traffic": avg_predicted,
        "total_samples_trained": total_samples_trained,
        "recent_samples_count": len(training_history),
        "model_ready": total_samples_trained > 10
    }

async def background_training_task():
    """Background task to periodically train on new data from database"""
    logger.info("Starting background training task (runs every 10 minutes)")

    # Initial delay to let system start up
    await asyncio.sleep(60)

    # On startup, train on existing historical data
    logger.info("Performing initial training on historical data...")
    try:
        response = requests.get(f"{DATA_MANAGER_URL}/training/hourly-with-weather?hours=168", timeout=30)
        if response.ok:
            training_response = response.json()
            data_points = training_response.get('data', [])

            trained_count = 0
            for data_point in data_points:
                if data_point.get('vehicle_count', 0) > 0:
                    training_data = TrainingData(
                        hour=data_point['hour'],
                        day_of_week=data_point['day_of_week'],
                        temperature=data_point.get('temperature', 20.0),
                        humidity=data_point.get('humidity', 50.0),
                        precipitation=data_point.get('precipitation', 0.0),
                        vehicle_count=data_point['vehicle_count']
                    )
                    train(training_data)
                    trained_count += 1

            logger.info(f"Initial training completed on {trained_count} historical samples")
    except Exception as e:
        logger.error(f"Error in initial training: {e}")

    # Periodic training loop
    while True:
        try:
            await asyncio.sleep(600)  # 10 minutes

            # Fetch recent hourly stats with weather from database
            try:
                response = requests.get(f"{DATA_MANAGER_URL}/training/hourly-with-weather?hours=24", timeout=10)
                if response.ok:
                    training_response = response.json()
                    data_points = training_response.get('data', [])

                    trained_count = 0
                    for data_point in data_points:
                        if data_point.get('vehicle_count', 0) > 0:
                            training_data = TrainingData(
                                hour=data_point['hour'],
                                day_of_week=data_point['day_of_week'],
                                temperature=data_point.get('temperature', 20.0),
                                humidity=data_point.get('humidity', 50.0),
                                precipitation=data_point.get('precipitation', 0.0),
                                vehicle_count=data_point['vehicle_count']
                            )
                            train(training_data)
                            trained_count += 1

                    if trained_count > 0:
                        logger.info(f"Background training: {trained_count} samples. Total: {total_samples_trained}")
                else:
                    logger.warning(f"Failed to fetch training data: {response.status_code}")
            except Exception as e:
                logger.error(f"Error in background training: {e}")

        except Exception as e:
            logger.error(f"Error in background task: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(background_training_task())
