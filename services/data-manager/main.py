"""
Data Manager Service
Manages SQLite database for storing detections, weather data, and traffic statistics
"""
import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

app = FastAPI(title="Data Manager Service")

# Configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', '/data/flowium.db')
Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)

# Database setup
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    class_id = Column(Integer)
    class_name = Column(String)
    confidence = Column(Float)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)

class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    temperature = Column(Float)
    humidity = Column(Float)
    weather_condition = Column(String)
    precipitation = Column(Float)
    wind_speed = Column(Float)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class DetectionCreate(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    timestamp: Optional[str] = None

class DetectionBatch(BaseModel):
    detections: List[DetectionCreate]

class WeatherCreate(BaseModel):
    temperature: float
    humidity: float
    weather_condition: str
    precipitation: float = 0.0
    wind_speed: float = 0.0

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {
        "service": "Data Manager Service",
        "status": "running",
        "database": DATABASE_PATH
    }

@app.get("/health")
def health():
    db = SessionLocal()
    detection_count = db.query(Detection).count()
    weather_count = db.query(WeatherData).count()
    db.close()

    return {
        "status": "healthy",
        "total_detections": detection_count,
        "total_weather_records": weather_count
    }

@app.post("/detections")
def create_detections(batch: DetectionBatch):
    """Store vehicle detections"""
    db = SessionLocal()

    try:
        for det in batch.detections:
            db_detection = Detection(
                timestamp=datetime.fromisoformat(det.timestamp) if det.timestamp else datetime.utcnow(),
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox_x1=det.bbox[0],
                bbox_y1=det.bbox[1],
                bbox_x2=det.bbox[2],
                bbox_y2=det.bbox[3]
            )
            db.add(db_detection)

        db.commit()
        return {"status": "success", "count": len(batch.detections)}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/detections")
def get_detections(limit: int = 100, offset: int = 0):
    """Get recent detections"""
    db = SessionLocal()

    detections = db.query(Detection)\
        .order_by(Detection.timestamp.desc())\
        .limit(limit)\
        .offset(offset)\
        .all()

    db.close()

    return {
        "count": len(detections),
        "detections": [
            {
                "id": d.id,
                "timestamp": d.timestamp.isoformat(),
                "class_name": d.class_name,
                "confidence": d.confidence
            }
            for d in detections
        ]
    }

@app.post("/weather")
def create_weather(weather: WeatherCreate):
    """Store weather data"""
    db = SessionLocal()

    try:
        db_weather = WeatherData(
            temperature=weather.temperature,
            humidity=weather.humidity,
            weather_condition=weather.weather_condition,
            precipitation=weather.precipitation,
            wind_speed=weather.wind_speed
        )
        db.add(db_weather)
        db.commit()

        return {"status": "success"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/weather/latest")
def get_latest_weather():
    """Get most recent weather data"""
    db = SessionLocal()

    weather = db.query(WeatherData)\
        .order_by(WeatherData.timestamp.desc())\
        .first()

    db.close()

    if not weather:
        return {"status": "no_data"}

    return {
        "timestamp": weather.timestamp.isoformat(),
        "temperature": weather.temperature,
        "humidity": weather.humidity,
        "weather_condition": weather.weather_condition,
        "precipitation": weather.precipitation,
        "wind_speed": weather.wind_speed
    }

@app.get("/stats/current")
def get_current_stats():
    """Get current traffic statistics (last 5 minutes)"""
    from sqlalchemy import func
    from datetime import timedelta

    db = SessionLocal()

    # Get detections from last 5 minutes
    five_min_ago = datetime.utcnow() - timedelta(minutes=5)

    recent_detections = db.query(Detection)\
        .filter(Detection.timestamp >= five_min_ago)\
        .all()

    # Count by vehicle type
    vehicle_counts = {}
    for det in recent_detections:
        vehicle_counts[det.class_name] = vehicle_counts.get(det.class_name, 0) + 1

    db.close()

    return {
        "total_vehicles": len(recent_detections),
        "vehicle_types": vehicle_counts,
        "timeframe": "last_5_minutes"
    }

@app.get("/stats/hourly")
def get_hourly_stats(hours: int = 24):
    """Get traffic statistics by hour for the last N hours"""
    from sqlalchemy import func
    from datetime import timedelta

    db = SessionLocal()

    # Get detections from last N hours
    start_time = datetime.utcnow() - timedelta(hours=hours)

    # Group by hour and count
    stats = db.query(
        func.strftime('%Y-%m-%d %H:00:00', Detection.timestamp).label('hour'),
        func.count(Detection.id).label('total_count'),
        Detection.class_name,
        func.count(Detection.id).label('count')
    ).filter(Detection.timestamp >= start_time)\
     .group_by('hour', Detection.class_name)\
     .order_by('hour')\
     .all()

    db.close()

    # Organize data by hour with vehicle type breakdown
    hourly_data = {}
    for stat in stats:
        hour = stat.hour
        if hour not in hourly_data:
            hourly_data[hour] = {"timestamp": hour, "total": 0, "by_type": {}}

        hourly_data[hour]["by_type"][stat.class_name] = stat.count
        hourly_data[hour]["total"] += stat.count

    return {
        "hours": hours,
        "data": list(hourly_data.values())
    }

@app.get("/stats/vehicle_types")
def get_vehicle_type_stats(hours: int = 24):
    """Get vehicle type distribution for the last N hours"""
    from sqlalchemy import func
    from datetime import timedelta

    db = SessionLocal()

    start_time = datetime.utcnow() - timedelta(hours=hours)

    stats = db.query(
        Detection.class_name,
        func.count(Detection.id).label('count')
    ).filter(Detection.timestamp >= start_time)\
     .group_by(Detection.class_name)\
     .order_by(func.count(Detection.id).desc())\
     .all()

    db.close()

    total = sum(s.count for s in stats)

    return {
        "hours": hours,
        "total_vehicles": total,
        "breakdown": [
            {
                "vehicle_type": s.class_name,
                "count": s.count,
                "percentage": round((s.count / total * 100), 1) if total > 0 else 0
            }
            for s in stats
        ]
    }

@app.get("/stats/timeline")
def get_timeline_stats(hours: int = 24, interval_minutes: int = 60):
    """Get traffic timeline with configurable interval"""
    from sqlalchemy import func
    from datetime import timedelta

    db = SessionLocal()

    start_time = datetime.utcnow() - timedelta(hours=hours)

    # Create time buckets based on interval
    if interval_minutes == 60:
        time_format = '%Y-%m-%d %H:00:00'
    elif interval_minutes == 30:
        # For 30-minute intervals, we'll post-process
        time_format = '%Y-%m-%d %H:%M:00'
    else:
        time_format = '%Y-%m-%d %H:%M:00'

    stats = db.query(
        func.strftime(time_format, Detection.timestamp).label('time_bucket'),
        func.count(Detection.id).label('count')
    ).filter(Detection.timestamp >= start_time)\
     .group_by('time_bucket')\
     .order_by('time_bucket')\
     .all()

    db.close()

    return {
        "hours": hours,
        "interval_minutes": interval_minutes,
        "data": [
            {"timestamp": s.time_bucket, "count": s.count}
            for s in stats
        ]
    }

@app.get("/stats/advanced")
def get_advanced_stats(hours: int = 24):
    """Get advanced statistics for dashboard"""
    from sqlalchemy import func
    from datetime import timedelta

    db = SessionLocal()
    start_time = datetime.utcnow() - timedelta(hours=hours)

    # Total detections
    total = db.query(func.count(Detection.id))\
        .filter(Detection.timestamp >= start_time)\
        .scalar()

    # Average confidence
    avg_confidence = db.query(func.avg(Detection.confidence))\
        .filter(Detection.timestamp >= start_time)\
        .scalar()

    # Low confidence detections (< 0.6)
    low_confidence = db.query(func.count(Detection.id))\
        .filter(Detection.timestamp >= start_time)\
        .filter(Detection.confidence < 0.6)\
        .scalar()

    # Busiest hour
    busiest = db.query(
        func.strftime('%Y-%m-%d %H:00:00', Detection.timestamp).label('hour'),
        func.count(Detection.id).label('count')
    ).filter(Detection.timestamp >= start_time)\
     .group_by('hour')\
     .order_by(func.count(Detection.id).desc())\
     .first()

    # Slowest hour
    slowest = db.query(
        func.strftime('%Y-%m-%d %H:00:00', Detection.timestamp).label('hour'),
        func.count(Detection.id).label('count')
    ).filter(Detection.timestamp >= start_time)\
     .group_by('hour')\
     .order_by(func.count(Detection.id).asc())\
     .first()

    db.close()

    return {
        "timeframe_hours": hours,
        "total_detections": total or 0,
        "average_confidence": round(avg_confidence or 0, 3),
        "low_confidence_count": low_confidence or 0,
        "low_confidence_percentage": round((low_confidence / total * 100), 1) if total > 0 else 0,
        "busiest_hour": {
            "timestamp": busiest.hour if busiest else None,
            "count": busiest.count if busiest else 0
        },
        "slowest_hour": {
            "timestamp": slowest.hour if slowest else None,
            "count": slowest.count if slowest else 0
        }
    }

@app.delete("/detections/all")
def clear_all_detections():
    """Clear all detections from database"""
    db = SessionLocal()

    try:
        count = db.query(Detection).count()
        db.query(Detection).delete()
        db.commit()

        return {
            "status": "success",
            "deleted_count": count,
            "message": f"Deleted {count} detections"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/detections/timerange")
def clear_detections_by_time(hours: int = 1):
    """Clear detections from the last N hours"""
    from datetime import timedelta

    db = SessionLocal()

    try:
        start_time = datetime.utcnow() - timedelta(hours=hours)
        count = db.query(Detection)\
            .filter(Detection.timestamp >= start_time)\
            .count()

        db.query(Detection)\
            .filter(Detection.timestamp >= start_time)\
            .delete()

        db.commit()

        return {
            "status": "success",
            "deleted_count": count,
            "timeframe_hours": hours,
            "message": f"Deleted {count} detections from last {hours} hours"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/detections/low_confidence")
def clear_low_confidence_detections(confidence_threshold: float = 0.6):
    """Clear detections below confidence threshold"""
    db = SessionLocal()

    try:
        count = db.query(Detection)\
            .filter(Detection.confidence < confidence_threshold)\
            .count()

        db.query(Detection)\
            .filter(Detection.confidence < confidence_threshold)\
            .delete()

        db.commit()

        return {
            "status": "success",
            "deleted_count": count,
            "threshold": confidence_threshold,
            "message": f"Deleted {count} low-confidence detections (< {confidence_threshold})"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/export/csv")
def export_detections_csv(hours: int = 24, limit: int = 10000):
    """Export detections as CSV"""
    from datetime import timedelta
    import csv
    from io import StringIO

    db = SessionLocal()
    start_time = datetime.utcnow() - timedelta(hours=hours)

    detections = db.query(Detection)\
        .filter(Detection.timestamp >= start_time)\
        .order_by(Detection.timestamp.desc())\
        .limit(limit)\
        .all()

    db.close()

    # Create CSV
    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(['timestamp', 'class_name', 'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])

    # Data
    for d in detections:
        writer.writerow([
            d.timestamp.isoformat(),
            d.class_name,
            d.confidence,
            d.bbox_x1,
            d.bbox_y1,
            d.bbox_x2,
            d.bbox_y2
        ])

    from fastapi.responses import Response
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=flowium_detections_{hours}h.csv"}
    )
