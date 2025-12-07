"""
Weather Service
Fetches weather data from Open-Meteo API (free, no API key required) and stores in Data Manager.
"""
import os
import asyncio
import logging
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weather Service")

LOCATION = os.getenv('LOCATION', 'Baja,HU')
DATA_MANAGER_URL = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')
FETCH_INTERVAL = int(os.getenv('WEATHER_FETCH_INTERVAL', '600'))

LATITUDE = float(os.getenv('LATITUDE', '46.18'))
LONGITUDE = float(os.getenv('LONGITUDE', '18.95'))

http_client = None
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

class WeatherData(BaseModel):
    temperature: float
    humidity: float
    weather_condition: str
    precipitation: float
    wind_speed: float

async def fetch_weather() -> WeatherData:
    """Fetch current weather from Open-Meteo API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'current': 'temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m',
        'timezone': 'Europe/Budapest'
    }

    try:
        response = await http_client.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data['current']
        weather_code = current['weather_code']
        weather_condition = WEATHER_CODES.get(weather_code, f"Unknown ({weather_code})")

        weather = WeatherData(
            temperature=current['temperature_2m'],
            humidity=current['relative_humidity_2m'],
            weather_condition=weather_condition,
            precipitation=current['precipitation'],
            wind_speed=current['wind_speed_10m']
        )

        return weather

    except httpx.RequestError as e:
        logger.error(f"Failed to fetch weather: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather: {e}")

async def send_to_data_manager(weather: WeatherData) -> bool:
    """Send weather data to Data Manager"""
    try:
        response = await http_client.post(
            f"{DATA_MANAGER_URL}/weather",
            json=weather.model_dump(),
            timeout=5
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to send weather to data manager: {e}")
        return False

@app.get("/")
def root():
    return {
        "service": "Weather Service",
        "status": "running",
        "location": LOCATION,
        "coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "api": "Open-Meteo (free, no API key required)",
        "fetch_interval_seconds": FETCH_INTERVAL
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "location": LOCATION,
        "coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE}
    }

@app.get("/weather/current")
async def get_current_weather():
    """Get current weather"""
    weather = await fetch_weather()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "location": LOCATION,
        "coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "weather": weather.model_dump()
    }

@app.post("/weather/fetch-and-store")
async def fetch_and_store():
    """Fetch weather and store in database"""
    weather = await fetch_weather()
    success = await send_to_data_manager(weather)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to store weather data")

    return {
        "status": "success",
        "weather": weather.model_dump()
    }

async def auto_weather_task():
    """Fetch weather automatically at configured interval"""
    logger.info(f"Starting automatic weather fetch task (interval: {FETCH_INTERVAL}s)...")

    while True:
        try:
            weather = await fetch_weather()
            success = await send_to_data_manager(weather)

            if success:
                logger.info(
                    f"Weather: {weather.temperature}°C, {weather.humidity}% humidity, "
                    f"{weather.weather_condition}, {weather.wind_speed} km/h wind"
                )
            else:
                logger.error("Failed to store weather data")

        except Exception as e:
            logger.error(f"Error in auto-weather task: {e}")

        await asyncio.sleep(FETCH_INTERVAL)

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    global http_client
    http_client = httpx.AsyncClient()
    asyncio.create_task(auto_weather_task())

@app.on_event("shutdown")
async def shutdown_event():
    """Close HTTP client"""
    if http_client:
        await http_client.close()
