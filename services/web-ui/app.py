"""
Streamlit Web UI for Flowium Traffic Monitoring System
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os

# Configuration
DATA_MANAGER_URL = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')
YOLO_DETECTOR_URL = os.getenv('YOLO_DETECTOR_URL', 'http://yolo-detector:8000')
ONLINE_LEARNER_URL = os.getenv('ONLINE_LEARNER_URL', 'http://online-learner:8000')
WEATHER_SERVICE_URL = os.getenv('WEATHER_SERVICE_URL', 'http://weather-service:8000')

# Page config
st.set_page_config(
    page_title="Flowium - Traffic Monitoring",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Flowium Traffic Monitoring System")
st.markdown("Real-time vehicle detection and traffic prediction using YOLOv8")

# Sidebar
st.sidebar.header("System Status")

def check_service_health(name, url):
    """Check if service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.ok
    except:
        return False

# Check all services
services = {
    "Data Manager": DATA_MANAGER_URL,
    "YOLO Detector": YOLO_DETECTOR_URL,
    "Online Learner": ONLINE_LEARNER_URL,
    "Weather Service": WEATHER_SERVICE_URL
}

for name, url in services.items():
    status = check_service_health(name, url)
    st.sidebar.markdown(f"{'✅' if status else '❌'} {name}")

st.sidebar.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Live Detection", "📈 Analytics", "⚙️ Settings"])

with tab1:
    st.header("System Dashboard")

    col1, col2, col3 = st.columns(3)

    # Fetch stats from data manager
    try:
        health_response = requests.get(f"{DATA_MANAGER_URL}/health", timeout=5)
        if health_response.ok:
            health_data = health_response.json()

            with col1:
                st.metric("Total Detections", health_data.get('total_detections', 0))

            with col2:
                st.metric("Weather Records", health_data.get('total_weather_records', 0))

            with col3:
                # Get latest weather
                weather_response = requests.get(f"{DATA_MANAGER_URL}/weather/latest", timeout=5)
                if weather_response.ok:
                    weather = weather_response.json()
                    if weather.get('status') != 'no_data':
                        st.metric("Temperature", f"{weather.get('temperature', 0):.1f}°C")
                    else:
                        st.metric("Temperature", "N/A")
                else:
                    st.metric("Temperature", "N/A")
    except Exception as e:
        st.error(f"Failed to fetch dashboard data: {e}")

    st.markdown("---")

    # Hourly traffic chart
    st.subheader("Hourly Traffic Statistics")

    try:
        stats_response = requests.get(f"{DATA_MANAGER_URL}/stats/hourly", timeout=5)
        if stats_response.ok:
            stats_data = stats_response.json()
            stats = stats_data.get('stats', [])

            if stats:
                df = pd.DataFrame(stats)
                fig = px.bar(df, x='hour', y='count',
                           labels={'hour': 'Hour of Day', 'count': 'Vehicle Count'},
                           title='Vehicle Detections by Hour')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No traffic data available yet")
    except Exception as e:
        st.error(f"Failed to fetch traffic stats: {e}")

with tab2:
    st.header("Live Vehicle Detection")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Detection Controls")

        if st.button("Run Single Detection"):
            with st.spinner("Running detection..."):
                try:
                    response = requests.post(f"{YOLO_DETECTOR_URL}/detect", timeout=10)
                    if response.ok:
                        result = response.json()
                        st.success(f"Detected {result.get('detections_count', 0)} vehicles")

                        if result.get('detections'):
                            df = pd.DataFrame(result['detections'])
                            st.dataframe(df[['class_name', 'confidence', 'timestamp']])
                    else:
                        st.error("Detection failed")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("Recent Detections")

        try:
            detections_response = requests.get(f"{DATA_MANAGER_URL}/detections?limit=10", timeout=5)
            if detections_response.ok:
                detections_data = detections_response.json()
                detections = detections_data.get('detections', [])

                if detections:
                    for det in detections[:5]:
                        st.text(f"{det['class_name']}: {det['confidence']:.2f}")
                else:
                    st.info("No detections yet")
        except Exception as e:
            st.error(f"Failed to fetch detections: {e}")

with tab3:
    st.header("Traffic Analytics")

    st.subheader("Traffic Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pred_hour = st.slider("Hour", 0, 23, datetime.now().hour)
        pred_day = st.selectbox("Day of Week",
                               ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                               index=datetime.now().weekday())

    with col2:
        pred_temp = st.number_input("Temperature (°C)", -20.0, 40.0, 20.0)
        pred_humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)

    if st.button("Predict Traffic"):
        day_mapping = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_num = day_mapping.index(pred_day)

        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    f"{ONLINE_LEARNER_URL}/predict",
                    json={
                        "hour": pred_hour,
                        "day_of_week": day_num,
                        "temperature": pred_temp,
                        "humidity": pred_humidity
                    },
                    timeout=5
                )

                if response.ok:
                    result = response.json()
                    prediction = result.get('prediction', 0)
                    mae = result.get('mae', 0)

                    st.success(f"Predicted vehicle count: **{prediction:.1f}**")
                    st.info(f"Model MAE: {mae:.2f}")
                else:
                    st.error("Prediction failed")
            except Exception as e:
                st.error(f"Error: {e}")

with tab4:
    st.header("System Settings")

    st.subheader("Background Tasks")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Vehicle Detection**")
        if st.button("Start Continuous Detection"):
            try:
                response = requests.post(f"{YOLO_DETECTOR_URL}/start-continuous", timeout=5)
                if response.ok:
                    st.success("Detection started")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Stop Continuous Detection"):
            try:
                response = requests.post(f"{YOLO_DETECTOR_URL}/stop-continuous", timeout=5)
                if response.ok:
                    st.success("Detection stopped")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.markdown("**Weather Monitoring**")
        if st.button("Start Weather Monitoring"):
            try:
                response = requests.post(f"{WEATHER_SERVICE_URL}/start-continuous", timeout=5)
                if response.ok:
                    st.success("Weather monitoring started")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Stop Weather Monitoring"):
            try:
                response = requests.post(f"{WEATHER_SERVICE_URL}/stop-continuous", timeout=5)
                if response.ok:
                    st.success("Weather monitoring stopped")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    st.subheader("Manual Training")
    if st.button("Train Model with Current Data"):
        with st.spinner("Training..."):
            try:
                response = requests.post(f"{ONLINE_LEARNER_URL}/auto-train", timeout=10)
                if response.ok:
                    result = response.json()
                    st.success(f"Trained on {result.get('trained_samples', 0)} samples")
            except Exception as e:
                st.error(f"Error: {e}")

# Auto-refresh option
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")

if auto_refresh:
    time.sleep(30)
    st.rerun()
