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
st.title("Flowium Traffic Monitoring System")
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
    st.sidebar.markdown(f"{'Online' if status else 'Offline'} - {name}")

st.sidebar.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Live Detection", "Analytics", "Predictions", "Settings"])

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
    st.header("Traffic Predictions & Machine Learning")

    # Check if online learner is available
    try:
        health_response = requests.get(f"{ONLINE_LEARNER_URL}/health", timeout=2)
        if not health_response.ok:
            st.error("Online Learner service is not available")
        else:
            health_data = health_response.json()

            # Model Performance Metrics
            st.subheader("Model Performance")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Samples Trained", health_data.get('total_samples', 0))

            with col2:
                mae = health_data.get('mae', 0)
                st.metric("Mean Absolute Error", f"{mae:.2f}")

            with col3:
                model_trained = health_data.get('model_trained', False)
                st.metric("Model Status", "Ready" if model_trained else "Training")

            with col4:
                # Get performance stats
                try:
                    perf_response = requests.get(f"{ONLINE_LEARNER_URL}/stats/performance", timeout=5)
                    if perf_response.ok:
                        perf_data = perf_response.json()
                        if perf_data.get('status') != 'no_data':
                            avg_error = perf_data.get('average_error', 0)
                            st.metric("Avg Error", f"{avg_error:.1f} vehicles")
                        else:
                            st.metric("Avg Error", "N/A")
                except:
                    st.metric("Avg Error", "N/A")

            st.markdown("---")

            # Traffic Predictions for Next 24 Hours
            st.subheader("Traffic Forecast - Next 24 Hours")

            try:
                predictions_response = requests.get(f"{ONLINE_LEARNER_URL}/predictions/next-hours?hours=24", timeout=10)
                if predictions_response.ok:
                    predictions_data = predictions_response.json()
                    predictions = predictions_data.get('predictions', [])

                    if predictions:
                        # Create chart data
                        chart_data = pd.DataFrame(predictions)

                        # Line chart for predictions
                        fig = px.line(
                            chart_data,
                            x='time',
                            y='predicted_vehicles',
                            title='Predicted Traffic Volume (Next 24 Hours)',
                            labels={'time': 'Time', 'predicted_vehicles': 'Predicted Vehicles'},
                            markers=True
                        )

                        fig.update_layout(
                            xaxis_title="Time",
                            yaxis_title="Predicted Vehicle Count",
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Show predictions table
                        with st.expander("View Detailed Predictions"):
                            display_predictions = chart_data[['time', 'day_name', 'predicted_vehicles', 'temperature']].copy()
                            display_predictions['predicted_vehicles'] = display_predictions['predicted_vehicles'].round(1)
                            display_predictions.columns = ['Time', 'Day', 'Predicted Vehicles', 'Temperature (°C)']
                            st.dataframe(display_predictions, use_container_width=True, hide_index=True)

                    else:
                        st.info("Not enough data to make predictions. The model will improve as more data is collected.")
                else:
                    st.warning("Could not load predictions")
            except Exception as e:
                st.error(f"Error loading predictions: {e}")

            st.markdown("---")

            # Training History
            st.subheader("Training History & Model Performance")

            col1, col2 = st.columns(2)

            with col1:
                # Recent training history
                st.markdown("**Recent Training Samples**")
                try:
                    history_response = requests.get(f"{ONLINE_LEARNER_URL}/training/history", timeout=5)
                    if history_response.ok:
                        history_data = history_response.json()
                        history = history_data.get('history', [])

                        if history:
                            # Show last 20 training samples
                            history_df = pd.DataFrame(history[-20:])

                            if 'predicted' in history_df.columns and 'actual' in history_df.columns:
                                # Chart showing predicted vs actual
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    y=history_df['actual'],
                                    mode='lines+markers',
                                    name='Actual',
                                    line=dict(color='blue')
                                ))

                                fig.add_trace(go.Scatter(
                                    y=history_df['predicted'],
                                    mode='lines+markers',
                                    name='Predicted',
                                    line=dict(color='red', dash='dash')
                                ))

                                fig.update_layout(
                                    title="Predicted vs Actual (Last 20 Samples)",
                                    yaxis_title="Vehicle Count",
                                    xaxis_title="Sample Index",
                                    hovermode='x unified'
                                )

                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Collecting training data...")
                        else:
                            st.info("No training history yet. The model will start learning from traffic data automatically.")
                except Exception as e:
                    st.error(f"Error loading training history: {e}")

            with col2:
                # Performance stats
                st.markdown("**Model Performance Statistics**")
                try:
                    perf_response = requests.get(f"{ONLINE_LEARNER_URL}/stats/performance", timeout=5)
                    if perf_response.ok:
                        perf_data = perf_response.json()

                        if perf_data.get('status') != 'no_data':
                            st.metric("Mean Absolute Error", f"{perf_data.get('mae', 0):.2f}")
                            st.metric("Average Actual Traffic", f"{perf_data.get('average_actual_traffic', 0):.1f}")
                            st.metric("Average Predicted Traffic", f"{perf_data.get('average_predicted_traffic', 0):.1f}")
                            st.metric("Recent Samples", perf_data.get('recent_samples_count', 0))

                            model_ready = perf_data.get('model_ready', False)
                            if model_ready:
                                st.success("Model is ready for accurate predictions")
                            else:
                                st.info("Model is still learning... collecting more data")

                            # Show error distribution
                            try:
                                history_response = requests.get(f"{ONLINE_LEARNER_URL}/training/history", timeout=5)
                                if history_response.ok:
                                    history_data = history_response.json()
                                    history = history_data.get('history', [])

                                    if history and len(history) > 5:
                                        errors = [h.get('error', 0) for h in history if h.get('error') is not None]
                                        if errors:
                                            fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=20)])
                                            fig.update_layout(
                                                title="Prediction Error Distribution",
                                                xaxis_title="Error (vehicles)",
                                                yaxis_title="Frequency",
                                                showlegend=False
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                            except:
                                pass

                        else:
                            st.info("Waiting for training data...")
                except Exception as e:
                    st.error(f"Error loading performance stats: {e}")

            st.markdown("---")

            # Manual Training Control
            st.subheader("Manual Training")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Train on Current Data**")
                if st.button("Train Now"):
                    with st.spinner("Training model on recent data..."):
                        try:
                            response = requests.post(f"{ONLINE_LEARNER_URL}/auto-train", timeout=30)
                            if response.ok:
                                result = response.json()
                                st.success(f"Training complete! Trained on {result.get('trained_samples', 0)} samples")
                            else:
                                st.error("Training failed")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with col2:
                st.markdown("**Model Information**")
                try:
                    info_response = requests.get(f"{ONLINE_LEARNER_URL}/model/info", timeout=5)
                    if info_response.ok:
                        info_data = info_response.json()
                        st.caption(f"Model Type: River Linear Regression")
                        st.caption(f"Total Samples: {info_data.get('total_samples_trained', 0)}")
                        st.caption(f"MAE: {info_data.get('mae', 0):.2f}")
                except:
                    pass

    except Exception as e:
        st.error(f"Error connecting to Online Learner service: {e}")

with tab5:
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
