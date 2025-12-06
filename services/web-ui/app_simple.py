"""
Simplified Streamlit UI - Stream Test Version
Shows live video stream from YouTube capture with crop configuration tool
"""
import streamlit as st
import requests
import time
import os
from PIL import Image, ImageDraw
from io import BytesIO
from analytics_tab import render_analytics_tab

# Configuration
PREPROCESSING_URL = os.getenv('PREPROCESSING_URL', 'http://preprocessing:8000')
YOLO_DETECTOR_URL = os.getenv('YOLO_DETECTOR_URL', 'http://yolo-detector:8000')
DATA_MANAGER_URL = os.getenv('DATA_MANAGER_URL', 'http://data-manager:8000')

# Page config
st.set_page_config(
    page_title="Flowium - Stream Test",
    page_icon="📹",
    layout="wide"
)

# Title
st.title("Flowium - Live Stream Test")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Live Stream", "Crop Configuration", "Analytics", "Predictions"])

# ============================================================================
# TAB 1: Live Stream View
# ============================================================================
with tab1:
    st.markdown("Testing YouTube stream capture with yt-dlp")

    # Sidebar
    st.sidebar.header("Stream Controls")

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_fps = st.sidebar.slider("Refresh rate (FPS)", 1, 10, 5, help="Updates per second (higher = smoother but more CPU)")

    st.sidebar.markdown("---")

    # Manual refresh button
    if st.sidebar.button("Refresh Now"):
        st.rerun()

    st.sidebar.markdown("---")

    # Background Subtraction Control
    st.sidebar.header("Background Subtraction")

    try:
        # Get current status
        status_response = requests.get(f"{PREPROCESSING_URL}/background-subtraction/status", timeout=2)
        if status_response.ok:
            status_data = status_response.json()
            current_status = status_data.get('enabled', False)

            # Toggle button
            bg_subtraction = st.sidebar.checkbox(
                "Enable Background Subtraction",
                value=current_status,
                help="Removes static obstacles (trees, poles) to isolate moving vehicles. Adapts to day/night changes."
            )

            # Update status if changed
            if bg_subtraction != current_status:
                if bg_subtraction:
                    enable_response = requests.post(f"{PREPROCESSING_URL}/background-subtraction/enable", timeout=2)
                    if enable_response.ok:
                        st.sidebar.success("Background subtraction enabled! Learning scene...")
                else:
                    disable_response = requests.post(f"{PREPROCESSING_URL}/background-subtraction/disable", timeout=2)
                    if disable_response.ok:
                        st.sidebar.info("Background subtraction disabled")

            # Reset button (only show if enabled)
            if bg_subtraction:
                if st.sidebar.button("Reset Background Model", help="Re-learn the background (useful after scene changes)"):
                    reset_response = requests.post(f"{PREPROCESSING_URL}/background-subtraction/reset", timeout=2)
                    if reset_response.ok:
                        st.sidebar.success("Background model reset!")

                st.sidebar.caption("The model adapts over ~100 seconds. Reset if scene changes significantly.")

    except Exception as e:
        st.sidebar.error(f"Background subtraction unavailable: {e}")

    st.sidebar.markdown("---")

    # Performance Metrics
    st.sidebar.header("Performance")

    try:
        # Get performance data from all services
        preprocess_perf = requests.get(f"{PREPROCESSING_URL}/performance", timeout=2)
        yolo_perf = requests.get(f"{YOLO_DETECTOR_URL}/performance", timeout=2)

        if preprocess_perf.ok and yolo_perf.ok:
            prep_data = preprocess_perf.json()
            yolo_data = yolo_perf.json()

            # Calculate overall time (preprocessing + detection)
            overall_time = prep_data.get('last_processing_time_ms', 0) + yolo_data.get('last_detection_time_ms', 0)

            st.sidebar.metric(
                "Preprocessing",
                f"{prep_data.get('last_processing_time_ms', 0):.1f} ms",
                delta=f"avg: {prep_data.get('avg_processing_time_ms', 0):.1f} ms"
            )

            st.sidebar.metric(
                "YOLO Detection",
                f"{yolo_data.get('last_detection_time_ms', 0):.1f} ms",
                delta=f"avg: {yolo_data.get('avg_detection_time_ms', 0):.1f} ms"
            )

            st.sidebar.metric(
                "Overall",
                f"{overall_time:.1f} ms",
                delta=f"{1000/overall_time:.1f} FPS" if overall_time > 0 else "N/A"
            )

            # Show frame drop rate (adaptive FPS indicator)
            drop_rate = yolo_data.get('drop_rate', 0)
            if drop_rate > 0:
                st.sidebar.metric(
                    "Frame Drop Rate",
                    f"{drop_rate:.1f}%",
                    delta=f"{yolo_data.get('frames_dropped', 0)} dropped",
                    delta_color="inverse"
                )
                if drop_rate > 20:
                    st.sidebar.caption("High drop rate - system is optimizing for speed")
                else:
                    st.sidebar.caption("Adaptive FPS working well")

    except Exception as e:
        st.sidebar.error(f"Performance data unavailable")

    st.sidebar.markdown("---")

    # Main content - 3 columns for Raw, Preprocessed, and Detected views
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Raw Stream Capture")

        # Check if service is available
        try:
            health_response = requests.get(f"{PREPROCESSING_URL}/health", timeout=2)
            if health_response.ok:
                health_data = health_response.json()

                if health_data.get('latest_frame_exists'):
                    # Initialize session state for last good image
                    if 'last_raw_image' not in st.session_state:
                        st.session_state.last_raw_image = None
                        st.session_state.last_raw_time = None

                    # Try to load and display the image with robust error handling
                    try:
                        # Add timestamp to prevent caching
                        cache_buster = int(time.time() * 1000)  # Use milliseconds for high FPS
                        img_response = requests.get(
                            f"{PREPROCESSING_URL}/image/latest?t={cache_buster}",
                            timeout=2,
                            headers={'Cache-Control': 'no-cache'}
                        )
                        if img_response.ok and len(img_response.content) > 1000:  # Ensure we have a complete image
                            try:
                                image = Image.open(BytesIO(img_response.content))
                                # Verify image is valid
                                image.verify()
                                # Reopen for display (verify() closes the file)
                                image = Image.open(BytesIO(img_response.content))

                                # Store as last good image
                                st.session_state.last_raw_image = img_response.content
                                st.session_state.last_raw_time = time.strftime('%H:%M:%S')

                                st.image(image, caption=f"Raw - {st.session_state.last_raw_time}", use_container_width=True)
                                st.info(f"Full Resolution: {image.size[0]} x {image.size[1]}")
                            except (IOError, OSError) as e:
                                # Corrupted JPEG or incomplete read - use last good image
                                if st.session_state.last_raw_image:
                                    image = Image.open(BytesIO(st.session_state.last_raw_image))
                                    st.image(image, caption=f"Raw - {st.session_state.last_raw_time} (cached)", use_container_width=True)
                                    st.warning("Using cached frame (current frame incomplete)")
                                else:
                                    st.error(f"Image corrupted, no cache available")
                        else:
                            # Show last good image if available
                            if st.session_state.last_raw_image:
                                image = Image.open(BytesIO(st.session_state.last_raw_image))
                                st.image(image, caption=f"Raw - {st.session_state.last_raw_time} (cached)", use_container_width=True)
                            else:
                                st.error("Could not load image")
                    except Exception as e:
                        # Show last good image on any error
                        if st.session_state.last_raw_image:
                            try:
                                image = Image.open(BytesIO(st.session_state.last_raw_image))
                                st.image(image, caption=f"Raw - {st.session_state.last_raw_time} (cached)", use_container_width=True)
                                st.warning(f"Using cached frame")
                            except:
                                st.error(f"Error: {str(e)[:100]}")
                        else:
                            st.error(f"Error loading image: {str(e)[:100]}")
                else:
                    st.warning("No frames captured yet. Waiting for stream...")
                    st.info("Make sure stream-capture service is running and YOUTUBE_URL is configured")
            else:
                st.error("Preprocessing service not responding")
        except Exception as e:
            st.error(f"Cannot connect to preprocessing service: {e}")

    with col2:
        st.subheader("Preprocessed View")

        # Initialize session state for last good processed image
        if 'last_processed_image' not in st.session_state:
            st.session_state.last_processed_image = None
            st.session_state.last_processed_time = None

        # Try to load processed image with robust error handling
        try:
            # Add timestamp to prevent caching
            cache_buster = int(time.time() * 1000)  # Use milliseconds for high FPS
            img_response = requests.get(
                f"{PREPROCESSING_URL}/image/processed?t={cache_buster}",
                timeout=2,
                headers={'Cache-Control': 'no-cache'}
            )
            if img_response.ok and len(img_response.content) > 1000:  # Ensure we have a complete image
                try:
                    image = Image.open(BytesIO(img_response.content))
                    # Verify image is valid
                    image.verify()
                    # Reopen for display
                    image = Image.open(BytesIO(img_response.content))

                    # Store as last good image
                    st.session_state.last_processed_image = img_response.content
                    st.session_state.last_processed_time = time.strftime('%H:%M:%S')

                    st.image(image, caption=f"Processed - {st.session_state.last_processed_time}", use_container_width=True)

                    # Show crop info
                    try:
                        preview_response = requests.get(f"{PREPROCESSING_URL}/preview", timeout=2)
                        if preview_response.ok:
                            preview_data = preview_response.json()
                            crop = preview_data.get('crop_region', {})
                            st.info(f"Crop: {crop.get('width')}x{crop.get('height')} at ({crop.get('x')}, {crop.get('y')})")
                    except:
                        pass
                except (IOError, OSError) as e:
                    # Corrupted JPEG - use last good image
                    if st.session_state.last_processed_image:
                        image = Image.open(BytesIO(st.session_state.last_processed_image))
                        st.image(image, caption=f"Processed - {st.session_state.last_processed_time} (cached)", use_container_width=True)
                        st.warning("Using cached frame")
                    else:
                        st.warning("Waiting for automatic preprocessing...")
            else:
                # Show last good image if available
                if st.session_state.last_processed_image:
                    image = Image.open(BytesIO(st.session_state.last_processed_image))
                    st.image(image, caption=f"Processed - {st.session_state.last_processed_time} (cached)", use_container_width=True)
                else:
                    st.warning("Waiting for automatic preprocessing...")
                    st.info("Frames are processed automatically")
        except Exception as e:
            # Show last good image on any error
            if st.session_state.last_processed_image:
                try:
                    image = Image.open(BytesIO(st.session_state.last_processed_image))
                    st.image(image, caption=f"Processed - {st.session_state.last_processed_time} (cached)", use_container_width=True)
                except:
                    st.warning("Waiting for automatic preprocessing...")
            else:
                st.warning("Waiting for automatic preprocessing...")

    with col3:
        st.subheader("Vehicle Detection")

        # Initialize session state for last good detected image
        if 'last_detected_image' not in st.session_state:
            st.session_state.last_detected_image = None
            st.session_state.last_detected_time = None

        # Try to load detected image with robust error handling
        try:
            # Add timestamp to prevent caching
            cache_buster = int(time.time() * 1000)  # Use milliseconds for high FPS
            img_response = requests.get(
                f"{YOLO_DETECTOR_URL}/image/detected?t={cache_buster}",
                timeout=2,
                headers={'Cache-Control': 'no-cache'}
            )
            if img_response.ok and len(img_response.content) > 1000:  # Ensure we have a complete image
                try:
                    image = Image.open(BytesIO(img_response.content))
                    # Verify image is valid
                    image.verify()
                    # Reopen for display
                    image = Image.open(BytesIO(img_response.content))

                    # Store as last good image
                    st.session_state.last_detected_image = img_response.content
                    st.session_state.last_detected_time = time.strftime('%H:%M:%S')

                    st.image(image, caption=f"Detected - {st.session_state.last_detected_time}", use_container_width=True)

                    # Show detection statistics
                    try:
                        stats_response = requests.get(f"{YOLO_DETECTOR_URL}/stats", timeout=2)
                        if stats_response.ok:
                            stats = stats_response.json()
                            if stats.get('detections_available'):
                                vehicle_count = stats.get('vehicle_count', 0)
                                vehicle_types = stats.get('vehicle_types', {})

                                # Display stats
                                st.info(f"Total Vehicles: {vehicle_count}")

                                # Display breakdown by vehicle type
                                if vehicle_types:
                                    type_text = " | ".join([f"{vtype}: {count}" for vtype, count in vehicle_types.items()])
                                    st.caption(f"{type_text}")
                    except:
                        pass
                except (IOError, OSError) as e:
                    # Corrupted JPEG - use last good image
                    if st.session_state.last_detected_image:
                        image = Image.open(BytesIO(st.session_state.last_detected_image))
                        st.image(image, caption=f"Detected - {st.session_state.last_detected_time} (cached)", use_container_width=True)
                        st.warning("Using cached frame")
                    else:
                        st.warning("Waiting for vehicle detection...")
            else:
                # Show last good image if available
                if st.session_state.last_detected_image:
                    image = Image.open(BytesIO(st.session_state.last_detected_image))
                    st.image(image, caption=f"Detected - {st.session_state.last_detected_time} (cached)", use_container_width=True)
                else:
                    st.warning("Waiting for vehicle detection...")
                    st.info("Detection runs automatically at processing FPS")
        except Exception as e:
            # Show last good image on any error
            if st.session_state.last_detected_image:
                try:
                    image = Image.open(BytesIO(st.session_state.last_detected_image))
                    st.image(image, caption=f"Detected - {st.session_state.last_detected_time} (cached)", use_container_width=True)
                except:
                    st.warning("Waiting for vehicle detection...")
            else:
                st.warning("Waiting for vehicle detection...")

    # Status section
    st.markdown("---")
    st.subheader("System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            health_response = requests.get(f"{PREPROCESSING_URL}/health", timeout=2)
            if health_response.ok:
                st.success("Preprocessing Service")
            else:
                st.error("Preprocessing Service")
        except:
            st.error("Preprocessing Service")

    with col2:
        # Check if frames are available
        try:
            health_response = requests.get(f"{PREPROCESSING_URL}/health", timeout=2)
            if health_response.ok:
                health_data = health_response.json()
                if health_data.get('latest_frame_exists', False):
                    st.success("Frames Available")
                else:
                    st.warning("No Frames Yet")
            else:
                st.warning("No Frames Yet")
        except:
            st.warning("No Frames Yet")

    with col3:
        # Check YOLO detector service
        try:
            health_response = requests.get(f"{YOLO_DETECTOR_URL}/health", timeout=2)
            if health_response.ok:
                st.success("YOLO Detector")
            else:
                st.error("YOLO Detector")
        except:
            st.error("YOLO Detector")

    # Recent Detections section
    st.markdown("---")
    st.subheader("Recent Detections (Last 10)")

    try:
        detections_response = requests.get(f"{DATA_MANAGER_URL}/detections?limit=10", timeout=2)
        if detections_response.ok:
            detections_data = detections_response.json()
            detections = detections_data.get('detections', [])

            if detections:
                # Create a dataframe for display
                import pandas as pd
                from datetime import datetime as dt

                # Format data for display
                display_data = []
                for detection in detections:
                    timestamp = detection.get('timestamp', '')
                    # Convert ISO timestamp to readable format
                    try:
                        ts = dt.fromisoformat(timestamp)
                        time_str = ts.strftime('%H:%M:%S')
                        date_str = ts.strftime('%Y-%m-%d')
                    except:
                        time_str = timestamp
                        date_str = ''

                    display_data.append({
                        'Time': time_str,
                        'Date': date_str,
                        'Vehicle': detection.get('class_name', 'Unknown'),
                        'Confidence': f"{detection.get('confidence', 0):.2f}",
                        'ID': detection.get('id', '')
                    })

                df = pd.DataFrame(display_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No detections yet. Waiting for vehicles...")
        else:
            st.warning("Could not load recent detections")
    except Exception as e:
        st.error(f"Error loading detections: {e}")

# ============================================================================
# TAB 2: Crop Configuration Tool
# ============================================================================
with tab2:
    st.markdown("### Interactive Crop Region Selector")
    st.markdown("Adjust the sliders below to select the area where vehicles pass. The red rectangle shows the selected region.")

    # Get the latest frame
    try:
        cache_buster = int(time.time())
        img_response = requests.get(
            f"{PREPROCESSING_URL}/image/latest?t={cache_buster}",
            timeout=5,
            headers={'Cache-Control': 'no-cache'}
        )

        if img_response.ok:
            original_image = Image.open(BytesIO(img_response.content))
            img_width, img_height = original_image.size

            st.info(f"Full Image Size: {img_width} x {img_height}")

            # Initialize session state for crop values from preprocessing service
            if 'crop_x' not in st.session_state:
                # Read default values from environment variables (current config)
                default_x = int(os.getenv('CROP_X', 560))
                default_y = int(os.getenv('CROP_Y', 340))
                default_width = int(os.getenv('CROP_WIDTH', 800))
                default_height = int(os.getenv('CROP_HEIGHT', 400))

                try:
                    # Get current crop settings from preprocessing service
                    preview_response = requests.get(f"{PREPROCESSING_URL}/preview", timeout=2)
                    if preview_response.ok:
                        preview_data = preview_response.json()
                        crop = preview_data.get('crop_region', {})
                        st.session_state.crop_x = crop.get('x', default_x)
                        st.session_state.crop_y = crop.get('y', default_y)
                        st.session_state.crop_width = crop.get('width', default_width)
                        st.session_state.crop_height = crop.get('height', default_height)
                    else:
                        # Fallback to env defaults if service not available
                        st.session_state.crop_x = default_x
                        st.session_state.crop_y = default_y
                        st.session_state.crop_width = default_width
                        st.session_state.crop_height = default_height
                except:
                    # Fallback to env defaults on error
                    st.session_state.crop_x = default_x
                    st.session_state.crop_y = default_y
                    st.session_state.crop_width = default_width
                    st.session_state.crop_height = default_height

            # Crop controls in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Position")
                crop_x = st.slider("X Position (left)", 0, img_width-100, st.session_state.crop_x, key='slider_x')
                crop_y = st.slider("Y Position (top)", 0, img_height-100, st.session_state.crop_y, key='slider_y')

            with col2:
                st.markdown("#### Size")
                max_width = img_width - crop_x
                max_height = img_height - crop_y
                crop_width = st.slider("Width", 100, max_width, min(st.session_state.crop_width, max_width), key='slider_width')
                crop_height = st.slider("Height", 100, max_height, min(st.session_state.crop_height, max_height), key='slider_height')

            # Update session state
            st.session_state.crop_x = crop_x
            st.session_state.crop_y = crop_y
            st.session_state.crop_width = crop_width
            st.session_state.crop_height = crop_height

            # Draw rectangle on image
            preview_image = original_image.copy()
            draw = ImageDraw.Draw(preview_image)

            # Draw red rectangle
            draw.rectangle(
                [crop_x, crop_y, crop_x + crop_width, crop_y + crop_height],
                outline='red',
                width=3
            )

            # Show preview
            st.markdown("---")
            st.markdown("#### Preview with Selected Crop Region (Red Rectangle)")
            st.image(preview_image, use_container_width=True)

            # Show cropped region
            cropped_preview = original_image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
            st.markdown("#### Cropped Region Preview")
            st.image(cropped_preview, use_container_width=True)

            # Configuration output
            st.markdown("---")
            st.markdown("### Configuration for .env file")
            st.markdown("Click the copy button in the top-right corner of each code block:")

            # .env format with copy button
            config_text = f"""# Preprocessing crop region
CROP_X={crop_x}
CROP_Y={crop_y}
CROP_WIDTH={crop_width}
CROP_HEIGHT={crop_height}"""
            st.code(config_text, language="bash")

            st.markdown("**Or add to docker-compose.yml preprocessing service:**")
            docker_compose_text = f"""environment:
  - CROP_X={crop_x}
  - CROP_Y={crop_y}
  - CROP_WIDTH={crop_width}
  - CROP_HEIGHT={crop_height}"""
            st.code(docker_compose_text, language="yaml")

            st.info("After updating the configuration, restart the preprocessing service with: `docker compose restart preprocessing`")

        else:
            st.error("Could not load image. Make sure stream-capture is running.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure the preprocessing service is running and frames are being captured.")

# ============================================================================
# TAB 3: Professional Analytics Dashboard
# ============================================================================
with tab3:
    should_rerun = render_analytics_tab(DATA_MANAGER_URL)
    if should_rerun:
        st.rerun()

# ============================================================================
# TAB 4: Traffic Predictions & Machine Learning
# ============================================================================
with tab4:
    st.header("Traffic Predictions & Machine Learning")

    ONLINE_LEARNER_URL = os.getenv('ONLINE_LEARNER_URL', 'http://online-learner:8000')

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
                        import pandas as pd
                        import plotly.express as px
                        import plotly.graph_objects as go

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
                            import pandas as pd
                            import plotly.graph_objects as go

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
                                        import plotly.graph_objects as go
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

# Auto-refresh logic (only for Live Stream tab)
if auto_refresh:
    time.sleep(1.0 / refresh_fps)
    st.rerun()
