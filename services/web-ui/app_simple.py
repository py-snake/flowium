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
st.title("📹 Flowium - Live Stream Test")

# Tabs
tab1, tab2, tab3 = st.tabs(["📹 Live Stream", "✂️ Crop Configuration", "📊 Analytics"])

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
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    st.sidebar.markdown("---")

    # Background Subtraction Control
    st.sidebar.header("🎭 Background Subtraction")

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
                        st.sidebar.success("✅ Background subtraction enabled! Learning scene...")
                else:
                    disable_response = requests.post(f"{PREPROCESSING_URL}/background-subtraction/disable", timeout=2)
                    if disable_response.ok:
                        st.sidebar.info("Background subtraction disabled")

            # Reset button (only show if enabled)
            if bg_subtraction:
                if st.sidebar.button("🔄 Reset Background Model", help="Re-learn the background (useful after scene changes)"):
                    reset_response = requests.post(f"{PREPROCESSING_URL}/background-subtraction/reset", timeout=2)
                    if reset_response.ok:
                        st.sidebar.success("Background model reset!")

                st.sidebar.caption("💡 The model adapts over ~100 seconds. Reset if scene changes significantly.")

    except Exception as e:
        st.sidebar.error(f"⚠️ Background subtraction unavailable: {e}")

    st.sidebar.markdown("---")

    # Performance Metrics
    st.sidebar.header("⚡ Performance")

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
                "🎨 Preprocessing",
                f"{prep_data.get('last_processing_time_ms', 0):.1f} ms",
                delta=f"avg: {prep_data.get('avg_processing_time_ms', 0):.1f} ms"
            )

            st.sidebar.metric(
                "🚗 YOLO Detection",
                f"{yolo_data.get('last_detection_time_ms', 0):.1f} ms",
                delta=f"avg: {yolo_data.get('avg_detection_time_ms', 0):.1f} ms"
            )

            st.sidebar.metric(
                "⏱️ Overall",
                f"{overall_time:.1f} ms",
                delta=f"{1000/overall_time:.1f} FPS" if overall_time > 0 else "N/A"
            )

    except Exception as e:
        st.sidebar.error(f"⚠️ Performance data unavailable")

    st.sidebar.markdown("---")

    # Main content - 3 columns for Raw, Preprocessed, and Detected views
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📹 Raw Stream Capture")

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
                                st.info(f"📐 Full Resolution: {image.size[0]} x {image.size[1]}")
                            except (IOError, OSError) as e:
                                # Corrupted JPEG or incomplete read - use last good image
                                if st.session_state.last_raw_image:
                                    image = Image.open(BytesIO(st.session_state.last_raw_image))
                                    st.image(image, caption=f"Raw - {st.session_state.last_raw_time} (cached)", use_container_width=True)
                                    st.warning("⚠️ Using cached frame (current frame incomplete)")
                                else:
                                    st.error(f"❌ Image corrupted, no cache available")
                        else:
                            # Show last good image if available
                            if st.session_state.last_raw_image:
                                image = Image.open(BytesIO(st.session_state.last_raw_image))
                                st.image(image, caption=f"Raw - {st.session_state.last_raw_time} (cached)", use_container_width=True)
                            else:
                                st.error("❌ Could not load image")
                    except Exception as e:
                        # Show last good image on any error
                        if st.session_state.last_raw_image:
                            try:
                                image = Image.open(BytesIO(st.session_state.last_raw_image))
                                st.image(image, caption=f"Raw - {st.session_state.last_raw_time} (cached)", use_container_width=True)
                                st.warning(f"⚠️ Using cached frame")
                            except:
                                st.error(f"❌ Error: {str(e)[:100]}")
                        else:
                            st.error(f"❌ Error loading image: {str(e)[:100]}")
                else:
                    st.warning("⚠️ No frames captured yet. Waiting for stream...")
                    st.info("Make sure stream-capture service is running and YOUTUBE_URL is configured")
            else:
                st.error("❌ Preprocessing service not responding")
        except Exception as e:
            st.error(f"❌ Cannot connect to preprocessing service: {e}")

    with col2:
        st.subheader("🔍 Preprocessed View")

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
                            st.info(f"📐 Crop: {crop.get('width')}x{crop.get('height')} at ({crop.get('x')}, {crop.get('y')})")
                    except:
                        pass
                except (IOError, OSError) as e:
                    # Corrupted JPEG - use last good image
                    if st.session_state.last_processed_image:
                        image = Image.open(BytesIO(st.session_state.last_processed_image))
                        st.image(image, caption=f"Processed - {st.session_state.last_processed_time} (cached)", use_container_width=True)
                        st.warning("⚠️ Using cached frame")
                    else:
                        st.warning("⚠️ Waiting for automatic preprocessing...")
            else:
                # Show last good image if available
                if st.session_state.last_processed_image:
                    image = Image.open(BytesIO(st.session_state.last_processed_image))
                    st.image(image, caption=f"Processed - {st.session_state.last_processed_time} (cached)", use_container_width=True)
                else:
                    st.warning("⚠️ Waiting for automatic preprocessing...")
                    st.info("Frames are processed automatically")
        except Exception as e:
            # Show last good image on any error
            if st.session_state.last_processed_image:
                try:
                    image = Image.open(BytesIO(st.session_state.last_processed_image))
                    st.image(image, caption=f"Processed - {st.session_state.last_processed_time} (cached)", use_container_width=True)
                except:
                    st.warning("⚠️ Waiting for automatic preprocessing...")
            else:
                st.warning("⚠️ Waiting for automatic preprocessing...")

    with col3:
        st.subheader("🚗 Vehicle Detection")

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
                                st.info(f"🚗 Total Vehicles: {vehicle_count}")

                                # Display breakdown by vehicle type
                                if vehicle_types:
                                    type_text = " | ".join([f"{vtype}: {count}" for vtype, count in vehicle_types.items()])
                                    st.caption(f"📊 {type_text}")
                    except:
                        pass
                except (IOError, OSError) as e:
                    # Corrupted JPEG - use last good image
                    if st.session_state.last_detected_image:
                        image = Image.open(BytesIO(st.session_state.last_detected_image))
                        st.image(image, caption=f"Detected - {st.session_state.last_detected_time} (cached)", use_container_width=True)
                        st.warning("⚠️ Using cached frame")
                    else:
                        st.warning("⚠️ Waiting for vehicle detection...")
            else:
                # Show last good image if available
                if st.session_state.last_detected_image:
                    image = Image.open(BytesIO(st.session_state.last_detected_image))
                    st.image(image, caption=f"Detected - {st.session_state.last_detected_time} (cached)", use_container_width=True)
                else:
                    st.warning("⚠️ Waiting for vehicle detection...")
                    st.info("Detection runs automatically at processing FPS")
        except Exception as e:
            # Show last good image on any error
            if st.session_state.last_detected_image:
                try:
                    image = Image.open(BytesIO(st.session_state.last_detected_image))
                    st.image(image, caption=f"Detected - {st.session_state.last_detected_time} (cached)", use_container_width=True)
                except:
                    st.warning("⚠️ Waiting for vehicle detection...")
            else:
                st.warning("⚠️ Waiting for vehicle detection...")

    # Status section
    st.markdown("---")
    st.subheader("📊 System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            health_response = requests.get(f"{PREPROCESSING_URL}/health", timeout=2)
            if health_response.ok:
                st.success("✅ Preprocessing Service")
            else:
                st.error("❌ Preprocessing Service")
        except:
            st.error("❌ Preprocessing Service")

    with col2:
        # Check if frames are available
        try:
            health_response = requests.get(f"{PREPROCESSING_URL}/health", timeout=2)
            if health_response.ok:
                health_data = health_response.json()
                if health_data.get('latest_frame_exists', False):
                    st.success("✅ Frames Available")
                else:
                    st.warning("⚠️ No Frames Yet")
            else:
                st.warning("⚠️ No Frames Yet")
        except:
            st.warning("⚠️ No Frames Yet")

    with col3:
        # Check YOLO detector service
        try:
            health_response = requests.get(f"{YOLO_DETECTOR_URL}/health", timeout=2)
            if health_response.ok:
                st.success("✅ YOLO Detector")
            else:
                st.error("❌ YOLO Detector")
        except:
            st.error("❌ YOLO Detector")

    # Recent Detections section
    st.markdown("---")
    st.subheader("🚗 Recent Detections (Last 10)")

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
            st.warning("⚠️ Could not load recent detections")
    except Exception as e:
        st.error(f"❌ Error loading detections: {e}")

# ============================================================================
# TAB 2: Crop Configuration Tool
# ============================================================================
with tab2:
    st.markdown("### 🎯 Interactive Crop Region Selector")
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

            st.info(f"📐 Full Image Size: {img_width} x {img_height}")

            # Initialize session state for crop values from preprocessing service
            if 'crop_x' not in st.session_state:
                try:
                    # Get current crop settings from preprocessing service
                    preview_response = requests.get(f"{PREPROCESSING_URL}/preview", timeout=2)
                    if preview_response.ok:
                        preview_data = preview_response.json()
                        crop = preview_data.get('crop_region', {})
                        st.session_state.crop_x = crop.get('x', 560)
                        st.session_state.crop_y = crop.get('y', 340)
                        st.session_state.crop_width = crop.get('width', 800)
                        st.session_state.crop_height = crop.get('height', 400)
                    else:
                        # Fallback to defaults if service not available
                        st.session_state.crop_x = 560
                        st.session_state.crop_y = 340
                        st.session_state.crop_width = 800
                        st.session_state.crop_height = 400
                except:
                    # Fallback to defaults on error
                    st.session_state.crop_x = 560
                    st.session_state.crop_y = 340
                    st.session_state.crop_width = 800
                    st.session_state.crop_height = 400

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
            st.markdown("### 📋 Configuration for .env file")
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

            st.info("💡 After updating the configuration, restart the preprocessing service with: `docker compose restart preprocessing`")

        else:
            st.error("❌ Could not load image. Make sure stream-capture is running.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Make sure the preprocessing service is running and frames are being captured.")

# ============================================================================
# TAB 3: Professional Analytics Dashboard
# ============================================================================
with tab3:
    should_rerun = render_analytics_tab(DATA_MANAGER_URL)
    if should_rerun:
        st.rerun()

# Auto-refresh logic (only for Live Stream tab)
if auto_refresh:
    time.sleep(1.0 / refresh_fps)
    st.rerun()
