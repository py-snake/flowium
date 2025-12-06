"""
Professional Analytics Dashboard Tab
Advanced statistics and data management for Flowium
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def render_analytics_tab(DATA_MANAGER_URL):
    """Render the professional analytics dashboard"""

    st.markdown("## 📊 Professional Traffic Analytics Dashboard")

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Dashboard Controls")

    time_range = st.sidebar.selectbox(
        "Time Range",
        [1, 6, 12, 24, 48, 72, 168],
        index=3,
        format_func=lambda x: f"Last {x} hours" if x < 168 else "Last Week"
    )

    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False, key="auto_refresh_analytics")

    # Check data manager availability
    try:
        health_response = requests.get(f"{DATA_MANAGER_URL}/health", timeout=2)
        if not health_response.ok:
            st.error("❌ Data Manager service not available")
            return False

        health_data = health_response.json()
        total_detections = health_data.get('total_detections', 0)

        if total_detections == 0:
            st.info("📊 No data collected yet. Start the stream and wait for data to accumulate.")
            return False

    except Exception as e:
        st.error(f"❌ Cannot connect to Data Manager: {e}")
        return False

    # ==========================================================================
    # TOP KPI METRICS
    # ==========================================================================
    st.markdown("### 🎯 Key Performance Indicators")

    try:
        advanced_stats = requests.get(f"{DATA_MANAGER_URL}/stats/advanced?hours={time_range}", timeout=5).json()
        current_stats = requests.get(f"{DATA_MANAGER_URL}/stats/current", timeout=2).json()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "📊 Total Detections",
                f"{advanced_stats['total_detections']:,}",
                delta=f"Last {time_range}h"
            )

        with col2:
            st.metric(
                "🚦 Current Traffic",
                current_stats['total_vehicles'],
                delta="Last 5 min",
                delta_color="normal"
            )

        with col3:
            st.metric(
                "📈 Avg Confidence",
                f"{advanced_stats['average_confidence']:.1%}",
                delta="Quality score"
            )

        with col4:
            busiest = advanced_stats['busiest_hour']
            if busiest['timestamp']:
                hour = datetime.fromisoformat(busiest['timestamp']).strftime('%H:%M')
                st.metric(
                    "⚡ Peak Traffic",
                    busiest['count'],
                    delta=f"At {hour}"
                )
            else:
                st.metric("⚡ Peak Traffic", "N/A")

        with col5:
            low_pct = advanced_stats['low_confidence_percentage']
            st.metric(
                "⚠️ Low Quality",
                f"{low_pct}%",
                delta=f"{advanced_stats['low_confidence_count']} detections",
                delta_color="inverse"
            )

    except Exception as e:
        st.error(f"Error loading KPIs: {e}")

    st.markdown("---")

    # ==========================================================================
    # WEATHER CONDITIONS
    # ==========================================================================
    st.markdown("### 🌤️ Current Weather Conditions")

    try:
        weather_response = requests.get(f"{DATA_MANAGER_URL}/weather/latest", timeout=2)
        if weather_response.ok:
            weather = weather_response.json()

            if weather.get('timestamp'):
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(
                        "🌡️ Temperature",
                        f"{weather['temperature']:.1f}°C"
                    )

                with col2:
                    st.metric(
                        "💧 Humidity",
                        f"{weather['humidity']:.0f}%"
                    )

                with col3:
                    # Weather condition with emoji
                    condition = weather['weather_condition']
                    emoji = "☀️" if "clear" in condition.lower() else \
                            "⛅" if "partly" in condition.lower() else \
                            "☁️" if "cloudy" in condition.lower() or "overcast" in condition.lower() else \
                            "🌧️" if "rain" in condition.lower() else \
                            "⛈️" if "thunder" in condition.lower() else \
                            "🌫️" if "fog" in condition.lower() else \
                            "❄️" if "snow" in condition.lower() else "🌤️"

                    st.metric(
                        f"{emoji} Conditions",
                        condition
                    )

                with col4:
                    st.metric(
                        "💨 Wind Speed",
                        f"{weather['wind_speed']:.1f} km/h"
                    )

                with col5:
                    st.metric(
                        "🌧️ Precipitation",
                        f"{weather['precipitation']:.1f} mm"
                    )

                # Show last update time
                last_update = datetime.fromisoformat(weather['timestamp'])
                st.caption(f"📅 Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info("No weather data available yet")
        else:
            st.warning("⚠️ Weather data temporarily unavailable")

    except Exception as e:
        st.warning(f"⚠️ Could not load weather data: {e}")

    st.markdown("---")

    # ==========================================================================
    # TRAFFIC ANALYSIS
    # ==========================================================================
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 📈 Traffic Timeline")

        try:
            timeline = requests.get(
                f"{DATA_MANAGER_URL}/stats/timeline?hours={time_range}&interval_minutes=60",
                timeout=5
            ).json()

            if timeline['data']:
                df_timeline = pd.DataFrame(timeline['data'])
                df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])

                # Create advanced line chart with area fill
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_timeline['timestamp'],
                    y=df_timeline['count'],
                    mode='lines+markers',
                    name='Vehicle Count',
                    fill='tozeroy',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))

                fig.update_layout(
                    title='Vehicles Detected Over Time',
                    xaxis_title='Time',
                    yaxis_title='Vehicles per Hour',
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Statistics cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total", f"{df_timeline['count'].sum():,}")
                with col2:
                    st.metric("📈 Average", f"{df_timeline['count'].mean():.1f}/h")
                with col3:
                    st.metric("🔝 Peak", f"{df_timeline['count'].max()}")
                with col4:
                    st.metric("🔻 Min", f"{df_timeline['count'].min()}")

            else:
                st.info(f"No timeline data for the last {time_range} hours")

        except Exception as e:
            st.error(f"Error loading timeline: {e}")

    with col_right:
        st.markdown("### 🚗 Vehicle Distribution")

        try:
            type_stats = requests.get(
                f"{DATA_MANAGER_URL}/stats/vehicle_types?hours={time_range}",
                timeout=5
            ).json()

            if type_stats['total_vehicles'] > 0:
                df_types = pd.DataFrame(type_stats['breakdown'])

                # Pie chart
                fig = px.pie(
                    df_types,
                    values='count',
                    names='vehicle_type',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )

                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
                )

                fig.update_layout(height=400, showlegend=False)

                st.plotly_chart(fig, use_container_width=True)

                # Breakdown table
                st.markdown("**Detailed Breakdown:**")
                for item in type_stats['breakdown']:
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.text(f"🚗 {item['vehicle_type']}")
                    with cols[1]:
                        st.text(f"{item['count']:,}")
                    with cols[2]:
                        st.text(f"{item['percentage']}%")

            else:
                st.info(f"No data for last {time_range}h")

        except Exception as e:
            st.error(f"Error loading vehicle types: {e}")

    st.markdown("---")

    # ==========================================================================
    # DATA MANAGEMENT & EXPORT
    # ==========================================================================
    st.markdown("### 🗄️ Data Management & Export")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📤 Export Data")

        export_hours = st.selectbox("Export timeframe", [1, 6, 12, 24, 48, 168], index=3)

        if st.button("📥 Download CSV", use_container_width=True):
            try:
                csv_url = f"{DATA_MANAGER_URL}/export/csv?hours={export_hours}"
                st.markdown(f"[📥 Click here to download CSV]({csv_url})")
                st.success(f"✅ Exporting data from last {export_hours} hours")
            except Exception as e:
                st.error(f"Export failed: {e}")

        st.markdown("---")

        st.markdown("#### 💾 Database Info")
        st.info(f"**Total Records:** {total_detections:,}")
        st.info(f"**Storage:** Unlimited retention")

    with col2:
        st.markdown("#### 🧹 Data Cleanup")

        st.warning("⚠️ **Warning:** These actions are irreversible!")

        # Clear by time range
        with st.expander("🕐 Clear by Time Range"):
            clear_hours = st.number_input("Hours to clear", min_value=1, max_value=168, value=1)
            if st.button(f"Delete last {clear_hours} hours", key="clear_time"):
                try:
                    response = requests.delete(
                        f"{DATA_MANAGER_URL}/detections/timerange?hours={clear_hours}",
                        timeout=5
                    )
                    if response.ok:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.rerun()
                    else:
                        st.error("❌ Delete failed")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Clear low confidence
        with st.expander("📉 Clear Low Quality"):
            confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05
            )
            if st.button(f"Delete detections < {confidence_threshold:.0%}", key="clear_confidence"):
                try:
                    response = requests.delete(
                        f"{DATA_MANAGER_URL}/detections/low_confidence?confidence_threshold={confidence_threshold}",
                        timeout=5
                    )
                    if response.ok:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        st.rerun()
                    else:
                        st.error("❌ Delete failed")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Clear all
        with st.expander("🗑️ Clear ALL Data"):
            st.error("⛔ This will delete ALL detections permanently!")
            confirm = st.text_input("Type 'DELETE ALL' to confirm")
            if st.button("🗑️ Delete Everything", key="clear_all", type="primary"):
                if confirm == "DELETE ALL":
                    try:
                        response = requests.delete(f"{DATA_MANAGER_URL}/detections/all", timeout=5)
                        if response.ok:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                            st.rerun()
                        else:
                            st.error("❌ Delete failed")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please type 'DELETE ALL' to confirm")

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(5)
        return True  # Signal to rerun

    return False  # No rerun needed
