#!/bin/bash
# Entrypoint script to choose which Streamlit app to run

APP_FILE=${STREAMLIT_APP:-app_simple.py}

echo "Starting Streamlit with: $APP_FILE"

exec streamlit run "$APP_FILE" --server.port=8501 --server.address=0.0.0.0
