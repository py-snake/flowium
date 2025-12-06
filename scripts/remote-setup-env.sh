#!/bin/bash
# Setup .env file on remote server

REMOTE_HOST="dockerserver"
REMOTE_DIR="flowium"

echo "Setup Environment Variables on Remote"
echo "========================================="
echo ""

# Check if .env already exists
if ssh ${REMOTE_HOST} "[ -f ${REMOTE_DIR}/.env ]"; then
    echo "WARNING: .env file already exists on remote server"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Prompt for values
echo "Please enter your configuration:"
echo ""

read -p "YouTube Stream URL: " YOUTUBE_URL
read -p "Proxy URL (optional, leave empty if not needed): " PROXY_URL
read -p "OpenWeatherMap API Key: " WEATHER_API_KEY

# Create .env content
ENV_CONTENT="# Flowium Environment Variables

# YouTube Stream URL
YOUTUBE_URL=${YOUTUBE_URL}

# Proxy URL for yt-dlp (optional, leave empty if not needed)
# Example: PROXY_URL=http://10.0.0.100:8888
PROXY_URL=${PROXY_URL}

# Weather API Key (get from openweathermap.org)
WEATHER_API_KEY=${WEATHER_API_KEY}

# Optional: Override default settings
# FRAME_RATE=5
# CONFIDENCE_THRESHOLD=0.5
"

# Write to remote
echo "${ENV_CONTENT}" | ssh ${REMOTE_HOST} "cat > ${REMOTE_DIR}/.env"

echo ""
echo "Environment variables configured on remote server"
echo ""
echo "You can edit them anytime with:"
echo "  ssh ${REMOTE_HOST} 'cd ${REMOTE_DIR} && nano .env'"
echo ""
