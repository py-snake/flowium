#!/bin/bash
# Flowium Setup Script

set -e

echo "Flowium Setup"
echo "================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ".env file created"
    echo ""
    echo "Please edit .env and add your:"
    echo "   - YOUTUBE_URL (YouTube live stream URL)"
    echo "   - WEATHER_API_KEY (from https://openweathermap.org/api)"
    echo ""
    read -p "Press Enter after editing .env file..."
else
    echo ".env file already exists"
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Docker Compose is not installed"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "Docker is installed"
echo ""

# Create volume directories
echo "Creating volume directories..."
mkdir -p volumes/frames volumes/models volumes/db-data
chmod -R 777 volumes/
echo "Volume directories created"
echo ""

# Build and start services
echo "Building Docker images (this may take a while)..."
docker-compose build

echo ""
echo "Setup complete!"
echo ""
echo "To start the system:"
echo "  docker-compose up -d"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To access the Web UI:"
echo "  http://localhost:8501"
echo ""
