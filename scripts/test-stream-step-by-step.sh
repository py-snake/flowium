#!/bin/bash
# Step-by-step test script for Flowium stream capture

REMOTE_HOST="dockerserver"
REMOTE_DIR="flowium"

echo "🧪 Flowium Stream Capture Test - Step by Step"
echo "=============================================="
echo ""
echo "This script will guide you through testing the YouTube stream capture."
echo ""

# Step 1: Check connection
echo "📌 Step 1: Checking connection to remote server..."
if ! ssh -o ConnectTimeout=5 ${REMOTE_HOST} "echo 'Connected'" > /dev/null 2>&1; then
    echo "❌ Cannot connect to ${REMOTE_HOST}"
    exit 1
fi
echo "✅ Connected to ${REMOTE_HOST}"
echo ""

read -p "Press Enter to continue..."
echo ""

# Step 2: Check if .env exists
echo "📌 Step 2: Checking environment configuration..."
if ssh ${REMOTE_HOST} "[ -f ${REMOTE_DIR}/.env ]"; then
    echo "✅ .env file exists"

    # Show current configuration (without API keys)
    echo ""
    echo "Current configuration:"
    ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && grep -E 'YOUTUBE_URL|PROXY_URL' .env | grep -v '^#'"
    echo ""
else
    echo "❌ .env file not found!"
    echo ""
    echo "Please configure environment first:"
    echo "  ./scripts/remote-setup-env.sh"
    echo ""
    exit 1
fi

read -p "Is this configuration correct? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please update configuration:"
    echo "  ssh ${REMOTE_HOST} 'cd ${REMOTE_DIR} && nano .env'"
    exit 0
fi
echo ""

# Step 3: Deploy latest changes
echo "📌 Step 3: Deploying latest code..."
echo "Running: ./scripts/deploy-remote.sh"
./scripts/deploy-remote.sh > /dev/null 2>&1
echo "✅ Code deployed"
echo ""

read -p "Press Enter to continue..."
echo ""

# Step 4: Stop any running containers
echo "📌 Step 4: Stopping any running containers..."
./scripts/remote-docker.sh down > /dev/null 2>&1
echo "✅ Containers stopped"
echo ""

read -p "Press Enter to continue..."
echo ""

# Step 5: Start only required services
echo "📌 Step 5: Starting required services (stream-capture, preprocessing, web-ui)..."
echo "This may take a few minutes on first run (building images, downloading Deno, etc.)"
echo ""

./scripts/remote-docker.sh up -d --build stream-capture preprocessing web-ui

echo ""
echo "✅ Services started"
echo ""

read -p "Press Enter to continue..."
echo ""

# Step 6: Wait for services to be ready
echo "📌 Step 6: Waiting for services to be ready..."
echo "Waiting 10 seconds for services to start..."
sleep 10
echo "✅ Services should be ready"
echo ""

# Step 7: Check service health
echo "📌 Step 7: Checking service health..."
echo ""

echo "Preprocessing service:"
ssh ${REMOTE_HOST} "curl -s http://localhost:8001/health" | head -1
echo ""

echo "Web UI service:"
if ssh ${REMOTE_HOST} "curl -s http://localhost:8501" > /dev/null 2>&1; then
    echo "✅ Web UI is responding"
else
    echo "⚠️ Web UI not responding yet (may need more time)"
fi
echo ""

read -p "Press Enter to continue..."
echo ""

# Step 8: Check stream capture logs
echo "📌 Step 8: Checking stream-capture logs..."
echo "Last 20 lines of stream-capture logs:"
echo "----------------------------------------"
./scripts/remote-docker.sh logs --tail=20 stream-capture
echo "----------------------------------------"
echo ""

echo "Look for:"
echo "  ✅ 'Got stream URL: ...' = Stream is being captured"
echo "  ✅ 'Captured frame ...' = Frames are being saved"
echo "  ❌ 'ERROR' = Something went wrong"
echo ""

read -p "Press Enter to continue..."
echo ""

# Step 9: Check if frames exist
echo "📌 Step 9: Checking if frames are being captured..."
FRAME_CHECK=$(ssh ${REMOTE_HOST} "ls -la ${REMOTE_DIR}/volumes/frames/latest.jpg 2>/dev/null")
if [ ! -z "$FRAME_CHECK" ]; then
    echo "✅ Frames are being captured!"
    echo "$FRAME_CHECK"
else
    echo "⚠️ No frames captured yet"
    echo "Check the logs above for errors"
fi
echo ""

read -p "Press Enter to continue..."
echo ""

# Step 10: Access Web UI
echo "📌 Step 10: Accessing the Web UI..."
echo ""
echo "You can now view the live stream in your browser!"
echo ""
echo "Option 1: SSH Tunnel (Recommended)"
echo "  Run in a new terminal:"
echo "    ssh -L 8501:localhost:8501 ${REMOTE_HOST}"
echo "  Then open: http://localhost:8501"
echo ""
echo "Option 2: Direct Access (if firewall allows)"
echo "  Get server IP: ssh ${REMOTE_HOST} 'hostname -I'"
echo "  Then open: http://<server-ip>:8501"
echo ""

read -p "Press Enter to open SSH tunnel now (or Ctrl+C to skip)..."

echo ""
echo "Creating SSH tunnel to port 8501..."
echo "Keep this terminal open and access http://localhost:8501 in your browser"
echo ""
echo "Press Ctrl+C to close the tunnel when done"
echo ""

ssh -L 8501:localhost:8501 ${REMOTE_HOST}
