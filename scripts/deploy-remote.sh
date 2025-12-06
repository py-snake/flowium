#!/bin/bash
# Deploy Flowium to Remote Docker Server

set -e

REMOTE_HOST="dockerserver"
REMOTE_DIR="flowium"

echo "🚀 Deploying Flowium to Remote Server"
echo "======================================"
echo ""

# Check SSH connection
echo "Testing SSH connection to ${REMOTE_HOST}..."
if ! ssh -o ConnectTimeout=5 ${REMOTE_HOST} "echo 'Connected'" > /dev/null 2>&1; then
    echo "❌ Cannot connect to ${REMOTE_HOST}"
    echo "Please check your SSH configuration"
    exit 1
fi
echo "✅ SSH connection successful"
echo ""

# Create archive
echo "Creating deployment archive..."
tar czf /tmp/flowium-deploy.tar.gz \
  --exclude='idea.txt' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.vscode' \
  --exclude='*.log' \
  --exclude='volumes/frames/*' \
  --exclude='volumes/models/*' \
  --exclude='volumes/db-data/*' \
  docker-compose.yml \
  .env.example \
  .gitignore \
  README.md \
  services \
  scripts \
  volumes 2>/dev/null

echo "✅ Archive created"
echo ""

# Copy to remote
echo "Copying files to ${REMOTE_HOST}:${REMOTE_DIR}/..."
scp /tmp/flowium-deploy.tar.gz ${REMOTE_HOST}:${REMOTE_DIR}/

echo "✅ Files copied"
echo ""

# Extract on remote
echo "Extracting files on remote server..."
ssh ${REMOTE_HOST} "cd ${REMOTE_DIR} && tar xzf flowium-deploy.tar.gz && rm flowium-deploy.tar.gz"

echo "✅ Files extracted"
echo ""

# Clean up local archive
rm /tmp/flowium-deploy.tar.gz

echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Configure .env on remote:"
echo "   ssh ${REMOTE_HOST} 'cd ${REMOTE_DIR} && cp .env.example .env && nano .env'"
echo ""
echo "2. Build and start services:"
echo "   ./scripts/remote-docker.sh up -d --build"
echo ""
echo "3. View logs:"
echo "   ./scripts/remote-docker.sh logs -f"
echo ""
