#!/bin/bash
# Execute docker compose commands on remote server

REMOTE_HOST="dockerserver"
REMOTE_DIR="flowium"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <docker compose command>"
    echo ""
    echo "Examples:"
    echo "  $0 up -d                  # Start services in background"
    echo "  $0 up -d --build          # Build and start services"
    echo "  $0 down                   # Stop and remove services"
    echo "  $0 logs -f                # Follow logs"
    echo "  $0 logs yolo-detector     # Logs for specific service"
    echo "  $0 ps                     # List services"
    echo "  $0 restart yolo-detector  # Restart specific service"
    echo "  $0 build yolo-detector    # Rebuild specific service"
    echo "  $0 exec web-ui /bin/bash  # Shell into container"
    exit 1
fi

# Pass all arguments to docker compose on remote
ssh -t ${REMOTE_HOST} "cd ${REMOTE_DIR} && docker compose $*"
