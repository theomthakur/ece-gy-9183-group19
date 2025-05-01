#!/bin/bash

# Exit on any error
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <service_name>"
  exit 1
fi

SERVICE_NAME=$1


echo "Stopping the service (if running)..."
sudo docker compose down $SERVICE_NAME

echo "Rebuilding the service: $SERVICE_NAME..."
sudo docker compose build $SERVICE_NAME

echo "Starting the service with --force-recreate..."
sudo docker compose up -d --force-recreate $SERVICE_NAME

echo "Done. Service '$SERVICE_NAME' rebuilt and restarted."
