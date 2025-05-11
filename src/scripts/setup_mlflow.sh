#!/bin/bash
# Script to setup MLflow server on Chameleon Cloud

# Exit on error
set -e

# Default values
PORT=${1:-5000}
BACKEND_URI=${2:-sqlite:///mlflow.db}
ARTIFACTS_DIR=${3:-./mlflow-artifacts}

echo "Setting up MLflow server..."
echo "Port: $PORT"
echo "Backend URI: $BACKEND_URI"
echo "Artifacts Directory: $ARTIFACTS_DIR"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in for group changes to take effect."
fi

# Create artifacts directory if it doesn't exist
mkdir -p $ARTIFACTS_DIR

# Create a docker-compose.yml file for MLflow
cat > docker-compose.yml << EOL
version: '3'
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "$PORT:$PORT"
    volumes:
      - $ARTIFACTS_DIR:/mlflow/artifacts
    environment:
      - MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts
    command: mlflow server --backend-store-uri $BACKEND_URI --artifacts-destination /mlflow/artifacts --host 0.0.0.0 --port $PORT
EOL

# Start MLflow server
echo "Starting MLflow server..."
docker compose up -d

# Get the IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}')
echo "MLflow server started at: http://$IP_ADDRESS:$PORT"
echo "To view MLflow UI, navigate to: http://$IP_ADDRESS:$PORT"
echo "To use this server in your training code, set mlflow_tracking_uri to: http://$IP_ADDRESS:$PORT"

# Add additional information
echo ""
echo "To stop the MLflow server, run: docker compose down"
echo "To view MLflow server logs, run: docker compose logs -f mlflow"
echo "Artifacts are stored in: $ARTIFACTS_DIR"
echo ""