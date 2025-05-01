#!/bin/bash
# Script to setup Ray cluster on Chameleon Cloud

# Exit on error
set -e

# Default values
NUM_WORKERS=${1:-2}
HEAD_PORT=${2:-6379}
DASHBOARD_PORT=${3:-8265}

echo "Setting up Ray cluster..."
echo "Number of workers: $NUM_WORKERS"
echo "Head port: $HEAD_PORT"
echo "Dashboard port: $DASHBOARD_PORT"

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

# Get the IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Create a docker-compose.yml file for Ray cluster
cat > docker-compose.yml << EOL
version: '3'
services:
  ray-head:
    image: rayproject/ray:latest
    ports:
      - "$HEAD_PORT:$HEAD_PORT"
      - "$DASHBOARD_PORT:$DASHBOARD_PORT"
      - "10001:10001"
    volumes:
      - ./ray_shared:/ray_shared
      - /tmp/ray:/tmp/ray
    command: >
      bash -c "
        ray start --head 
          --port=$HEAD_PORT 
          --dashboard-port=$DASHBOARD_PORT 
          --dashboard-host=0.0.0.0 
          --include-dashboard=true 
          --block
      "
    shm_size: '2gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
EOL

# Add worker services
for ((i=1; i<=NUM_WORKERS; i++)); do
  cat >> docker-compose.yml << EOL
  ray-worker-$i:
    image: rayproject/ray:latest
    volumes:
      - ./ray_shared:/ray_shared
      - /tmp/ray:/tmp/ray
    command: >
      bash -c "
        ray start 
          --address=ray-head:$HEAD_PORT 
          --block
      "
    depends_on:
      - ray-head
    shm_size: '2gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
EOL
done

# Create shared directory
mkdir -p ray_shared

# Start Ray cluster
echo "Starting Ray cluster..."
docker compose up -d

echo "Ray cluster started!"
echo "Head node: $IP_ADDRESS:$HEAD_PORT"
echo "Dashboard: http://$IP_ADDRESS:$DASHBOARD_PORT"
echo "To use this cluster in your training code, set ray_address to: ray://$IP_ADDRESS:$HEAD_PORT"

# Add additional information
echo ""
echo "To stop the Ray cluster, run: docker compose down"
echo "To view Ray logs, run: docker compose logs -f"
echo "Shared storage is available in: ./ray_shared"
echo ""

# Wait for Ray dashboard to become available
echo "Waiting for Ray dashboard to become available..."
timeout=60
elapsed=0
while ! curl -s http://$IP_ADDRESS:$DASHBOARD_PORT > /dev/null; do
    sleep 1
    elapsed=$((elapsed+1))
    if [ $elapsed -ge $timeout ]; then
        echo "Timed out waiting for Ray dashboard."
        break
    fi
done

if [ $elapsed -lt $timeout ]; then
    echo "Ray dashboard is now available at: http://$IP_ADDRESS:$DASHBOARD_PORT"
fi

# Create a script to submit a job to the Ray cluster
cat > submit_job.sh << EOL
#!/bin/bash
# Script to submit a job to the Ray cluster

if [ \$# -lt 1 ]; then
    echo "Usage: \$0 <script.py> [args...]"
    exit 1
fi

SCRIPT=\$1
shift
ARGS="\$@"

# Run the script on the Ray cluster
docker exec -it \$(docker compose ps -q ray-head) python \$SCRIPT \$ARGS --ray_address=ray://localhost:$HEAD_PORT
EOL

chmod +x submit_job.sh

echo "Created submit_job.sh script to submit jobs to the Ray cluster."
echo "Usage: ./submit_job.sh <script.py> [args...]"