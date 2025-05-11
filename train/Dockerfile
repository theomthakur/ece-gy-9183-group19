FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create required directories
RUN mkdir -p /app/data /app/weights /app/runs /app/config

# Copy application files
COPY flow.py /app/flow.py

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV MODEL_PATH="/app/weights/yolo12n.pt"
ENV CONFIG_PATH="/app/config/config.yaml"
ENV DATA_PATH="/app/config/data.yaml"
ENV OUTPUT_DIR="/app/runs"
ENV SERVICE_PORT=8000
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_EXPERIMENT_NAME="YOLO12-Training"

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI server
ENTRYPOINT ["python", "flow.py"]