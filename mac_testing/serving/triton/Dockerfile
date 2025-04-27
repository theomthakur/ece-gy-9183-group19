FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl git libgl1-mesa-glx libglib2.0-0 libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install tritonclient and other required packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tritonclient[http] ultralytics torch torchvision pillow

# Create a directory for the model repository
WORKDIR /models

# Create necessary directories for Triton
RUN mkdir -p /opt/tritonserver/backends/python

# Install a simpler Triton server for Mac compatibility
RUN pip install --no-cache-dir nvidia-pytritonserver

# Expose Triton's ports
EXPOSE 8000 8001 8002

# Start Triton server when the container starts
CMD ["python", "-m", "tritonserver.server", "--model-repository=/models", "--http-port=8000"]