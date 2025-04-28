FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the Python backend
RUN pip install --no-cache-dir ultralytics torch torchvision pillow

# Create a directory for the model repository
WORKDIR /models

# Expose Triton's ports
EXPOSE 8500 8501 8502

# Start Triton server when the container starts
# Model repository will be mounted at runtime
CMD ["tritonserver", "--model-repository=/models", "--http-port=8500", "--grpc-port=8501", "--metrics-port=8502", "--log-verbose=1"]