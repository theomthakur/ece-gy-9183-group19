FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install Python dependencies for the Python backend
RUN pip install --no-cache-dir ultralytics torch torchvision pillow

# Create a directory for the model repository
WORKDIR /models

# Expose Triton's ports
EXPOSE 8000 8001 8002

# Start Triton server when the container starts
# Model repository will be mounted at runtime
CMD ["tritonserver", "--model-repository=/models", "--http-port=8500", "--grpc-port=8501", "--metrics-port=8502", "--log-verbose=1"]