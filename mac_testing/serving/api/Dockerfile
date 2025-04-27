FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV TRITON_SERVER_URL=triton_server:8000
ENV MODEL_NAME=chest_xray_detector
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Expose port
EXPOSE 8000

# Command to run the application with hot-reload enabled
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]