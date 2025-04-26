FROM python:3.12-slim

WORKDIR /app

# Get build arguments from docker-compose
ARG MODEL_DIR
ARG MODEL_PATH

# Install system dependencies - include additional libs for Mac compatibility
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

# Create a volume mount point for models
RUN mkdir -p ${MODEL_DIR}
VOLUME ${MODEL_DIR}

# Set environment variables (these will be overridden by docker-compose)
ENV MODEL_DIR=${MODEL_DIR}
ENV MODEL_PATH=${MODEL_PATH}
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Command to run the application with hot-reload enabled
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]