FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ONNX Runtime with GPU support
RUN pip install onnxruntime-gpu==1.16.3

COPY ./app /app/app

# Make port 8000 available for the app
EXPOSE 8000

# Define environment variable
ENV MODEL_PATH=/models/yolov11_cxr.onnx
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]