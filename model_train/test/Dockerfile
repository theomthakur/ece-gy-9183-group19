FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py model_trainer.py ./

RUN mkdir -p /app/weights /app/runs /app/config /app/datasets

ENV PYTHONPATH "${PYTHONPATH}:/app"

ENTRYPOINT ["python", "main.py"]
