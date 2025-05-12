FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
python3-pip \
python3-dev \
git \
wget \
curl \
libgl1-mesa-glx \
libglib2.0-0 \
tzdata \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir \
"ray[tune,train,air]==2.42.1" \
"ray[default]==2.42.1" \
"pytorch_lightning" \
"bayesian-optimization" \
"hyperopt" \
"botorch" \
"python-dotenv"

RUN mkdir -p /app/datasets /app/runs /app/weights /app/config

COPY yolo_ray_main.py /app/
COPY ray_setup.py /app/
COPY ray_trainer.py /app/
COPY ray_tune.py /app/
COPY ray_main.py /app/