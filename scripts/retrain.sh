#!/bin/bash
# Script to retrain model on new production data

# Default values
CONFIG=${1:-config.yaml}
PRODUCTION_DATA_CSV=${2:-data/production.csv}
PRODUCTION_IMAGES_DIR=${3:-data/production}
MODEL_PATH=${4:-runs/finetune/best.pt}
EPOCHS=${5:-10}

echo "Retraining model on production data..."
echo "Config: $CONFIG"
echo "Production data CSV: $PRODUCTION_DATA_CSV"
echo "Production images directory: $PRODUCTION_IMAGES_DIR"
echo "Pre-trained model: $MODEL_PATH"
echo "Epochs: $EPOCHS"

# Ensure Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if production data exists
if [ ! -f "$PRODUCTION_DATA_CSV" ]; then
    echo "Production data CSV not found: $PRODUCTION_DATA_CSV"
    exit 1
fi

if [ ! -d "$PRODUCTION_IMAGES_DIR" ]; then
    echo "Production images directory not found: $PRODUCTION_IMAGES_DIR"
    exit 1
fi

# Check if pre-trained model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Pre-trained model not found: $MODEL_PATH"
    exit 1
fi

# Run retraining in Docker container
docker compose run --rm train python3 src/retrain.py \
    --config $CONFIG \
    --production_data_csv $PRODUCTION_DATA_CSV \
    --production_images_dir $PRODUCTION_IMAGES_DIR \
    --model_path $MODEL_PATH \
    --epochs $EPOCHS \
    --ray_address ray://ray-head:6379 \
    --mlflow_tracking_uri http://mlflow:5000

echo "Retraining complete!"