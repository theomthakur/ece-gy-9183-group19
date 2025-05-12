# MODEL TRAINING

We selected YOLO12 for our medical x-ray analysis system because of its great performance across benchmarks in object detection tasks and practical implementation benefits.

### Why YOLO12?

- Top-tier accuracy in identifying objects within images
- The Ultralytics package simplifies implementation across our entire workflow
- Ready-to-use configurations for each MLOps pipeline stage (training, validation, deployment)

### How It Works
When a radiologist uploads an x-ray image (PNG format), our system:

- Processes the image through the YOLO12 model
- Identifies potential abnormalities
- Draws bounding boxes around suspicious areas
- Labels each area with the suspected condition
- Reports "No findings" when no abnormalities are detected

### Performance Optimization

We improved processing efficiency by:

- Increasing batch size to analyze multiple images simultaneously
- Adding more worker threads to utilize available computing resources
- Implementing reduced precision calculations without sacrificing accuracy

### Results
- mAP50 = 0.31053
- mAP50-95 = 0.15599
- Number of epochs: 50
- Training time: 3 hours

**NOTE:** We did all our testing with a smaller model (yolo12n) and trained it for just one resources because we had limited GPU power. For your actual project, you can simply change to the more powerful model (~yolo12x~) by editing the config/config.yaml files in each folder. You can also adjust how many GPUs to use with the "device:" setting and how many training runs to do with the "epochs" setting.

**NOTE:** Rename the .env.example file and fill in your credentials for the respective fields

### Folders:

- ray (Ray Train, Ray Tune and Ray Head setup):
  - `cd ray`
  - run `docker compose -f docker-compose-ray-cuda.yaml up` (sets up a Ray cluster with GPU support, MinIO object storage, and Grafana monitoring on a single network with specific port mappings.)
  - run `docker compose -f docker-compose-ray-tune.yaml up` (creates a separate Ray environment dedicated to hyperparameter tuning of YOLO models using Ray Tune, with its own MinIO storage, Grafana dashboard, and GPU worker.)
  - run `docker compose -f docker-compose-yolo-ray.yaml up` (creates a separate YOLO training environment using Ray for distributed training, complete with its own MinIO storage, Grafana monitoring, and GPU-enabled worker node.)
- src (non-interactive pipeline model training):
  - `cd src`
  - run `docker compose -f docker-compose-train.yaml`
- train (trigger automatic training and finetuning)
  - `cd train`
  - run `docker compose up`
- test (copy of src folder, without artifact logging, for experimenting running times):
  - `cd test`
  - run `docker compose -f docker-compose-train.yaml`

### Training Workflow
The training workflow in this project follows a pipeline that handles everything from model initialization to MLflow logging. Here's how it works in practice:

When you run the training container using docker-compose-train.yaml, the system:

- Allocates GPU resources via the NVIDIA runtime
- Mounts your datasets from /mnt/object into the container at /app/datasets
- Sets up persistent volumes for model weights and training runs
- Configures MLflow tracking environment variables

The main.py script serves as the entry point and performs these tasks:

- Parses command-line arguments for configuration paths and experiment naming
- Validates that necessary configuration files exist
- Sets up the MLflow connection and experiment tracking
- Calls the main training function from model_trainer.py

The model_trainer.py script then:

- Loads and validates the configuration settings using load_config_with_defaults()
- Attempts to locate or download the base YOLO weights through download_yolo_weights()
- Initializes the YOLO model from Ultralytics using the specified architecture (e.g., YOLOv12)

The training process:

- Creates a unique run name based on the model name and timestamp (or uses your custom name)
- Executes the YOLO training process with parameters from config.yaml
- Applies any data augmentation settings specified in the configuration
- Monitors for early stopping based on validation performance

After training completion:

- Validates the model on the test set to calculate performance metrics
- Runs inference on sample validation images to verify detection capabilities
- Exports the trained model to ONNX format for deployment

The system logs all relevant data to MLflow:

- Training parameters (model name, epochs, batch size, etc.)
- Performance metrics (mAP, precision, recall)
- Model artifacts (best weights in PyTorch and ONNX formats)
- Configuration files for reproducibility
- Optionally registers the model in the MLflow Model Registry
- Applies metadata tags and descriptions to the registered model

### Data Flow
The data flow through the training pipeline follows this path:

- Dataset is read from the mounted volume at /app/datasets
- Configuration files from /app/config determine training parameters
- Training produces artifacts in the persistent volume /app/runs
- Model weights are saved to /app/weights
- All metrics and models are logged to the configured MLflow server

### Error Handling
The workflow includes robust error handling:

- Validates dataset format and configuration files before training starts
- Gracefully handles MLflow connection issues while still completing training
- Provides detailed error messages for troubleshooting

UNIT 5:

Experiment Tracking: 
Implemented mlflow for both training and fine-tuning [here] and [here]

Scheduling training jobs: 
Implemented ray cluster to 

Use Ray Train: 
Implemented Ray Train [here]

Scheduling Hyperparameter tuning job: 
Implemented Ray Tune
