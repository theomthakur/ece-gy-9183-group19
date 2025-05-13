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

### Results (yolo12n)
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
 
## UNIT 4

### Training Workflow
The training workflow in this project follows a pipeline that handles everything from model initialization to MLflow logging. Here's how it works in practice:

When you run the training container using `docker-compose-train.yaml`, the system:

- Allocates GPU resources via the NVIDIA runtime
- Mounts your datasets from `/mnt/object` into the container at `/app/datasets`
- Sets up persistent volumes for model weights and training runs
- Configures MLflow tracking environment variables

The main.py script serves as the entry point and performs these tasks:

- Parses command-line arguments for configuration paths and experiment naming
- Validates that necessary configuration files exist
- Sets up the MLflow connection and experiment tracking
- Calls the main training function from `model_trainer.py`

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

- Dataset is read from the mounted volume at `/app/datasets`
- Configuration files from `/app/config` determine training parameters
- Training produces artifacts in the persistent volume `/app/runs`
- Model weights are saved to `/app/weights`
- All metrics and models are logged to the configured MLflow server

### Error Handling
The workflow includes robust error handling:

- Validates dataset format and configuration files before training starts
- Gracefully handles MLflow connection issues while still completing training
- Provides detailed error messages for troubleshooting

## UNIT 5:

Experiment Tracking: 

Our training system uses MLflow to:

- Track training parameters and performance metrics
- Store trained models and configurations
- Manage model versions through the Model Registry
- Support fine-tuning workflows with model lineage

We maintain separate experiments for training (YOLO12-Training) and fine-tuning (YOLO12-Finetuning) to keep our workflows organized.

For each training session, we log:

- Parameters: Model architecture, batch size, image size, epochs, etc.
- Metrics: mAP, mAP50, mAP75, and other performance metrics
- Artifacts:

  - Model files (PyTorch .pt and ONNX .onnx formats)
  - Configuration files (YAML)
  - Dataset specifications

Our pipeline automatically registers successful models in the MLflow Model Registry:

- Base models are registered under their architecture name (e.g., `yolo12n`)
- Fine-tuned models get a `-finetuned` suffix
- We use aliases to identify model stages:
  - development for base trained models
  - staging for fine-tuned models

Access the MLflow UI at your configured tracking URI to:

- Compare model performance across runs
- View training parameters
- Download model artifacts
- Manage model versions

Models can be retrieved from the registry for inference or fine-tuning:

`model_path, model_version = get_latest_model(model_name)`
`model_path, model_version = get_latest_model(model_name, alias="staging")`

Our service provides endpoints to trigger training and access model information:

- `/trigger-training`: Automatically determines whether to train from scratch or fine-tune
- `/trigger-finetune`: Explicitly starts a fine-tuning pipeline
- `/config`: View current MLflow configuration and model settings

### Ray Setup
In our `docker-compose-ray-cuda.yaml`, we established the foundation of our Ray cluster:

- We configured a Ray head node as the cluster coordinator
- We set up GPU-enabled worker nodes to execute computations in parallel
- We integrated MinIO for S3-compatible storage of model artifacts
- We incorporated Grafana for real-time monitoring of cluster performance

Our ray_setup.py handles the connection to the Ray cluster with fault tolerance:
```
def setup_ray_cluster():
    try:
        ray.init(address="ray://ray-head:6379", ignore_reinit_error=True)
        print("Connected to existing Ray cluster")
    except Exception as e:
        # Fall back to local initialization if connection fails
        ray.init(ignore_reinit_error=True)
```

We built a robust distributed training pipeline in `ray_trainer.py` that:

- Distributes YOLO model training across multiple GPUs
- Implements checkpointing to recover from potential failures
- Tracks all experiments through MLflow integration
- Exports trained models for deployment

We defined a worker function that runs on each node:
```
def train_func(config):
    # Initialize model based on configuration
    # Train the model and track metrics
    # Save checkpoints and report results
```
For our production environment, we created a dedicated Docker Compose configuration (`docker-compose-yolo-ray.yaml`) that:

- Establishes a Ray cluster specifically for training
- Mounts our datasets and configuration files
- Provides containers for both initial training and retraining scenarios

To optimize our models, we implemented hyperparameter tuning in `ray_tune.py`:

- We defined a search space covering learning rate, batch size, and image resolution
- We utilized Bayesian optimization to efficiently explore parameter combinations
- We employed aggressive early stopping for ineffective trials
- We tracked all experiments in MLflow for analysis and reproducibility

```
def run_hyperparameter_tuning(config, data_yaml, num_samples=10, max_concurrent=None):
    # Configure search space and optimization algorithm
    # Run trials in parallel across our cluster
    # Return the optimal configuration found
```

Our team created a separate environment for tuning in `docker-compose-ray-tune.yaml` that:

- Sets up a dedicated Ray cluster for these resource-intensive experiments
- Scales automatically based on available GPU resources
- Operates independently from our training cluster

Our `yolo_ray_main.py` provides a unified interface for our team to:

- Train new models: `--mode train`
- Continue training from existing checkpoints: `--mode retrain --model_path /path/to/model.pt`
- Run hyperparameter optimization: `--mode tune --num_samples 20`

## Infrastructure Setup on Chameleon Cloud (Optional)

If you plan to run the training and deployment workflows on Chameleon Cloud, you'll first need to provision and configure a GPU-enabled server. Below are the steps to achieve this, derived from the setup process used in our project.

**1. Prerequisites on Chameleon Cloud:**

*   **Account and Project:** You need an active Chameleon Cloud account and an allocated project.
*   **Lease a GPU Node:**
    *   Create a lease for at least one GPU-enabled bare metal node (e.g., `gpu_rtx_6000` or similar).
    *   Note your **Lease Name** and **Reservation ID** for the node. You will need the Reservation ID when creating the server.
    *   Ensure your lease is active before proceeding.
*   **SSH Keypair:** Ensure you have an SSH keypair configured in your Chameleon profile and that you have the private key accessible on your local machine. This keypair will be associated with the server.

**2. Provision the Server Instance:**

Using the Chameleon Cloud portal (or `openstack` CLI if you are familiar with it):

*   **Navigate to "Instances"** in your project.
*   **Launch Instance:**
    *   **Instance Name:** Choose a name (e.g., `my-llm-node`).
    *   **Source:** Select "Image" and search for `CC-Ubuntu24.04-CUDA`. This image comes with NVIDIA drivers and CUDA pre-installed.
    *   **Flavor:** Select `baremetal`.
    *   **Networks:** Select `sharednet1` (or the appropriate public network for your site).
    *   **Key Pair:** Select the SSH keypair you prepared earlier.
    *   **Scheduler Hints (Advanced Options):** Under "Advanced Options" or "Scheduler Hints", add a hint: `reservation=<YOUR_NODE_RESERVATION_ID>`. Replace `<YOUR_NODE_RESERVATION_ID>` with the ID from your lease. This ensures the server is launched on your reserved hardware.
*   **Launch** the instance. Wait for it to become "ACTIVE".
*   **Associate Floating IP:** Once the instance is active, associate a Floating IP address to it so it's accessible from the internet. Note down this Floating IP.

**3. Configure the Server Software (via SSH):**

SSH into your newly provisioned server using its Floating IP:
```
ssh cc@<YOUR_SERVER_FLOATING_IP> -i /path/to/your/chameleon_private_key
```

Replace <YOUR_SERVER_FLOATING_IP> and /path/to/your/chameleon_private_key. The default username for Chameleon appliances is cc.
Once logged in, execute the following commands to set up Docker, NVIDIA Container Toolkit, and other utilities:

Install Docker Engine:
```
curl -fsSL https://get.docker.com | sudo sh
sudo groupadd -f docker
sudo usermod -aG docker $USER
```

Note: After running usermod, you might need to log out and log back in, or run newgrp docker in your current session for the group changes to take effect immediately for your user. Otherwise, you'll need to prefix docker commands with sudo.
Install NVIDIA Container Toolkit:
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list and install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU Access within Docker:
```
sudo docker run --rm --gpus all ubuntu nvidia-smi
```

This command should output the nvidia-smi details, showing your server's GPU(s).
Install nvtop (GPU process monitor):
```
sudo apt-get update
sudo apt-get install -y nvtop
```
4. Pull Required Docker Images for the Project:
These are the primary images used by different parts of this project:
Jupyter PyTorch Notebook (for single-GPU experimentation/interactive work):
```
sudo docker pull quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1
```
PyTorch Devel Image (base for multi-GPU training, e.g., with DeepSpeed):
```
sudo docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```
Note: This image pull can take a significant amount of time.
