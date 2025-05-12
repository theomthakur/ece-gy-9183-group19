## Unit 8: Data Pipeline

### Persistent Storage
We use persistent storage on Chameleon to manage data effectively:

1. **Swift Object Store** (`chi_tacc:object-persist-project19`):
   - **Purpose**: Stores the organized VinBigData dataset for training, evaluation, and production, as well as re-training data.
   - **Used By**: Offline Data, Data Pipeline (Offline and Production), and Online Data to store and retrieve dataset splits and re-training data.

2. **Block Storage Volume** (`/mnt/mydata` on `vdb`):
   - **Purpose**: Stores simulator feedback and logs generated during online data processing.
   - **Used By**: Online Data and Data Pipeline (Production) to save feedback JSONs and logs.

### Offline Data
- **Training Dataset**: We use the [VinBigData Chest X-ray Abnormalities Detection dataset](https://www.kaggle.com/datasets/xhlulu/vinbigdata-chest-xray-resized-png-1024x1024), resized to 1024x1024.
- **Data Lineage**:
  1. Downloaded from Kaggle.
  2. Transformed into YOLO format with splits: `training` (~11500 images), `validation` (~750), `test` (~750), `staging` (~1000), `canary` (~500), `production` (~500).
  3. Uploaded to `chi_tacc:object-persist-project19/organized/`.
- **Example Sample**:
  - **Image**: `organized/training/images/<image_id>_R1.png`.
  - **Label**: `organized/training/labels/<image_id>_R1.txt` (YOLO format: `class_id x_center y_center width height`).
  - **Relevance to Customer**: The radiologist uses this data to train a model for detecting chest X-ray abnormalities. The sample represents a typical X-ray with annotations, supporting diagnostic accuracy.
- **Production Sample Lifetime**: Production samples (from the `production` split) are processed by the online simulator, generating feedback (`<image_id>.json`) with predicted `class_id`, `confidence`, and `bbox`. Labels are refined via simulated radiologist feedback processed in the production pipeline.
- **Used By**: Data Pipeline (Offline) to process and organize the dataset.

### Data Pipeline
- **Offline Pipeline**:
  - **Implementation**: Managed by the `docker-compose-etl.yaml` file.
    - **What It Does**: This Docker Compose file orchestrates the offline ETL pipeline with three services: one downloads the VinBigData dataset from Kaggle and extracts it; another processes a CSV file to convert bounding boxes to YOLO format, split the dataset into `training`, `validation`, `test`, `staging`, `canary`, and `production`, and generate a YOLO configuration file; the third uploads the organized dataset to Swift storage.
  - **Steps**:
    1. **Extract**: Downloads VinBigData from Kaggle using `kaggle datasets download`.
    2. **Transform**: Converts to YOLO format, splits into `training`, `validation`, `test`, `staging`, `canary`, and `production` sets (scaled to ~15,000 unique images). Avoids data leakage by shuffling unique `image_id`s before splitting.
    3. **Load**: Uploads to `chi_tacc:object-persist-project19/organized/` using `rclone`.
  - **Pre-processing**: Images are pre-resized to 1024x1024 (in the dataset), and bounding boxes are normalized to YOLO format.
  - **Used By**: Data Pipeline (Offline) to execute the ETL process.

- **Production Pipeline (Closing the Loop)**:
  - **Implementation**: Managed by the `docker-compose-production.yaml` file.
    - **What It Does**: This Docker Compose file runs the production ETL pipeline, collecting feedback JSONs, transforming them into a re-training CSV by filtering high-confidence predictions (`confidence > 0.9`), and uploading the result to `chi_tacc:object-persist-project19/organized/retrain/` using `rclone`.
  - **Steps**:
    1. **Extract**: Collects feedback JSONs from `/mnt/mydata/model-checkpoints/feedback/`.
    2. **Transform**: Filters high-confidence predictions (`confidence > 0.9`) into `retrain.csv`.
    3. **Load**: Uploads to `chi_tacc:object-persist-project19/organized/retrain/`.
  - **Purpose**: Prepares data for model re-training based on production feedback, aligning with Lab 7’s “closing the loop” concept.
  - **Used By**: Data Pipeline (Production) to process feedback and prepare re-training data.

### Data Dashboard
- **Implementation**: Managed by the `data_dashboard.py` script and supporting files `Dockerfile`, `requirements.txt`, and `docker-compose-data-dashboard.yaml`.
  - **Main Script (`data_dashboard.py`)**:
    - **What It Does**: Creates an interactive Streamlit dashboard, loading a CSV file to display split sizes, class distribution, annotator agreement, data drift analysis, bounding box heatmaps, simulator feedback from a feedback directory, and optional sample images with bounding boxes from an image directory.
  - **Supporting Files**:
    - **Dockerfile**: Builds the Docker image, installs dependencies, and configures Streamlit to run on port `8501`.
    - **requirements.txt**: Lists Python dependencies required for the dashboard.
    - **docker-compose-data-dashboard.yaml**: Configures the dashboard service, maps a port, and mounts the CSV file, feedback directory, log file, and optional image directory.
- **Description**: The interactive Streamlit dashboard visualizes data quality and feedback for our radiologist:
  - Displays split sizes (`training: 11500`, `validation: 750`, etc.), reflecting data distribution.
  - Shows simulator feedback with predicted `class_id`, `confidence`, and `bbox`.
  - Optionally displays re-training data.
- **Customer Insight**: The radiologist can assess data quality (e.g., sample distribution) and review simulator predictions to validate model performance before re-training.
- **Used By**: Data Dashboard to provide visual insights and feedback analysis.

### Online Data
- **Implementation**: Managed by the `data_simulation.py` script and the `docker-compose-data-simulation.yaml` configuration file.
  - **Main Script (`data_simulation.py`)**:
    - **What It Does**: Simulates online data by downloading `test` split images from Swift, mocking inference with random predictions, and generating feedback JSONs in a feedback directory using a configurable worker pattern (e.g., `[1,2,3,5,3,2,1]` workers).
  - **Configuration File (`docker-compose-data-simulation.yaml`)**:
    - **What It Does**: Configures the simulator service, mounts feedback and log directories, sets environment variables for load patterns and timeouts, installs dependencies, and runs the simulation script.
- **Description**: The online data simulator processes the `test` split (~750 images) from `chi_tacc:object-persist-project19/organized/test/images/`:
  - Downloads images using `rclone`.
  - Simulates inference with a configurable `LOAD_PATTERN` (e.g., `[1,2,3,5,3,2,1]` workers, 60s stages).
  - Generates feedback (`/mnt/mydata/model-checkpoints/feedback/<image_id>.json`) with `class_id`, `confidence`, `bbox`, and `timestamp`.
- **Customer Relevance**: Mimics real-time X-ray analysis in a hospital, providing feedback for model evaluation and re-training.
- **Used By**: Online Data to simulate real-time processing and generate feedback.

### Follow these steps to run the data components:


1. **Run Offline ETL Pipeline**:
   ```bash
   cd deployment/docker
   docker-compose -f docker-compose-etl.yaml up
   ```

2. **Run Online Data Simulator**:
   ```bash
   cd ../../data-pipeline/data-simulation
   docker-compose -f docker-compose-data-simulation.yaml up -d
   ```

3. **Run Production ETL Pipeline**:
   ```bash
   cd ../../deployment/docker
   docker-compose -f docker-compose-production.yaml up
   ```

4. **Run Data Dashboard**:
   ```bash
   cd ../../data-pipeline/streamlit-dashboard
   docker-compose -f docker-compose-data-dashboard.yaml up -d
   ```
