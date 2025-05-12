## Unit 8: Data Pipeline

### Persistent Storage
We use persistent storage on Chameleon to manage data effectively:

1. **Swift Object Store** (`chi_tacc:object-persist-project19`):
   - **Purpose**: Stores the organized VinBigData dataset for training, evaluation, and production, as well as re-training data.

2. **Block Storage Volume** (`/mnt/mydata` on `vdb`):
   - **Purpose**: Stores simulator feedback and logs generated during online data processing.

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

### Data Pipeline
- **Offline Pipeline**:
  - **Implementation**: See [ece-gy-9183-group19/deployment/docker/docker-compose-etl.yaml](ece-gy-9183-group19/deployment/docker/docker-compose-etl.yaml).
  - **Steps**:
    1. **Extract**: Downloads VinBigData from Kaggle using `kaggle datasets download`.
    2. **Transform**: Converts to YOLO format, splits into `training`, `validation`, `test`, `staging`, `canary`, and `production` sets (scaled to ~15,000 unique images). Avoids data leakage by shuffling unique `image_id`s before splitting.
    3. **Load**: Uploads to `chi_tacc:object-persist-project19/organized/` using `rclone`.
  - **Pre-processing**: Images are pre-resized to 1024x1024 (in the dataset), and bounding boxes are normalized to YOLO format.

- **Production Pipeline (Closing the Loop)**:
  - **Implementation**: See [deployment/docker/docker-compose-production.yaml](deployment/docker/docker-compose-production.yaml).
  - **Steps**:
    1. **Extract**: Collects feedback JSONs from `/mnt/mydata/model-checkpoints/feedback/`.
    2. **Transform**: Filters high-confidence predictions (`confidence > 0.9`) into `retrain.csv`.
    3. **Load**: Uploads to `chi_tacc:object-persist-project19/organized/retrain/`.
  - **Purpose**: Prepares data for model re-training based on production feedback, aligning with Lab 7’s “closing the loop” concept.

### Data Dashboard
- **Implementation**: See [data-pipeline/streamlit-dashboard/data_dashboard.py](data-pipeline/streamlit-dashboard/data_dashboard.py) and supporting files [Dockerfile](data-pipeline/streamlit-dashboard/Dockerfile), [requirements.txt](data-pipeline/streamlit-dashboard/requirements.txt), and [docker-compose-data-dashboard.yaml](data-pipeline/streamlit-dashboard/docker-compose-data-dashboard.yaml).
- **Description**: The interactive Streamlit dashboard visualizes data quality and feedback for our radiologist:
  - Displays split sizes (`training: 11500`, `validation: 750`, etc.), reflecting data distribution.
  - Shows simulator feedback (`/app/feedback/<image_id>.json`) with predicted `class_id`, `confidence`, and `bbox`.
  - Optionally displays re-training data (`/app/data/organized_prod/retrain.csv`).
- **Customer Insight**: The radiologist can assess data quality (e.g., sample distribution) and review simulator predictions to validate model performance before re-training.

### Online Data
- **Implementation**: See [data-pipeline/data-simulation/data_simulation.py](data-pipeline/data-simulation/data_simulation.py) and [docker-compose-data-simulation.yaml](data-pipeline/data-simulation/docker-compose-data-simulation.yaml).
- **Description**: The online data simulator processes the `test` split (~750 images) from `chi_tacc:object-persist-project19/organized/test/images/`:
  - Downloads images using `rclone`.
  - Simulates inference with a configurable `LOAD_PATTERN` (e.g., `[1,2,3,5,3,2,1]` workers, 60s stages).
  - Generates feedback (`/mnt/mydata/model-checkpoints/feedback/<image_id>.json`) with `class_id`, `confidence`, `bbox`, and `timestamp`.
- **Customer Relevance**: Mimics real-time X-ray analysis in a hospital, providing feedback for model evaluation and re-training.

### Instructions for Running on Chameleon
Follow these steps to deploy and run the data components on Chameleon:

1. **Prerequisites**:
   - Provision a Chameleon VM (`node1-project19`) at CHI@TACC with a floating IP.
   - Configure `~/.config/rclone/rclone.conf` with `chi_tacc` for Swift access.
   - Upload your SSH key (`~/.ssh/<your-key>.pem`) to Chameleon.

2. **Clone Repository**:
   ```bash
   ssh cc@<floating-ip> -i ~/.ssh/<your-key>.pem
   git clone https://github.com/theomthakur/ece-gy-9183-group19.git
   cd ece-gy-9183-group19
   ```

3. **Set Up Storage**:
   - Ensure `/mnt/mydata` is mounted (as per `lsblk`).
   - Verify Swift access: `rclone ls chi_tacc:object-persist-project19/`.

4. **Run Offline ETL Pipeline**:
   ```bash
   cd deployment/docker
   docker-compose -f docker-compose-etl.yaml up
   ```
   - Verifies: `rclone ls chi_tacc:object-persist-project19/organized/`.

5. **Run Online Data Simulator**:
   ```bash
   cd ../../data-pipeline/data-simulation
   docker-compose -f docker-compose-data-simulation.yaml up -d
   ```
   - Verifies: `ls -l /mnt/mydata/model-checkpoints/feedback/`.

6. **Run Production ETL Pipeline**:
   ```bash
   cd ../../deployment/docker
   docker-compose -f docker-compose-production.yaml up
   ```
   - Verifies: `rclone ls chi_tacc:object-persist-project19/organized/retrain/`.

7. **Run Data Dashboard**:
   ```bash
   cd ../../data-pipeline/streamlit-dashboard
   docker-compose -f docker-compose-data-dashboard.yaml up -d
   ```
