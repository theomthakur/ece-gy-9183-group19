import os
import time
import random
import uuid
import subprocess
import numpy as np
from PIL import Image
import io
import json
import logging
import requests
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "http://129.114.26.168:8000")
SWIFT_CONTAINER = os.environ.get("SWIFT_CONTAINER", "object-persist-project19")
TEST_IMAGES_PATH = "organized/test/images/"
LOG_DIR = "/mnt/model-checkpoints/logs"
FEEDBACK_DIR = "/mnt/model-checkpoints/feedback"
FEEDBACK_SWIFT_PATH = "feedback/"
MODEL_NAME = "chest_xray_detector"
LOAD_PATTERN = [int(x) for x in os.environ.get("LOAD_PATTERN", "1,2,3,5,3,2,1").split(",")]
DELAY_BETWEEN_STEPS = int(os.environ.get("DELAY_BETWEEN_STEPS", "60"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))

# --- Logging setup ---
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
except Exception as e:
    print(f"Error creating directories {LOG_DIR} or {FEEDBACK_DIR}: {e}")
    exit(1)

logging.basicConfig(
    filename=f"{LOG_DIR}/online_data.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("online_data_simulator")
logger.info("Initializing online data simulator")

# --- Image utilities ---
def fetch_image(image_key):
    """Fetch image from Chameleon Swift using rclone"""
    try:
        local_path = f"/tmp/{os.path.basename(image_key)}"
        subprocess.run(
            ["rclone", "copyto", f"chi_tacc:{SWIFT_CONTAINER}/{image_key}", local_path],
            check=True, capture_output=True
        )
        with open(local_path, "rb") as f:
            image_data = f.read()
        os.remove(local_path)
        logger.info(f"Fetched image {image_key}")
        return image_data
    except Exception as e:
        logger.error(f"Failed to fetch image {image_key}: {e}")
        return None

def preprocess_image(image_data):
    """Preprocess image for Triton inference (resize to 416x416, normalize)"""
    try:
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img.verify()
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((416, 416), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        if random.random() < 0.05:
            noise = np.random.normal(0, 0.01, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            logger.info("Applied Gaussian noise to image")
        if random.random() < 0.01:
            raise ValueError("Simulated corrupted image")
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        temp_path = f"/tmp/{uuid.uuid4()}.png"
        img.save(temp_path)
        logger.info(f"Preprocessed image to {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

# --- Request sending ---
def simulate_hospital_metadata():
    """Generate realistic hospital metadata"""
    metadata = {
        "patient_id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hospital_id": random.choice(["HOSP001", "HOSP002", "HOSP003", "HOSP004", "HOSP005"]),
        "scan_type": random.choice(["frontal", "lateral"])
    }
    logger.info(f"Generated metadata: {metadata}")
    return metadata

def send_request(image_key):
    """Send image to Triton server for inference"""
    image_data = fetch_image(image_key)
    if image_data is None:
        return False, None, None
    temp_path = preprocess_image(image_data)
    if temp_path is None:
        return False, None, None
    metadata = simulate_hospital_metadata()
    try:
        model = YOLO(f"{TRITON_SERVER_URL}/{MODEL_NAME}", task="detect")
        results = model(temp_path)
        detections = [
            {
                "class": results.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "box": box.xyxy[0].tolist()
            } for box in results[0].boxes
        ]
        result = {"metadata": metadata, "detections": detections}
        image_id = os.path.basename(image_key).split(".")[0]
        store_feedback(image_id, metadata, result)
        logger.info(f"Inference result for {image_key}: {result}")
        return True, result, image_id
    except Exception as e:
        logger.error(f"Triton inference failed for {image_key}: {e}")
        return False, None, None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Continuous request worker ---
def send_continuous_requests(image_keys, duration_sec):
    """Send requests continuously for the specified duration"""
    start = time.time()
    successes, failures = 0, 0
    while time.time() - start < duration_sec:
        image_key = random.choice(image_keys)
        ok, result, image_id = send_request(image_key)
        if ok:
            successes += 1
        else:
            failures += 1
        time.sleep(random.uniform(0.5, 2))
    logger.info(f"Worker completed: {successes} successes, {failures} failures")
    return successes, failures

# --- Load stage runner ---
def run_load_stage(image_keys, concurrent_workers, duration_sec):
    """Run a load stage with specified concurrent workers"""
    logger.info(f"Starting stage: {concurrent_workers} workers for {duration_sec} seconds")
    with ThreadPoolExecutor(max_workers=concurrent_workers) as pool:
        futures = [pool.submit(send_continuous_requests, image_keys, duration_sec) for _ in range(concurrent_workers)]
        total_success, total_failure = 0, 0
        for f in futures:
            s, f_ = f.result()
            total_success += s
            total_failure += f_
    logger.info(f"Stage done: {total_success} successes, {total_failure} failures")
    return total_success, total_failure

# --- Feedback storage ---
def store_feedback(image_id, metadata, result):
    """Store inference result and feedback in Swift and Cinder volume"""
    try:
        feedback_data = {
            "image_id": image_id,
            "metadata": metadata,
            "prediction": result,
            "needs_review": random.random() < 0.1
        }
        feedback_file = f"{FEEDBACK_DIR}/{image_id}_feedback.json"
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)
        feedback_key = f"{FEEDBACK_SWIFT_PATH}{image_id}_feedback.json"
        subprocess.run(
            ["rclone", "copyto", feedback_file, f"chi_tacc:{SWIFT_CONTAINER}/{feedback_key}"],
            check=True, capture_output=True
        )
        logger.info(f"Stored feedback for {image_id}")
    except Exception as e:
        logger.error(f"Failed to store feedback for {image_id}: {e}")

# --- Triton server check ---
def check_triton_status():
    """Check if Triton server is running and model is ready"""
    try:
        response = requests.get(f"{TRITON_SERVER_URL}/v2/health/ready", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            model_response = requests.get(f"{TRITON_SERVER_URL}/v2/models/{MODEL_NAME}", timeout=REQUEST_TIMEOUT)
            if model_response.status_code == 200:
                logger.info("Triton server and model are ready")
                return True
            else:
                logger.error("Model 'chest_xray_detector' not ready")
                return False
        else:
            logger.error("Triton server not ready")
            return False
    except Exception as e:
        logger.error(f"Error checking Triton server: {e}")
        return False

# --- List test images ---
def list_test_images():
    """List test images in Swift container"""
    try:
        result = subprocess.run(
            ["rclone", "ls", f"chi_tacc:{SWIFT_CONTAINER}/{TEST_IMAGES_PATH}"],
            check=True, capture_output=True, text=True
        )
        image_keys = [line.split()[1] for line in result.stdout.splitlines() if line.endswith(".png")]
        logger.info(f"Found {len(image_keys)} test images in Swift")
        return image_keys
    except Exception as e:
        logger.error(f"Failed to list test images: {e}")
        return []

# --- Main runner ---
def main():
    """Run the online data simulator with load pattern"""
    logger.info("Starting online data simulator")
    logger.info("Waiting 10s for Triton server to be ready...")
    time.sleep(10)
    if not check_triton_status():
        logger.error("Aborting simulation: Triton server not available")
        return
    image_keys = list_test_images()
    if not image_keys:
        logger.error("Aborting simulation: No test images found in Swift")
        return
    logger.info(f"Loaded {len(image_keys)} test images from Swift")
    total_success, total_failure = 0, 0
    for load in LOAD_PATTERN:
        s, f = run_load_stage(image_keys, load, DELAY_BETWEEN_STEPS)
        total_success += s
        total_failure += f
        logger.info(f"Total so far: {total_success} successes, {total_failure} failures")
    logger.info("Simulation complete.")

if __name__ == "__main__":
    main()