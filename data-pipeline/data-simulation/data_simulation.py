import os
import time
import base64
import logging
import requests
import glob
import random
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import subprocess

# --- Configuration ---
TRITON_URL = os.environ.get("TRITON_URL", "http://129.114.25.124:8000")  
SWIFT_CONTAINER = os.environ.get("RCLONE_CONTAINER", "object-persist-project19")
TEST_IMAGES_PATH = "/app/test_images"
FEEDBACK_DIR = "/app/feedback"
LOG_FILE = "/app/logs/online_data.log"
LOAD_PATTERN = [int(x) for x in os.environ.get("LOAD_PATTERN", "1,2,3,5,3,2,1").split(",")]
DELAY_BETWEEN_STEPS = int(os.environ.get("DELAY_BETWEEN_STEPS", "60"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Ensure directories ---
os.makedirs(TEST_IMAGES_PATH, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# --- Image utilities ---
def load_and_encode_image(image_path):
    """Encode image to base64 for inference"""
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded
    except Exception as e:
        logger.warning(f"Could not encode image {image_path}: {e}")
        return None

# --- Download test images ---
def download_test_images():
    """Download test split images from Swift"""
    logger.info("Downloading test images from Swift...")
    try:
        subprocess.run([
            "rclone", "sync",
            f"chi_tacc:{SWIFT_CONTAINER}/organized/test/images/",
            TEST_IMAGES_PATH,
            "--progress"
        ], check=True)
        image_files = glob.glob(f"{TEST_IMAGES_PATH}/*.png")
        logger.info(f"Downloaded {len(image_files)} test images")
        return image_files
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download images: {e}")
        return []

# --- Inference ---
def send_inference_request(image_path):
    """Send image to Triton for inference (mock if Triton unavailable)"""
    image_id = os.path.basename(image_path).split("_")[0]
    encoded_str = load_and_encode_image(image_path)
    if not encoded_str:
        return False, None, image_id
    
    # Mock inference (replace with Triton client)
    try:
        # Random predictions for 15 classes (0-14, including 'No finding')
        class_id = random.randint(0, 14)
        confidence = round(random.uniform(0.5, 1.0), 4)
        bbox = [
            round(random.uniform(0.1, 0.9), 4),  # x_center
            round(random.uniform(0.1, 0.9), 4),  # y_center
            round(random.uniform(0.05, 0.2), 4), # width
            round(random.uniform(0.05, 0.2), 4)  # height
        ]
        
        # Uncomment for Triton integration
        """
        payload = {"image": encoded_str}
        resp = requests.post(f"{TRITON_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        class_id = result.get("class_id")
        confidence = result.get("confidence")
        bbox = result.get("bbox")
        """
        
        logger.info(f"Inference for {image_id}: class_id={class_id}, confidence={confidence}, bbox={bbox}")
        return True, {
            "image_id": image_id,
            "class_id": class_id,
            "confidence": confidence,
            "bbox": bbox,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, image_id
    except Exception as e:
        logger.error(f"Error during inference for {image_id}: {e}")
        return False, None, image_id

# --- Continuous request worker ---
def process_images(image_files, duration_sec):
    """Process images for the specified duration"""
    start = time.time()
    successes, failures = 0, 0
    processed = []
    
    while time.time() - start < duration_sec and image_files:
        img_path = random.choice(image_files)
        ok, result, image_id = send_inference_request(img_path)
        if ok:
            successes += 1
            feedback_path = os.path.join(FEEDBACK_DIR, f"{image_id}.json")
            with open(feedback_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved feedback: {feedback_path}")
            processed.append(img_path)
        else:
            failures += 1
        time.sleep(0.1)  # Avoid overwhelming the server
    
    return successes, failures, processed

# --- Load stage runner ---
def run_load_stage(image_files, concurrent_workers, duration_sec):
    """Run a load stage with concurrent workers"""
    logger.info(f"Starting stage: {concurrent_workers} workers for {duration_sec} seconds")
    with ThreadPoolExecutor(max_workers=concurrent_workers) as pool:
        futures = [pool.submit(process_images, image_files, duration_sec) for _ in range(concurrent_workers)]
        total_success, total_failure = 0, 0
        all_processed = []
        for f in futures:
            s, f_, p = f.result()
            total_success += s
            total_failure += f_
            all_processed.extend(p)
    logger.info(f"Stage done: {total_success} successes, {total_failure} failures")
    return total_success, total_failure, all_processed

# --- Main runner ---
def main():
    """Run the online data simulator"""
    logger.info("Starting online data simulator...")
    
    # Download test images
    image_files = download_test_images()
    if not image_files:
        logger.error("No test images available. Exiting.")
        return
    
    logger.info(f"Loaded {len(image_files)} test images")
    total_success, total_failure = 0, 0
    remaining_images = image_files.copy()
    
    # Run load pattern stages
    for i, workers in enumerate(LOAD_PATTERN, 1):
        if not remaining_images:
            logger.info("No more images to process.")
            break
        s, f, processed = run_load_stage(remaining_images, workers, DELAY_BETWEEN_STEPS)
        total_success += s
        total_failure += f
        remaining_images = [img for img in remaining_images if img not in processed]
        logger.info(f"Total so far: {total_success} successes, {total_failure} failures")
        logger.info(f"Remaining images: {len(remaining_images)}")
    
    logger.info("Simulation complete.")
    logger.info(f"Final stats: {total_success} successes, {total_failure} failures")

if __name__ == "__main__":
    logger.info("Waiting 10s for inference server to be ready...")
    time.sleep(10)
    main()