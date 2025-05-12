import os
import cv2
import streamlit as st
import tempfile
from pathlib import Path
from ultralytics import YOLO
import time
from PIL import Image
import boto3
from datetime import datetime
from mimetypes import guess_type
import uuid
from concurrent.futures import ThreadPoolExecutor

# Configuration from environment variables
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "triton_server:8000")
MODEL_NAME = os.getenv("CHEST_XRAY_MODEL_NAME", "chest_xray_detector")
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "PROSadmin19")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "PROSadmin19")
BUCKET_NAME = os.getenv("BUCKET_NAME", "production")

# Initialize MinIO client
s3 = boto3.client(
    's3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name='us-east-1'
)

# Thread pool for asynchronous uploads
executor = ThreadPoolExecutor(max_workers=2)

def upload_to_minio(img_path, predictions, confidences, prediction_id):
    """Upload image to MinIO with metadata tags."""
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    predicted_class = predictions[0] if predictions else "Unknown"
    confidence = confidences[0] if confidences else 0.0
    class_dir = f"class_{predicted_class.replace(' ', '_')}"

    _, ext = os.path.splitext(img_path)
    content_type = guess_type(img_path)[0] or 'application/octet-stream'
    s3_key = f"{class_dir}/{prediction_id}{ext}"

    try:
        with open(img_path, 'rb') as f:
            s3.upload_fileobj(
                f,
                BUCKET_NAME,
                s3_key,
                ExtraArgs={'ContentType': content_type}
            )
        s3.put_object_tagging(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Tagging={
                'TagSet': [
                    {'Key': 'predicted_class', 'Value': predicted_class},
                    {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                    {'Key': 'timestamp', 'Value': timestamp}
                ]
            }
        )
        st.write(f"Image uploaded to MinIO with key: {s3_key}")
        return s3_key
    except Exception as e:
        st.error(f"Failed to upload to MinIO: {e}")
        return None

def flag_object(s3_key):
    """Tag an image as flagged in MinIO."""
    try:
        current_tags = s3.get_object_tagging(Bucket=BUCKET_NAME, Key=s3_key)['TagSet']
        tags = {t['Key']: t['Value'] for t in current_tags}
        if "flagged" not in tags:
            tags["flagged"] = "true"
            tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
            s3.put_object_tagging(Bucket=BUCKET_NAME, Key=s3_key, Tagging={'TagSet': tag_set})
            st.success(f"Image {s3_key} flagged successfully.")
        else:
            st.warning(f"Image {s3_key} was already flagged.")
    except Exception as e:
        st.error(f"Failed to flag image {s3_key}: {e}")

st.title("Chest X-Ray Detection using YOLOV11")

# Initialize session state to store detection results and result image
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 's3_key' not in st.session_state:
    st.session_state.s3_key = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'confidences' not in st.session_state:
    st.session_state.confidences = []
if 'result_images' not in st.session_state:
    st.session_state.result_images = []  # Store detection result images

server_url = st.text_input("Triton Server URL", value=TRITON_SERVER_URL)
model_name = st.text_input("Model Name", value=MODEL_NAME)
MINIO_URL = st.text_input("MinIO URL", value=MINIO_URL)
# MINIO_ACCESS_KEY = st.text_input("MinIO Access Key", value=MINIO_ACCESS_KEY)
# MINIO_SECRET_KEY = st.text_input("MinIO Secret Key", value=MINIO_SECRET_KEY)
BUCKET_NAME = st.text_input("Bucket Name", value=BUCKET_NAME)

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name

    if st.button("Detect"):
        try:
            st.info(f"Connecting to Triton Server at: http://{server_url}/{model_name}")
            model = YOLO(f"http://{server_url}/{model_name}", task="detect")
            start_time = time.time()
            results = model(temp_path)
            inference_time = time.time() - start_time

            st.success(f"Detection completed in {inference_time:.4f} seconds")

            predictions = []
            confidences = []
            result_images = []  # Temporary list to store result images
            for result in results:
                result_img = result.plot()
                result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                st.image(result_img_bgr, caption="Detection Results")
                result_images.append(result_img_bgr)  # Store the result image

                boxes = result.boxes
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = result.names[cls]
                        predictions.append(label)
                        confidences.append(conf)
                else:
                    predictions.append("None")
                    confidences.append(0.0)

            prediction_id = str(uuid.uuid4())
            s3_key = upload_to_minio(temp_path, predictions, confidences, prediction_id)

            # Store results in session state
            st.session_state.detection_results = results
            st.session_state.s3_key = s3_key
            st.session_state.predictions = predictions
            st.session_state.confidences = confidences
            st.session_state.result_images = result_images  # Store result images

            # Trigger a rerun to avoid duplicate display
            st.rerun()

        except Exception as e:
            import traceback
            st.error(f"Error: {e} \n {traceback.format_exc()}")
            st.warning("Make sure the Triton server is running and the model is loaded correctly.")
        finally:
            os.unlink(temp_path)

CLASS_TO_DISEASE_MAPPING = {
    'class0': 'Aortic enlargement',
    'class1': 'Atelectasis',
    'class2': 'Calcification',
    'class3': 'Cardiomegaly',
    'class4': 'Consolidation',
    'class5': 'ILD',
    'class6': 'Infiltration',
    'class7': 'Lung Opacity',
    'class8': 'Nodule/Mass',
    'class9': 'Other lesion',
    'class10': 'Pleural effusion',
    'class11': 'Pleural thickening',
    'class12': 'Pneumothorax',
    'class13': 'Pulmonary fibrosis',
    'class14': 'No finding'
}

# Display detection results and result images from session state
if st.session_state.detection_results is not None:
    # Display stored result images
    for idx, result_img in enumerate(st.session_state.result_images):
        st.image(result_img, caption=f"Detection Results {idx+1}")

    # Display detection labels and confidences
    for i, (label, conf) in enumerate(zip(st.session_state.predictions, st.session_state.confidences)):
        st.write(f"- Detection {i+1}: {CLASS_TO_DISEASE_MAPPING.get(label, label)} (Confidence: {conf:.3f})")
    
    # Display a single Flag button if s3_key exists
    if st.session_state.s3_key and st.button("Flag Image", key=f"flag_{st.session_state.s3_key}"):
        flag_object(st.session_state.s3_key)

st.markdown("---")
st.caption("Using YOLOV11 model deployed on NVIDIA Triton Inference Server")