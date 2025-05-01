import os
import streamlit as st
import tempfile
from pathlib import Path
from ultralytics import YOLO
import time
from PIL import Image

# Configuration from environment variables
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "triton_server:8000")
MODEL_NAME = os.getenv("CHEST_XRAY_MODEL_NAME", "chest_xray_detector")

st.title("Chest X-Ray Detection using YOLOV11")

# Server URL input (can use environment variable)
server_url = st.text_input("Triton Server URL", value=TRITON_SERVER_URL)
model_name = st.text_input("Model Name", value=MODEL_NAME)

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    # Process image when button is clicked
    if st.button("Detect"):
        try:
            st.info(f"Connecting to Triton Server at: http://{server_url}/{model_name}")
            
            # Load the Triton Server model following the documentation example
            model = YOLO(f"http://{server_url}/{model_name}", task="detect")
            
            # Run inference on the server
            start_time = time.time()
            results = model(temp_path)
            inference_time = time.time() - start_time
            
            # Display results
            st.success(f"Detection completed in {inference_time:.4f} seconds")
            
            # Display the result image
            for result in results:
                # Using the built-in plotting from YOLO results
                result_img = result.plot()
                st.image(result_img, caption="Detection Results")
                
                # Show the detection details
                boxes = result.boxes
                if len(boxes) > 0:
                    st.write(f"Found {len(boxes)} detections:")
                    for i, box in enumerate(boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = result.names[cls]
                        st.write(f"- Detection {i+1}: {label} (Confidence: {conf:.3f})")
                else:
                    st.write("No detections found.")
        
        except Exception as e:
            import traceback
            st.error(f"Error: {e} \n {traceback.format_exc()}")
            st.warning("Make sure the Triton server is running and the model is loaded correctly.")
    
    # Clean up the temporary file
    os.unlink(temp_path)

# Footer with information
st.markdown("---")
st.caption("Using YOLOV11 model deployed on NVIDIA Triton Inference Server")