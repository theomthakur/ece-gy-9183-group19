import requests
import base64
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import numpy as np

def encode_image(image_path):
    """Encode an image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def visualize_detections(image_path, detections):
    """Visualize bounding boxes on the image"""
    # Load the image
    img = Image.open(image_path)
    img_np = np.array(img)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_np)
    
    # Define colors for different classes (add more if needed)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    # Draw bounding boxes
    for detection in detections:
        x_min = detection['x_min']
        y_min = detection['y_min']
        width = detection['x_max'] - x_min
        height = detection['y_max'] - y_min
        
        # Create rectangle patch
        color_idx = detection['class_id'] % len(colors)
        rect = patches.Rectangle(
            (x_min, y_min), width, height, 
            linewidth=2, edgecolor=colors[color_idx], facecolor='none'
        )
        
        # Add rectangle to the image
        ax.add_patch(rect)
        
        # Add label
        label = f"{detection['class_name']} ({detection['confidence']:.2f})"
        ax.text(
            x_min, y_min-5, label, 
            bbox=dict(facecolor=colors[color_idx], alpha=0.5),
            fontsize=10, color='white'
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("detection_result.png", bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test Chest X-Ray Detection API')
    parser.add_argument('--image', required=True, help='Path to X-ray image file')
    parser.add_argument('--url', default='http://localhost:8080/predict', help='API endpoint URL')
    args = parser.parse_args()
    
    # Encode the image
    image_base64 = encode_image(args.image)
    
    # Prepare the payload
    payload = {
        "image": image_base64
    }
    
    # Make the API request
    try:
        response = requests.post(args.url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Process the response
        result = response.json()
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            detections = result["detections"]
            processing_time = result["processing_time"]
            
            print(f"Processing time: {processing_time:.4f} seconds")
            print(f"Found {len(detections)} detections:")
            
            for i, detection in enumerate(detections):
                print(f"Detection {i+1}:")
                print(f"  Class: {detection['class_name']} (ID: {detection['class_id']})")
                print(f"  Confidence: {detection['confidence']:.4f}")
                print(f"  Bounding Box: ({detection['x_min']:.1f}, {detection['y_min']:.1f}) to ({detection['x_max']:.1f}, {detection['y_max']:.1f})")
            
            # Visualize the results
            visualize_detections(args.image, detections)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()