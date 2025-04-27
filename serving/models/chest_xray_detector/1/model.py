import os
import json
import numpy as np
import base64
import io
import time
import torch
from PIL import Image
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """Initialize the model. This will be called during model loading.

        Args:
            args (dict): Both keys and values are strings. The dictionary contains:
                * model_config: A JSON string containing the model configuration
                * model_instance_kind: A string containing model instance kind
                * model_instance_device_id: A string containing model instance device ID
                * model_repository: Model repository path
                * model_version: Model version
                * model_name: Model name
        """
        # Parse the model configuration
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get the model instance kind and device ID
        instance_kind = args.get('model_instance_kind', 'CPU').upper()
        instance_device_id = args.get('model_instance_device_id', '0')
        
        # Set the device based on instance kind
        if instance_kind == 'GPU':
            self.device = f'cuda:{instance_device_id}'
        else:
            self.device = 'cpu'
            
        print(f"Using device: {self.device}")
        
        # Load the YOLO model
        self.load_model()

    def load_model(self):
        """Load the YOLO model"""
        try:
            # Get model directory path
            model_dir = os.path.dirname(__file__)
            model_path = os.path.join(model_dir, "yolov11n.pt")
            
            # Import here to avoid import issues
            from ultralytics import YOLO
            
            # Load the model and move to the appropriate device
            self.model = YOLO(model_path)
            
            # Set device via model.to() if needed
            if hasattr(self.model, 'to') and callable(getattr(self.model, 'to')):
                self.model.to(self.device)
                
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def execute(self, requests):
        """
        This function is called when an inference request is made
        for this model.
        """
        responses = []
        
        # Process each request in the batch
        for request in requests:
            # Get the input image
            input_image = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            conf_threshold = pb_utils.get_input_tensor_by_name(request, "CONFIDENCE_THRESHOLD")
            iou_threshold = pb_utils.get_input_tensor_by_name(request, "IOU_THRESHOLD")
            
            # Extract values
            image_data = input_image.as_numpy()[0][0]  # Assuming shape [1, 1]
            conf_threshold_value = float(conf_threshold.as_numpy()[0][0])
            iou_threshold_value = float(iou_threshold.as_numpy()[0][0])
            
            # Decode and process the image
            try:
                detection_result = self.process_image(
                    image_data, 
                    conf_threshold_value,
                    iou_threshold_value
                )
                
                # Convert the detection result to JSON string
                result_json = json.dumps(detection_result["detections"])
                inference_time = detection_result["inference_time"]
                
                # Create output tensors
                detections_tensor = pb_utils.Tensor("DETECTIONS", 
                                                   np.array([[result_json]], dtype=np.object_))
                time_tensor = pb_utils.Tensor("INFERENCE_TIME", 
                                             np.array([[inference_time]], dtype=np.float32))
                
                # Create InferenceResponse
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[detections_tensor, time_tensor]
                )
                responses.append(inference_response)
                
            except Exception as e:
                # Return error response
                error = str(e)
                error_tensor = pb_utils.Tensor("DETECTIONS", 
                                              np.array([[json.dumps([])]], dtype=np.object_))
                time_tensor = pb_utils.Tensor("INFERENCE_TIME", 
                                             np.array([[0.0]], dtype=np.float32))
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[error_tensor, time_tensor]
                )
                responses.append(inference_response)
                print(f"Error processing request: {error}")
        
        # Return all responses
        return responses

    def process_image(self, base64_image, confidence_threshold=0.25, iou_threshold=0.45):
        """
        Process image with YOLO model and return detections
        """
        try:
            # Decode base64 image
            if isinstance(base64_image, bytes):
                base64_image = base64_image.decode("utf-8")
                
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Measure inference time
            start_time = time.time()
            
            # Run inference using ultralytics YOLO model
            results = self.model.predict(
                source=np.array(image),
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Process results
            detections_list = []
            
            # Get the first result (assuming single image input)
            result = results[0]
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Extract boxes, confidences, and class IDs
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates, normalized to [0,1]
                box = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2] format
                
                # Normalize coordinates to [0,1]
                normalized_box = [
                    box[0] / img_width,
                    box[1] / img_height,
                    box[2] / img_width,
                    box[3] / img_height
                ]
                
                # Get class ID and confidence
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                
                # Get class name
                class_name = result.names[class_id]
                
                # Add detection to list
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": normalized_box
                }
                detections_list.append(detection)
            
            return {
                "detections": detections_list,
                "inference_time": inference_time,
                "model_name": "yolov11n"
            }
        
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

    def finalize(self):
        """
        This function is called when the model is being unloaded.
        """
        print("Cleaning up...")