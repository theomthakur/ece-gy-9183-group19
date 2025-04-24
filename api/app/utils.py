import numpy as np
import cv2

def preprocess_image(image, input_size=(640, 640)):
    """
    Preprocess image for YOLOv11-L model
    
    Args:
        image: RGB image
        input_size: Model input size (width, height)
        
    Returns:
        Preprocessed image ready for model inference
    """
    # Resize image
    resized = cv2.resize(image, input_size)
    
    # Convert to float32 and normalize to [0, 1]
    img_norm = resized.astype(np.float32) / 255.0
    
    # Expand dimensions to create batch of size 1
    img_input = np.expand_dims(img_norm, axis=0)
    
    return img_input

def postprocess_predictions(outputs, original_shape):
    """
    Process YOLOv11-L outputs to get bounding boxes, scores, and class IDs
    
    Args:
        outputs: Model outputs
        original_shape: Original image shape (height, width)
        
    Returns:
        Tuple of (boxes, scores, class_ids)
    """
    # Assuming outputs[0] contains the predictions with shape [batch, num_boxes, num_classes+5]
    predictions = outputs[0][0]  # Get predictions for batch 0
    
    # Extract boxes, scores, and class ids
    boxes = []
    scores = []
    class_ids = []
    
    # Filter predictions based on confidence threshold
    confidence_threshold = 0.25
    
    # YOLOv11-L output format: [x, y, w, h, confidence, class_1, class_2, ..., class_n]
    for prediction in predictions:
        confidence = prediction[4]
        
        if confidence >= confidence_threshold:
            # Get class with highest score
            class_scores = prediction[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # Skip if class score is below threshold
            if class_score < confidence_threshold:
                continue
            
            # Convert predictions to bounding box coordinates
            x, y, w, h = prediction[0:4]
            
            # Normalized coordinates to absolute coordinates
            img_height, img_width = original_shape
            model_input_size = 640  # YOLOv11-L input size
            
            # Convert from center_x, center_y, width, height to x1, y1, x2, y2
            x1 = max(0, (x - w/2) * img_width / model_input_size)
            y1 = max(0, (y - h/2) * img_height / model_input_size)
            x2 = min(img_width, (x + w/2) * img_width / model_input_size)
            y2 = min(img_height, (y + h/2) * img_height / model_input_size)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(class_score)
            class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, 0.45)
    
    if len(indices) > 0:
        # Convert indices format based on OpenCV version
        if isinstance(indices, list):
            selected_indices = indices
        else:
            selected_indices = indices.flatten()
        
        return [boxes[i] for i in selected_indices], [scores[i] for i in selected_indices], [class_ids[i] for i in selected_indices]
    else:
        return [], [], []