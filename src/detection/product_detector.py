from src.detection.classifier import classify_crop

def detect_products(image_source, image_rgb, yolo_model, confidence_threshold, cnn_transform, cnn_model, class_names, device):
    """Detect and classify products using YOLO + CNN"""
    yolo_results = yolo_model(image_source, conf=confidence_threshold)

    detections = []

    for result in yolo_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()

                # Crop the detected region
                cropped_image = image_rgb[y1:y2, x1:x2]

                if cropped_image.size > 0:  # Check if crop is valid
                    # Classify the cropped image
                    subclass_label, subclass_confidence = classify_crop(
                        cropped_image, cnn_transform, cnn_model, class_names, device
                    )

                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'area': (x2 - x1) * (y2 - y1),
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'yolo_confidence': confidence,
                        'subclass': subclass_label,
                        'subclass_confidence': subclass_confidence,
                        'combined_confidence': confidence * subclass_confidence
                    }
                    detections.append(detection)

    return detections
