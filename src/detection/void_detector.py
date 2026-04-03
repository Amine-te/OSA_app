def detect_voids(image_path, void_model, void_confidence_threshold):
    """Detect void areas using void detection model"""
    void_results = void_model(image_path, conf=void_confidence_threshold)

    void_detections = []

    for result in void_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()

                void_detection = {
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'area': (x2 - x1) * (y2 - y1),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'confidence': confidence
                }
                void_detections.append(void_detection)

    return void_detections
