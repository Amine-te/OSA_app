import torch
import cv2
from shared.config import get_device, DEFAULT_CNN_TRANSFORM
from shared.networks.cnn import LightweightCNN
from shared.detection.product_detector import detect_products
import numpy as np

class YOLOCNNPipeline:
    """Complete pipeline combining YOLO detection with CNN classification"""

    def __init__(self, yolo_model_path, cnn_model_path, class_names, confidence_threshold=0.5):
        self.device = get_device()

        # Load YOLO model
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)

        # Load CNN classifier
        self.num_classes = len(class_names)
        self.cnn_model = LightweightCNN(self.num_classes)
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

        # CNN preprocessing
        self.cnn_transform = DEFAULT_CNN_TRANSFORM

    def detect_and_classify(self, image_path):
        """
        Main pipeline: detect objects with YOLO, then classify subclasses with CNN
        """
        # Load image
        import cv2
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = detect_products(image_path, image_rgb, self.yolo_model,
                                     self.confidence_threshold, self.cnn_transform,
                                     self.cnn_model, self.class_names, self.device)

        return detections, image_rgb

    def visualize_results(self, image, detections, save_path=None):
        """Visualize detection and classification results"""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        class_color_map = {class_name: colors[i] for i, class_name in enumerate(self.class_names)}

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            subclass = detection['subclass']
            yolo_conf = detection['yolo_confidence']
            subclass_conf = detection['subclass_confidence']

            import matplotlib.patches as patches
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color=class_color_map[subclass], linewidth=2)
            ax.add_patch(rect)

            # Add label
            label = f'{subclass}\nYOLO: {yolo_conf:.2f}\nCNN: {subclass_conf:.2f}'
            ax.text(x1, y1-10, label, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=class_color_map[subclass], alpha=0.7))

        ax.set_title('YOLO Detection + CNN Classification Results')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()
