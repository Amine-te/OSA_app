import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import get_device, DEFAULT_ASSIGNMENT_PARAMS, DEFAULT_CNN_TRANSFORM
from src.models.cnn import LightweightCNN
from src.detection.product_detector import detect_products
from src.detection.void_detector import detect_voids
from src.analysis.shelf_patterns import analyze_shelf_patterns
from src.analysis.spatial_context import analyze_spatial_context
from src.analysis.void_assignment import intelligent_void_assignment_with_spatial_context, filter_isolated_voids
from src.reporting.summary import generate_summary, print_detailed_summary
from src.visualization.results_visualizer import visualize_complete_results

class EnhancedRetailPipeline:
    """Complete pipeline combining YOLO detection, CNN classification, and intelligent void area detection"""

    def __init__(self, yolo_model_path, cnn_model_path, void_model_path, class_names,
                 confidence_threshold=0.5, void_confidence_threshold=0.5):
        self.device = get_device()

        # Load YOLO model for product detection
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)

        # Load CNN classifier for product subclasses
        self.num_classes = len(class_names)
        self.cnn_model = LightweightCNN(self.num_classes)
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

        # Load void detection model (assuming it's also a YOLO model)
        self.void_model = YOLO(void_model_path)

        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.void_confidence_threshold = void_confidence_threshold

        # CNN preprocessing
        self.cnn_transform = DEFAULT_CNN_TRANSFORM

        # Color mapping for visualization
        self.product_colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        self.void_color = [1.0, 0.0, 0.0, 0.7]  # Red with transparency

        # Enhanced assignment parameters with spatial context priority
        self.assignment_params = DEFAULT_ASSIGNMENT_PARAMS

    def detect_and_classify_complete(self, image_path):
        """
        Complete pipeline: detect products, classify subclasses, detect voids, and analyze relationships
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: YOLO product detection + CNN classification
        product_detections = detect_products(
            image_path, image_rgb, self.yolo_model, self.confidence_threshold,
            self.cnn_transform, self.cnn_model, self.class_names, self.device
        )

        # Step 2: Void area detection
        void_detections = detect_voids(
            image_path, self.void_model, self.void_confidence_threshold
        )

        # Step 3: Analyze shelf patterns and product clusters
        shelf_analysis = analyze_shelf_patterns(
            product_detections, image_rgb.shape, self.class_names, self.assignment_params
        )

        # Step 4: NEW - Analyze spatial context for each void
        spatial_context_analysis = analyze_spatial_context(
            product_detections, void_detections, self.assignment_params
        )

        # Step 5: Enhanced intelligent void-product assignment with spatial priority
        void_analysis = intelligent_void_assignment_with_spatial_context(
            product_detections, void_detections, shelf_analysis, spatial_context_analysis, image_rgb.shape, self.assignment_params
        )

        # Step 6: Filter out isolated voids (NEW)
        filtered_void_detections, filtered_void_analysis = filter_isolated_voids(
            void_detections, product_detections, void_analysis, self.assignment_params
        )

        # Step 7: Generate comprehensive summary
        summary = generate_summary(
            product_detections, filtered_void_detections, filtered_void_analysis, self.class_names
        )

        return {
            'image': image_rgb,
            'product_detections': product_detections,
            'void_detections': filtered_void_detections,  # Use filtered voids
            'shelf_analysis': shelf_analysis,
            'spatial_context_analysis': spatial_context_analysis,
            'void_analysis': filtered_void_analysis,      # Use filtered analysis
            'summary': summary
        }

    def visualize_complete_results(self, results, save_path=None, figsize=(30, 22)):
        visualize_complete_results(results, self.class_names, self.product_colors, save_path, figsize)

    def print_detailed_summary(self, results):
        print_detailed_summary(results)
