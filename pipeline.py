import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import defaultdict, Counter
import os
import glob
import random
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import streamlit as st
import tempfile
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Force CPU usage
torch.cuda.is_available = lambda: False
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class SubclassDataset(Dataset):
    """Custom dataset for training the CNN classifier on subclasses"""

    def __init__(self, image_paths, labels, transform=None, class_names=None):
        print(f"Initializing dataset with {len(image_paths)} images...")
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or [f"class_{i}" for i in range(len(set(labels)))]

        # Quick validation - don't check every image to avoid hanging
        print("Dataset initialized successfully!")
        print(f"Total images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image and label as fallback
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label

class LightweightCNN(nn.Module):

    """Lightweight CNN classifier optimized for small datasets"""

    def __init__(self, num_classes):
        super(LightweightCNN, self).__init__()

        # Much simpler architecture
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            # Second block
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            # Third block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            # Global Average Pooling instead of large FC layer
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Much smaller classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
class YOLOCNNPipeline:
    """Complete pipeline combining YOLO detection with CNN classification"""

    def __init__(self, yolo_model_path, cnn_model_path, class_names, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load YOLO model
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)

        # Load CNN classifier (updated for LightweightCNN)
        self.num_classes = len(class_names)
        self.cnn_model = LightweightCNN(self.num_classes)
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

        self.class_names = class_names
        self.confidence_threshold = confidence_threshold

        # CNN preprocessing
        self.cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_and_classify(self, image_path):
        """
        Main pipeline: detect objects with YOLO, then classify subclasses with CNN
        """
        # Load image
        import cv2
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO detection
        yolo_results = self.yolo_model(image_path, conf=self.confidence_threshold)

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
                        subclass_label, subclass_confidence = self.classify_crop(cropped_image)

                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'yolo_confidence': confidence,
                            'subclass': subclass_label,
                            'subclass_confidence': subclass_confidence,
                            'combined_confidence': confidence * subclass_confidence
                        }
                        detections.append(detection)

        return detections, image_rgb

    def classify_crop(self, cropped_image):
        """Classify a cropped image using the CNN classifier"""
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cropped_image)

        # Preprocess
        input_tensor = self.cnn_transform(pil_image).unsqueeze(0).to(self.device)

        # Classify
        with torch.no_grad():
            outputs = self.cnn_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()

        return predicted_class, confidence_score

    def visualize_results(self, image, detections, save_path=None):
        """Visualize detection and classification results"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        class_color_map = {class_name: colors[i] for i, class_name in enumerate(self.class_names)}

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            subclass = detection['subclass']
            yolo_conf = detection['yolo_confidence']
            subclass_conf = detection['subclass_confidence']

            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
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

# ENHANCED PIPELINE: YOLO + CNN + VOID DETECTION WITH INTELLIGENT ASSIGNMENT
# Added spatial context priority for void assignment

class EnhancedRetailPipeline:
    """Complete pipeline combining YOLO detection, CNN classification, and intelligent void area detection"""

    def __init__(self, yolo_model_path, cnn_model_path, void_model_path, class_names,
                 confidence_threshold=0.5, void_confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.cnn_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Color mapping for visualization
        self.product_colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        self.void_color = [1.0, 0.0, 0.0, 0.7]  # Red with transparency

        # Enhanced assignment parameters with spatial context priority
        self.assignment_params = {
            'spatial_context_weight': 0.5,     # NEW: Weight for spatial context (left/right neighbors)
            'proximity_weight': 0.25,          # Reduced to make room for spatial context
            'scarcity_weight': 0.15,           # Reduced
            'pattern_weight': 0.1,             # Reduced
            'confidence_weight': 0.05,         # Reduced but still present
            'clustering_eps': 80,              # DBSCAN clustering parameter for grouping products
            'min_cluster_size': 2,             # Minimum products needed to form a cluster
            'max_assignment_distance': 200,    # Maximum distance for valid assignment
            'spatial_context_threshold': 100,  # NEW: Max distance to consider products as neighbors
            'neighbor_alignment_tolerance': 50, # NEW: Y-axis tolerance for horizontal alignment
            # ADD THESE NEW PARAMETERS:
            'isolation_distance_threshold': 100,  # Max distance to nearest product to consider void
            'min_assignment_confidence': 0.3,     # Minimum confidence to keep void assignment
            'high_confidence_threshold': 0.5      # Never filter voids with assignments above this
        }

    def detect_and_classify_complete(self, image_path):
        """
        Complete pipeline: detect products, classify subclasses, detect voids, and analyze relationships
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 1: YOLO product detection + CNN classification
        product_detections = self._detect_products(image_path, image_rgb)

        # Step 2: Void area detection
        void_detections = self._detect_voids(image_path)

        # Step 3: Analyze shelf patterns and product clusters
        shelf_analysis = self._analyze_shelf_patterns(product_detections, image_rgb.shape)

        # Step 4: NEW - Analyze spatial context for each void
        spatial_context_analysis = self._analyze_spatial_context(product_detections, void_detections)

        # Step 5: Enhanced intelligent void-product assignment with spatial priority
        void_analysis = self._intelligent_void_assignment_with_spatial_context(
            product_detections, void_detections, shelf_analysis, spatial_context_analysis, image_rgb.shape
        )

        # Step 6: Filter out isolated voids (NEW)
        filtered_void_detections, filtered_void_analysis = self._filter_isolated_voids(
            void_detections, product_detections, void_analysis
        )

        # Step 7: Generate comprehensive summary (updated step number)
        summary = self._generate_summary(product_detections, filtered_void_detections, filtered_void_analysis)

        return {
            'image': image_rgb,
            'product_detections': product_detections,
            'void_detections': filtered_void_detections,  # Use filtered voids
            'shelf_analysis': shelf_analysis,
            'spatial_context_analysis': spatial_context_analysis,
            'void_analysis': filtered_void_analysis,      # Use filtered analysis
            'summary': summary
        }

    def _analyze_spatial_context(self, product_detections, void_detections):
        """
        NEW: Analyze spatial context - identify products that surround each void
        """
        spatial_context = []

        for void_idx, void in enumerate(void_detections):
            void_center = void['center']
            void_bbox = void['bbox']

            context_info = {
                'void_id': void_idx,
                'left_neighbors': [],
                'right_neighbors': [],
                'top_neighbors': [],
                'bottom_neighbors': [],
                'horizontal_context': None,  # Will be set if same product on left and right
                'vertical_context': None,    # Will be set if same product on top and bottom
                'dominant_context': None     # The strongest spatial context
            }

            # Find products in each direction
            for prod_idx, product in enumerate(product_detections):
                prod_center = product['center']
                prod_bbox = product['bbox']

                # Calculate distances and check alignment
                horizontal_distance = abs(void_center[0] - prod_center[0])
                vertical_distance = abs(void_center[1] - prod_center[1])

                # Check if products are horizontally aligned (same row)
                if vertical_distance <= self.assignment_params['neighbor_alignment_tolerance']:
                    # Product is to the left
                    if prod_center[0] < void_center[0] and horizontal_distance <= self.assignment_params['spatial_context_threshold']:
                        context_info['left_neighbors'].append({
                            'product_id': prod_idx,
                            'product': product,
                            'distance': horizontal_distance,
                            'product_type': product['subclass']
                        })

                    # Product is to the right
                    elif prod_center[0] > void_center[0] and horizontal_distance <= self.assignment_params['spatial_context_threshold']:
                        context_info['right_neighbors'].append({
                            'product_id': prod_idx,
                            'product': product,
                            'distance': horizontal_distance,
                            'product_type': product['subclass']
                        })

                # Check if products are vertically aligned (same column)
                if horizontal_distance <= self.assignment_params['neighbor_alignment_tolerance']:
                    # Product is above
                    if prod_center[1] < void_center[1] and vertical_distance <= self.assignment_params['spatial_context_threshold']:
                        context_info['top_neighbors'].append({
                            'product_id': prod_idx,
                            'product': product,
                            'distance': vertical_distance,
                            'product_type': product['subclass']
                        })

                    # Product is below
                    elif prod_center[1] > void_center[1] and vertical_distance <= self.assignment_params['spatial_context_threshold']:
                        context_info['bottom_neighbors'].append({
                            'product_id': prod_idx,
                            'product': product,
                            'distance': vertical_distance,
                            'product_type': product['subclass']
                        })

            # Sort neighbors by distance (closest first)
            context_info['left_neighbors'].sort(key=lambda x: x['distance'])
            context_info['right_neighbors'].sort(key=lambda x: x['distance'])
            context_info['top_neighbors'].sort(key=lambda x: x['distance'])
            context_info['bottom_neighbors'].sort(key=lambda x: x['distance'])

            # Analyze horizontal context (left and right)
            if context_info['left_neighbors'] and context_info['right_neighbors']:
                closest_left = context_info['left_neighbors'][0]
                closest_right = context_info['right_neighbors'][0]

                # Check if same product type on both sides
                if closest_left['product_type'] == closest_right['product_type']:
                    context_info['horizontal_context'] = {
                        'product_type': closest_left['product_type'],
                        'confidence': 1.0,  # Maximum confidence for direct neighbors
                        'left_distance': closest_left['distance'],
                        'right_distance': closest_right['distance'],
                        'context_strength': 'strong'  # Same product on both sides
                    }
                    context_info['dominant_context'] = context_info['horizontal_context']

            # Analyze vertical context (top and bottom) - only if no strong horizontal context
            if not context_info['horizontal_context'] and context_info['top_neighbors'] and context_info['bottom_neighbors']:
                closest_top = context_info['top_neighbors'][0]
                closest_bottom = context_info['bottom_neighbors'][0]

                if closest_top['product_type'] == closest_bottom['product_type']:
                    context_info['vertical_context'] = {
                        'product_type': closest_top['product_type'],
                        'confidence': 0.9,  # Slightly lower than horizontal
                        'top_distance': closest_top['distance'],
                        'bottom_distance': closest_bottom['distance'],
                        'context_strength': 'strong'
                    }
                    if not context_info['dominant_context']:
                        context_info['dominant_context'] = context_info['vertical_context']

            # Analyze single-sided context (weaker but still valuable)
            if not context_info['dominant_context']:
                single_side_contexts = []

                # Check single left neighbor
                if context_info['left_neighbors']:
                    closest_left = context_info['left_neighbors'][0]
                    single_side_contexts.append({
                        'product_type': closest_left['product_type'],
                        'confidence': 0.6,
                        'direction': 'left',
                        'distance': closest_left['distance'],
                        'context_strength': 'moderate'
                    })

                # Check single right neighbor
                if context_info['right_neighbors']:
                    closest_right = context_info['right_neighbors'][0]
                    single_side_contexts.append({
                        'product_type': closest_right['product_type'],
                        'confidence': 0.6,
                        'direction': 'right',
                        'distance': closest_right['distance'],
                        'context_strength': 'moderate'
                    })

                # Choose the closest single-sided context
                if single_side_contexts:
                    context_info['dominant_context'] = min(single_side_contexts, key=lambda x: x['distance'])

            spatial_context.append(context_info)

        return spatial_context

    def _calculate_border_to_center_distance(self, void_bbox, product_center):
        """Calculate minimum distance from void border to product center"""
        x1, y1, x2, y2 = void_bbox
        px, py = product_center
        
        # Calculate distance from point to rectangle
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        
        return np.sqrt(dx * dx + dy * dy)

    def _filter_isolated_voids(self, void_detections, product_detections, void_analysis):
        """Filter out isolated voids that are too far from any product"""
        if not product_detections:
            return [], []
        
        filtered_voids = []
        filtered_analysis = []
        
        for i, (void, analysis) in enumerate(zip(void_detections, void_analysis)):
            # Never filter voids with high confidence assignments
            if (analysis['final_assignment'] and 
                analysis['final_assignment']['confidence'] >= self.assignment_params['high_confidence_threshold']):
                filtered_voids.append(void)
                filtered_analysis.append(analysis)
                continue
            
            # Calculate minimum border-to-center distance to any product
            min_distance = float('inf')
            for product in product_detections:
                distance = self._calculate_border_to_center_distance(void['bbox'], product['center'])
                min_distance = min(min_distance, distance)
            
            # Keep void if it's close enough to products OR has decent assignment confidence
            keep_void = (
                min_distance <= self.assignment_params['isolation_distance_threshold'] or
                (analysis['final_assignment'] and 
                analysis['final_assignment']['confidence'] >= self.assignment_params['min_assignment_confidence'])
            )
            
            if keep_void:
                filtered_voids.append(void)
                filtered_analysis.append(analysis)
        
        return filtered_voids, filtered_analysis


    def _intelligent_void_assignment_with_spatial_context(self, product_detections, void_detections,
                                                         shelf_analysis, spatial_context_analysis, image_shape):
        """
        Enhanced void assignment that prioritizes spatial context
        """
        void_analysis = []

        if not product_detections or not void_detections:
            return void_analysis

        # Prepare data for assignment
        product_centers = np.array([p['center'] for p in product_detections])
        void_centers = np.array([v['center'] for v in void_detections])

        # Calculate distance matrix between voids and products
        distance_matrix = cdist(void_centers, product_centers, metric='euclidean')

        for void_idx, void in enumerate(void_detections):
            spatial_context = spatial_context_analysis[void_idx]

            void_info = {
                'void_id': void_idx,
                'void_bbox': void['bbox'],
                'void_area': void['area'],
                'spatial_context': spatial_context,
                'assignment_candidates': [],
                'final_assignment': None,
                'assignment_confidence': 0.0,
                'assignment_reasoning': [],
                'estimated_product_count': 0
            }

            # PRIORITY 1: Check for strong spatial context (surrounded by same product)
            if spatial_context['dominant_context'] and spatial_context['dominant_context']['context_strength'] == 'strong':
                dominant_context = spatial_context['dominant_context']

                void_info['final_assignment'] = {
                    'product_type': dominant_context['product_type'],
                    'confidence': dominant_context['confidence'],
                    'assignment_method': 'spatial_context_priority',
                    'primary_factors': ['spatial_context'],
                    'context_type': 'horizontal' if 'left_distance' in dominant_context else 'vertical'
                }

                void_info['estimated_product_count'] = self._estimate_product_count_from_context(
                    void, dominant_context, product_detections
                )

                void_info['assignment_confidence'] = dominant_context['confidence']

                if 'left_distance' in dominant_context:
                    void_info['assignment_reasoning'] = [
                        f"Strong spatial context: {dominant_context['product_type']} products on both left and right",
                        f"Left distance: {dominant_context['left_distance']:.0f}px, Right distance: {dominant_context['right_distance']:.0f}px",
                        "Direct neighbor analysis indicates clear product continuation"
                    ]
                else:
                    void_info['assignment_reasoning'] = [
                        f"Strong spatial context: {dominant_context['product_type']} products above and below",
                        f"Top distance: {dominant_context['top_distance']:.0f}px, Bottom distance: {dominant_context['bottom_distance']:.0f}px",
                        "Vertical alignment indicates clear product continuation"
                    ]

                void_analysis.append(void_info)
                continue

            # PRIORITY 2: Check for moderate spatial context
            if spatial_context['dominant_context'] and spatial_context['dominant_context']['context_strength'] == 'moderate':
                dominant_context = spatial_context['dominant_context']

                # Still assign based on spatial context but with lower confidence
                void_info['final_assignment'] = {
                    'product_type': dominant_context['product_type'],
                    'confidence': dominant_context['confidence'],
                    'assignment_method': 'spatial_context_moderate',
                    'primary_factors': ['spatial_context']
                }

                void_info['estimated_product_count'] = self._estimate_product_count_from_context(
                    void, dominant_context, product_detections
                )

                void_info['assignment_confidence'] = dominant_context['confidence']
                void_info['assignment_reasoning'] = [
                    f"Moderate spatial context: {dominant_context['product_type']} product on {dominant_context['direction']} side",
                    f"Distance: {dominant_context['distance']:.0f}px",
                    "Single-sided neighbor analysis suggests likely product type"
                ]

                void_analysis.append(void_info)
                continue

            # PRIORITY 3: Fall back to original intelligent scoring system
            void_center = void['center']
            void_distances = distance_matrix[void_idx]

            # Find all products within reasonable assignment distance
            nearby_indices = np.where(void_distances <= self.assignment_params['max_assignment_distance'])[0]

            if len(nearby_indices) == 0:
                # No products within reasonable distance - use fallback
                void_info['final_assignment'] = self._fallback_assignment(
                    void, product_detections, shelf_analysis
                )
                void_info['assignment_reasoning'].append("No spatial context or nearby products - used fallback")
                void_analysis.append(void_info)
                continue

            # Analyze each nearby product as a potential assignment candidate
            candidates = []

            for prod_idx in nearby_indices:
                product = product_detections[prod_idx]
                distance = void_distances[prod_idx]
                product_type = product['subclass']

                # Calculate various scoring factors (with updated weights)
                scores = self._calculate_assignment_scores_with_spatial_context(
                    void, product, distance, shelf_analysis, spatial_context, void_idx, prod_idx
                )

                candidate = {
                    'product_id': prod_idx,
                    'product': product,
                    'distance': distance,
                    'product_type': product_type,
                    'scores': scores,
                    'total_score': sum(scores.values()),
                    'reasoning': self._generate_assignment_reasoning(scores, product_type, distance)
                }

                candidates.append(candidate)

            # Sort candidates by total score (higher is better)
            candidates.sort(key=lambda x: x['total_score'], reverse=True)
            void_info['assignment_candidates'] = candidates

            # Select the best assignment
            if candidates:
                best_candidate = candidates[0]
                void_info['final_assignment'] = {
                    'product_type': best_candidate['product_type'],
                    'confidence': min(best_candidate['total_score'], 1.0),
                    'assignment_method': 'intelligent_scoring',
                    'primary_factors': self._identify_primary_factors(best_candidate['scores'])
                }

                # Estimate product count
                void_info['estimated_product_count'] = self._estimate_product_count(
                    void, best_candidate['product']
                )

                # Compile reasoning
                void_info['assignment_reasoning'] = best_candidate['reasoning']
                void_info['assignment_confidence'] = best_candidate['total_score']

                # Add comparison with other candidates if significant
                if len(candidates) > 1:
                    second_best = candidates[1]
                    score_diff = best_candidate['total_score'] - second_best['total_score']
                    if score_diff < 0.2:  # Close competition
                        void_info['assignment_reasoning'].append(
                            f"Close competition with {second_best['product_type']} "
                            f"(score difference: {score_diff:.3f})"
                        )

            void_analysis.append(void_info)

        return void_analysis

    def _estimate_product_count(self, void, product):
        """Estimate how many products could fit in the void area"""
        if product['area'] == 0:
            return 1
        
        area_ratio = void['area'] / product['area']
        
        # Consider both area and dimensional constraints
        void_width, void_height = void['width'], void['height']
        prod_width, prod_height = product['width'], product['height']
        
        # Calculate how many could fit in each dimension
        width_fit = max(1, void_width // prod_width)
        height_fit = max(1, void_height // prod_height)
        
        # Use the more conservative estimate
        dimensional_estimate = min(width_fit * height_fit, area_ratio)
        
        return max(1, round(dimensional_estimate))


    def _estimate_product_count_from_context(self, void, spatial_context, product_detections):
        """
        Estimate product count based on spatial context
        """
        # Find a representative product of the assigned type for size estimation
        product_type = spatial_context['product_type']
        representative_products = [p for p in product_detections if p['subclass'] == product_type]

        if not representative_products:
            return 1

        # Use the average size of products of this type
        avg_area = np.mean([p['area'] for p in representative_products])
        avg_width = np.mean([p['width'] for p in representative_products])
        avg_height = np.mean([p['height'] for p in representative_products])

        if avg_area == 0:
            return 1

        # Calculate estimates based on area and dimensions
        area_ratio = void['area'] / avg_area
        width_fit = max(1, void['width'] // avg_width)
        height_fit = max(1, void['height'] // avg_height)

        # Use the more conservative estimate
        dimensional_estimate = min(width_fit * height_fit, area_ratio)

        return max(1, round(dimensional_estimate))

    def _calculate_assignment_scores_with_spatial_context(self, void, product, distance, shelf_analysis,
                                                         spatial_context, void_idx, prod_idx):
        """
        Calculate assignment scores with spatial context consideration
        """
        scores = {}

        # 1. Spatial Context Score (NEW - highest priority)
        spatial_score = 0.0
        if spatial_context['dominant_context']:
            context = spatial_context['dominant_context']
            if product['subclass'] == context['product_type']:
                if context['context_strength'] == 'strong':
                    spatial_score = 1.0
                elif context['context_strength'] == 'moderate':
                    spatial_score = 0.7
                else:
                    spatial_score = 0.4

        scores['spatial_context'] = spatial_score * self.assignment_params['spatial_context_weight']

        # 2. Proximity Score (closer = better)
        max_distance = self.assignment_params['max_assignment_distance']
        proximity_score = max(0, (max_distance - distance) / max_distance)
        scores['proximity'] = proximity_score * self.assignment_params['proximity_weight']

        # 3. Scarcity Score (less present products get higher priority)
        product_type = product['subclass']
        scarcity_score = shelf_analysis['scarcity_scores'].get(product_type, 0.5)
        scores['scarcity'] = scarcity_score * self.assignment_params['scarcity_weight']

        # 4. Pattern Alignment Score
        pattern_score = self._calculate_pattern_alignment_score(
            void, product, shelf_analysis['spatial_patterns']
        )
        scores['pattern'] = pattern_score * self.assignment_params['pattern_weight']

        # 5. Confidence Score
        confidence_score = product['combined_confidence']
        scores['confidence'] = confidence_score * self.assignment_params['confidence_weight']

        return scores

    def visualize_complete_results(self, results, save_path=None, figsize=(30, 22)):
            """Visualize all detection results with spatial context annotations"""
            fig = plt.figure(figsize=figsize)

            # Create single axes for the main image only
            ax_main = fig.add_subplot(111)
            ax_main.imshow(results['image'])

            class_color_map = {class_name: self.product_colors[i] for i, class_name in enumerate(self.class_names)}

            # Draw product detections
            for i, detection in enumerate(results['product_detections']):
                x1, y1, x2, y2 = detection['bbox']
                subclass = detection['subclass']

                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, color=class_color_map[subclass], linewidth=2)
                ax_main.add_patch(rect)

                label = f'{subclass}\n{detection["combined_confidence"]:.2f}'
                ax_main.text(x1, max(0, y1 - 12), label, fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=class_color_map[subclass], alpha=0.7))

            # Draw void detections with enhanced spatial context information
            for i, void in enumerate(results['void_detections']):
                x1, y1, x2, y2 = void['bbox']

                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        fill=False, color='red', linewidth=3, linestyle='--')
                ax_main.add_patch(rect)

                void_info = results['void_analysis'][i]
                spatial_context = void_info['spatial_context']

                if void_info['final_assignment']:
                    assignment = void_info['final_assignment']

                    # Enhanced symbols for different assignment methods
                    method_symbols = {
                        'spatial_context_priority': '🎯',  # Strong spatial context
                        'spatial_context_moderate': '📍',  # Moderate spatial context
                        'intelligent_scoring': '🧠',      # Traditional intelligent scoring
                        'scarcity_fallback': '⚠️'        # Fallback method
                    }

                    symbol = method_symbols.get(assignment['assignment_method'], '?')

                    # Add context information
                    context_info = ""
                    if spatial_context['dominant_context']:
                        context = spatial_context['dominant_context']
                        if 'left_distance' in context:
                            context_info = f"L/R: {context['product_type']}"
                        elif 'top_distance' in context:
                            context_info = f"T/B: {context['product_type']}"
                        elif 'direction' in context:
                            context_info = f"{context['direction']}: {context['product_type']}"

                    void_label = (f"VOID {i + 1} {symbol}\n{assignment['product_type']}\n"
                                f"Est: {void_info['estimated_product_count']} items\n"
                                f"Conf: {assignment['confidence']:.2f}\n"
                                f"{context_info}")
                else:
                    void_label = f"VOID {i + 1}\nNo Assignment"

                # Draw spatial context connections
                if spatial_context['horizontal_context']:
                    # Draw lines to left and right neighbors
                    void_center = void['center']
                    if spatial_context['left_neighbors']:
                        left_prod = spatial_context['left_neighbors'][0]['product']
                        left_center = left_prod['center']
                        ax_main.plot([void_center[0], left_center[0]], [void_center[1], left_center[1]],
                                'g--', linewidth=2, alpha=0.7)

                    if spatial_context['right_neighbors']:
                        right_prod = spatial_context['right_neighbors'][0]['product']
                        right_center = right_prod['center']
                        ax_main.plot([void_center[0], right_center[0]], [void_center[1], right_center[1]],
                                'g--', linewidth=2, alpha=0.7)

                # Position label
                label_y = y2 + 5 if y2 + 80 < results['image'].shape[0] else y1 - 60
                ax_main.text(x1, label_y, void_label, fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7, edgecolor='white'))

            ax_main.set_title('🛒 Enhanced Shelf Analysis with Spatial Context\n🎯 = Strong Context | 📍 = Moderate Context | 🧠 = Intelligent | ⚠️ = Fallback',
                            fontsize=14, fontweight='bold')
            ax_main.axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()
    def print_detailed_summary(self, results):
        """Print detailed text summary with intelligent assignment explanations"""
        summary = results['summary']

        print("="*100)
        print("INTELLIGENT SHELF ANALYSIS REPORT")
        print("="*100)

        print(f"\n📊 OVERVIEW:")
        print(f"   • Total Products Detected: {summary['total_products_detected']}")
        print(f"   • Total Void Areas: {summary['total_void_areas']}")
        print(f"   • Estimated Missing Products: {summary['estimated_missing_products']}")
        print(f"   • Overall Stock Level: {summary['overall_stock_percentage']:.1f}%")
        print(f"   • Average Assignment Confidence: {summary['average_assignment_confidence']:.2f}")

        print(f"\n🏷️ PRODUCT INVENTORY:")
        for product_type, count in summary['product_counts_by_type'].items():
            print(f"   • {product_type}: {count} items")

        print(f"\n📈 STOCK LEVEL ANALYSIS:")
        for product_type, data in summary['stock_levels'].items():
            status = "🟢 GOOD" if data['stock_percentage'] >= 80 else "🟡 LOW" if data['stock_percentage'] >= 50 else "🔴 CRITICAL"
            print(f"   • {product_type}: {data['stock_percentage']:.1f}% stocked {status}")
            print(f"     - Present: {data['current_count']} | Missing: {data['missing_count']} | Full Capacity: {data['estimated_full_count']}")

        if summary['missing_by_product_type']:
            print(f"\n🕳️ INTELLIGENT VOID ASSIGNMENTS:")
            for product_type, missing_count in summary['missing_by_product_type'].items():
                print(f"   • {missing_count} missing {product_type} items intelligently assigned to void areas")

        print(f"\n🧠 ASSIGNMENT METHOD ANALYSIS:")
        method_descriptions = {
            'intelligent_scoring': 'Multi-factor intelligent scoring (proximity, scarcity, patterns, confidence)',
            'scarcity_fallback': 'Scarcity-based fallback (when no products nearby)'
        }

        for method, count in summary['assignment_methods'].items():
            desc = method_descriptions.get(method, method)
            print(f"   • {desc}: {count} assignments")

        print(f"\n📋 DETAILED VOID ANALYSIS:")

        for i, void_info in enumerate(results['void_analysis']):
            print(f"\n   🕳️ VOID AREA #{i+1}:")
            print(f"      Location: {void_info['void_bbox']}")
            print(f"      Area: {void_info['void_area']} pixels²")

            if void_info['final_assignment']:
                assignment = void_info['final_assignment']
                print(f"      ✅ ASSIGNMENT:")
                print(f"         - Product Type: {assignment['product_type']}")
                print(f"         - Estimated Count: {void_info['estimated_product_count']} items")
                print(f"         - Confidence: {assignment['confidence']:.3f}")
                print(f"         - Method: {assignment['assignment_method']}")
                print(f"         - Primary Factors: {', '.join(assignment.get('primary_factors', []))}")

                if void_info['assignment_reasoning']:
                    print(f"         - Reasoning: {'; '.join(void_info['assignment_reasoning'])}")

                # Show top assignment candidates for context
                if len(void_info['assignment_candidates']) > 1:
                    print(f"         - Alternative Candidates:")
                    for j, candidate in enumerate(void_info['assignment_candidates'][1:3]):  # Show top 2 alternatives
                        print(f"           {j+2}. {candidate['product_type']} (score: {candidate['total_score']:.3f})")
            else:
                print(f"      ❌ NO ASSIGNMENT POSSIBLE")

        # Additional insights
        print(f"\n💡 INSIGHTS & RECOMMENDATIONS:")

        # Identify critical stock situations
        critical_products = [pt for pt, data in summary['stock_levels'].items()
                           if data['stock_percentage'] < 50]
        if critical_products:
            print(f"   🔴 CRITICAL STOCK ALERT: {', '.join(critical_products)} need immediate restocking")

        # Identify products with high void assignment confidence
        high_confidence_assignments = {}
        for void_info in results['void_analysis']:
            if (void_info['final_assignment'] and
                void_info['final_assignment']['confidence'] > 0.7):
                product_type = void_info['final_assignment']['product_type']
                high_confidence_assignments[product_type] = high_confidence_assignments.get(product_type, 0) + 1

        if high_confidence_assignments:
            print(f"   ✅ HIGH CONFIDENCE ASSIGNMENTS: ", end="")
            confidence_list = [f"{count} {product}" for product, count in high_confidence_assignments.items()]
            print(", ".join(confidence_list))

        # Pattern analysis insights
        if 'shelf_analysis' in results:
            spatial_pattern = results['shelf_analysis']['spatial_patterns']['dominant_pattern']
            print(f"   📐 SHELF LAYOUT: Detected {spatial_pattern} arrangement pattern")

            if results['shelf_analysis']['clusters']:
                cluster_count = len(results['shelf_analysis']['clusters'])
                print(f"   🎯 PRODUCT CLUSTERING: {cluster_count} distinct product clusters identified")

        print("="*100)

    def _generate_summary(self, product_detections, void_detections, void_analysis):
        """Generate comprehensive analysis summary"""
        # Product counts by type
        product_counts = Counter([p['subclass'] for p in product_detections])

        # Void analysis summary
        void_assignments = [v['final_assignment'] for v in void_analysis if v['final_assignment']]
        estimated_missing_by_type = defaultdict(int)

        for i, void_info in enumerate(void_analysis):
            if void_info['final_assignment']:
                product_type = void_info['final_assignment']['product_type']
                count = void_info['estimated_product_count']
                estimated_missing_by_type[product_type] += count

        total_estimated_missing = sum(estimated_missing_by_type.values())

        # Calculate potential full inventory
        potential_full_inventory = dict(product_counts)
        for product_type, missing_count in estimated_missing_by_type.items():
            potential_full_inventory[product_type] = potential_full_inventory.get(product_type, 0) + missing_count

        # Stock level analysis
        stock_levels = {}
        for product_type in self.class_names:
            current = product_counts.get(product_type, 0)
            potential = potential_full_inventory.get(product_type, current)
            if potential > 0:
                stock_level = (current / potential) * 100
                stock_levels[product_type] = {
                    'current_count': current,
                    'estimated_full_count': potential,
                    'stock_percentage': stock_level,
                    'missing_count': potential - current
                }

        # Assignment method statistics
        assignment_methods = Counter()
        for void_info in void_analysis:
            if void_info['final_assignment']:
                method = void_info['final_assignment']['assignment_method']
                assignment_methods[method] += 1

        summary = {
            'total_products_detected': len(product_detections),
            'total_void_areas': len(void_detections),
            'product_counts_by_type': dict(product_counts),
            'estimated_missing_products': total_estimated_missing,
            'missing_by_product_type': dict(estimated_missing_by_type),
            'stock_levels': stock_levels,
            'overall_stock_percentage': (len(product_detections) / (len(product_detections) + total_estimated_missing) * 100) if total_estimated_missing > 0 else 100.0,
            'assignment_methods': dict(assignment_methods),
            'average_assignment_confidence': np.mean([v['assignment_confidence'] for v in void_analysis if v['assignment_confidence'] > 0]) if void_analysis else 0.0
        }

        return summary

    def _detect_products(self, image_path, image_rgb):
        """Detect and classify products using YOLO + CNN"""
        yolo_results = self.yolo_model(image_path, conf=self.confidence_threshold)

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
                        subclass_label, subclass_confidence = self._classify_crop(cropped_image)

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

    def _detect_voids(self, image_path):
        """Detect void areas using void detection model"""
        void_results = self.void_model(image_path, conf=self.void_confidence_threshold)

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

    def _classify_crop(self, cropped_image):
        """Classify a cropped image using the CNN classifier"""
        pil_image = Image.fromarray(cropped_image)
        input_tensor = self.cnn_transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.cnn_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()

        return predicted_class, confidence_score

    def _analyze_shelf_patterns(self, product_detections, image_shape):
        """Analyze shelf patterns and product clustering"""
        if not product_detections:
            return {
                'clusters': [],
                'product_counts': {},
                'scarcity_scores': {},
                'spatial_patterns': {}
            }

        # Extract product centers and types
        centers = np.array([p['center'] for p in product_detections])
        product_types = [p['subclass'] for p in product_detections]

        # Perform spatial clustering to identify product groups
        clustering = DBSCAN(
            eps=self.assignment_params['clustering_eps'],
            min_samples=self.assignment_params['min_cluster_size']
        )
        cluster_labels = clustering.fit_predict(centers)

        # Analyze clusters
        clusters = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue

            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_products = [product_detections[i] for i in cluster_indices]
            cluster_centers = centers[cluster_indices]
            cluster_types = [product_types[i] for i in cluster_indices]

            # Calculate cluster statistics
            cluster_info = {
                'cluster_id': cluster_id,
                'products': cluster_products,
                'center': np.mean(cluster_centers, axis=0),
                'product_types': Counter(cluster_types),
                'dominant_type': Counter(cluster_types).most_common(1)[0][0],
                'size': len(cluster_products),
                'bbox': self._calculate_cluster_bbox(cluster_products)
            }
            clusters.append(cluster_info)

        # Calculate product counts and scarcity scores
        product_counts = Counter(product_types)
        total_products = len(product_detections)

        scarcity_scores = {}
        for product_type in self.class_names:
            count = product_counts.get(product_type, 0)
            # Higher score = more scarce (less present)
            scarcity_scores[product_type] = 1.0 - (count / total_products) if total_products > 0 else 1.0

        # Analyze spatial patterns (horizontal vs vertical arrangements)
        spatial_patterns = self._analyze_spatial_patterns(product_detections, image_shape)

        return {
            'clusters': clusters,
            'product_counts': product_counts,
            'scarcity_scores': scarcity_scores,
            'spatial_patterns': spatial_patterns
        }

    def _calculate_cluster_bbox(self, cluster_products):
        """Calculate bounding box that encompasses all products in a cluster"""
        if not cluster_products:
            return None

        x1_min = min(p['bbox'][0] for p in cluster_products)
        y1_min = min(p['bbox'][1] for p in cluster_products)
        x2_max = max(p['bbox'][2] for p in cluster_products)
        y2_max = max(p['bbox'][3] for p in cluster_products)

        return (x1_min, y1_min, x2_max, y2_max)

    def _analyze_spatial_patterns(self, product_detections, image_shape):
        """Analyze spatial arrangement patterns of products"""
        if len(product_detections) < 2:
            return {'dominant_pattern': 'insufficient_data'}

        centers = np.array([p['center'] for p in product_detections])

        # Calculate horizontal and vertical spreads
        horizontal_spread = np.std(centers[:, 0])
        vertical_spread = np.std(centers[:, 1])

        # Determine dominant arrangement pattern
        if horizontal_spread > vertical_spread * 1.5:
            dominant_pattern = 'horizontal'
        elif vertical_spread > horizontal_spread * 1.5:
            dominant_pattern = 'vertical'
        else:
            dominant_pattern = 'mixed'

        return {
            'dominant_pattern': dominant_pattern,
            'horizontal_spread': horizontal_spread,
            'vertical_spread': vertical_spread
        }

    def _calculate_assignment_scores(self, void, product, distance, shelf_analysis, void_idx, prod_idx):
        """Calculate various scoring factors for void-product assignment"""
        scores = {}

        # 1. Proximity Score (closer = better)
        max_distance = self.assignment_params['max_assignment_distance']
        proximity_score = max(0, (max_distance - distance) / max_distance)
        scores['proximity'] = proximity_score * self.assignment_params['proximity_weight']

        # 2. Scarcity Score (less present products get higher priority)
        product_type = product['subclass']
        scarcity_score = shelf_analysis['scarcity_scores'].get(product_type, 0.5)
        scores['scarcity'] = scarcity_score * self.assignment_params['scarcity_weight']

        # 3. Pattern Alignment Score
        pattern_score = self._calculate_pattern_alignment_score(
            void, product, shelf_analysis['spatial_patterns']
        )
        scores['pattern'] = pattern_score * self.assignment_params['pattern_weight']

        # 4. Confidence Score
        confidence_score = product['combined_confidence']
        scores['confidence'] = confidence_score * self.assignment_params['confidence_weight']

        # 5. Cluster Coherence Score (bonus if void is near a cluster of same product type)
        cluster_score = self._calculate_cluster_coherence_score(
            void, product, shelf_analysis['clusters']
        )
        scores['cluster_coherence'] = cluster_score * 0.1  # Small bonus

        # 6. Size Compatibility Score
        size_score = self._calculate_size_compatibility_score(void, product)
        scores['size_compatibility'] = size_score * 0.1  # Small bonus

        return scores

    def _calculate_pattern_alignment_score(self, void, product, spatial_patterns):
        """Calculate how well the void-product assignment aligns with shelf patterns"""
        if spatial_patterns['dominant_pattern'] == 'insufficient_data':
            return 0.5

        void_center = void['center']
        product_center = product['center']

        horizontal_distance = abs(void_center[0] - product_center[0])
        vertical_distance = abs(void_center[1] - product_center[1])

        if spatial_patterns['dominant_pattern'] == 'horizontal':
            # Prefer horizontal alignment
            if vertical_distance < 50:  # Same row
                return 1.0
            elif horizontal_distance < 100:  # Close horizontally
                return 0.7
            else:
                return 0.3

        elif spatial_patterns['dominant_pattern'] == 'vertical':
            # Prefer vertical alignment
            if horizontal_distance < 50:  # Same column
                return 1.0
            elif vertical_distance < 100:  # Close vertically
                return 0.7
            else:
                return 0.3

        else:  # Mixed pattern
            total_distance = horizontal_distance + vertical_distance
            return max(0, 1.0 - (total_distance / 200))

    def _calculate_cluster_coherence_score(self, void, product, clusters):
        """Calculate bonus score if void is near a cluster of the same product type"""
        if not clusters:
            return 0.0

        void_center = void['center']
        product_type = product['subclass']

        max_coherence = 0.0

        for cluster in clusters:
            if product_type in cluster['product_types']:
                # Calculate distance from void to cluster center
                cluster_center = cluster['center']
                distance = np.sqrt((void_center[0] - cluster_center[0])**2 +
                                 (void_center[1] - cluster_center[1])**2)

                # Calculate coherence score based on:
                # 1. Distance to cluster
                # 2. Proportion of this product type in cluster
                type_proportion = cluster['product_types'][product_type] / cluster['size']
                distance_factor = max(0, 1.0 - (distance / 150))  # 150px max cluster influence

                coherence = distance_factor * type_proportion
                max_coherence = max(max_coherence, coherence)

        return max_coherence

    def _calculate_size_compatibility_score(self, void, product):
        """Calculate how well the void size matches the product size"""
        void_area = void['area']
        product_area = product['area']

        if product_area == 0:
            return 0.5

        area_ratio = void_area / product_area

        # Ideal ratio is between 0.5 and 3.0 (void can fit 0.5 to 3 products)
        if 0.5 <= area_ratio <= 3.0:
            return 1.0
        elif 0.25 <= area_ratio <= 5.0:
            return 0.7
        else:
            return 0.3

    def _generate_assignment_reasoning(self, scores, product_type, distance):
        """Generate human-readable reasoning for the assignment"""
        reasoning = []

        # Identify the top factors
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for factor, score in sorted_scores[:3]:  # Top 3 factors
            if score > 0.1:  # Only mention significant factors
                if factor == 'proximity':
                    reasoning.append(f"Close proximity ({distance:.0f}px)")
                elif factor == 'scarcity':
                    reasoning.append(f"Low stock priority for {product_type}")
                elif factor == 'pattern':
                    reasoning.append("Good spatial pattern alignment")
                elif factor == 'confidence':
                    reasoning.append("High detection confidence")
                elif factor == 'cluster_coherence':
                    reasoning.append("Near similar product cluster")
                elif factor == 'size_compatibility':
                    reasoning.append("Compatible size match")

        return reasoning

    def _identify_primary_factors(self, scores):
        """Identify the primary factors that influenced the assignment"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [factor for factor, score in sorted_scores[:2] if score > 0.1]

    def _fallback_assignment(self, void, product_detections, shelf_analysis):
        """Fallback assignment when no products are nearby"""
        if not product_detections or not shelf_analysis['scarcity_scores']:
            return None

        # Only consider product types that are actually present in the image
        present_product_types = set(p['subclass'] for p in product_detections)
        present_scarcity_scores = {k: v for k, v in shelf_analysis['scarcity_scores'].items() 
                                if k in present_product_types}
        
        if not present_scarcity_scores:
            return None

        # Assign to the most scarce (least present) product type among those actually detected
        most_scarce_type = max(present_scarcity_scores.items(), key=lambda x: x[1])[0]

        return {
            'product_type': most_scarce_type,
            'confidence': 0.2,  # Low confidence for fallback
            'assignment_method': 'scarcity_fallback',
            'primary_factors': ['scarcity']
        }