import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

def analyze_shelf_patterns(product_detections, image_shape, class_names, assignment_params):
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
        eps=assignment_params['clustering_eps'],
        min_samples=assignment_params['min_cluster_size']
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
            'bbox': calculate_cluster_bbox(cluster_products)
        }
        clusters.append(cluster_info)

    # Calculate product counts and scarcity scores
    product_counts = Counter(product_types)
    total_products = len(product_detections)

    scarcity_scores = {}
    for product_type in class_names:
        count = product_counts.get(product_type, 0)
        # Higher score = more scarce (less present)
        scarcity_scores[product_type] = 1.0 - (count / total_products) if total_products > 0 else 1.0

    # Analyze spatial patterns (horizontal vs vertical arrangements)
    spatial_patterns = analyze_spatial_patterns(product_detections, image_shape)

    return {
        'clusters': clusters,
        'product_counts': product_counts,
        'scarcity_scores': scarcity_scores,
        'spatial_patterns': spatial_patterns
    }

def calculate_cluster_bbox(cluster_products):
    """Calculate bounding box that encompasses all products in a cluster"""
    if not cluster_products:
        return None

    x1_min = min(p['bbox'][0] for p in cluster_products)
    y1_min = min(p['bbox'][1] for p in cluster_products)
    x2_max = max(p['bbox'][2] for p in cluster_products)
    y2_max = max(p['bbox'][3] for p in cluster_products)

    return (x1_min, y1_min, x2_max, y2_max)

def analyze_spatial_patterns(product_detections, image_shape):
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
