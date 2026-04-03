import numpy as np

def calculate_border_to_center_distance(void_bbox, product_center):
    """Calculate minimum distance from void border to product center"""
    x1, y1, x2, y2 = void_bbox
    px, py = product_center
    
    # Calculate distance from point to rectangle
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    
    return np.sqrt(dx * dx + dy * dy)

def estimate_product_count(void, product):
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

def estimate_product_count_from_context(void, spatial_context, product_detections):
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

def calculate_assignment_scores_with_spatial_context(void, product, distance, shelf_analysis,
                                                     spatial_context, void_idx, prod_idx, assignment_params):
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

    scores['spatial_context'] = spatial_score * assignment_params['spatial_context_weight']

    # 2. Proximity Score (closer = better)
    max_distance = assignment_params['max_assignment_distance']
    proximity_score = max(0, (max_distance - distance) / max_distance)
    scores['proximity'] = proximity_score * assignment_params['proximity_weight']

    # 3. Scarcity Score (less present products get higher priority)
    product_type = product['subclass']
    scarcity_score = shelf_analysis['scarcity_scores'].get(product_type, 0.5)
    scores['scarcity'] = scarcity_score * assignment_params['scarcity_weight']

    # 4. Pattern Alignment Score
    pattern_score = calculate_pattern_alignment_score(
        void, product, shelf_analysis['spatial_patterns']
    )
    scores['pattern'] = pattern_score * assignment_params['pattern_weight']

    # 5. Confidence Score
    confidence_score = product['combined_confidence']
    scores['confidence'] = confidence_score * assignment_params['confidence_weight']

    return scores

def calculate_pattern_alignment_score(void, product, spatial_patterns):
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

def calculate_cluster_coherence_score(void, product, clusters):
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

def calculate_size_compatibility_score(void, product):
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

def generate_assignment_reasoning(scores, product_type, distance):
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

def identify_primary_factors(scores):
    """Identify the primary factors that influenced the assignment"""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [factor for factor, score in sorted_scores[:2] if score > 0.1]
