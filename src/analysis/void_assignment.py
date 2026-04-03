import numpy as np
from scipy.spatial.distance import cdist
from src.analysis.scoring import (
    calculate_border_to_center_distance,
    estimate_product_count,
    estimate_product_count_from_context,
    calculate_assignment_scores_with_spatial_context,
    generate_assignment_reasoning,
    identify_primary_factors
)

def filter_isolated_voids(void_detections, product_detections, void_analysis, assignment_params):
    """Filter out isolated voids that are too far from any product"""
    if not product_detections:
        return [], []
    
    filtered_voids = []
    filtered_analysis = []
    
    for i, (void, analysis) in enumerate(zip(void_detections, void_analysis)):
        # Never filter voids with high confidence assignments
        if (analysis['final_assignment'] and 
            analysis['final_assignment']['confidence'] >= assignment_params['high_confidence_threshold']):
            filtered_voids.append(void)
            filtered_analysis.append(analysis)
            continue
        
        # Calculate minimum border-to-center distance to any product
        min_distance = float('inf')
        for product in product_detections:
            distance = calculate_border_to_center_distance(void['bbox'], product['center'])
            min_distance = min(min_distance, distance)
        
        # Keep void if it's close enough to products OR has decent assignment confidence
        keep_void = (
            min_distance <= assignment_params['isolation_distance_threshold'] or
            (analysis['final_assignment'] and 
            analysis['final_assignment']['confidence'] >= assignment_params['min_assignment_confidence'])
        )
        
        if keep_void:
            filtered_voids.append(void)
            filtered_analysis.append(analysis)
    
    return filtered_voids, filtered_analysis


def fallback_assignment(void, product_detections, shelf_analysis):
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


def intelligent_void_assignment_with_spatial_context(product_detections, void_detections,
                                                     shelf_analysis, spatial_context_analysis,
                                                     image_shape, assignment_params):
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

            void_info['estimated_product_count'] = estimate_product_count_from_context(
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

            void_info['estimated_product_count'] = estimate_product_count_from_context(
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
        nearby_indices = np.where(void_distances <= assignment_params['max_assignment_distance'])[0]

        if len(nearby_indices) == 0:
            # No products within reasonable distance - use fallback
            void_info['final_assignment'] = fallback_assignment(
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
            scores = calculate_assignment_scores_with_spatial_context(
                void, product, distance, shelf_analysis, spatial_context, void_idx, prod_idx, assignment_params
            )

            candidate = {
                'product_id': prod_idx,
                'product': product,
                'distance': distance,
                'product_type': product_type,
                'scores': scores,
                'total_score': sum(scores.values()),
                'reasoning': generate_assignment_reasoning(scores, product_type, distance)
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
                'primary_factors': identify_primary_factors(best_candidate['scores'])
            }

            # Estimate product count
            void_info['estimated_product_count'] = estimate_product_count(
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
