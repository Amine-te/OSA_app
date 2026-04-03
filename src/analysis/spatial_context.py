def analyze_spatial_context(product_detections, void_detections, assignment_params):
    """
    Analyze spatial context - identify products that surround each void
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
            if vertical_distance <= assignment_params['neighbor_alignment_tolerance']:
                # Product is to the left
                if prod_center[0] < void_center[0] and horizontal_distance <= assignment_params['spatial_context_threshold']:
                    context_info['left_neighbors'].append({
                        'product_id': prod_idx,
                        'product': product,
                        'distance': horizontal_distance,
                        'product_type': product['subclass']
                    })

                # Product is to the right
                elif prod_center[0] > void_center[0] and horizontal_distance <= assignment_params['spatial_context_threshold']:
                    context_info['right_neighbors'].append({
                        'product_id': prod_idx,
                        'product': product,
                        'distance': horizontal_distance,
                        'product_type': product['subclass']
                    })

            # Check if products are vertically aligned (same column)
            if horizontal_distance <= assignment_params['neighbor_alignment_tolerance']:
                # Product is above
                if prod_center[1] < void_center[1] and vertical_distance <= assignment_params['spatial_context_threshold']:
                    context_info['top_neighbors'].append({
                        'product_id': prod_idx,
                        'product': product,
                        'distance': vertical_distance,
                        'product_type': product['subclass']
                    })

                # Product is below
                elif prod_center[1] > void_center[1] and vertical_distance <= assignment_params['spatial_context_threshold']:
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
