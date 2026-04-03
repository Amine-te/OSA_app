import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_complete_results(results, class_names, product_colors, save_path=None, figsize=(30, 22)):
    """Visualize all detection results with spatial context annotations"""
    fig = plt.figure(figsize=figsize)

    # Create single axes for the main image only
    ax_main = fig.add_subplot(111)
    ax_main.imshow(results['image'])

    class_color_map = {class_name: product_colors[i] for i, class_name in enumerate(class_names)}

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
