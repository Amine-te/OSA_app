from collections import Counter, defaultdict
import numpy as np

def generate_summary(product_detections, void_detections, void_analysis, class_names):
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
    for product_type in class_names:
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


def print_detailed_summary(results):
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
        'scarcity_fallback': 'Scarcity-based fallback (when no products nearby)',
        'spatial_context_priority': 'Strong spatial context (both sides matching)',
        'spatial_context_moderate': 'Moderate spatial context (single side match)'
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
