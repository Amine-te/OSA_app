def normalize_stock_level(data):
    """Map pipeline stock_levels keys to dashboard field names."""
    current = data.get('current_count', data.get('current', 0))
    missing = data.get('missing_count', data.get('missing', 0))
    capacity = data.get(
        'estimated_full_count',
        data.get('capacity', current + missing),
    )
    return {
        'current': current,
        'missing': missing,
        'capacity': capacity,
        'stock_pct': data.get('stock_percentage', 0.0),
    }
