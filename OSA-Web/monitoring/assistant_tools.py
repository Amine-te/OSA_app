import json
from django.utils import timezone
from .models import CameraConfig, MonitoringSession, AnalyticsSample, AlertRule, AlertEvent
from .utils import normalize_stock_level

def get_inventory_status():
    """
    Get the current inventory status, including all products, their current stock,
    missing count, total capacity, stock percentage, and status (Critical, Warning, OK).
    
    Returns:
        dict: A dictionary containing details of current inventory.
    """
    latest_session = MonitoringSession.objects.filter(status='running').order_by('-started_at').first()
    if not latest_session:
        latest_session = MonitoringSession.objects.filter(status='ended').order_by('-ended_at').first()
        
    latest_sample = None
    products_list = []
    
    if latest_session:
        latest_sample = latest_session.samples.order_by('-timestamp').first()
        
    if latest_sample:
        stock_levels = latest_sample.summary_json.get('stock_levels', {})
        for name, data in stock_levels.items():
            fields = normalize_stock_level(data)
            pct = fields['stock_pct']
            status = 'Critical' if pct < 50 else 'Warning' if pct < 80 else 'OK'
            products_list.append({
                'name': name,
                'current': fields['current'],
                'missing': fields['missing'],
                'capacity': fields['capacity'],
                'stock_pct': round(pct, 1),
                'status': status
            })
            
    return {
        'session_id': latest_session.id if latest_session else None,
        'session_status': latest_session.status if latest_session else 'none',
        'camera_name': latest_session.camera.name if latest_session else 'none',
        'timestamp': latest_sample.timestamp.strftime('%Y-%m-%d %H:%M:%S') if latest_sample else None,
        'products': sorted(products_list, key=lambda x: x['name'])
    }


def get_active_sessions():
    """
    Get a list of all currently active (running) monitoring sessions.
    
    Returns:
        list: A list of active sessions with their start time, camera name, and session ID.
    """
    sessions = MonitoringSession.objects.filter(status='running').order_by('-started_at')
    active_list = []
    for s in sessions:
        active_list.append({
            'session_id': s.id,
            'camera_name': s.camera.name,
            'started_at': s.started_at.strftime('%Y-%m-%d %H:%M:%S'),
            'status': s.status
        })
    return active_list


def get_session_summary(session_id=None):
    """
    Get a summary of a specific monitoring session or the most recent session if no ID is specified.
    
    Args:
        session_id (int, optional): The ID of the session. Defaults to None (gets latest).
        
    Returns:
        dict: Session details, total samples, average stock level, average FPS, latency, and triggered alerts.
    """
    if session_id:
        session = MonitoringSession.objects.filter(pk=session_id).first()
    else:
        session = MonitoringSession.objects.order_by('-started_at').first()
        
    if not session:
        return {'status': 'error', 'message': 'No monitoring sessions found.'}
        
    samples = session.samples.all()
    sample_count = len(samples)
    
    avg_stock_pct = sum(s.stock_pct for s in samples) / sample_count if sample_count > 0 else 0.0
    avg_fps = sum(s.fps for s in samples) / sample_count if sample_count > 0 else 0.0
    avg_latency = sum(s.latency_ms for s in samples) / sample_count if sample_count > 0 else 0.0
    
    alerts = AlertEvent.objects.filter(session=session).order_by('-triggered_at')
    alert_list = []
    for a in alerts:
        alert_list.append({
            'product_name': a.product_name,
            'triggered_at': a.triggered_at.strftime('%H:%M:%S'),
            'stock_pct': round(a.stock_pct_at_trigger, 1),
            'severity': a.severity
        })
        
    return {
        'session_id': session.id,
        'camera_name': session.camera.name,
        'started_at': session.started_at.strftime('%Y-%m-%d %H:%M:%S'),
        'ended_at': session.ended_at.strftime('%Y-%m-%d %H:%M:%S') if session.ended_at else None,
        'status': session.status,
        'sample_count': sample_count,
        'avg_stock_pct': round(avg_stock_pct, 1),
        'avg_fps': round(avg_fps, 1),
        'avg_latency_ms': round(avg_latency, 1),
        'alerts': alert_list
    }


def get_recent_alerts(limit=10):
    """
    Get a list of the most recent alert events triggered in the system.
    
    Args:
        limit (int, optional): Maximum number of alerts to return. Defaults to 10.
        
    Returns:
        list: Recent alert events, sorted by trigger time descending.
    """
    alerts = AlertEvent.objects.select_related('session', 'session__camera').order_by('-triggered_at')[:limit]
    alert_list = []
    for a in alerts:
        alert_list.append({
            'alert_id': a.id,
            'session_id': a.session.id,
            'camera_name': a.session.camera.name,
            'product_name': a.product_name,
            'triggered_at': a.triggered_at.strftime('%Y-%m-%d %H:%M:%S'),
            'stock_pct': round(a.stock_pct_at_trigger, 1),
            'severity': a.severity
        })
    return alert_list


def get_camera_configs():
    """
    Get all configured cameras in the system.
    
    Returns:
        list: List of cameras with confidence thresholds and RTSP URLs.
    """
    cameras = CameraConfig.objects.all().order_by('-created_at')
    camera_list = []
    for c in cameras:
        camera_list.append({
            'camera_id': c.id,
            'name': c.name,
            'rtsp_url': c.rtsp_url,
            'frame_skip': c.frame_skip,
            'confidence_threshold': c.confidence_threshold,
            'created_at': c.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    return camera_list


def get_product_history(product_name):
    """
    Retrieve historical stock level data for a specific product over recent samples.
    This is extremely useful for plotting graphs of stock level changes.
    
    Args:
        product_name (str): The name of the product (case-insensitive).
        
    Returns:
        dict: A dictionary with labels (timestamps) and stock_pct (percentages) for plotting.
    """
    samples = AnalyticsSample.objects.select_related('session').order_by('timestamp')
    labels = []
    stock_series = []
    missing_series = []
    
    for s in samples:
        stock_levels = s.summary_json.get('stock_levels', {})
        target_key = None
        for k in stock_levels.keys():
            if k.lower() == product_name.lower():
                target_key = k
                break
                
        if not target_key:
            continue
            
        data = stock_levels[target_key]
        fields = normalize_stock_level(data)
        pct = fields['stock_pct']
        missing = int(fields['missing'])
        
        labels.append(s.timestamp.strftime('%H:%M:%S'))
        stock_series.append(round(pct, 1))
        missing_series.append(missing)
        
    # Cap to last 30 samples to avoid overloading charts
    return {
        'product_name': product_name,
        'labels': labels[-30:],
        'stock': stock_series[-30:],
        'missing': missing_series[-30:]
    }
