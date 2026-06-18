import csv
import io
import json
from datetime import datetime

from django.http import HttpResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .models import AlertEvent, AnalyticsSample, MonitoringSession
from .utils import normalize_stock_level


def build_kpi_table(samples=None):
    if samples is None:
        samples = AnalyticsSample.objects.select_related('session').order_by('timestamp')

    product_stats = {}
    for s in samples:
        stock_levels = s.summary_json.get('stock_levels', {})
        for prod_name, data in stock_levels.items():
            if prod_name not in product_stats:
                product_stats[prod_name] = {
                    'stock_above_80': [],
                    'stock_below_80': [],
                    'missing_products': [],
                    'samples_below_80': 0,
                }

            pct = data.get('stock_percentage', 100.0)
            fields = normalize_stock_level(data)
            missing = fields['missing']

            if pct >= 80:
                product_stats[prod_name]['stock_above_80'].append(pct)
            else:
                product_stats[prod_name]['stock_below_80'].append(pct)
                product_stats[prod_name]['samples_below_80'] += 1

            product_stats[prod_name]['missing_products'].append(missing)

    kpi_table = []
    for prod_name, stats in product_stats.items():
        above = stats['stock_above_80']
        below = stats['stock_below_80']
        missing = stats['missing_products']

        kpi_table.append({
            'name': prod_name,
            'osa_rate': round(sum(above) / len(above), 1) if above else 0.0,
            'oos_rate': round(sum(below) / len(below), 1) if below else 0.0,
            'peak_missing': max(missing) if missing else 0,
            'time_below_80': stats['samples_below_80'],
        })

    kpi_table.sort(key=lambda x: x['name'])
    return kpi_table


def build_analytics_timeseries_rows(selected_product):
    rows = []
    if not selected_product:
        return rows

    samples = AnalyticsSample.objects.select_related('session').order_by('timestamp')
    for s in samples:
        stock_levels = s.summary_json.get('stock_levels', {})
        if selected_product not in stock_levels:
            continue
        fields = normalize_stock_level(stock_levels[selected_product])
        pct = fields['stock_pct']
        rows.append({
            'timestamp': s.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': s.session_id,
            'product': selected_product,
            'stock_pct': round(pct, 1),
            'oos_rate_pct': round(pct, 1) if pct < 80 else '',
            'missing': int(fields['missing']),
            'below_threshold': 'Yes' if pct < 80 else 'No',
        })
    return rows


def build_inventory_rows():
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
                'status': status,
            })

    products_list.sort(key=lambda x: x['name'])
    meta = {
        'session_id': latest_session.id if latest_session else None,
        'camera': latest_session.camera.name if latest_session else None,
        'sample_timestamp': latest_sample.timestamp.isoformat() if latest_sample else None,
        'total_products': latest_sample.total_products if latest_sample else 0,
        'total_missing': latest_sample.missing_products if latest_sample else 0,
        'overall_stock_pct': round(latest_sample.stock_pct, 1) if latest_sample else 0,
    }
    return products_list, meta


def build_sessions_rows():
    rows = []
    qs = MonitoringSession.objects.select_related('camera').order_by('-started_at')
    for s in qs:
        samples = list(s.samples.all())
        sample_count = len(samples)
        avg_stock_pct = (
            round(sum(samp.stock_pct for samp in samples) / sample_count, 1)
            if sample_count else 0.0
        )
        rows.append({
            'id': s.id,
            'camera': s.camera.name,
            'started_at': s.started_at.strftime('%Y-%m-%d %H:%M'),
            'ended_at': s.ended_at.strftime('%Y-%m-%d %H:%M') if s.ended_at else '',
            'samples': sample_count,
            'avg_stock_pct': avg_stock_pct,
            'status': s.status.upper(),
        })
    return rows


def build_session_samples_rows(session_id):
    session = MonitoringSession.objects.get(pk=session_id)
    rows = []
    for s in session.samples.order_by('timestamp'):
        rows.append({
            'timestamp': s.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'total_products': s.total_products,
            'missing_products': s.missing_products,
            'void_detections': s.void_detections,
            'stock_pct': round(s.stock_pct, 1),
            'fps': round(s.fps, 1),
            'latency_ms': round(s.latency_ms, 1),
        })
    return rows, session


def build_session_alerts_rows(session_id):
    session = MonitoringSession.objects.get(pk=session_id)
    rows = []
    for a in session.alerts.select_related('rule').order_by('-triggered_at'):
        rows.append({
            'triggered_at': a.triggered_at.strftime('%Y-%m-%d %H:%M:%S'),
            'severity': a.rule.severity.upper(),
            'product': a.product_name or 'Global',
            'stock_pct_at_trigger': round(a.stock_pct_at_trigger, 1),
        })
    return rows, session


EXPORT_DATASETS = {
    'analytics_kpi': {
        'title': 'Analytics KPI Summary',
        'filename': 'analytics_kpi',
        'columns': [
            {'key': 'name', 'label': 'Product'},
            {'key': 'osa_rate', 'label': 'OSA Rate (%)'},
            {'key': 'oos_rate', 'label': 'OOS Rate (%)'},
            {'key': 'peak_missing', 'label': 'Peak Missing'},
            {'key': 'time_below_80', 'label': 'Time Below 80% (sec)'},
        ],
    },
    'analytics_timeseries': {
        'title': 'Analytics Time Series',
        'filename': 'analytics_timeseries',
        'columns': [
            {'key': 'timestamp', 'label': 'Timestamp'},
            {'key': 'session_id', 'label': 'Session ID'},
            {'key': 'product', 'label': 'Product'},
            {'key': 'stock_pct', 'label': 'Stock (%)'},
            {'key': 'oos_rate_pct', 'label': 'OOS Rate (%)'},
            {'key': 'missing', 'label': 'Missing'},
            {'key': 'below_threshold', 'label': 'Below 80%'},
        ],
    },
    'inventory': {
        'title': 'Inventory Snapshot',
        'filename': 'inventory_snapshot',
        'columns': [
            {'key': 'name', 'label': 'Product'},
            {'key': 'current', 'label': 'Current'},
            {'key': 'missing', 'label': 'Missing'},
            {'key': 'capacity', 'label': 'Capacity'},
            {'key': 'stock_pct', 'label': 'Stock (%)'},
            {'key': 'status', 'label': 'Status'},
        ],
    },
    'sessions': {
        'title': 'Monitoring Sessions',
        'filename': 'monitoring_sessions',
        'columns': [
            {'key': 'id', 'label': 'ID'},
            {'key': 'camera', 'label': 'Camera'},
            {'key': 'started_at', 'label': 'Started'},
            {'key': 'ended_at', 'label': 'Ended'},
            {'key': 'samples', 'label': 'Samples'},
            {'key': 'avg_stock_pct', 'label': 'Avg Stock (%)'},
            {'key': 'status', 'label': 'Status'},
        ],
    },
    'session_samples': {
        'title': 'Session Samples',
        'filename': 'session_samples',
        'columns': [
            {'key': 'timestamp', 'label': 'Timestamp'},
            {'key': 'total_products', 'label': 'Total Products'},
            {'key': 'missing_products', 'label': 'Missing Products'},
            {'key': 'void_detections', 'label': 'Void Detections'},
            {'key': 'stock_pct', 'label': 'Stock (%)'},
            {'key': 'fps', 'label': 'FPS'},
            {'key': 'latency_ms', 'label': 'Latency (ms)'},
        ],
    },
    'session_alerts': {
        'title': 'Session Alerts',
        'filename': 'session_alerts',
        'columns': [
            {'key': 'triggered_at', 'label': 'Time'},
            {'key': 'severity', 'label': 'Severity'},
            {'key': 'product', 'label': 'Product'},
            {'key': 'stock_pct_at_trigger', 'label': 'Stock (%)'},
        ],
    },
}


def get_export_payload(dataset, request, session_id=None):
    config = EXPORT_DATASETS[dataset]
    meta = {}

    if dataset == 'analytics_kpi':
        rows = build_kpi_table()
    elif dataset == 'analytics_timeseries':
        selected_product = request.GET.get('selected_product', '')
        if not selected_product:
            raise ValueError('Select a product before exporting time series data.')
        rows = build_analytics_timeseries_rows(selected_product)
        config = {
            **config,
            'title': f'Analytics Time Series - {selected_product}',
            'filename': f'analytics_{selected_product.replace(" ", "_").lower()}',
        }
    elif dataset == 'inventory':
        rows, meta = build_inventory_rows()
    elif dataset == 'sessions':
        rows = build_sessions_rows()
    elif dataset == 'session_samples':
        if not session_id:
            raise ValueError('Session ID is required.')
        rows, session = build_session_samples_rows(session_id)
        config = {
            **config,
            'title': f'Session #{session.id} Samples',
            'filename': f'session_{session.id}_samples',
        }
    elif dataset == 'session_alerts':
        if not session_id:
            raise ValueError('Session ID is required.')
        rows, session = build_session_alerts_rows(session_id)
        config = {
            **config,
            'title': f'Session #{session.id} Alerts',
            'filename': f'session_{session.id}_alerts',
        }
    else:
        raise ValueError('Unknown export dataset.')

    return config, rows, meta


def _timestamp_slug():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def response_csv(config, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=[c['key'] for c in config['columns']])
    writer.writeheader()
    for row in rows:
        writer.writerow({col['key']: row.get(col['key'], '') for col in config['columns']})

    response = HttpResponse(buffer.getvalue(), content_type='text/csv; charset=utf-8')
    filename = f"{config['filename']}_{_timestamp_slug()}.csv"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def response_json(config, rows, meta=None):
    payload = {
        'title': config['title'],
        'exported_at': datetime.now().isoformat(),
        'columns': config['columns'],
        'rows': rows,
    }
    if meta:
        payload['meta'] = meta

    response = HttpResponse(
        json.dumps(payload, indent=2),
        content_type='application/json; charset=utf-8',
    )
    filename = f"{config['filename']}_{_timestamp_slug()}.json"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def response_pdf(config, rows):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(config['title'], styles['Title']),
        Paragraph(f'Exported: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', styles['Normal']),
        Spacer(1, 12),
    ]

    table_data = [[col['label'] for col in config['columns']]]
    for row in rows:
        table_data.append([str(row.get(col['key'], '')) for col in config['columns']])

    if len(table_data) == 1:
        table_data.append(['No data available'] + [''] * (len(config['columns']) - 1))

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1d2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    doc.build(story)

    response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
    filename = f"{config['filename']}_{_timestamp_slug()}.pdf"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def export_table_response(dataset, request, session_id=None):
    fmt = request.GET.get('format', 'csv').lower()
    if fmt not in {'csv', 'json', 'pdf'}:
        raise ValueError('Unsupported export format. Use csv, json, or pdf.')

    config, rows, meta = get_export_payload(dataset, request, session_id=session_id)

    if fmt == 'csv':
        return response_csv(config, rows)
    if fmt == 'json':
        return response_json(config, rows, meta=meta or None)
    return response_pdf(config, rows)
