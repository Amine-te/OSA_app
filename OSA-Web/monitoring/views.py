import json
import logging
import redis

logger = logging.getLogger(__name__)
from django.conf import settings
from django.contrib import messages
from django.http import HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.clickjacking import xframe_options_sameorigin
from django.views.decorators.http import require_POST

from .exports import build_kpi_table, export_table_response
from .forms import AlertRuleForm, CameraConfigForm
from .models import AlertEvent, AlertRule, AnalyticsSample, CameraConfig, MonitoringSession
from .tasks import run_pipeline_session
from .utils import normalize_stock_level

def dashboard(request):
    cameras = CameraConfig.objects.all()
    camera_id = request.GET.get('camera_id')
    
    if camera_id:
        selected_camera = cameras.filter(id=camera_id).first()
    else:
        selected_camera = cameras.first()

    active_session = None
    latest_sample = None
    recent_alerts = []

    if selected_camera:
        active_session = MonitoringSession.objects.filter(
            camera=selected_camera, status='running'
        ).first()

        if active_session:
            latest_sample = active_session.samples.order_by('-timestamp').first()
            recent_alerts = AlertEvent.objects.filter(session=active_session).order_by('-triggered_at')[:10]

    context = {
        'cameras': cameras,
        'selected_camera': selected_camera,
        'active_session': active_session,
        'latest_sample': latest_sample,
        'recent_alerts': recent_alerts,
    }
    return render(request, 'monitoring/dashboard.html', context)

@require_POST
def start_session(request):
    camera_id = request.POST.get('camera_id')
    camera = get_object_or_404(CameraConfig, pk=camera_id)

    if MonitoringSession.objects.filter(camera=camera, status='running').exists():
        messages.error(request, f"A session is already running for camera '{camera.name}'.")
        return redirect(f"/?camera_id={camera.id}")

    session = MonitoringSession.objects.create(camera=camera)
    run_pipeline_session.delay(session.id)
    messages.success(request, f"Started monitoring session for '{camera.name}'.")
    return redirect(f"/?camera_id={camera.id}")

@require_POST
def stop_session(request):
    session_id = request.POST.get('session_id')
    session = get_object_or_404(MonitoringSession, pk=session_id)

    r = redis.from_url(settings.REDIS_URL)
    r.set(f"stop_session_{session_id}", "1", ex=300)

    if session.status == 'running':
        session.status = 'ended'
        session.ended_at = timezone.now()
        session.save(update_fields=['status', 'ended_at'])

    messages.info(request, f"Monitoring stopped for '{session.camera.name}'.")
    return redirect(f"/?camera_id={session.camera.id}")

def sessions_list(request):
    qs = MonitoringSession.objects.select_related('camera').order_by('-started_at')
    
    sessions = []
    for s in qs:
        samples = s.samples.all()
        s.sample_count = len(samples)
        if s.sample_count > 0:
            s.avg_stock_pct = sum(samp.stock_pct for samp in samples) / s.sample_count
        else:
            s.avg_stock_pct = 0.0
        sessions.append(s)

    return render(request, 'monitoring/sessions.html', {'sessions': sessions})

def session_detail(request, pk):
    session = get_object_or_404(MonitoringSession.objects.select_related('camera'), pk=pk)
    samples = session.samples.order_by('timestamp')

    chart_data = json.dumps({
        "labels": [s.timestamp.strftime('%H:%M:%S') for s in samples],
        "stock": [s.stock_pct for s in samples],
        "missing": [s.missing_products for s in samples],
        "fps": [s.fps for s in samples],
    })

    alerts = session.alerts.select_related('rule').order_by('-triggered_at')

    context = {
        'session': session,
        'chart_data': chart_data,
        'alerts': alerts,
    }
    return render(request, 'monitoring/session_detail.html', context)

def analytics(request):
    samples = AnalyticsSample.objects.select_related('session').order_by('timestamp')
    selected_product = request.GET.get('selected_product')
    kpi_table = build_kpi_table(samples)

    product_chart_data = None
    has_chart_data = False
    if selected_product:
        labels = []
        stock_series = []
        oos_series = []
        missing_series = []
        threshold_events = []

        for s in samples:
            stock_levels = s.summary_json.get('stock_levels', {})
            if selected_product not in stock_levels:
                continue
            data = stock_levels[selected_product]
            fields = normalize_stock_level(data)
            pct = fields['stock_pct']
            missing = int(fields['missing'])

            labels.append(s.timestamp.strftime('%H:%M:%S'))
            stock_series.append(round(pct, 1))
            oos_series.append(round(pct, 1) if pct < 80 else None)
            missing_series.append(missing)
            threshold_events.append(1 if pct < 80 else 0)

        has_chart_data = len(labels) > 0
        product_chart_data = json.dumps({
            'labels': labels,
            'stock': stock_series,
            'oos': oos_series,
            'missing': missing_series,
            'threshold_events': threshold_events,
        })

    running_session = MonitoringSession.objects.filter(status='running').order_by('-started_at').first()

    context = {
        'kpi_table': kpi_table,
        'selected_product': selected_product,
        'product_chart_data': product_chart_data,
        'has_chart_data': has_chart_data,
        'running_session_id': running_session.id if running_session else None,
    }
    return render(request, 'monitoring/analytics.html', context)

def inventory(request):
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
                'stock_pct': pct,
                'status': status
            })

    context = {
        'latest_session': latest_session,
        'latest_sample': latest_sample,
        'products': sorted(products_list, key=lambda x: x['name'])
    }
    return render(request, 'monitoring/inventory.html', context)

def settings_view(request):
    camera_form = CameraConfigForm()
    alert_form = AlertRuleForm()
    
    if request.method == 'POST':
        form_type = request.POST.get('form_type')
        if form_type == 'camera':
            camera_form = CameraConfigForm(request.POST)
            if camera_form.is_valid():
                camera_form.save()
                messages.success(request, 'Camera configuration saved.')
                return redirect('monitoring:settings')
        elif form_type == 'alert':
            alert_form = AlertRuleForm(request.POST)
            if alert_form.is_valid():
                alert_form.save()
                messages.success(request, 'Alert rule saved.')
                return redirect('monitoring:settings')

    cameras = CameraConfig.objects.all()
    alert_rules = AlertRule.objects.all()
    
    context = {
        'camera_form': camera_form,
        'alert_form': alert_form,
        'cameras': cameras,
        'alert_rules': alert_rules,
    }
    return render(request, 'monitoring/settings.html', context)

@require_POST
def delete_camera(request, pk):
    camera = get_object_or_404(CameraConfig, pk=pk)
    try:
        camera.delete()
        messages.success(request, f"Camera '{camera.name}' deleted.")
    except Exception as e:
        messages.error(request, f"Cannot delete camera: {str(e)}")
    return redirect('monitoring:settings')

@require_POST
def delete_alert(request, pk):
    rule = get_object_or_404(AlertRule, pk=pk)
    rule.delete()
    messages.success(request, "Alert rule deleted.")
    return redirect('monitoring:settings')


def export_table(request, dataset):
    try:
        return export_table_response(dataset, request)
    except ValueError as exc:
        return HttpResponseBadRequest(str(exc))


def export_session_table(request, pk, dataset):
    dataset_map = {
        'samples': 'session_samples',
        'alerts': 'session_alerts',
    }
    mapped = dataset_map.get(dataset)
    if not mapped:
        return HttpResponseBadRequest('Unknown session export dataset.')

    get_object_or_404(MonitoringSession, pk=pk)
    try:
        return export_table_response(mapped, request, session_id=pk)
    except ValueError as exc:
        return HttpResponseBadRequest(str(exc))


# ---------------------------------------------------------------------------
# AI Co-Pilot Views
# ---------------------------------------------------------------------------
from django.http import JsonResponse
from .assistant import query_co_pilot

@xframe_options_sameorigin
def copilot(request):
    """Render the main AI Co-pilot dashboard (also embedded in the floating drawer iframe)."""
    api_key_set = bool(settings.GROQ_API_KEY)
    is_embed = request.GET.get('embed') == 'true'
    template = 'monitoring/copilot_embed.html' if is_embed else 'monitoring/copilot.html'
    context = {
        'api_key_set': api_key_set,
        'page_title': 'AI Co-pilot',
        'is_embed': is_embed,
    }
    return render(request, template, context)


@require_POST
def copilot_chat_api(request):
    """API endpoint to receive chat messages and invoke the Groq AI engine."""
    try:
        body = json.loads(request.body)
        prompt = body.get('prompt', '').strip()
    except (json.JSONDecodeError, TypeError):
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON request.'}, status=400)
        
    if not prompt:
        return JsonResponse({'status': 'error', 'message': 'Prompt cannot be empty.'}, status=400)
        
    # Get conversation history from the session
    chat_history = request.session.get('assistant_chat_history', [])
    
    try:
        # Query the AI Co-pilot engine (which handles database tool calling)
        result = query_co_pilot(prompt, conversation_history=chat_history)
        
        # Save interaction to session history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": result['message']})
        
        # Prevent session bloating (keep last 20 messages)
        request.session['assistant_chat_history'] = chat_history[-20:]
        
        return JsonResponse({
            'status': 'success',
            'message': result['message'],
            'chart_data': result['chart_data'],
            'tool_calls': result['tool_calls']
        })
        
    except ValueError as val_err:
        return JsonResponse({
            'status': 'error',
            'error_type': 'missing_api_key',
            'message': str(val_err)
        }, status=400)
    except Exception as e:
        logger.exception("Error in AI Co-pilot chat view")
        return JsonResponse({
            'status': 'error',
            'error_type': 'server_error',
            'message': f"An error occurred while communicating with the AI: {str(e)}"
        }, status=500)


@require_POST
def clear_chat_history(request):
    """Clear chat history in the user's session."""
    if 'assistant_chat_history' in request.session:
        del request.session['assistant_chat_history']
    return JsonResponse({'status': 'success', 'message': 'Chat history cleared successfully.'})
