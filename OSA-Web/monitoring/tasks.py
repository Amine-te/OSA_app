import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import redis
from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from django.conf import settings
from django.utils import timezone

# Add repo root to path so shared module is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.models import AlertEvent, AlertRule, AnalyticsSample, MonitoringSession
from monitoring.utils import normalize_stock_level
from shared.pipelines.enhanced_pipeline import EnhancedRetailPipeline
from shared.visualization.frame_annotator import annotate_frame_bgr


@shared_task(bind=True, name='monitoring.tasks.run_pipeline_session')
def run_pipeline_session(self, session_id):
    """
    Main Celery task that runs the detection/classification pipeline
    against a camera feed for the given MonitoringSession.

    The loop reads frames, runs the EnhancedRetailPipeline, persists
    AnalyticsSamples, evaluates AlertRules, and pushes live updates
    to a WebSocket group via Django Channels.

    The task can be stopped gracefully by setting the Redis key
    ``stop_session_<session_id>`` to any value.
    """

    # ------------------------------------------------------------------
    # 1. Connect to Redis
    # ------------------------------------------------------------------
    r = redis.from_url(settings.REDIS_URL)

    # ------------------------------------------------------------------
    # 2. Fetch session
    # ------------------------------------------------------------------
    try:
        session = MonitoringSession.objects.select_related('camera').get(pk=session_id)
    except MonitoringSession.DoesNotExist:
        return

    # ------------------------------------------------------------------
    # 3-4. Stop key (clean up any stale key from a previous crash)
    # ------------------------------------------------------------------
    stop_key = f"stop_session_{session_id}"
    r.delete(stop_key)

    # ------------------------------------------------------------------
    # 5-6. Channel layer + group name
    # ------------------------------------------------------------------
    channel_layer = get_channel_layer()
    group_name = f"session_{session_id}"

    # ------------------------------------------------------------------
    # 7. Build absolute model paths
    # ------------------------------------------------------------------
    repo_root = Path(settings.BASE_DIR) / '..'
    yolo = repo_root / session.camera.yolo_model_path
    cnn = repo_root / session.camera.cnn_model_path
    void = repo_root / session.camera.void_model_path

    # ------------------------------------------------------------------
    # 8. Instantiate pipeline
    # ------------------------------------------------------------------
    try:
        pipeline = EnhancedRetailPipeline(
            yolo_model_path=str(yolo),
            cnn_model_path=str(cnn),
            void_model_path=str(void),
            class_names=settings.CLASS_NAMES,
            confidence_threshold=session.camera.confidence_threshold,
            void_confidence_threshold=session.camera.void_confidence_threshold,
        )
    except Exception:
        session.status = 'failed'
        session.save()
        return

    # ------------------------------------------------------------------
    # 9. Open video capture
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(session.camera.rtsp_url)
    if not cap.isOpened():
        session.status = 'failed'
        session.save()
        return

    # ------------------------------------------------------------------
    # 10. Main processing loop
    # ------------------------------------------------------------------
    frame_count = 0

    try:
        while True:
            # (a) Check for stop signal
            if r.exists(stop_key):
                break

            # (b) Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # (c) Skip frames per camera config
            if frame_count % session.camera.frame_skip != 0:
                frame_count += 1
                continue

            # (d) Start timer
            t_start = time.time()

            # (e) Write frame to temp file
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = tmp.name
            try:
                cv2.imwrite(temp_path, frame)

                # (f) Run pipeline
                result = pipeline.detect_and_classify_complete(temp_path)
            finally:
                # (g) Clean up temp file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

            # (h) Compute latency & FPS
            latency_ms = (time.time() - t_start) * 1000
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            # (i) Extract summary metrics
            summary = result.get('summary', {})
            total_products = summary.get('total_products_detected', 0)
            missing_products = summary.get('estimated_missing_products', 0)
            void_detections = summary.get('void_detections', 0)
            stock_pct = summary.get('overall_stock_percentage', 0.0)
            stock_levels = summary.get('stock_levels', {})

            # (j) Persist analytics sample
            AnalyticsSample.objects.create(
                session=session,
                total_products=total_products,
                missing_products=missing_products,
                void_detections=void_detections,
                stock_pct=stock_pct,
                fps=fps,
                latency_ms=latency_ms,
                summary_json=summary,
            )

            # (k) Check alert rules
            for rule in AlertRule.objects.filter(enabled=True):
                if rule.product_name is None:
                    # Global rule
                    if stock_pct < rule.threshold_pct:
                        AlertEvent.objects.create(
                            session=session,
                            rule=rule,
                            stock_pct_at_trigger=stock_pct,
                        )
                else:
                    # Per-product rule
                    prod_data = stock_levels.get(rule.product_name, {})
                    prod_pct = prod_data.get('stock_percentage', 100.0)
                    if prod_pct < rule.threshold_pct:
                        AlertEvent.objects.create(
                            session=session,
                            rule=rule,
                            product_name=rule.product_name,
                            stock_pct_at_trigger=prod_pct,
                        )

            # (l) Encode annotated frame as base64 JPEG
            annotated_frame = annotate_frame_bgr(frame, result, settings.CLASS_NAMES)
            _, buf = cv2.imencode(
                '.jpg',
                annotated_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70],
            )
            img_b64 = base64.b64encode(buf).decode('utf-8')

            # (m) Build per-product list for inventory table
            products_list = [
                {
                    'name': k,
                    **normalize_stock_level(v),
                }
                for k, v in stock_levels.items()
            ]

            # (n) Push live update to WebSocket group
            async_to_sync(channel_layer.group_send)(group_name, {
                'type': 'stream.update',
                'data': {
                    'annotated_image_b64': img_b64,
                    'stock_pct': stock_pct,
                    'missing_count': missing_products,
                    'total_products': total_products,
                    'void_detections': void_detections,
                    'fps': round(fps, 1),
                    'latency_ms': round(latency_ms, 1),
                    'products': products_list,
                    'alert_events': [],
                },
            })

            frame_count += 1
            time.sleep(0)

    except Exception:
        # 11. Outer exception handler — mark session failed & re-raise
        session.refresh_from_db()
        session.status = 'failed'
        session.save()
        raise

    finally:
        # Always clean up
        cap.release()
        r.delete(stop_key)
        session.refresh_from_db()
        if session.status != 'failed':
            session.status = 'ended'
        session.ended_at = timezone.now()
        session.save()
