# OSA-Web

**Django web dashboard** for the On-Shelf Availability (OSA) system.  
Multi-camera monitoring with real-time WebSocket frame streaming, Celery task queue, and AI-powered shelf analysis.

---

## Prerequisites

- Python 3.10+
- Redis 6+ (Channels layer + Celery broker)

---

## Quick Start

```bash
# From the repo root
cd OSA-Web
pip install -r requirements.txt

# Copy and edit environment config
cp .env.example .env

# Run database migrations
python manage.py migrate

# Terminal 1 — Redis
redis-server

# Terminal 2 — Celery worker
# (use --pool=solo on macOS to avoid OpenCV/PyTorch fork issues)
python -m celery -A osa_web worker --loglevel=info --pool=solo

# Terminal 3 — ASGI server
daphne -b 0.0.0.0 -p 8000 osa_web.asgi:application

# Open in browser
open http://localhost:8000
```

---

## Project Structure

```
OSA-Web/
├── manage.py
├── .env.example              # Environment template — copy to .env and fill in secrets
├── requirements.txt
├── osa_web/                  # Django project package
│   ├── settings.py           # Adds repo root to sys.path so shared/ is importable
│   ├── urls.py
│   ├── asgi.py
│   ├── celery.py
│   └── wsgi.py
└── monitoring/               # Main Django application
    ├── models.py             # MonitoringSession, Camera, AnalyticsSample, AlertRule
    ├── views.py
    ├── urls.py
    ├── forms.py
    ├── consumers.py          # WebSocket consumers (Django Channels)
    ├── tasks.py              # Celery tasks — runs shared.pipelines.EnhancedRetailPipeline
    ├── routing.py            # WebSocket URL routing
    └── templates/
        └── monitoring/
            ├── base.html
            ├── dashboard.html
            ├── sessions.html
            ├── session_detail.html
            ├── analytics.html
            ├── inventory.html
            └── settings.html
```

---

## Shared AI Engine

`monitoring/tasks.py` imports the detection pipeline from the shared library at the repo root:

```python
from shared.pipelines.enhanced_pipeline import EnhancedRetailPipeline
from shared.visualization.frame_annotator import annotate_frame_bgr
```

`osa_web/settings.py` inserts the repo root into `sys.path` at startup so these imports resolve both for the Django dev server and for Celery workers.

---

## Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | `change-me-in-production` | Django secret key |
| `DEBUG` | `True` | Debug mode |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `GROQ_API_KEY` | *(none)* | Optional AI co-pilot key |
