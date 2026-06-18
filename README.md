# On-Shelf Availability (OSA) Analysis

> **AI-powered retail shelf monitoring** — real-time detection of missing products and void spaces on retail shelves, delivered as both a desktop control centre and a web dashboard.

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue?logo=readthedocs)](https://osa-app.readthedocs.io/fr/latest/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/UI-PyQt6-green)](https://www.riverbankcomputing.com/software/pyqt/)
[![Django](https://img.shields.io/badge/web-Django%204%2B-green?logo=django)](https://www.djangoproject.com/)

---

## Repository Structure

```
OSA_app/
├── OSA-Desktop/        # PyQt6 desktop control centre (RTSP live feed)
├── OSA-Web/            # Django web dashboard (multi-camera monitoring)
├── shared/             # Shared AI engine — detection, classification, pipelines
├── models/             # Trained model weights (YOLO, CNN, Void)
├── notebooks/          # Research & experimentation notebooks
├── docs/               # Sphinx documentation (ReadTheDocs)
├── .readthedocs.yaml
└── .gitignore
```

---

## Applications

### 🖥️ OSA-Desktop
**PyQt6 industrial control centre** for single-camera live RTSP monitoring.

| Feature | Details |
|---|---|
| Live feed | RTSP stream with bounding-box HUD |
| AI pipeline | YOLO (products + voids) + CNN classifier |
| Themes | Docker Light / OSA Dark (toggle `Ctrl+Shift+T`) |
| Alerts | Threshold-based notification centre + toast popups |
| Export | CSV / JSON / PDF session reports |

**Quick start:**
```bash
cd OSA-Desktop
pip install -r requirements.txt
python main.py
```

→ Full details: [OSA-Desktop/README.md](OSA-Desktop/README.md)

---

### 🌐 OSA-Web
**Django web dashboard** for multi-camera fleet monitoring with real-time WebSocket updates.

| Feature | Details |
|---|---|
| Live feed | WebSocket-streamed annotated frames |
| Task queue | Celery + Redis for async pipeline execution |
| Analytics | Session history, KPI charts, alert rules |
| Export | CSV / JSON report exports |

**Quick start:**
```bash
cd OSA-Web
pip install -r requirements.txt
cp .env.example .env          # fill in secrets
python manage.py migrate
redis-server &
python -m celery -A osa_web worker --loglevel=info --pool=solo &
daphne -b 0.0.0.0 -p 8000 osa_web.asgi:application
```

→ Full details: [OSA-Web/README.md](OSA-Web/README.md)

---

## Shared AI Engine (`shared/`)

Both applications import the same detection and classification core:

```
shared/
├── pipelines/          # EnhancedRetailPipeline (main entry point)
├── detection/          # YOLO-based shelf and void detection
├── networks/           # CNN classifier heads
├── analysis/           # Scoring, shelf pattern analysis, void assignment
├── frame_sources/      # Frame ingestion (RTSP, file, webcam)
├── visualization/      # Frame annotation utilities
├── reporting/          # Report builders
└── config.py           # Shared configuration dataclasses
```

Both apps add the repo root to `sys.path` at startup so that `from shared.pipelines...` resolves correctly.

---

## Model Weights (`models/`)

```
models/
├── classifier/   # CNN product classifier (.pth)
├── sku/          # YOLO SKU detector (.pt)
└── void/         # YOLO void-space detector (.pt)
```

Model paths are configured in `OSA-Desktop/config.yaml` and via the OSA-Web camera settings.  
See [models/README.md](models/README.md) for the expected file layout and how to plug in new weights.

---

## Documentation

Full documentation (installation, architecture, model details, API reference) is hosted on ReadTheDocs:

**[📖 https://osa-app.readthedocs.io/fr/latest/](https://osa-app.readthedocs.io/fr/latest/)**

To build locally:
```bash
cd docs
pip install -r requirements.txt
make html
```

---

## Requirements

| Component | Minimum version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0 |
| Ultralytics (YOLO) | 8.0 |
| Redis | 6.0 (OSA-Web only) |

Each application has its own `requirements.txt`. Install only what you need.

---

_Revolutionizing Retail with AI-Powered Solutions_
