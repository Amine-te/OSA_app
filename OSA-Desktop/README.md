# OSA-Desktop

**PyQt6 Industrial Control Centre** for the On-Shelf Availability (OSA) system.  
Designed for single-operator, single-camera live RTSP monitoring with AI-powered shelf analysis.

---

## Quick Start

```bash
# From the repo root
cd OSA-Desktop
pip install -r requirements.txt
python main.py
```

> The application adds the repo root to `sys.path` automatically, so the `shared/` AI engine is importable without any extra setup.

---

## Application Structure

```
OSA-Desktop/
├── main.py                     # Entry point — loads config, sets theme, launches UI
├── config.yaml                 # Model paths, class lists, notification thresholds
├── requirements.txt            # All Python dependencies
├── core/                       # Core application logic
│   ├── app_state.py            # Centralised mutable state (pipeline, source, detections)
│   ├── event_bus.py            # Decoupled PyQt signal bus (UI ↔ worker communication)
│   ├── history_store.py        # WAL-enabled SQLite persistence for analytics sessions
│   ├── notification_engine.py  # Threshold checking, cooldowns, recovery detection
│   └── session_manager.py      # Save/restore window layouts and application state
├── ui/                         # PyQt6 user interface modules
│   ├── main_window.py          # Root window — live RTSP viewer + docking system
│   ├── auxiliary_windows.py    # Detached windows for Configuration, Analytics, Inventory
│   ├── notification_center.py  # Dockable panel with alert cards and badge counter
│   ├── sidebar.py              # Config panel for pipeline init and threshold settings
│   ├── viewer.py               # Split-view, HUD overlays, zoom/pan
│   ├── widgets.py              # Tables, toast notifications, log console, etc.
│   ├── error_banner.py         # Retryable error banner
│   ├── performance_panel.py    # Real-time FPS and latency graphs
│   └── styles.py               # QSS tokens for Light and Dark themes
├── workers/                    # Background QThreads
│   ├── pipeline_worker.py      # Runs YOLO/CNN inference on RTSP frames
│   └── stream_manager.py       # Graceful RTSP frame ingestion
└── utils/
    └── path_utils.py           # OS-aware path resolution
```

---

## Core Features

### AI Inference Pipeline
Powered by `shared.pipelines.EnhancedRetailPipeline`:
- **Live RTSP** — continuous CCTV stream analysis (no static media overhead)
- **Hardware acceleration** — auto-selects MPS → CUDA → CPU
- **Model routing** — separate YOLO instances for products and void spaces, CNN for SKU classification

### Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+,` | Open Configuration window |
| `Ctrl+N` | Toggle Notification Center |
| `Ctrl+Shift+T` | Toggle Light / Dark theme |
| `Ctrl+E` | Quick export report |
| `Ctrl+S` | Screenshot of live annotated feed |
| `` Ctrl+` `` | Toggle Log Console dock |
| `F11` | Toggle fullscreen |
| `Ctrl+/` | View all shortcuts |

### Alert System
- **Warning** (default 70%) and **Critical** (default 50%) stock thresholds
- Per-product and aggregate monitoring, smart cooldowns (default 30 s)
- Recovery notifications when stock returns above warning level
- Channels: Notification Center dock, toast popups, log console, system beep

---

## Configuration (`config.yaml`)

```yaml
models:
  yolo:    ../models/sku/best.pt
  cnn:     ../models/classifier/model.pth
  void:    ../models/void/best.pt

notifications:
  warning_threshold: 70   # Stock % below this → warning
  critical_threshold: 50  # Stock % below this → critical
  cooldown_seconds: 30    # Min gap between repeated alerts per product
```
