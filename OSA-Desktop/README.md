# OSA Desktop Application

Welcome to the **OSA (On-Shelf Availability) Desktop Application**, an Industrial Control Center designed with PyQt6 for intelligent retail shelf analysis. It gives users complete insight into their live retail inventory deployments through a fast, real-time RTSP AI pipeline.

## 🚀 Application Structure

The application is structured into the following key directories and files:

```text
OSA-Desktop/
├── main.py                     # Primary entry point. Loads configurations, sets up the Light theme, and launches the UI.
├── config.yaml                 # Centralized configuration file mapping model paths, class lists, and default preferences.
├── ui/                         # User Interface modules (PyQt6)
│   ├── main_window.py          # The core window connecting the live RTSP viewer and docking system.
│   ├── auxiliary_windows.py    # Detached modular windows for Configuration, Analytics, and Inventory Reporting.
│   ├── sidebar.py              # Configuration panel for initializing the ML pipeline and setting thresholds.
│   ├── viewer.py               # Custom visual viewer classes offering split-comparisons, HUD overlays, and zoom/pan functionality.
│   ├── widgets.py              # Modular interactive widgets (Tables, Toast Notifications, Log Console, etc).
│   └── styles.py               # Global QSS and color tokens defining the dual-theme (Docker Light / OSA Dark) layout.
├── workers/                    # Background asynchronous processors
│   ├── pipeline_worker.py      # QThread runner executing the inference pipeline (YOLO/CNN) exclusively for RTSP streams.
│   └── stream_manager.py       # Manager for pulling frames from live RTSP sources gracefully.
└── utils/                      # Helper functionalities
    └── path_utils.py           # Evaluates paths efficiently dynamically mapping the user's OS directory layout.
```

## ✨ Core Functionalities

### 1. Robust AI Inference Pipeline
The desktop application uses the `EnhancedRetailPipeline` to execute object detection and classification models optimized for:
* **Live RTSP Feeds**: Exclusively optimized for analyzing continuous streams from CCTV cameras without the overhead of static media.
* **Hardware Acceleration**: Auto-detects and runs inference on the best available hardware accelerator (MPS, CUDA, or CPU).
* **Model Routing**: Incorporates separate instances for *Individual Products* (YOLO), *Void Spaces* (YOLO), and explicit *Product Classes* (CNN).

### 2. Live Monitoring & Interaction
- **Live Detection Feed**: Connect straight into an active real-time CCTV `rtsp://` link. Allows bounding box interactivity (clicking table rows highlights the product bounding box) and live HUD tracking.
- **Detached Analytics**: History, KPIs, and performance hardware trends (latency/fps) are collected in a dedicated, multi-tabbed analytics window.

### 3. Comprehensive Dashboard Panels
The dashboard translates raw model detections into simple reporting structures, exposing:
- **Inventory & Report Window**: View live updating stock tables detailing exact capacity and missing items.
- **Export Data**: Automatically generate PDF, CSV, or JSON summaries out of the session report.
- **Trend KPIs**: Real-time plots tracking stock health trends and missing product events.

### 4. Modern, Responsive UI
- **Dual-Theme Support**: Built-in support for a sleek, Docker-inspired Light Theme (default) and a deep, OSA Original Dark Theme.
- **Log Console**: Displays background thread debug logs dynamically in a dockable pane.
- **HUD Overlays**: Displays FPS drop reports, exact latency timing in MS, and hardware execution status over the video window.
- **Toast Notifications**: Interactive non-obstructive popup alarms for pipeline success/errors.

## ⚙️ Configuration Setup (`config.yaml`)

Edit `config.yaml` to tailor your tracking behaviors and AI setups:
* Map new `.pt` / `.pth` models targeting different SKUs.
* Adjust confidence threshold logic.
* Manage classes matching your CNN classification pipeline requirements.

## 🏁 How to Run

Activate your environment and use:
```bash
python -m OSA-Desktop.main
```
