# OSA Desktop Application

Welcome to the **OSA (On-Shelf Availability) Desktop Application**, an Industrial Control Center designed with PyQt6 for intelligent retail shelf analysis. It gives users complete insight into their real-time and static inventory deployments through a fast and robust local AI pipeline.

## 🚀 Application Structure

The application is structured into the following key directories and files:

```text
OSA-Desktop/
├── main.py                     # Primary entry point. Loads configurations, styles, and launches the UI.
├── config.yaml                 # Centralized configuration file mapping model paths, class lists, and default preferences.
├── assets/                     # Directory for storing icons and interface graphical assets.
├── ui/                         # User Interface modules (PyQt6)
│   ├── main_window.py          # The core window containing the main visual layouts, image/video tabs, and user interactions.
│   ├── sidebar.py              # The configuration sidebar for loading the ML pipeline and setting properties.
│   ├── viewer.py               # Custom visual viewer classes offering split-comparisons, HUD overlays, and zoom/pan functionality.
│   ├── widgets.py              # Modular interactive widgets (Charts, Gauges, Data Tables, Toast Notifications, etc).
│   └── styles.py               # Global QSS and color tokens defining the dark glassmorphism interface layout.
├── workers/                    # Background asynchronous processors
│   ├── pipeline_worker.py      # QThread runner executing the inference pipeline (YOLO/CNN) to prevent GUI freezing.
│   └── stream_manager.py       # Manager for pulling frames from Video and RTSP sources gracefully.
└── utils/                      # Helper functionalities
    └── path_utils.py           # Evaluates paths efficiently dynamically mapping the user's OS directory layout.
```

## ✨ Core Functionalities

### 1. Robust AI Inference Pipeline
The desktop application uses the `EnhancedRetailPipeline` to execute dual-camera object detection and classification models optimized for:
* **MPS (Apple Silicon), CUDA, and CPU**: The app auto-detects the hardware executing inference on the best available hardware accelerator.
* **Model Routing**: Incorporates separate instances for *Individual Products* (YOLO), *Void Spaces* (YOLO), and explicit *Product Classes* (CNN).

### 2. Deep Interactive Views
- **Image Tab**: Supports `.jpg/.png` uploads. Allows deep inspection using zooming, "Before/After" split-views, bounding box interactivity (clicking table rows highlights the product image bounding box), Heatmap representations, and manual ROI (Region of Interest) cropping.
- **Video & RTSP Tab**: Analyze recorded videos or connect straight into an active real-time CCTV `rtsp://` link. Allows frame scrubbing, trend analysis plotting, and automatic logging of percentage stocks.

### 3. Comprehensive Dashboard Panels
The dashboard translates raw model detections into simple reporting structures, exposing:
- **Alert Metrics**: Total missing products vs fully stocked percentages displayed in numeric cards and circular gauges.
- **Detailed Inventories**: Clickable stock tables exposing exactly how much inventory of each class exists per picture frame.
- **Exports Data**: Automatically generates summary data out to CSV or PDF via the Export modules.

### 4. Non-Blocking Information UI
- **Log Console**: Displays background thread debug logs without cluttering the UI. 
- **HUD Overlays**: Displays FPS drop reports, exact latency timing in MS, and hardware execution status over image/video windows.
- **Toast Notifications**: Interactive non-obstructive popup alarms in corners showcasing pipeline success/disconnection logic.

## ⚙️ Configuration Setup (`config.yaml`)

Edit `config.yaml` to tailor your tracking behaviors and AI setups:
* Map new `.pt` / `.pth` models targeting different SKUs.
* Adjust confidence threshold logic directly without entering python code.
* Manage classes matching your CNN classification pipeline requirements.

## 🏁 How to Run

Activate your environment and use:
```bash
python main.py
```
