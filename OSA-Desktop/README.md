# OSA Desktop Application

Welcome to the **OSA (On-Shelf Availability) Desktop Application**, an Industrial Control Center designed with PyQt6 for intelligent retail shelf analysis. It gives users complete insight into their live retail inventory deployments through a fast, real-time RTSP AI pipeline.

## 🚀 Application Structure

The application is structured into the following key directories and files:

```text
OSA-Desktop/
├── main.py                     # Primary entry point. Loads configurations, sets up the Light theme, and launches the UI.
├── config.yaml                 # Centralized configuration file mapping model paths, class lists, and default preferences.
├── core/                       # Core application logic
│   ├── app_state.py            # Centralized mutable application state (pipeline state, source, detections).
│   ├── event_bus.py            # Decoupled PyQt signal bus for UI ↔ worker communication.
│   ├── history_store.py        # WAL-enabled SQLite persistence layer for analytics session data.
│   ├── notification_engine.py  # Alert evaluation engine — threshold checking, cooldowns, and recovery detection.
│   └── session_manager.py      # Session save/restore for window layouts and application state.
├── ui/                         # User Interface modules (PyQt6)
│   ├── main_window.py          # The core window connecting the live RTSP viewer and docking system.
│   ├── auxiliary_windows.py    # Detached modular windows for Configuration, Analytics, and Inventory Reporting.
│   ├── notification_center.py  # Dockable notification panel with alert cards, badge counter, and inline settings.
│   ├── sidebar.py              # Configuration panel for initializing the ML pipeline and setting thresholds.
│   ├── viewer.py               # Custom visual viewer classes offering split-comparisons, HUD overlays, and zoom/pan.
│   ├── widgets.py              # Modular interactive widgets (Tables, Toast Notifications, Log Console, etc).
│   ├── error_banner.py         # Retryable error banner displayed at the top of the main window.
│   ├── performance_panel.py    # Real-time FPS and latency monitoring panel with live graphs.
│   └── styles.py               # Global QSS and color tokens defining the dual-theme (Docker Light / OSA Dark) layout.
├── workers/                    # Background asynchronous processors
│   ├── pipeline_worker.py      # QThread runner executing the inference pipeline (YOLO/CNN) for RTSP streams.
│   └── stream_manager.py       # Manager for pulling frames from live RTSP sources gracefully.
└── utils/                      # Helper functionalities
    └── path_utils.py           # Evaluates paths efficiently, dynamically mapping the user's OS directory layout.
```

## ✨ Core Functionalities

### 1. Robust AI Inference Pipeline
The desktop application uses the `EnhancedRetailPipeline` to execute object detection and classification models optimized for:
* **Live RTSP Feeds**: Exclusively optimized for analyzing continuous streams from CCTV cameras without the overhead of static media.
* **Hardware Acceleration**: Auto-detects and runs inference on the best available hardware accelerator (MPS, CUDA, or CPU).
* **Model Routing**: Incorporates separate instances for *Individual Products* (YOLO), *Void Spaces* (YOLO), and explicit *Product Classes* (CNN).

### 2. Live Monitoring & Interaction
- **Live Detection Feed**: Connect straight into an active real-time CCTV `rtsp://` link. Allows bounding box interactivity (clicking table rows highlights the product bounding box) and live HUD tracking.
- **RTSP Connection History**: The application remembers the last 10 unique RTSP streams you've connected to, available via a convenient dropdown for one-click reconnects.
- **Quick Controls & Shortcuts**:
  - `Ctrl+,` : Open Configuration window.
  - `Ctrl+N` : Toggle Notification Center.
  - `Ctrl+Shift+T` : Toggle Light/Dark Theme dynamically.
  - `Ctrl+E` : Quick export report.
  - `Ctrl+S` : Instantly save a screenshot of the live annotated feed.
  - `Ctrl+`` ` : Toggle Log Console dock.
  - `F11` : Toggle edge-to-edge fullscreen mode.
  - `Ctrl+/` : View all keyboard shortcuts.

### 3. Comprehensive Dashboard Panels
The dashboard translates raw model detections into simple reporting structures, exposing:
- **Inventory & Report Window**: View live updating stock tables detailing exact capacity and missing items.
- **KPI Evolution Dashboard**: A dynamic analytics tab that calculates industry-standard metrics:
  - **OSA Rate (%)**, **OOS Rate (%)**, **Peak Missing**, and **Threshold Events (<80%)**.
  - Selecting "All products" displays a comprehensive comparative data table.
  - Selecting a specific product generates a 2x2 grid of time-series graphs mapping the evolution of that item's availability.
- **Export Data**: Instantly export the filtered KPI tables or session reports to CSV, JSON, or PDF formats.

### 4. Stock Alert Notification System
A real-time notification system that monitors product stock levels against configurable thresholds and alerts the user when intervention is needed:
- **Threshold-Based Alerts**: Configurable **warning** (default 70%) and **critical** (default 50%) stock percentage thresholds.
- **Per-Product & Aggregate Monitoring**: Alerts can fire for individual products or the overall stock level, independently togglable.
- **Smart Cooldowns**: Prevents notification spam with a configurable cooldown period (default 30 seconds) per product.
- **Recovery Notifications**: Optional alerts when a product's stock recovers above the warning threshold.
- **Multi-Channel Display**:
  - 🔔 **Notification Center Dock** — Persistent scrollable panel with color-coded alert cards (critical/warning/info), dismiss buttons, unread badge counter, and a "Clear all" action.
  - 🔔 **Toast Popups** — Non-blocking bottom-right toast notifications with severity-appropriate icons.
  - 🔔 **Log Console** — All alerts are logged with appropriate severity level.
  - 🔊 **System Sound** — Critical alerts trigger an audible system beep (configurable).
- **Inline Settings**: Expand the ⚙ panel inside the Notification Center to adjust thresholds, cooldown, and behavior without leaving the dock.

### 5. Modern, Responsive UI
- **Dual-Theme Support**: Built-in support for a sleek, Docker-inspired Light Theme (default) and a deep, OSA Original Dark Theme. All components — including dock panels, notification cards, and pyqtgraph plots — consistently follow the active theme.
- **Log Console**: Displays background thread debug logs dynamically in a dockable pane with level filtering and search.
- **HUD Overlays**: Displays FPS, exact latency timing in ms, and hardware execution status over the video window.
- **Toast Notifications**: Interactive non-obstructive popup alarms for pipeline success/errors.
- **Session Persistence**: Window layouts, dock visibility, and auxiliary window positions are saved and restored across sessions.

## ⚙️ Configuration Setup (`config.yaml`)

Edit `config.yaml` to tailor your tracking behaviors and AI setups:
* Map new `.pt` / `.pth` models targeting different SKUs.
* Adjust confidence threshold logic.
* Manage classes matching your CNN classification pipeline requirements.
* Configure notification thresholds under the `notifications` key:
  ```yaml
  notifications:
    warning_threshold: 70    # Stock % below this → warning alert
    critical_threshold: 50   # Stock % below this → critical alert
    cooldown_seconds: 30     # Min seconds between repeated alerts per product
  ```

## 🏁 How to Run

Activate your environment and use:
```bash
python -m OSA-Desktop.main
```
