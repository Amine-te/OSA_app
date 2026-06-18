# Shared AI Engine

Shared Python library used by both **OSA-Desktop** and **OSA-Web**.  
Contains the complete computer-vision pipeline for retail shelf analysis.

## Modules

| Module | Description |
|---|---|
| `pipelines/` | `EnhancedRetailPipeline` — main orchestrator |
| `detection/` | YOLO-based product and void-space detection |
| `networks/` | CNN classifier head for SKU identification |
| `analysis/` | Scoring, shelf pattern analysis, void assignment |
| `frame_sources/` | Frame ingestion adapters (RTSP, file, webcam) |
| `visualization/` | `annotate_frame_bgr` and other overlay utilities |
| `reporting/` | Report builders (CSV, JSON, PDF) |
| `config.py` | Shared configuration dataclasses |

## Usage

Both apps add the **repository root** to `sys.path` at startup, so imports work from either location:

```python
from shared.pipelines.enhanced_pipeline import EnhancedRetailPipeline
from shared.visualization.frame_annotator import annotate_frame_bgr
```

## Requirements

Install the core ML dependencies before importing:

```bash
# From the repo root
pip install torch>=2.0.0 torchvision>=0.15.0 ultralytics>=8.0.0
pip install opencv-python>=4.8.0 numpy>=1.24.0 pillow>=10.0.0
```
