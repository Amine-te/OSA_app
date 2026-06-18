# Model Weights

This directory holds the trained model weight files used by both OSA apps.

> [!NOTE]
> Model `.pt` / `.pth` files are excluded from version control via `.gitignore`.  
> Use **Git LFS** or store them in a shared object-storage bucket (S3, GCS, etc.).

## Directory Layout

```
models/
├── classifier/
│   ├── model.pth          # CNN product classifier weights
│   └── model_info.json    # {"class_names": ["cola", "juice", ...]}
├── sku/
│   └── best.pt            # YOLO SKU detector weights
└── void/
    └── best.pt            # YOLO void-space detector weights
```

## Adding a New Model

1. Drop the weight file into the appropriate subdirectory.
2. Update the path in:
   - **OSA-Desktop**: `OSA-Desktop/config.yaml` under `models:`
   - **OSA-Web**: Camera settings in the admin panel (`/admin/monitoring/camera/`)
3. For the CNN classifier, update `classifier/model_info.json` with the new `class_names` list.

## `model_info.json` Schema

```json
{
  "class_names": ["product_a", "product_b", "product_c"],
  "num_classes": 3,
  "input_size": [224, 224]
}
```

`OSA-Web/osa_web/settings.py` reads `class_names` from this file at startup.
