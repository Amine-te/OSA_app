"""Persist and restore sessions (layout, ROI, config snapshot, detections)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt6.QtCore import QByteArray, QSettings
from PyQt6.QtWidgets import QMainWindow

from core.app_state import AppState, SourceType


def sessions_root(base: Path) -> Path:
    p = base / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _serialize_state(state: AppState) -> Dict[str, Any]:
    return {
        "source_type": state.current_source.name,
        "source_path": state.source_path,
        "heatmap_enabled": state.heatmap_enabled,
        "focus_mode": state.focus_mode,
    }


def _deserialize_state(data: Dict[str, Any], state: AppState) -> None:
    try:
        state.current_source = SourceType[data.get("source_type", "NONE")]
    except KeyError:
        state.current_source = SourceType.NONE
    state.source_path = data.get("source_path", "") or ""
    # Backwards-compat: clear any legacy ROI fields if present.
    state.rois = []
    state.active_roi_preset = ""
    state.heatmap_enabled = bool(data.get("heatmap_enabled", False))
    state.focus_mode = bool(data.get("focus_mode", False))


def save_session(
    base_dir: Path,
    state: AppState,
    config: dict,
    main_window: QMainWindow,
    detections_payload: Optional[dict] = None,
    auxiliary_windows: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write session folder with state.json, config.yaml copy, layout.dat, optional results.json."""
    root = sessions_root(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / f"session_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "saved_at": datetime.now().isoformat(),
        "state": _serialize_state(state),
    }
    if auxiliary_windows:
        payload["auxiliary_windows"] = auxiliary_windows
    (folder / "state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        import yaml

        (folder / "config_snapshot.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    except Exception:
        (folder / "config_snapshot.yaml").write_text("{}", encoding="utf-8")

    if detections_payload is not None:
        (folder / "detections.json").write_text(
            json.dumps(detections_payload, default=str, indent=2), encoding="utf-8"
        )

    geom = main_window.saveState()
    with open(folder / "layout.dat", "wb") as f:
        f.write(bytes(geom))

    # Pointer to last session for auto-load
    (root / "last_session.txt").write_text(str(folder), encoding="utf-8")
    return folder


def load_last_session_path(base_dir: Path) -> Optional[Path]:
    root = base_dir / "sessions"
    last = root / "last_session.txt"
    if not last.exists():
        return None
    p = Path(last.read_text(encoding="utf-8").strip())
    return p if p.is_dir() else None


def load_session_state(folder: Path, state: AppState) -> None:
    path = folder / "state.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    _deserialize_state(data.get("state", {}), state)


def load_auxiliary_windows_payload(folder: Path) -> Dict[str, Any]:
    """Geometry / visibility for detached windows and optional docks (from state.json)."""
    path = folder / "state.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("auxiliary_windows")
    return raw if isinstance(raw, dict) else {}


def restore_window_layout(main_window: QMainWindow, folder: Path) -> bool:
    layout_file = folder / "layout.dat"
    if not layout_file.exists():
        return False
    data = layout_file.read_bytes()
    return main_window.restoreState(QByteArray(data))


def settings_for_app(base_dir: Path) -> QSettings:
    return QSettings(str(base_dir / "osa_desktop.ini"), QSettings.Format.IniFormat)
