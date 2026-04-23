"""Centralized mutable application state for the OSA control center."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional


class PipelineState(Enum):
    """High-level pipeline / transport lifecycle."""

    IDLE = auto()
    LOADING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()


class SourceType(Enum):
    NONE = auto()
    IMAGE = auto()
    VIDEO = auto()
    RTSP = auto()


@dataclass
class ROIRecord:
    """Single region of interest in image coordinates."""

    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class AppState:
    """
    Single source of truth for UI and session persistence.
    Mutated only on the main thread; workers communicate via EventBus.
    """

    current_source: SourceType = SourceType.NONE
    source_path: str = ""
    current_frame: Optional[Any] = None  # last raw frame / result image (numpy) if needed
    detections: List[dict] = field(default_factory=list)
    void_detections: List[dict] = field(default_factory=list)
    rois: List[ROIRecord] = field(default_factory=list)
    active_roi_preset: str = ""
    pipeline_state: PipelineState = PipelineState.IDLE
    device: str = "—"
    last_error: str = ""
    current_workspace_index: int = 0
    video_frame_index: int = 0
    video_is_playing: bool = True
    focus_mode: bool = False
    selected_detection_index: int = -1
    viewer_compare_mode: str = "slider"  # slider | side | toggle
    heatmap_enabled: bool = False

    # Analytics (video/live): current session id for history storage
    analytics_session_id: str = ""

    # Last full pipeline result dict for reports / export
    last_results: Optional[dict] = None

    def reset_session_media(self) -> None:
        self.current_source = SourceType.NONE
        self.source_path = ""
        self.current_frame = None
        self.detections = []
        self.void_detections = []
        self.last_results = None
        self.video_frame_index = 0

    def set_pipeline(self, state: PipelineState, error: str = "") -> None:
        self.pipeline_state = state
        self.last_error = error if state == PipelineState.ERROR else ""
