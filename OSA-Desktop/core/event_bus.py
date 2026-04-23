"""Decouples UI from workers via typed PyQt signals (main thread)."""

from PyQt6.QtCore import QObject, pyqtSignal


class EventBus(QObject):
    """
    All UI-facing updates from inference should flow through these signals.
    MainWindow bridges PipelineWorker → EventBus → widgets/state.
    """

    frame_updated = pyqtSignal(object)  # pipeline results dict
    detections_updated = pyqtSignal(list, list)  # product_dets, void_dets
    pipeline_status_changed = pyqtSignal(object)  # PipelineState
    error_occurred = pyqtSignal(str)
