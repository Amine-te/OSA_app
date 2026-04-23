"""Central application state and event infrastructure for OSA Desktop."""

from core.app_state import AppState, PipelineState, SourceType
from core.event_bus import EventBus

__all__ = ["AppState", "PipelineState", "SourceType", "EventBus"]
