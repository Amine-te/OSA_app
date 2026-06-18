from django.contrib import admin

from .models import (
    AlertEvent,
    AlertRule,
    AnalyticsSample,
    CameraConfig,
    MonitoringSession,
)


@admin.register(CameraConfig)
class CameraConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'rtsp_url', 'frame_skip', 'confidence_threshold', 'created_at')
    search_fields = ('name', 'rtsp_url')
    list_filter = ('created_at',)


@admin.register(MonitoringSession)
class MonitoringSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'camera', 'status', 'started_at', 'ended_at')
    list_filter = ('status', 'started_at')
    search_fields = ('camera__name',)
    raw_id_fields = ('camera',)


@admin.register(AnalyticsSample)
class AnalyticsSampleAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'timestamp', 'total_products', 'missing_products', 'stock_pct', 'fps')
    list_filter = ('timestamp',)
    raw_id_fields = ('session',)


@admin.register(AlertRule)
class AlertRuleAdmin(admin.ModelAdmin):
    list_display = ('product_name', 'threshold_pct', 'severity', 'enabled')
    list_filter = ('severity', 'enabled')
    search_fields = ('product_name',)


@admin.register(AlertEvent)
class AlertEventAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'rule', 'product_name', 'stock_pct_at_trigger', 'triggered_at')
    list_filter = ('triggered_at', 'rule__severity')
    raw_id_fields = ('session', 'rule')
