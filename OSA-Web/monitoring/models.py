from django.db import models


class CameraConfig(models.Model):
    """Configuration for a single camera feed used in monitoring."""

    name = models.CharField(max_length=100)
    rtsp_url = models.CharField(max_length=500)
    frame_skip = models.IntegerField(default=5)
    yolo_model_path = models.CharField(
        max_length=500,
        default='models/sku/individual_products.pt',
    )
    cnn_model_path = models.CharField(
        max_length=500,
        default='models/classifier/best_lightweight_cnn.pth',
    )
    void_model_path = models.CharField(
        max_length=500,
        default='models/void/void_0,95_best_one.pt',
    )
    confidence_threshold = models.FloatField(default=0.5)
    void_confidence_threshold = models.FloatField(default=0.5)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class MonitoringSession(models.Model):
    """A single monitoring run tied to a camera configuration."""

    STATUS_CHOICES = [
        ('running', 'Running'),
        ('ended', 'Ended'),
        ('failed', 'Failed'),
    ]

    camera = models.ForeignKey(
        CameraConfig,
        on_delete=models.PROTECT,
    )
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default='running',
    )

    def __str__(self):
        return f"Session {self.id} \u2013 {self.camera.name} ({self.status})"


class AnalyticsSample(models.Model):
    """A single point-in-time snapshot of pipeline metrics for a session."""

    session = models.ForeignKey(
        MonitoringSession,
        on_delete=models.CASCADE,
        related_name='samples',
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    total_products = models.IntegerField(default=0)
    missing_products = models.IntegerField(default=0)
    void_detections = models.IntegerField(default=0)
    stock_pct = models.FloatField(default=0.0)
    fps = models.FloatField(default=0.0)
    latency_ms = models.FloatField(default=0.0)
    summary_json = models.JSONField(default=dict)
    # summary_json stores the full pipeline output dict including per-product
    # breakdown, e.g.:
    # {
    #   "stock_levels": {
    #     "productA": {
    #       "current": 5,
    #       "missing": 1,
    #       "capacity": 6,
    #       "stock_percentage": 83.3
    #     },
    #     ...
    #   }
    # }

    def __str__(self):
        return f"Sample {self.id} @ {self.timestamp:%H:%M:%S}"


class AlertRule(models.Model):
    """Configurable threshold rule that triggers alerts when stock drops."""

    SEVERITY_CHOICES = [
        ('warning', 'Warning'),
        ('critical', 'Critical'),
    ]

    product_name = models.CharField(
        max_length=100,
        null=True,
        blank=True,
    )  # null = global rule
    threshold_pct = models.FloatField(default=80.0)
    severity = models.CharField(
        max_length=10,
        choices=SEVERITY_CHOICES,
        default='warning',
    )
    enabled = models.BooleanField(default=True)

    def __str__(self):
        scope = self.product_name if self.product_name else 'Global'
        return f"{scope} < {self.threshold_pct}% ({self.severity})"


class AlertEvent(models.Model):
    """Record of an alert that fired during a monitoring session."""

    session = models.ForeignKey(
        MonitoringSession,
        on_delete=models.CASCADE,
        related_name='alerts',
    )
    rule = models.ForeignKey(
        AlertRule,
        on_delete=models.CASCADE,
    )
    triggered_at = models.DateTimeField(auto_now_add=True)
    product_name = models.CharField(
        max_length=100,
        null=True,
        blank=True,
    )
    stock_pct_at_trigger = models.FloatField()

    @property
    def severity(self):
        return self.rule.severity

    def __str__(self):
        return (
            f"Alert {self.severity} \u2013 "
            f"{self.product_name or 'Global'} "
            f"@ {self.triggered_at:%H:%M:%S}"
        )
