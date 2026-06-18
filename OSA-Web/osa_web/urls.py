"""
URL configuration for osa_web project.
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(('monitoring.urls', 'monitoring'), namespace='monitoring')),
]
