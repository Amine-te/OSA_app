"""
ASGI config for osa_web project.

Exposes the ASGI callable as a module-level variable named ``application``.
Routes HTTP traffic to Django and WebSocket traffic through Channels.
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'osa_web.settings')

# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing consumers.
django_asgi_app = get_asgi_application()

from monitoring.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter({
    'http': django_asgi_app,
    'websocket': AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})
