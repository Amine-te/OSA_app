import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'osa_web.settings')

app = Celery('osa_web')

# Read config from Django settings, using the CELERY_ namespace so that
# all Celery-related settings keys must be prefixed with CELERY_.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all installed apps.
app.autodiscover_tasks()
