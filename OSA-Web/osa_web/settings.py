"""
Django settings for osa_web project.

Configured for the OSA (On-Shelf Availability) monitoring system.
All secrets loaded via python-decouple from .env file.
"""

import json
import sys
from pathlib import Path

from decouple import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Add the repo root so that `shared/` is importable by Celery workers
# (e.g.  from shared.detection import ShelfDetector)
REPO_ROOT = BASE_DIR / '..'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Directory containing trained model artifacts
MODEL_DIR = REPO_ROOT / 'models'

# ---------------------------------------------------------------------------
# Class names – loaded from the classifier model_info.json
# ---------------------------------------------------------------------------

_MODEL_INFO_PATH = MODEL_DIR / 'classifier' / 'model_info.json'
try:
    with open(_MODEL_INFO_PATH, 'r') as f:
        CLASS_NAMES = json.load(f).get(
            'class_names',
            ['product1', 'product2', 'product3'],
        )
except (FileNotFoundError, json.JSONDecodeError):
    CLASS_NAMES = ['product1', 'product2', 'product3']

# ---------------------------------------------------------------------------
# OSA thresholds
# ---------------------------------------------------------------------------

OSA_THRESHOLD = 80.0  # Stock % below this triggers warning/critical alerts

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

SECRET_KEY = config('DJANGO_SECRET_KEY', default='change-me-in-production')

DEBUG = config('DEBUG', default=True, cast=bool)

ALLOWED_HOSTS = ['*']

# Allow same-origin iframe embedding (floating AI co-pilot drawer on all pages)
X_FRAME_OPTIONS = 'SAMEORIGIN'

# ---------------------------------------------------------------------------
# Application definition
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third-party
    'channels',
    # Local
    'monitoring',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'osa_web.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'osa_web.wsgi.application'
ASGI_APPLICATION = 'osa_web.asgi.application'

# ---------------------------------------------------------------------------
# Database – SQLite for development
# ---------------------------------------------------------------------------

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ---------------------------------------------------------------------------
# Django Channels – Redis as channel layer
# ---------------------------------------------------------------------------

REDIS_URL = config('REDIS_URL', default='redis://localhost:6379/0')

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [REDIS_URL],
        },
    },
}

# ---------------------------------------------------------------------------
# Celery – Redis as broker and result backend
# ---------------------------------------------------------------------------

CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'

# macOS + OpenCV/PyTorch: prefork workers crash on fork (SIGABRT/SIGSEGV).
# Use a single-process pool locally; Linux/production can override via env.
import sys
if sys.platform == 'darwin':
    CELERY_WORKER_POOL = 'solo'
    CELERY_WORKER_CONCURRENCY = 1

# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ---------------------------------------------------------------------------
# Internationalization
# ---------------------------------------------------------------------------

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

STATIC_URL = 'static/'

# ---------------------------------------------------------------------------
# Default primary key field type
# ---------------------------------------------------------------------------

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Groq AI Co-Pilot API Key
GROQ_API_KEY = config('GROQ_API_KEY', default=None)
