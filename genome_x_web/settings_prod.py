from .settings import *

DEBUG = False

ALLOWED_HOSTS = ["*"]

STATIC_ROOT = BASE_DIR / "staticfiles"

MIDDLEWARE = [
    "whitenoise.middleware.WhiteNoiseMiddleware",
] + MIDDLEWARE
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
