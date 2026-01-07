import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE',
    'genome_x_web.settings_prod'
)

application = get_wsgi_application()
