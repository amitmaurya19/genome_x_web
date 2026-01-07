from django.contrib import admin
from django.urls import path
from dashboard import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('download/', views.download_csv, name='download_csv'),
]

# IMPORTANT: enable media handling (for file uploads)
if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )
