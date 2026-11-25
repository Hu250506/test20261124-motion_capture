from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

from analyses.page_views import protected_media

admin.site.site_header = "Vision Action 관리자"
admin.site.site_title = "Vision Action 관리자"

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="analyses:index", permanent=False)),
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("", include("analyses.urls")),
]

# 업로드·수집 영상은 작업의 주인과 관리자만 볼 수 있다.
urlpatterns += [path("media/<path:path>", protected_media, name="media")]
