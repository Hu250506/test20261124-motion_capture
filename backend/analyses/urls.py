from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import page_views
from .views import AnalysisJobViewSet

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysis")

app_name = "analyses"

urlpatterns = [
    path("analyses/", page_views.index, name="index"),
    path("dashboard/", page_views.dashboard, name="dashboard"),
    path("cameras/<str:name>/snapshot.jpg", page_views.camera_snapshot, name="camera_snapshot"),
    path("cameras/<str:name>/pose.json", page_views.camera_pose, name="camera_pose"),
    path("api/v1/", include(router.urls)),
]
