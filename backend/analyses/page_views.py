"""HTML 화면: 사용자 분석 화면과 관리자 전용 대시보드(실제 DB 집계)."""

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count
from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.shortcuts import render
from django.views.static import serve

from .models import ACTION_CHOICES, AnalysisJob, PersonPrediction


@login_required
def protected_media(request, path):
    """영상 파일 제공: 해당 작업의 주인이거나 관리자일 때만."""
    jobs = AnalysisJob.objects.filter(video=path)
    if not request.user.is_staff:
        jobs = jobs.filter(user=request.user)
    if not jobs.exists():
        raise Http404
    return serve(request, path, document_root=settings.MEDIA_ROOT)


@login_required
def index(request):
    return render(request, "analyses/index.html", {
        "cameras": list(settings.ALLOWED_CAMERAS.keys()),
        "capture_seconds": settings.CAMERA_CAPTURE_SECONDS,
        "actions": [a for a, _ in ACTION_CHOICES],
        "max_upload_mb": settings.MAX_UPLOAD_MB,
    })


@login_required
def dashboard(request):
    # 로그인한 일반 회원은 로그인 화면으로 보내지 않고 403(권한 없음)을 보여 준다(리디렉션 반복 방지).
    if not request.user.is_staff:
        raise PermissionDenied("관리자만 볼 수 있습니다.")
    # 1) 동작별 예측 건수 (17개 모두 표시, 0건 포함)
    counts = dict(PersonPrediction.objects.values_list("action_label").annotate(n=Count("id")))
    action_labels = [a for a, _ in ACTION_CHOICES]
    action_counts = [counts.get(a, 0) for a in action_labels]

    # 2) 상태별 작업 건수
    status_rows = dict(AnalysisJob.objects.values_list("status").annotate(n=Count("id")))
    status_labels = [label for _, label in AnalysisJob.Status.choices]
    status_counts = [status_rows.get(value, 0) for value, _ in AnalysisJob.Status.choices]

    # 선택: 입력 방식별 건수·평균 처리 시간
    by_input = (AnalysisJob.objects.values("input_type")
                .annotate(n=Count("id"), avg_ai=Avg("ai_processing_sec")).order_by("input_type"))

    chart_data = {
        "actions": {"labels": action_labels, "counts": action_counts},
        "statuses": {"labels": status_labels, "counts": status_counts},
    }
    return render(request, "analyses/dashboard.html", {
        "chart_data": chart_data,
        "total_jobs": AnalysisJob.objects.count(),
        "total_predictions": PersonPrediction.objects.count(),
        "uncertain_predictions": PersonPrediction.objects.filter(uncertain=True).count(),
        "by_input": [
            {"label": AnalysisJob.InputType(r["input_type"]).label, "n": r["n"],
             "avg_ai": round(r["avg_ai"], 1) if r["avg_ai"] is not None else None}
            for r in by_input
        ],
        "recent_jobs": (AnalysisJob.objects.select_related("user")
                        .annotate(n_persons=Count("predictions")).order_by("-created_at")[:15]),
    })
