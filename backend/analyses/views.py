"""DRF 분석 API.

POST   /api/v1/analyses/              작업 생성 → DB 저장 완료 후 Celery 등록 → 202 + 작업 ID
GET    /api/v1/analyses/              내 작업 목록 (관리자는 전체)
GET    /api/v1/analyses/{id}/         상세 + 사람별 결과
PATCH  /api/v1/analyses/{id}/         제목·메모만 수정
DELETE /api/v1/analyses/{id}/         삭제 (예측·영상 파일도 삭제), 처리 중이면 409
GET    /api/v1/analyses/{id}/status/  상태만 가볍게 조회 (화면 폴링용)
"""
import logging

from django.db import transaction
from django.utils import timezone
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import AnalysisJob
from .serializers import AnalysisJobSerializer, AnalysisJobUpdateSerializer, AnalysisStatusSerializer
from .tasks import run_analysis

logger = logging.getLogger(__name__)

class AnalysisJobViewSet(mixins.CreateModelMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin,
                         mixins.UpdateModelMixin, mixins.DestroyModelMixin, viewsets.GenericViewSet):
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]

    def get_queryset(self):
        qs = AnalysisJob.objects.select_related("user").prefetch_related("predictions")
        # 일반 회원은 본인 작업만 보인다. 다른 사람 작업 ID로 접근하면 404.
        return qs if self.request.user.is_staff else qs.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action == "partial_update":
            return AnalysisJobUpdateSerializer
        if self.action == "status":
            return AnalysisStatusSerializer
        return AnalysisJobSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save(user=request.user)
        # DB 저장(커밋)이 끝난 뒤에 작업을 큐에 넣어, Worker가 아직 없는 행을 찾는 일을 막는다.
        transaction.on_commit(lambda: self._enqueue(job.pk))
        data = AnalysisJobSerializer(job, context={"request": request}).data
        return Response(data, status=status.HTTP_202_ACCEPTED)

    @staticmethod
    def _enqueue(job_id):
        try:
            result = run_analysis.delay(job_id)
        except Exception as exc:  # Redis가 꺼져 있으면 '대기'에 멈추지 않도록 실패로 기록한다.
            logger.exception("job #%s 작업 등록 실패", job_id)
            AnalysisJob.objects.filter(pk=job_id).update(
                status=AnalysisJob.Status.FAILED, error_code="queue_unavailable",
                error_message=f"작업 대기열(Redis)에 등록하지 못했습니다: {type(exc).__name__}",
                completed_at=timezone.now(),
            )
            return
        AnalysisJob.objects.filter(pk=job_id, celery_task_id="").update(celery_task_id=result.id or "")

    def partial_update(self, request, *args, **kwargs):
        job = self.get_object()
        serializer = AnalysisJobUpdateSerializer(job, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(AnalysisJobSerializer(job, context={"request": request}).data)

    def destroy(self, request, *args, **kwargs):
        job = self.get_object()
        if job.status == AnalysisJob.Status.PROCESSING:
            return Response({"detail": "처리 중인 작업은 삭제할 수 없습니다. 완료 후 다시 시도하세요."},
                            status=status.HTTP_409_CONFLICT)
        job.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        return Response(AnalysisStatusSerializer(self.get_object()).data)
