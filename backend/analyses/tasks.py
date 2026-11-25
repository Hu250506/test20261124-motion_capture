"""Celery Worker가 실행하는 분석 작업.

PENDING → PROCESSING → (영상 확보) → FastAPI 추론 → COMPLETED / FAILED
결과 저장은 여기서 Django ORM으로 한다(FastAPI는 DB에 접근하지 않는다).
"""
import logging
from pathlib import Path

from celery import shared_task
from celery.exceptions import Retry
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .camera_capture import CameraError, capture_clip
from .models import ACTION_NAMES, AnalysisJob, PersonPrediction
from .services import AIServiceError, predict_video

logger = logging.getLogger(__name__)

AI_RETRIES = 3
AI_RETRY_SECONDS = 10


def _fail(job: AnalysisJob, code: str, message: str):
    job.status = AnalysisJob.Status.FAILED
    job.error_code = code
    job.error_message = message
    job.completed_at = timezone.now()
    job.save(update_fields=["status", "error_code", "error_message", "completed_at"])
    logger.warning("job #%s FAILED %s: %s", job.pk, code, message)


@shared_task(bind=True, acks_late=True)
def run_analysis(self, job_id: int):
    # 같은 작업이 두 번 실행되지 않도록 PENDING일 때만 PROCESSING으로 바꾼다.
    updated = AnalysisJob.objects.filter(pk=job_id, status=AnalysisJob.Status.PENDING).update(
        status=AnalysisJob.Status.PROCESSING, started_at=timezone.now(), celery_task_id=self.request.id or ""
    )
    if not updated:
        logger.info("job #%s: PENDING이 아니어서 건너뜀", job_id)
        return
    job = AnalysisJob.objects.get(pk=job_id)
    logger.info("job #%s PROCESSING (%s)", job.pk, job.input_type)

    try:
        # 1) 분석할 영상 확보
        if job.input_type == AnalysisJob.InputType.CAMERA and not job.video:  # 재시도 때는 다시 수집하지 않음
            rel = f"camera/{timezone.now():%Y/%m/%d}/job_{job.pk}.mp4"
            info = capture_clip(job.camera_name, job.capture_seconds or settings.CAMERA_CAPTURE_SECONDS,
                                Path(settings.MEDIA_ROOT) / rel)
            job.video.name = rel
            job.save(update_fields=["video"])
            logger.info("job #%s camera captured %s", job.pk, info)
        if not job.video:
            return _fail(job, "no_video", "분석할 영상이 없습니다.")

        # 2) FastAPI 추론
        result = predict_video(job.video.path)

        # 3) 결과 저장 (작업 상태와 사람별 예측을 한 트랜잭션으로)
        with transaction.atomic():
            job.predictions.all().delete()
            PersonPrediction.objects.bulk_create([
                PersonPrediction(
                    job=job,
                    track_id=p["track_id"],
                    action_label=p["action_label"],
                    action_name=p.get("action_name") or ACTION_NAMES.get(p["action_label"], ""),
                    confidence=p["confidence"],
                    uncertain=p.get("uncertain", p["confidence"] < settings.UNCERTAIN_BELOW),
                    top3=p.get("top3", []),
                    frames_seen=p.get("frames_seen", 0),
                    valid_frames=p.get("valid_frames", 0),
                    first_seen_sec=p.get("first_seen_sec"),
                    last_seen_sec=p.get("last_seen_sec"),
                )
                for p in result["persons"]
            ])
            video = result.get("video", {})
            job.video_duration_sec = video.get("duration_sec")
            job.video_fps = video.get("fps")
            job.ai_processing_sec = result.get("processing_time_sec")
            job.status = AnalysisJob.Status.COMPLETED
            job.error_code = job.error_message = ""
            job.completed_at = timezone.now()
            job.save()
        logger.info("job #%s COMPLETED persons=%d", job.pk, len(result["persons"]))

    except AIServiceError as exc:
        # AI 서버가 잠깐 재시작 중일 수 있으므로 연결 실패는 10초 간격으로 최대 3번 다시 시도한다.
        if exc.code == "ai_unreachable" and self.request.retries < AI_RETRIES and not self.request.is_eager:
            AnalysisJob.objects.filter(pk=job.pk).update(status=AnalysisJob.Status.PENDING)
            logger.info("job #%s AI 서버 연결 실패 → %d초 후 재시도 (%d/%d)", job.pk, AI_RETRY_SECONDS,
                        self.request.retries + 1, AI_RETRIES)
            raise self.retry(countdown=AI_RETRY_SECONDS)
        _fail(job, exc.code, exc.message)
    except CameraError as exc:
        _fail(job, exc.code, exc.message)
    except Retry:
        raise
    except Exception as exc:  # 예상하지 못한 오류도 FAILED로 남겨 화면이 '처리 중'에 멈추지 않게 한다.
        logger.exception("job #%s unexpected error", job.pk)
        _fail(job, "internal_error", f"분석 중 서버 오류가 발생했습니다: {type(exc).__name__}")
