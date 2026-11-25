"""분석 기록 삭제 정책: 작업을 지우면 사람별 예측(CASCADE)과 업로드 영상 파일도 함께 지운다."""
from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import AnalysisJob


@receiver(post_delete, sender=AnalysisJob)
def delete_video_file(sender, instance, **kwargs):
    if instance.video:
        instance.video.delete(save=False)
