# Django가 시작될 때 Celery 앱도 함께 로드한다(@shared_task가 이 앱을 사용).
from .celery import app as celery_app

__all__ = ("celery_app",)
