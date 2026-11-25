"""분석 작업(AnalysisJob)과 사람별 예측(PersonPrediction).

auth_user 1 ─ N AnalysisJob 1 ─ N PersonPrediction
- 영상 파일은 MEDIA_ROOT에 두고 DB에는 경로만 저장한다.
- track_id는 한 작업(영상) 안에서만 의미 있는 추적 번호다.
"""
from django.conf import settings
from django.db import models


# 무인 매장 CCTV 이상행동 17종. ai_service/models/action17/action_labels.json과 같은 내용을 유지한다.
# 번호(A001~A017)는 모델이 학습한 클래스라서 바꾸지 않는다. 이름만 바꿀 때는 재학습이 필요 없다.
ACTION_NAMES = {
    "A001": "키오스크 두드리기",
    "A002": "한손 물건 던지기",
    "A003": "앞차기",
    "A004": "잡고 흔들기",
    "A005": "밀치기",
    "A006": "휘두르기",
    "A007": "바닥에 앉기",
    "A008": "위에 앉기",
    "A009": "앞으로 눕기",
    "A010": "뒤로 눕기",
    "A011": "몸통 박치기",
    "A012": "쪼그려 물건 빼내기",
    "A013": "샷건",
    "A014": "구토",
    "A015": "옆으로 쓰러지기",
    "A016": "뒤로 넘어지기",
    "A017": "앞으로 넘어지기",
}

# 영어 이름(action_key). ai_service/models/action17/action_keys.json과 같은 내용을 유지한다.
ACTION_KEYS = {
    "A001": "kiosk_tap",
    "A002": "throw_one_hand",
    "A003": "front_kick",
    "A004": "grab_shake",
    "A005": "push",
    "A006": "swing",
    "A007": "sit_floor",
    "A008": "sit_on_object",
    "A009": "lie_forward",
    "A010": "lie_backward",
    "A011": "body_slam",
    "A012": "crouch_take_item",
    "A013": "shotgun",
    "A014": "vomit",
    "A015": "fall_sideways",
    "A016": "fall_backward",
    "A017": "fall_forward",
}
ACTION_CHOICES = [(code, f"{code} {name}") for code, name in ACTION_NAMES.items()]


class AnalysisJob(models.Model):
    class InputType(models.TextChoices):
        UPLOAD = "UPLOAD", "영상 업로드"
        CAMERA = "CAMERA", "IP Webcam"

    class Status(models.TextChoices):
        PENDING = "PENDING", "대기"
        PROCESSING = "PROCESSING", "처리 중"
        COMPLETED = "COMPLETED", "완료"
        FAILED = "FAILED", "실패"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analysis_jobs")
    title = models.CharField("제목", max_length=100, blank=True)
    memo = models.TextField("메모", blank=True)
    input_type = models.CharField("입력 방식", max_length=10, choices=InputType.choices)
    video = models.FileField("영상", upload_to="uploads/%Y/%m/%d/", blank=True)
    camera_name = models.CharField("카메라", max_length=50, blank=True)
    capture_seconds = models.FloatField("수집 시간(초)", null=True, blank=True)
    expected_action = models.CharField(
        "사용자가 수행했다고 선택한 동작(정답 아님)", max_length=10, choices=ACTION_CHOICES, blank=True
    )

    status = models.CharField("상태", max_length=12, choices=Status.choices, default=Status.PENDING, db_index=True)
    celery_task_id = models.CharField(max_length=64, blank=True)
    error_code = models.CharField(max_length=50, blank=True)
    error_message = models.TextField(blank=True)

    video_duration_sec = models.FloatField(null=True, blank=True)
    video_fps = models.FloatField(null=True, blank=True)
    ai_processing_sec = models.FloatField("AI 추론 시간(초)", null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "분석 작업"
        verbose_name_plural = "분석 작업"

    def __str__(self):
        return f"#{self.pk} {self.user} {self.get_input_type_display()} {self.status}"

    @property
    def is_finished(self):
        return self.status in (self.Status.COMPLETED, self.Status.FAILED)

    @property
    def total_seconds(self):
        if self.completed_at and self.created_at:
            return round((self.completed_at - self.created_at).total_seconds(), 2)
        return None


class PersonPrediction(models.Model):
    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name="predictions")
    track_id = models.PositiveIntegerField("Track ID")
    action_label = models.CharField("예측 동작", max_length=10, db_index=True)
    action_name = models.CharField("동작 이름", max_length=100, blank=True)
    action_key = models.CharField("동작 영어 이름", max_length=40, blank=True)
    confidence = models.FloatField("신뢰도")
    uncertain = models.BooleanField("판단 불확실", default=False)
    top3 = models.JSONField(default=list, blank=True)
    frames_seen = models.PositiveIntegerField(default=0)
    valid_frames = models.PositiveIntegerField("처리한 유효 Frame 수", default=0)
    first_seen_sec = models.FloatField(null=True, blank=True)
    last_seen_sec = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["job", "track_id"]
        constraints = [models.UniqueConstraint(fields=["job", "track_id"], name="uniq_job_track")]
        verbose_name = "사람별 예측"
        verbose_name_plural = "사람별 예측"

    def __str__(self):
        return f"job#{self.job_id} track{self.track_id} {self.action_label} ({self.confidence:.2f})"
