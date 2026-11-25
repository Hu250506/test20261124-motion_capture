from pathlib import Path

from django.conf import settings
from rest_framework import serializers

from .models import AnalysisJob, PersonPrediction


class PersonPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonPrediction
        fields = ["track_id", "action_label", "action_name", "action_key", "confidence", "uncertain", "top3",
                  "frames_seen", "valid_frames", "first_seen_sec", "last_seen_sec"]
        read_only_fields = fields


class AnalysisJobSerializer(serializers.ModelSerializer):
    """작업 생성·조회용. 상태와 AI 결과는 읽기 전용이라 사용자가 바꿀 수 없다."""

    predictions = PersonPredictionSerializer(many=True, read_only=True)
    video_url = serializers.SerializerMethodField()
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    input_type_display = serializers.CharField(source="get_input_type_display", read_only=True)
    total_seconds = serializers.FloatField(read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = AnalysisJob
        fields = ["id", "username", "title", "memo", "input_type", "input_type_display", "video", "video_url",
                  "camera_name", "capture_seconds", "expected_action",
                  "status", "status_display", "error_code", "error_message",
                  "video_duration_sec", "video_fps", "ai_processing_sec", "total_seconds",
                  "created_at", "started_at", "completed_at", "predictions"]
        read_only_fields = ["status", "error_code", "error_message", "video_duration_sec", "video_fps",
                            "ai_processing_sec", "created_at", "started_at", "completed_at"]
        extra_kwargs = {"video": {"write_only": True, "required": False}}

    def get_video_url(self, obj):
        if not obj.video:
            return None
        request = self.context.get("request")
        return request.build_absolute_uri(obj.video.url) if request else obj.video.url

    def validate_video(self, f):
        ext = Path(f.name).suffix.lower()
        if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
            raise serializers.ValidationError(f"지원하지 않는 형식입니다. 허용: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}")
        if f.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise serializers.ValidationError(f"파일이 너무 큽니다. 최대 {settings.MAX_UPLOAD_MB}MB입니다.")
        if f.size == 0:
            raise serializers.ValidationError("빈 파일입니다.")
        return f

    def validate_capture_seconds(self, v):
        if v is not None and not (3 <= v <= 15):
            raise serializers.ValidationError("수집 시간은 3~15초 사이여야 합니다.")
        return v

    def validate(self, attrs):
        kind = attrs.get("input_type")
        if kind == AnalysisJob.InputType.UPLOAD:
            if not attrs.get("video"):
                raise serializers.ValidationError({"video": "영상 파일을 선택하세요."})
            attrs["camera_name"] = ""
            attrs["capture_seconds"] = None
        elif kind == AnalysisJob.InputType.CAMERA:
            if attrs.get("video"):
                raise serializers.ValidationError({"video": "IP Webcam 분석에는 파일을 올리지 않습니다."})
            name = attrs.get("camera_name", "")
            if name not in settings.ALLOWED_CAMERAS:
                allowed = ", ".join(settings.ALLOWED_CAMERAS) or "(설정된 카메라 없음)"
                raise serializers.ValidationError({"camera_name": f"허용된 카메라만 사용할 수 있습니다: {allowed}"})
            attrs.setdefault("capture_seconds", settings.CAMERA_CAPTURE_SECONDS)
            if attrs["capture_seconds"] is None:
                attrs["capture_seconds"] = settings.CAMERA_CAPTURE_SECONDS
        return attrs


class AnalysisJobUpdateSerializer(serializers.ModelSerializer):
    """PATCH: 제목·메모만 수정할 수 있다."""

    class Meta:
        model = AnalysisJob
        fields = ["title", "memo"]


class AnalysisStatusSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    is_finished = serializers.BooleanField(read_only=True)
    person_count = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisJob
        fields = ["id", "status", "status_display", "is_finished", "error_code", "error_message", "person_count"]

    def get_person_count(self, obj):
        return obj.predictions.count()
