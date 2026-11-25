from django.contrib import admin

from .models import AnalysisJob, PersonPrediction


class PersonPredictionInline(admin.TabularInline):
    model = PersonPrediction
    extra = 0
    can_delete = False
    readonly_fields = ("track_id", "action_label", "action_name", "action_key", "confidence", "uncertain",
                       "valid_frames", "first_seen_sec", "last_seen_sec")
    fields = readonly_fields


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "input_type", "status", "title", "expected_action",
                    "person_count", "created_at", "completed_at")
    list_filter = ("status", "input_type", "created_at")
    search_fields = ("user__username", "title", "memo", "camera_name")
    readonly_fields = ("celery_task_id", "error_code", "error_message", "video_duration_sec", "video_fps",
                       "ai_processing_sec", "created_at", "started_at", "completed_at")
    inlines = [PersonPredictionInline]
    date_hierarchy = "created_at"

    @admin.display(description="사람 수")
    def person_count(self, obj):
        return obj.predictions.count()


@admin.register(PersonPrediction)
class PersonPredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "track_id", "action_label", "action_name", "action_key", "confidence", "uncertain",
                    "valid_frames", "created_at")
    list_filter = ("action_label", "uncertain", "job__input_type")
    search_fields = ("job__user__username", "action_label", "action_name", "action_key")
    list_select_related = ("job", "job__user")
