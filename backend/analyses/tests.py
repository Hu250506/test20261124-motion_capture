"""분석 API·권한·작업 상태·결과 저장 테스트 (FastAPI는 가짜 응답으로 대체)."""
import shutil
import tempfile
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from .models import AnalysisJob, PersonPrediction
from .services import AIServiceError
from .tasks import run_analysis

FAKE_RESULT = {
    "video": {"fps": 30.0, "frame_count": 150, "duration_sec": 5.0, "width": 540, "height": 960},
    "person_count": 2, "skipped_tracks": 0, "processing_time_sec": 3.2,
    "persons": [
        {"track_id": 1, "action_label": "A003", "action_name": None, "confidence": 0.71, "uncertain": False,
         "top3": [{"action_label": "A003", "confidence": 0.71}], "frames_seen": 150, "valid_frames": 148,
         "first_seen_sec": 0.0, "last_seen_sec": 4.9},
        {"track_id": 2, "action_label": "A010", "action_name": None, "confidence": 0.22, "uncertain": True,
         "top3": [], "frames_seen": 90, "valid_frames": 80, "first_seen_sec": 1.0, "last_seen_sec": 4.0},
    ],
}


def video_file(name="clip.mp4", content=b"\x00\x00\x00\x18ftypmp42"):
    return SimpleUploadedFile(name, content, content_type="video/mp4")


class Base(TestCase):
    def setUp(self):
        self.media = tempfile.mkdtemp()
        self.override = override_settings(MEDIA_ROOT=self.media, ALLOWED_CAMERAS={"lab-phone": "http://10.0.0.5:8080/video"})
        self.override.enable()
        self.a = User.objects.create_user("alice", password="pw-alice-123")
        self.b = User.objects.create_user("bob", password="pw-bob-123")
        self.admin = User.objects.create_user("admin", password="pw-admin-123", is_staff=True)
        self.client = APIClient()

    def tearDown(self):
        self.override.disable()
        shutil.rmtree(self.media, ignore_errors=True)

    def create_job(self, user, **kw):
        return AnalysisJob.objects.create(user=user, input_type="UPLOAD", video=video_file(), **kw)


@mock.patch("analyses.views.run_analysis.delay", **{"return_value.id": "fake-task-id"})
class CreateAndScopeTests(Base):
    def test_requires_login(self, delay):
        self.assertEqual(self.client.get("/api/v1/analyses/").status_code, 403)

    def test_upload_creates_pending_and_enqueues_after_commit(self, delay):
        self.client.force_authenticate(self.a)
        with self.captureOnCommitCallbacks(execute=True):
            res = self.client.post("/api/v1/analyses/", {"input_type": "UPLOAD", "video": video_file()}, format="multipart")
        self.assertEqual(res.status_code, 202)
        job = AnalysisJob.objects.get(pk=res.data["id"])
        self.assertEqual(job.status, "PENDING")
        delay.assert_called_once_with(job.pk)

    def test_rejects_bad_input(self, delay):
        self.client.force_authenticate(self.a)
        cases = [
            {"input_type": "UPLOAD"},
            {"input_type": "UPLOAD", "video": video_file("notes.txt")},
            {"input_type": "CAMERA", "camera_name": "http://evil.example/stream"},
            {"input_type": "CAMERA", "camera_name": "lab-phone", "capture_seconds": 120},
        ]
        for data in cases:
            res = self.client.post("/api/v1/analyses/", data, format="multipart")
            self.assertEqual(res.status_code, 400, data)
        self.assertEqual(AnalysisJob.objects.count(), 0)
        delay.assert_not_called()

    def test_camera_job_uses_default_seconds(self, delay):
        self.client.force_authenticate(self.a)
        res = self.client.post("/api/v1/analyses/", {"input_type": "CAMERA", "camera_name": "lab-phone"}, format="multipart")
        self.assertEqual(res.status_code, 202)
        from django.conf import settings
        self.assertEqual(AnalysisJob.objects.get().capture_seconds, settings.CAMERA_CAPTURE_SECONDS)

    def test_user_sees_only_own_jobs(self, delay):
        mine, theirs = self.create_job(self.a), self.create_job(self.b)
        self.client.force_authenticate(self.a)
        ids = [j["id"] for j in self.client.get("/api/v1/analyses/").data["results"]]
        self.assertEqual(ids, [mine.pk])
        for method in ("get", "patch", "delete"):
            res = getattr(self.client, method)(f"/api/v1/analyses/{theirs.pk}/")
            self.assertEqual(res.status_code, 404, method)
        self.assertTrue(AnalysisJob.objects.filter(pk=theirs.pk).exists())

    def test_admin_sees_all(self, delay):
        self.create_job(self.a), self.create_job(self.b)
        self.client.force_authenticate(self.admin)
        self.assertEqual(self.client.get("/api/v1/analyses/").data["count"], 2)

    def test_patch_only_title_and_memo(self, delay):
        job = self.create_job(self.a, status="COMPLETED")
        self.client.force_authenticate(self.a)
        res = self.client.patch(f"/api/v1/analyses/{job.pk}/", {"title": "새 제목", "status": "FAILED"}, format="json")
        self.assertEqual(res.status_code, 200)
        job.refresh_from_db()
        self.assertEqual((job.title, job.status), ("새 제목", "COMPLETED"))

    def test_delete_removes_predictions_and_file_but_not_while_processing(self, delay):
        job = self.create_job(self.a, status="COMPLETED")
        PersonPrediction.objects.create(job=job, track_id=1, action_label="A001", confidence=0.9)
        path = job.video.path
        self.client.force_authenticate(self.a)
        busy = self.create_job(self.a, status="PROCESSING")
        self.assertEqual(self.client.delete(f"/api/v1/analyses/{busy.pk}/").status_code, 409)
        self.assertEqual(self.client.delete(f"/api/v1/analyses/{job.pk}/").status_code, 204)
        self.assertFalse(PersonPrediction.objects.exists())
        import os
        self.assertFalse(os.path.exists(path))

    def test_status_endpoint(self, delay):
        job = self.create_job(self.a)
        self.client.force_authenticate(self.a)
        res = self.client.get(f"/api/v1/analyses/{job.pk}/status/")
        self.assertEqual((res.data["status"], res.data["is_finished"]), ("PENDING", False))


class ActionLabelTests(TestCase):
    def test_names_match_ai_service_file(self):
        """Django의 동작 이름과 FastAPI의 action_labels.json이 같아야 한다(저장소에서 실행할 때만 확인)."""
        import json
        from pathlib import Path
        from .models import ACTION_NAMES
        path = Path(__file__).resolve().parents[2] / "ai_service" / "models" / "action17" / "action_labels.json"
        if not path.exists():
            self.skipTest("ai_service 폴더가 없는 환경(backend 컨테이너)")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), ACTION_NAMES)


class QueueDownTests(Base):
    @mock.patch("analyses.views.run_analysis.delay", side_effect=ConnectionError("redis down"))
    def test_queue_down_marks_failed(self, delay):
        self.client.force_authenticate(self.a)
        with self.captureOnCommitCallbacks(execute=True):
            res = self.client.post("/api/v1/analyses/", {"input_type": "UPLOAD", "video": video_file()}, format="multipart")
        self.assertEqual(res.status_code, 202)
        job = AnalysisJob.objects.get(pk=res.data["id"])
        self.assertEqual((job.status, job.error_code), ("FAILED", "queue_unavailable"))


class TaskTests(Base):
    @mock.patch("analyses.tasks.predict_video", return_value=FAKE_RESULT)
    def test_success_saves_predictions(self, predict):
        job = self.create_job(self.a)
        run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual(job.status, "COMPLETED")
        self.assertEqual(job.ai_processing_sec, 3.2)
        preds = list(job.predictions.values_list("track_id", "action_label", "action_name", "uncertain"))
        self.assertEqual(preds, [(1, "A003", "냉동고 앞차기", False), (2, "A010", "뒤로 누워서 폰보기", True)])

    @mock.patch("analyses.tasks.predict_video", side_effect=AIServiceError("ai_unreachable", "AI 서버에 연결할 수 없습니다."))
    def test_ai_down_marks_failed(self, predict):
        job = self.create_job(self.a)
        run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual((job.status, job.error_code), ("FAILED", "ai_unreachable"))
        self.assertIsNotNone(job.completed_at)

    @mock.patch("analyses.tasks.predict_video", return_value=FAKE_RESULT)
    def test_runs_only_once(self, predict):
        job = self.create_job(self.a)
        run_analysis.apply(args=[job.pk])
        run_analysis.apply(args=[job.pk])
        self.assertEqual(predict.call_count, 1)

    def test_camera_failure_marks_failed(self):
        job = AnalysisJob.objects.create(user=self.a, input_type="CAMERA", camera_name="lab-phone", capture_seconds=3)
        with override_settings(CAMERA_CONNECT_TIMEOUT=0.5, ALLOWED_CAMERAS={"lab-phone": "http://127.0.0.1:9/video"}):
            run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual((job.status, job.error_code), ("FAILED", "camera_connect_failed"))


class PageTests(Base):
    def test_pages_require_login_and_dashboard_requires_staff(self):
        self.assertEqual(self.client.get("/analyses/").status_code, 302)
        self.client.force_login(self.a)
        self.assertEqual(self.client.get("/analyses/").status_code, 200)
        self.assertEqual(self.client.get("/dashboard/").status_code, 403)
        self.client.logout()
        self.assertEqual(self.client.get("/dashboard/").status_code, 302)
        self.client.force_login(self.admin)
        job = self.create_job(self.a, status="COMPLETED")
        PersonPrediction.objects.create(job=job, track_id=1, action_label="A005", confidence=0.6)
        res = self.client.get("/dashboard/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.context["chart_data"]["actions"]["counts"][4], 1)

    def test_media_only_for_owner_or_staff(self):
        job = self.create_job(self.a)
        url = f"/media/{job.video.name}"
        self.client.force_login(self.b)
        self.assertEqual(self.client.get(url).status_code, 404)
        self.client.force_login(self.a)
        self.assertEqual(self.client.get(url).status_code, 200)


class CameraSnapshotTests(Base):
    """카메라 미리보기: 로그인 필요, 허용 목록의 이름만, 주소는 서버에만."""

    url = "/cameras/lab-phone/snapshot.jpg"

    def test_requires_login(self):
        self.assertEqual(self.client.get(self.url).status_code, 302)

    def test_unknown_camera_is_404(self):
        self.client.force_login(self.a)
        self.assertEqual(self.client.get("/cameras/other/snapshot.jpg").status_code, 404)

    @mock.patch("analyses.page_views.requests.get")
    def test_returns_jpeg_from_allowed_camera_only(self, get):
        get.return_value = mock.Mock(status_code=200, headers={"Content-Type": "image/jpeg"}, content=b"\xff\xd8jpg")
        self.client.force_login(self.a)
        res = self.client.get(self.url)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res["Content-Type"], "image/jpeg")
        self.assertEqual(res.content, b"\xff\xd8jpg")
        # 허용 목록의 host:port에만, redirect를 따라가지 않고 요청한다
        get.assert_called_once_with("http://10.0.0.5:8080/shot.jpg", timeout=3, allow_redirects=False)

    @mock.patch("analyses.page_views.requests.get")
    def test_camera_down_is_502(self, get):
        import requests
        get.side_effect = requests.ConnectionError()
        self.client.force_login(self.a)
        self.assertEqual(self.client.get(self.url).status_code, 502)
        get.side_effect = None
        get.return_value = mock.Mock(status_code=200, headers={"Content-Type": "text/html"}, content=b"<html>")
        self.assertEqual(self.client.get(self.url).status_code, 502)


class CameraPoseTests(Base):
    """스켈레톤 표시: 사진과 관절을 한 응답으로, FastAPI가 안 되면 사진만."""

    url = "/cameras/lab-phone/pose.json"

    def setUp(self):
        super().setUp()
        self.client.force_login(self.a)

    def shot(self):
        return mock.Mock(status_code=200, headers={"Content-Type": "image/jpeg"}, content=b"\xff\xd8jpg")

    def test_requires_login_and_allowed_name(self):
        self.client.logout()
        self.assertEqual(self.client.get(self.url).status_code, 302)
        self.client.force_login(self.a)
        self.assertEqual(self.client.get("/cameras/other/pose.json").status_code, 404)

    @mock.patch("analyses.page_views.requests.post")
    @mock.patch("analyses.page_views.requests.get")
    def test_returns_image_and_persons(self, get, post):
        get.return_value = self.shot()
        person = {"box": [1, 2, 3, 4], "confidence": 0.9, "keypoints": [[1, 2, 0.9]] * 17}
        post.return_value = mock.Mock(status_code=200, json=lambda: {"persons": [person]})
        with override_settings(AI_SERVICE_URL="http://fastapi:8001"):
            data = self.client.get(self.url).json()
        self.assertEqual(data["pose_status"], "ok")
        self.assertEqual(data["persons"], [person])
        self.assertTrue(data["image"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(post.call_args.args[0], "http://fastapi:8001/predict/pose-frame")

    @mock.patch("analyses.page_views.requests.post")
    @mock.patch("analyses.page_views.requests.get")
    def test_ai_busy_or_down_still_returns_image(self, get, post):
        import requests
        get.return_value = self.shot()
        post.return_value = mock.Mock(status_code=503, headers={"Content-Type": "application/json"},
                                      json=lambda: {"detail": {"code": "analysis_running"}})
        data = self.client.get(self.url).json()
        self.assertEqual((data["pose_status"], data["persons"]), ("analysis_running", None))
        post.side_effect = requests.ConnectionError()
        data = self.client.get(self.url).json()
        self.assertEqual(data["pose_status"], "ai_unavailable")
        self.assertIn("image", data)

    @mock.patch("analyses.page_views.requests.get")
    def test_camera_down_is_502(self, get):
        import requests
        get.side_effect = requests.ConnectionError()
        res = self.client.get(self.url)
        self.assertEqual((res.status_code, res.json()["error"]), (502, "camera_unreachable"))
