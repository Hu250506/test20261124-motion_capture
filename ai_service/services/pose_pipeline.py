"""영상 → YOLO Pose 추적 → 사람(Track ID)별 관절 시퀀스 → 17개 동작 예측.

학습(training/extract_keypoints.py)과 같은 조건을 지킨다.
- 프레임을 약 30fps로 샘플링 (60fps 영상이면 2프레임마다 1장)
- 같은 Pose 가중치(yolo11n-pose.pt)와 사람 검출 기준(conf 0.25)
- 관절 → Feature 계산은 services/action_features.py (학습과 같은 코드)

학습 영상은 한 사람씩 찍었으므로, 여러 사람이 나오면 Track ID별로 따로 예측한다.
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from services.action_inference import summarize_persons
from services.model_loader import ModelResources

TRACKER = "bytetrack.yaml"
MIN_VIDEO_SECONDS = 1.0

# 한 번에 영상 하나만 분석한다(CPU·메모리 급증 방지).
_ANALYSIS_LOCK = threading.Lock()


class VideoAnalysisError(Exception):
    status_code = 422
    code = "video_analysis_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class VideoOpenError(VideoAnalysisError):
    status_code = 400
    code = "video_open_failed"


class VideoTooShortError(VideoAnalysisError):
    code = "video_too_short"


class VideoTooLongError(VideoAnalysisError):
    code = "video_too_long"


class NoPersonDetectedError(VideoAnalysisError):
    code = "no_person_detected"


class InsufficientSequenceError(VideoAnalysisError):
    code = "insufficient_sequence"


def analyze_video(video_path: str | Path, resources: ModelResources, max_seconds: float = 60.0) -> dict:
    with _ANALYSIS_LOCK:
        started = time.perf_counter()
        result = _analyze(Path(video_path), resources, max_seconds)
        result["processing_time_sec"] = round(time.perf_counter() - started, 2)
        return result


def _analyze(video_path: Path, resources: ModelResources, max_seconds: float) -> dict:
    project = resources.project_config

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoOpenError("영상 파일을 열 수 없습니다. 올바른 영상인지 확인하세요.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise VideoOpenError("영상의 FPS를 읽을 수 없습니다.")

    if frame_count > 0:
        duration = frame_count / fps
        if duration > max_seconds:
            capture.release()
            raise VideoTooLongError(
                f"영상이 너무 깁니다({duration:.1f}초). 최대 {max_seconds:.0f}초까지 분석합니다."
            )
        if duration < MIN_VIDEO_SECONDS:
            capture.release()
            raise VideoTooShortError(
                f"영상이 너무 짧습니다({duration:.1f}초). 최소 {MIN_VIDEO_SECONDS:.0f}초 이상이어야 합니다."
            )

    step = max(1, round(fps / resources.target_fps))
    # 요청마다 새 모델 객체를 써서 이전 영상의 Tracker 상태가 섞이지 않게 한다.
    pose_model = YOLO(str(resources.pose_weights_path))

    records: dict[int, dict] = {}
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            now = frame_index / fps
            if frame_count <= 0 and now > max_seconds:
                raise VideoTooLongError(f"영상이 너무 깁니다. 최대 {max_seconds:.0f}초까지 분석합니다.")

            if frame_index % step == 0:
                result = pose_model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER,
                    imgsz=int(project["image_size"]),
                    conf=float(project["person_confidence"]),
                    device=resources.device,
                    verbose=False,
                )[0]
                boxes, keypoints = result.boxes, result.keypoints
                if boxes is not None and boxes.id is not None and keypoints is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                    xy = keypoints.xy.cpu().numpy()
                    kconf = (
                        keypoints.conf.cpu().numpy()
                        if keypoints.conf is not None
                        else np.zeros(xy.shape[:2], np.float32)
                    )
                    for i in range(min(len(ids), len(xy))):
                        rec = records.setdefault(int(ids[i]), {"kp": [], "conf": [], "t": []})
                        rec["kp"].append(xy[i])
                        rec["conf"].append(kconf[i])
                        rec["t"].append(now)
            frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise VideoOpenError("영상에서 Frame을 읽지 못했습니다.")
    duration = frame_index / fps
    if frame_count <= 0 and duration < MIN_VIDEO_SECONDS:
        raise VideoTooShortError(
            f"영상이 너무 짧습니다({duration:.1f}초). 최소 {MIN_VIDEO_SECONDS:.0f}초 이상이어야 합니다."
        )
    if not records:
        raise NoPersonDetectedError("영상에서 사람을 찾지 못했습니다.")

    persons, skipped = summarize_persons(records, resources)
    if not persons:
        raise InsufficientSequenceError(
            "사람은 보이지만 동작을 판단할 만큼 관절이 충분히 보이지 않습니다. "
            "전신이 더 오래, 선명하게 보이는 영상을 사용하세요."
        )

    return {
        "video": {
            "fps": round(fps, 3),
            "frame_count": frame_index,
            "duration_sec": round(duration, 3),
            "width": width,
            "height": height,
        },
        "person_count": len(persons),
        "skipped_tracks": skipped,
        "persons": persons,
    }
