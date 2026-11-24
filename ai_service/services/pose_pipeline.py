from __future__ import annotations

import math
import threading
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.multi_person_utils import (
    MultiPersonStateManager,
    feature_from_keypoints,
)
from src.realtime_utils import predict_from_window
from services.action_inference import summarize_persons
from services.model_loader import ModelResources

TRACKER = "bytetrack.yaml"
LOST_TIMEOUT_SECONDS = 2.0

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


def analyze_video(
    video_path: str | Path,
    resources: ModelResources,
    max_seconds: float = 60.0,
) -> dict:
    with _ANALYSIS_LOCK:
        return _analyze(Path(video_path), resources, max_seconds)


def _analyze(video_path: Path, resources: ModelResources, max_seconds: float) -> dict:
    cfg = resources.runtime_config
    project = resources.project_config

    window_seconds = float(cfg["window_seconds"])
    min_required = window_seconds * float(cfg["min_window_fill_ratio"])

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
        if duration < min_required:
            capture.release()
            raise VideoTooShortError(
                f"영상이 너무 짧습니다({duration:.1f}초). "
                f"최소 {min_required:.1f}초 이상이어야 합니다."
            )

    # 요청마다 새 모델 객체를 써서 이전 영상의 Tracker 상태가 섞이지 않게 한다.
    pose_model = YOLO(str(resources.pose_weights_path))

    manager = MultiPersonStateManager(
        window_seconds=window_seconds,
        stabilization_buffer=int(cfg["stabilization_buffer"]),
        stabilization_min_votes=int(cfg["stabilization_min_votes"]),
        lost_timeout_seconds=LOST_TIMEOUT_SECONDS,
    )

    records: dict[int, dict] = {}
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break

            now = frame_index / fps
            result = pose_model.track(
                source=frame,
                persist=True,
                tracker=TRACKER,
                imgsz=int(project["image_size"]),
                conf=float(project["person_confidence"]),
                device=resources.device,
                verbose=False,
            )[0]

            seen_ids: set[int] = set()
            boxes = result.boxes
            keypoints = result.keypoints

            if (
                boxes is not None
                and boxes.id is not None
                and keypoints is not None
                and keypoints.xy is not None
            ):
                track_ids = boxes.id.detach().cpu().numpy().astype(int)
                kp_xy = keypoints.xy.detach().cpu().numpy()

                for index in range(min(len(track_ids), len(kp_xy))):
                    track_id = int(track_ids[index])
                    seen_ids.add(track_id)

                    state = manager.get_or_create(track_id, now)
                    record = records.setdefault(
                        track_id,
                        {
                            "first_seen": now,
                            "last_seen": now,
                            "frames": 0,
                            "predictions": [],
                        },
                    )
                    record["last_seen"] = now
                    record["frames"] += 1

                    state.feature_buffer.append(
                        now, feature_from_keypoints(kp_xy[index])
                    )

                    ready = state.feature_buffer.is_ready(
                        min_window_fill_ratio=float(cfg["min_window_fill_ratio"]),
                        min_valid_feature_ratio=float(cfg["min_valid_feature_ratio"]),
                        min_valid_features=int(cfg["min_valid_features"]),
                    )
                    interval = float(cfg["prediction_interval_sec"])

                    if ready and (now - state.last_prediction_time) >= interval:
                        try:
                            label, raw_label, confidence, _ = predict_from_window(
                                model=resources.action_model,
                                feature_buffer=state.feature_buffer,
                                sequence_steps=resources.sequence_steps,
                                confidence_threshold=float(cfg["confidence_threshold"]),
                            )
                        except ValueError:
                            continue
                        state.last_prediction_time = now
                        record["predictions"].append(
                            {
                                "time": now,
                                "label": label,
                                "raw_label": raw_label,
                                "confidence": float(confidence),
                            }
                        )

            manager.mark_missing(now, seen_ids)
            manager.remove_lost(now)
            frame_index += 1

            if frame_count <= 0 and now > max_seconds:
                raise VideoTooLongError(
                    f"영상이 너무 깁니다. 최대 {max_seconds:.0f}초까지 분석합니다."
                )
    finally:
        capture.release()

    if frame_index == 0:
        raise VideoOpenError("영상에서 Frame을 읽지 못했습니다.")

    duration = frame_index / fps
    if frame_count <= 0 and duration < min_required:
        raise VideoTooShortError(
            f"영상이 너무 짧습니다({duration:.1f}초). "
            f"최소 {min_required:.1f}초 이상이어야 합니다."
        )

    if not records:
        raise NoPersonDetectedError("영상에서 사람을 찾지 못했습니다.")

    persons = summarize_persons(records)
    if not persons:
        raise InsufficientSequenceError(
            "사람은 보이지만 동작을 판단할 만큼 충분한 구간이 없습니다. "
            "사람이 더 오래, 선명하게 보이는 영상을 사용하세요."
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
        "persons": persons,
    }
