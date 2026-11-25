from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from services.pose_pipeline import VideoAnalysisError, analyze_video

logger = logging.getLogger("ai_service")

router = APIRouter(prefix="/predict", tags=["predict"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "60"))
CHUNK_SIZE = 1024 * 1024


class VideoInfo(BaseModel):
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


class TopPrediction(BaseModel):
    action_label: str
    confidence: float


class PersonResult(BaseModel):
    track_id: int
    action_label: str
    action_name: str | None = None
    action_key: str | None = None
    confidence: float
    uncertain: bool
    top3: list[TopPrediction]
    frames_seen: int
    valid_frames: int
    first_seen_sec: float
    last_seen_sec: float


class PredictResponse(BaseModel):
    video: VideoInfo
    person_count: int
    skipped_tracks: int
    processing_time_sec: float
    persons: list[PersonResult]


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


@router.post("/video", response_model=PredictResponse)
async def predict_video(request: Request, file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise _error(
            400,
            "unsupported_file_type",
            f"지원하지 않는 형식입니다. 허용: {sorted(ALLOWED_EXTENSIONS)}",
        )

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp:
            temp_path = temp.name
            total = 0
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise _error(
                        413,
                        "file_too_large",
                        f"파일이 너무 큽니다. 최대 {MAX_UPLOAD_MB}MB입니다.",
                    )
                temp.write(chunk)

        if total == 0:
            raise _error(400, "empty_file", "빈 파일입니다.")

        resources = request.app.state.resources
        try:
            return await run_in_threadpool(
                analyze_video, temp_path, resources, MAX_VIDEO_SECONDS
            )
        except VideoAnalysisError as exc:
            raise _error(exc.status_code, exc.code, exc.message)
        except HTTPException:
            raise
        except Exception:
            logger.exception("영상 분석 중 예기치 못한 오류")
            raise _error(500, "internal_error", "분석 중 서버 오류가 발생했습니다.")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
