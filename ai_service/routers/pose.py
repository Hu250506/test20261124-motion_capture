"""사진 한 장 → 사람별 관절(COCO 17개) 좌표. 카메라 미리보기의 스켈레톤 표시용.

- 동작 분류는 하지 않는다(관절 위치만). 분석 결과는 지금처럼 /predict/video가 만든다.
- 영상 분석이 진행 중이면 503(analysis_running)을 돌려준다. 미리보기 때문에 분석이 느려지지 않게 한다.
- DB에 접근하지 않는다(FastAPI는 추론만).
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from ultralytics import YOLO

from services.pose_pipeline import _ANALYSIS_LOCK

router = APIRouter(prefix="/predict", tags=["predict"])

MAX_IMAGE_MB = 5
_POSE_LOCK = threading.Lock()
_pose_model: YOLO | None = None  # 처음 요청 때 한 번 만들고 재사용한다


class PosePerson(BaseModel):
    box: list[float]              # [x1, y1, x2, y2] (원본 사진 픽셀)
    confidence: float             # 사람 검출 신뢰도
    keypoints: list[list[float]]  # 17개 × [x, y, 관절 신뢰도]


class PoseFrameResponse(BaseModel):
    width: int
    height: int
    processing_ms: float
    persons: list[PosePerson]


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _detect(image: np.ndarray, resources) -> list[dict]:
    global _pose_model
    if not _POSE_LOCK.acquire(timeout=3):
        raise _error(503, "pose_busy", "관절 검출 요청이 밀려 있습니다.")
    try:
        if _pose_model is None:
            _pose_model = YOLO(str(resources.pose_weights_path))
        project = resources.project_config
        result = _pose_model.predict(
            source=image,
            imgsz=int(project["image_size"]),
            conf=float(project["person_confidence"]),
            device=resources.device,
            verbose=False,
        )[0]
    finally:
        _POSE_LOCK.release()

    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    box_conf = result.boxes.conf.cpu().numpy()
    xy = result.keypoints.xy.cpu().numpy()
    kconf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
             else np.ones(xy.shape[:2], dtype=np.float32))
    persons = []
    for i in range(len(boxes)):
        persons.append({
            "box": [round(float(v), 1) for v in boxes[i]],
            "confidence": round(float(box_conf[i]), 3),
            "keypoints": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                          for (x, y), c in zip(xy[i], kconf[i])],
        })
    return persons


@router.post("/pose-frame", response_model=PoseFrameResponse)
async def predict_pose_frame(request: Request, file: UploadFile = File(...)):
    if _ANALYSIS_LOCK.locked():
        raise _error(503, "analysis_running", "영상 분석 중이라 관절 표시를 잠시 멈춥니다.")
    data = await file.read(MAX_IMAGE_MB * 1024 * 1024 + 1)
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise _error(413, "file_too_large", f"사진은 최대 {MAX_IMAGE_MB}MB입니다.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        raise _error(400, "image_open_failed", "사진을 읽을 수 없습니다(JPG·PNG만).")

    started = time.perf_counter()
    persons = await run_in_threadpool(_detect, image, request.app.state.resources)
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "processing_ms": round((time.perf_counter() - started) * 1000, 1),
        "persons": persons,
    }
