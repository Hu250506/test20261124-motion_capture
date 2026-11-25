"""허용 목록에 있는 IP Webcam에서 정해진 시간만큼 프레임을 받아 MP4로 저장한다.

- 누가 읽나: Celery Worker (브라우저가 아님). 그래서 Worker 컨테이너에서 카메라 주소에 접속할 수 있어야 한다.
- 언제 끝나나: 첫 프레임을 받은 시점부터 capture_seconds가 지나면 끝난다(연결 시간은 제외).
- 사용자는 카메라 '이름'만 고르고, 실제 URL은 서버 설정(ALLOWED_CAMERAS)에만 있다.
"""
import time
from pathlib import Path

import cv2
from django.conf import settings


class CameraError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


MIN_FRAMES = 30  # 모델은 유효 프레임 10개 이상이 필요하다. 여유를 두고 30장(약 1초) 이상을 요구한다.


def capture_clip(camera_name: str, seconds: float, out_path: Path) -> dict:
    url = settings.ALLOWED_CAMERAS.get(camera_name)
    if not url:
        raise CameraError("camera_not_allowed", f"허용되지 않은 카메라입니다: {camera_name}")

    cap = cv2.VideoCapture(url)
    deadline = time.monotonic() + settings.CAMERA_CONNECT_TIMEOUT
    ok, frame = False, None
    while time.monotonic() < deadline:
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        time.sleep(0.5)
        if not cap.isOpened():
            cap.open(url)
    if not ok or frame is None:
        cap.release()
        raise CameraError("camera_connect_failed", "카메라에 연결하지 못했습니다. 주소·네트워크·앱 실행 상태를 확인하세요.")

    frames = [frame]
    start = time.monotonic()
    misses = 0
    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            misses += 1
            if misses > 30:
                cap.release()
                raise CameraError("camera_disconnected", "수집 중 카메라 연결이 끊겼습니다.")
            time.sleep(0.05)
            continue
        misses = 0
        frames.append(frame)
    elapsed = time.monotonic() - start
    cap.release()

    if len(frames) < MIN_FRAMES:
        raise CameraError("camera_too_few_frames", f"{seconds:.0f}초 동안 프레임을 {len(frames)}장만 받았습니다.")

    # 실제로 받은 프레임 수 / 걸린 시간 = 실제 FPS (카메라 표기 FPS를 믿지 않는다)
    fps = max(1.0, len(frames) / max(elapsed, 1e-3))
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return {"frames": len(frames), "fps": round(fps, 2), "seconds": round(elapsed, 2)}
