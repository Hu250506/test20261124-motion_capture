"""FastAPI AI 서비스 호출. 네트워크·응답 오류를 AIServiceError 하나로 정리한다.

컨테이너 사이에서는 localhost가 아니라 Compose 서비스 이름(http://fastapi:8001)으로 접속한다.
"""
from pathlib import Path

import requests
from django.conf import settings


class AIServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def predict_video(path: str | Path) -> dict:
    url = settings.AI_SERVICE_URL.rstrip("/") + "/predict/video"
    path = Path(path)
    try:
        with path.open("rb") as f:
            response = requests.post(
                url,
                files={"file": (path.name, f, "video/mp4")},
                timeout=(10, settings.AI_REQUEST_TIMEOUT),
            )
    except requests.ConnectionError:
        raise AIServiceError("ai_unreachable", "AI 서버에 연결할 수 없습니다. FastAPI가 실행 중인지 확인하세요.")
    except requests.Timeout:
        raise AIServiceError("ai_timeout", f"AI 분석이 {settings.AI_REQUEST_TIMEOUT:.0f}초 안에 끝나지 않았습니다.")
    except requests.RequestException as exc:
        raise AIServiceError("ai_request_error", f"AI 서버 요청 오류: {exc}")

    if response.status_code == 200:
        return response.json()

    # FastAPI 오류 형식: {"detail": {"code": ..., "message": ...}}
    try:
        detail = response.json().get("detail", {})
    except ValueError:
        detail = {}
    if isinstance(detail, dict) and detail.get("code"):
        raise AIServiceError(detail["code"], detail.get("message", "AI 분석 실패"))
    raise AIServiceError(f"ai_http_{response.status_code}", f"AI 서버 오류(HTTP {response.status_code})")
