from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers.predictions import router as predictions_router
from services.model_loader import load_resources


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 켜질 때 모델·설정을 한 번만 읽고 입력 차원을 검사한다.
    app.state.resources = load_resources()
    yield


app = FastAPI(title="Vision Action AI Service", lifespan=lifespan)
app.include_router(predictions_router)


@app.get("/health")
def health():
    resources = getattr(app.state, "resources", None)
    if resources is None:
        return {"status": "loading", "model_ready": False}
    return {
        "status": "ok",
        "model_ready": True,
        "model_variant": resources.training_info.get("variant"),
        "classes": resources.class_names,
        "action_names": resources.action_names,
        "sequence_steps": resources.sequence_steps,
        "feature_count": resources.feature_count,
        "target_fps": resources.target_fps,
        "device": str(resources.device),
    }
