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
    resources = app.state.resources
    return {
        "status": "ok",
        "classes": resources.class_names,
        "sequence_steps": resources.sequence_steps,
        "feature_count": resources.feature_count,
        "device": resources.device,
    }
