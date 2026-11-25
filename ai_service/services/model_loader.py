"""서버 시작 시 17개 동작 모델과 설정을 한 번만 읽고, 입력 규격이 맞는지 검사한다.

모델 폴더(기본 models/action17/)
- action17_rf.joblib     : 학습된 RandomForest (training/train_action_model.py 결과)
- feature_columns.json   : 모델 입력 Feature의 이름과 순서
- training_info.json     : 클래스, Sequence 길이, 신뢰도 기준 등 학습 설정
- action_labels.json     : (선택) {"A001": "한국어 이름", ...}  없으면 ID만 표시
- action_keys.json       : (선택) {"A001": "kiosk_tap", ...}   영어 이름(action_key)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from services import action_features as af
from src.common import load_config, resolve_device

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("ACTION_MODEL_DIR", str(ROOT / "models" / "action17")))


@dataclass(frozen=True)
class ModelResources:
    action_model: object
    pose_weights_path: Path
    device: str
    project_config: dict
    training_info: dict
    class_names: list
    feature_count: int
    feature_index: list            # af.feature_names() 중 모델이 쓰는 열의 위치
    sequence_steps: int
    target_fps: float
    action_names: dict = field(default_factory=dict)
    action_keys: dict = field(default_factory=dict)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_resources() -> ModelResources:
    project_config = load_config()
    info = _read_json(MODEL_DIR / "training_info.json")
    columns = _read_json(MODEL_DIR / "feature_columns.json")
    model = joblib.load(MODEL_DIR / "action17_rf.joblib")

    # 1) 학습 때와 지금의 Feature 계산 규격이 같은지 확인
    if int(info["sequence_steps"]) != af.SEQUENCE_STEPS:
        raise RuntimeError(
            f"Sequence 길이 불일치: 학습={info['sequence_steps']}, 코드={af.SEQUENCE_STEPS}"
        )
    all_names = af.feature_names()
    position = {name: i for i, name in enumerate(all_names)}
    unknown = [c for c in columns if c not in position]
    if unknown:
        raise RuntimeError(f"action_features.py가 만들지 않는 Feature가 있습니다: {unknown[:5]}")
    n_features = int(model.n_features_in_)
    if not (n_features == len(columns) == int(info["n_features"])):
        raise RuntimeError(
            f"Feature 수 불일치: model={n_features}, columns={len(columns)}, info={info['n_features']}"
        )

    # 2) 클래스 확인
    class_names = [str(c) for c in model.classes_]
    if sorted(map(str, info["classes"])) != sorted(class_names):
        raise RuntimeError("학습 정보의 Class와 모델의 Class가 다릅니다.")

    labels_path = MODEL_DIR / "action_labels.json"
    action_names = _read_json(labels_path) if labels_path.exists() else {}
    keys_path = MODEL_DIR / "action_keys.json"
    action_keys = _read_json(keys_path) if keys_path.exists() else {}

    pose_weights_path = Path(project_config["pose_model"])
    if not pose_weights_path.is_absolute():
        pose_weights_path = ROOT / pose_weights_path
    if not pose_weights_path.exists():
        raise RuntimeError(f"Pose 가중치 파일이 없습니다: {pose_weights_path}")

    return ModelResources(
        action_model=model,
        pose_weights_path=pose_weights_path,
        device=resolve_device(project_config["device"]),
        project_config=project_config,
        training_info=info,
        class_names=class_names,
        feature_count=n_features,
        feature_index=[position[c] for c in columns],
        sequence_steps=af.SEQUENCE_STEPS,
        target_fps=float(info.get("target_fps", 30)),
        action_names=action_names,
        action_keys=action_keys,
    )
