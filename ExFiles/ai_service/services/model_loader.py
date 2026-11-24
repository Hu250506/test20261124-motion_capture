from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from src.common import load_config, resolve_device
from src.realtime_utils import validate_runtime_values

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"

REQUIRED_RUNTIME_KEYS = [
    "window_seconds",
    "min_window_fill_ratio",
    "min_valid_feature_ratio",
    "min_valid_features",
    "prediction_interval_sec",
    "confidence_threshold",
    "stabilization_buffer",
    "stabilization_min_votes",
]


@dataclass(frozen=True)
class ModelResources:
    action_model: object
    pose_weights_path: Path
    device: str
    project_config: dict
    runtime_config: dict
    sequence_steps: int
    class_names: list
    feature_count: int


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_resources() -> ModelResources:
    """서버 시작 시 한 번 호출해 모델과 설정을 읽고 입력 차원을 검사한다."""
    project_config = load_config()
    runtime_config = _read_json(CONFIGS_DIR / "day07_realtime.json")
    training_info = _read_json(MODELS_DIR / "rf_training_info.json")
    feature_columns = _read_json(MODELS_DIR / "rf_feature_columns.json")

    missing = [k for k in REQUIRED_RUNTIME_KEYS if k not in runtime_config]
    if missing:
        raise RuntimeError(f"day07_realtime.json에 없는 항목: {missing}")

    validate_runtime_values(
        window_seconds=float(runtime_config["window_seconds"]),
        confidence_threshold=float(runtime_config["confidence_threshold"]),
        stabilization_buffer=int(runtime_config["stabilization_buffer"]),
        stabilization_min_votes=int(runtime_config["stabilization_min_votes"]),
    )

    sequence_steps = int(training_info["sequence_steps"])
    expected_features = sequence_steps * 38

    action_model = joblib.load(MODELS_DIR / "rf_action_baseline.joblib")
    n_features = int(action_model.n_features_in_)
    if not (n_features == expected_features == len(feature_columns)):
        raise RuntimeError(
            "Feature 수 불일치: "
            f"model={n_features}, sequence_steps*38={expected_features}, "
            f"columns={len(feature_columns)}"
        )

    class_names = [str(c) for c in action_model.classes_]
    info_classes = training_info.get("classes")
    if info_classes is not None and sorted(map(str, info_classes)) != sorted(class_names):
        raise RuntimeError("학습 정보의 Class와 모델의 Class가 다릅니다.")

    pose_weights_path = Path(project_config["pose_model"])
    if not pose_weights_path.is_absolute():
        pose_weights_path = ROOT / pose_weights_path
    if not pose_weights_path.exists():
        raise RuntimeError(f"Pose 가중치 파일이 없습니다: {pose_weights_path}")

    return ModelResources(
        action_model=action_model,
        pose_weights_path=pose_weights_path,
        device=resolve_device(project_config["device"]),
        project_config=project_config,
        runtime_config=runtime_config,
        sequence_steps=sequence_steps,
        class_names=class_names,
        feature_count=n_features,
    )
