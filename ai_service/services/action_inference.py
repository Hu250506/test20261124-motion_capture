"""사람(Track ID)별 관절 시퀀스 → 17개 동작 예측과 신뢰도 정리."""
from __future__ import annotations

import numpy as np

from services import action_features as af
from services.model_loader import ModelResources

# 이 값보다 신뢰도가 낮으면 화면에 '판단 불확실'로 표시하도록 uncertain=True를 붙인다.
# RandomForest 17클래스는 확률이 넓게 퍼져 값 자체가 낮게 나온다(무작위 추측 ≈ 0.06).
# LOSO 평가(training/results/loso_predictions.csv)에서 신뢰도 구간별 정답률:
#   0.3 미만 0.59 (105건) · 0.3~0.4 0.88 (59건) · 0.4~0.5 0.96 (50건) · 0.5 이상 1.00 (24건)
UNCERTAIN_BELOW = 0.3


def predict_person(track_id: int, record: dict, resources: ModelResources) -> dict | None:
    """한 사람의 기록으로 동작 1개를 예측한다. 유효 프레임이 부족하면 None."""
    kp = np.asarray(record["kp"], np.float32)
    conf = np.asarray(record["conf"], np.float32)
    t = np.asarray(record["t"], np.float64)
    vector, quality = af.clip_feature(kp, conf, t)
    if vector is None:
        return None

    x = vector[resources.feature_index].reshape(1, -1)
    proba = resources.action_model.predict_proba(x)[0]
    classes = resources.action_model.classes_
    order = np.argsort(proba)[::-1]
    best = order[0]
    label = str(classes[best])
    confidence = float(proba[best])

    return {
        "track_id": int(track_id),
        "action_label": label,
        "action_name": resources.action_names.get(label),
        "confidence": round(confidence, 4),
        "uncertain": confidence < UNCERTAIN_BELOW,
        "top3": [
            {"action_label": str(classes[i]), "confidence": round(float(proba[i]), 4)}
            for i in order[:3]
        ],
        "frames_seen": quality["total_frames"],
        "valid_frames": quality["valid_frames"],
        "first_seen_sec": round(float(t[0]), 3),
        "last_seen_sec": round(float(t[-1]), 3),
    }


def summarize_persons(track_records: dict, resources: ModelResources) -> tuple[list, int]:
    """예측 가능한 사람만 Track ID 순서로 정리한다. (결과 목록, 판단 불가로 제외한 사람 수)"""
    persons, skipped = [], 0
    for track_id, record in sorted(track_records.items()):
        result = predict_person(track_id, record, resources)
        if result is None:
            skipped += 1
        else:
            persons.append(result)
    return persons, skipped
