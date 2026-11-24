from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean


def summarize_track(track_id: int, record: dict) -> dict:
    """한 사람(Track ID)의 예측 목록을 최종 동작 1개로 정리한다.

    UNKNOWN(신뢰도 기준 미달)을 제외한 예측 중 신뢰도 합이 가장 큰 동작을 선택한다.
    """
    predictions = record["predictions"]
    valid = [p for p in predictions if p["label"] != "UNKNOWN"]

    if valid:
        score = defaultdict(float)
        confidences = defaultdict(list)
        for p in valid:
            score[p["label"]] += p["confidence"]
            confidences[p["label"]].append(p["confidence"])
        action_label = max(score, key=score.get)
        confidence = mean(confidences[action_label])
    else:
        action_label = "UNKNOWN"
        confidence = mean(p["confidence"] for p in predictions)

    return {
        "track_id": int(track_id),
        "action_label": action_label,
        "confidence": round(float(confidence), 4),
        "prediction_count": len(predictions),
        "frames_seen": int(record["frames"]),
        "first_seen_sec": round(float(record["first_seen"]), 3),
        "last_seen_sec": round(float(record["last_seen"]), 3),
        "label_counts": dict(Counter(p["label"] for p in predictions)),
    }


def summarize_persons(track_records: dict) -> list:
    """예측이 1개 이상 만들어진 사람만 Track ID 순서로 정리한다."""
    return [
        summarize_track(track_id, record)
        for track_id, record in sorted(track_records.items())
        if record["predictions"]
    ]
