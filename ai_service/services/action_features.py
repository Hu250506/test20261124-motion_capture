"""17개 동작 모델의 입력 Feature를 만드는 공용 모듈.

학습(training/train_action_model.py)과 FastAPI 추론이 **같은 함수**를 쓰도록
이 파일 하나에 모았다. 여기 계산이 바뀌면 모델을 다시 학습해야 한다.

입력: 한 사람의 프레임별 COCO-17 관절 (kp [T,17,2] 픽셀, conf [T,17])
출력: 영상 한 건당 1차원 Feature 벡터

구성
1. 프레임 Feature 44개
   - 6일차와 같은 38-D: 골반 중심·어깨 폭 정규화 좌표 34 + 팔꿈치·무릎 각도 4
   - 추가 6개: 좌·우 고관절 각도, 좌·우 어깨(팔 올림) 각도, 상체 기울기, 몸 높이(정규화)
2. 유효 프레임을 시간축으로 30 Step에 맞춰 보간 → 30 × 44 = 1,320
3. 영상 전체 요약 통계: 프레임 Feature별 평균·표준편차·최솟값·최댓값·평균 변화량(5 × 44 = 220)
   + 골반 중심의 이동량(정규화) 3개
합계 1,543개
"""
from __future__ import annotations

import os
import sys

import numpy as np

_AI_SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AI_SERVICE_DIR not in sys.path:
    sys.path.insert(0, _AI_SERVICE_DIR)

from src.feature_utils import (  # noqa: E402
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    calculate_angle, normalize_by_shoulder_width,
)

SEQUENCE_STEPS = 30
MIN_VALID_FRAMES = 10
MIN_CORE_CONF = 0.3  # 골반·어깨 신뢰도가 이보다 낮으면 정규화 기준이 불안정 → 프레임 제외

COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
FRAME_FEATURE_NAMES = [f"{n}_{a}" for n in COCO_NAMES for a in ("x", "y")] + [
    "left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle", "left_shoulder_angle", "right_shoulder_angle",
    "trunk_tilt", "body_height",
]
N_FRAME = len(FRAME_FEATURE_NAMES)  # 44
_CORE = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]


def feature_names() -> list[str]:
    names = [f"s{s:02d}_{f}" for s in range(SEQUENCE_STEPS) for f in FRAME_FEATURE_NAMES]
    for stat in ("mean", "std", "min", "max", "dmean"):
        names += [f"{stat}_{f}" for f in FRAME_FEATURE_NAMES]
    names += ["hip_path_len", "hip_range_x", "hip_range_y"]
    return names


def frame_feature(kp: np.ndarray) -> np.ndarray | None:
    """관절 [17,2] → 44-D. 정규화할 수 없으면 None."""
    try:
        norm, _, scale = normalize_by_shoulder_width(kp)
    except ValueError:
        return None
    ang = lambda a, b, c: calculate_angle(kp[a], kp[b], kp[c])  # noqa: E731
    sh_mid = (kp[LEFT_SHOULDER] + kp[RIGHT_SHOULDER]) / 2
    hip_mid = (kp[LEFT_HIP] + kp[RIGHT_HIP]) / 2
    trunk = sh_mid - hip_mid
    tilt = float(np.degrees(np.arctan2(abs(trunk[0]), -trunk[1] + 1e-8)))  # 0=똑바로 섬, 90=수평
    ys = norm[:, 1]
    extra = [
        ang(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), ang(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        ang(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), ang(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        ang(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE), ang(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
        ang(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW), ang(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
        tilt, float(np.nanmax(ys) - np.nanmin(ys)),
    ]
    v = np.concatenate([norm.reshape(-1), np.asarray(extra, np.float32)]).astype(np.float32)
    return v if np.all(np.isfinite(v)) else None


def clip_feature(kp: np.ndarray, conf: np.ndarray, t: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """한 사람의 프레임 시퀀스 → (Feature 벡터, 품질 정보). 유효 프레임이 부족하면 벡터는 None."""
    feats, times, centers, scales = [], [], [], []
    for k, c, tt in zip(kp, conf, t):
        if not np.all(np.isfinite(k)) or np.min(c[_CORE]) < MIN_CORE_CONF:
            continue
        f = frame_feature(k)
        if f is None:
            continue
        feats.append(f)
        times.append(tt)
        centers.append((k[LEFT_HIP] + k[RIGHT_HIP]) / 2)
        scales.append(np.linalg.norm(k[LEFT_SHOULDER] - k[RIGHT_SHOULDER]))
    info = {"total_frames": int(len(kp)), "valid_frames": len(feats),
            "valid_ratio": round(len(feats) / max(1, len(kp)), 3)}
    if len(feats) < MIN_VALID_FRAMES:
        return None, info
    F = np.stack(feats)
    T = np.asarray(times, np.float64)
    grid = np.linspace(T[0], T[-1], SEQUENCE_STEPS)
    seq = np.stack([np.interp(grid, T, F[:, j]) for j in range(N_FRAME)], axis=1)
    d = np.abs(np.diff(F, axis=0)).mean(axis=0)
    C = np.stack(centers) / (np.median(scales) + 1e-8)
    motion = [float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()),
              float(np.ptp(C[:, 0])), float(np.ptp(C[:, 1]))]
    vec = np.concatenate([seq.reshape(-1), F.mean(0), F.std(0), F.min(0), F.max(0), d, motion])
    return vec.astype(np.float32), info
