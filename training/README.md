# 17개 동작 모델 재학습

기존 `clip_model.joblib`과 `action_model.joblib`은 학습에 쓴 Feature 계산 코드가 없었다. 그래서 같은 입력을 만들 수 없어 원본 영상으로 다시 학습했다.

## 순서

```bash
# 1) 영상에서 관절 추출 (CPU 약 30분, 이미 만든 파일은 건너뜀)
python3 training/extract_keypoints.py      # data/videos_small → data/keypoints/*.npz

# 2) Feature 생성, 사람 단위 평가, 최종 모델 저장
python3 training/train_action_model.py     # → ai_service/models/action17/, training/results/
```

## 데이터

- 영상 238개: S01·S03·S04·S05는 각 51개(17동작 × 3방향), S02는 34개(T03 없음)
- 30fps로 샘플링했다. 프레임마다 화면에서 가장 크게 보이는 사람의 COCO-17 관절을 추출했다(`yolo11n-pose.pt`).
- 영상에서 제외된 클립은 없다.

## Feature (`ai_service/services/action_features.py`, 학습과 추론에서 같은 코드를 쓴다)

| 구성 | 개수 |
|---|---:|
| 프레임 Feature 44개 = 6일차 38-D(정규화 좌표 34 + 팔꿈치·무릎 각도 4) + 고관절 각도 2 + 어깨 각도 2 + 상체 기울기 + 몸 높이 | 44 |
| 유효 프레임을 30 Step으로 보간한 시퀀스 | 30 × 44 = 1,320 |
| 영상 전체 요약: 평균·표준편차·최솟값·최댓값·평균 변화량 | 5 × 44 = 220 |
| 골반 중심 이동량 (경로 길이, x·y 범위) | 3 |
| **합계** | **1,543** |

골반이나 어깨의 신뢰도가 0.3보다 낮은 프레임은 정규화 기준이 불안정해서 뺐다. 유효 프레임이 10개 미만이면 예측하지 않는다.

## 평가: Leave-One-Subject-Out

한 사람을 통째로 빼고 나머지 4명으로 학습한 뒤, 뺀 사람으로 평가했다. 이것을 5명 모두 반복했다.

| 설정 | Feature 수 | 정확도 | Macro F1 |
|---|---:|---:|---:|
| 6일차 방식 (30 × 38) | 1,140 | 0.664 | 0.662 |
| **최종: 44-D 시퀀스 + 요약 통계** | 1,543 | **0.782** | **0.777** |
| 최종 + 좌우 반전 증강 | 1,543 | 0.748 | 0.738 |

- 사람별 정확도: S01 0.824 · S02 0.735 · S03 0.765 · S04 0.784 · S05 0.784
- 촬영 방향별 정확도: T01 0.788 · T02 0.788 · T03 0.765
- 잘 맞히는 동작: A001·A002·A003·A007·A008·A010·A016 (F1 0.89 이상)
- 약한 동작: A004·A005·A009·A012·A013 (F1 0.4~0.6)
- 자주 헷갈리는 쌍: A009→A012 (6건), A004↔A005 (8건), A004→A013 (4건)

자세한 내용은 `results/` 폴더에 있다.

- `metrics.json`: 전체 지표
- `classification_report.txt`: 동작별 정밀도·재현율
- `confusion_matrix.csv`: 혼동행렬
- `loso_predictions.csv`: 영상별 예측과 신뢰도
- `misclassified.csv`: 틀린 영상 목록

## 최종 모델 (`ai_service/models/action17/`)

- `action17_rf.joblib`: RandomForest(트리 300개, class_weight=balanced). 238개 전체로 학습했다.
- `feature_columns.json`: 입력 1,543개의 이름과 순서
- `training_info.json`: 클래스, Sequence 길이, 신뢰도 기준, sklearn 버전 등

## 주의

- 기존 모델의 `person_clip_acc 0.906`과 이 결과를 바로 비교하지 않는다. 평가 방식과 Feature가 다르다.
- 학습한 사람이 5명뿐이다. 새로운 사람이나 새로운 환경에서는 정확도가 더 낮을 수 있다.
