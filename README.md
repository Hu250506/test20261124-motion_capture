# Vision Action Platform

## 1. 프로젝트 소개

브라우저에서 **영상 파일을 올리거나 IP Webcam으로 몇 초간 촬영**하면, AI가 화면 속 사람마다 **17개 동작(A001~A017) 중 어떤 동작인지**와 신뢰도를 알려 주는 웹 서비스입니다.
분석은 백그라운드에서 처리되므로 사용자는 기다리지 않고 진행 상태를 볼 수 있습니다. 결과는 PostgreSQL에 저장되고, 관리자는 대시보드에서 전체 통계를 확인합니다.

- 입력: 영상 업로드(MP4·MOV·AVI·MKV, 최대 100MB·60초), IP Webcam(허용된 카메라, 3~15초 수집, 기본 8초)
- 결과: 사람(Track ID)별 예측 동작, 신뢰도, 판단 불확실 표시, 후보 Top 3, 처리 시간

## 2. 실행 방법

준비물: Docker Desktop(또는 Docker Engine + Compose v2)

```bash
git clone https://github.com/Hu250506/test20261124-motion_capture.git
cd test20261124-motion_capture
cp .env.example .env          # 비밀번호·SECRET_KEY를 바꾸고, 카메라를 쓸 경우 ALLOWED_CAMERAS 입력
docker compose up --build -d  # 처음 빌드는 PyTorch 설치 때문에 수 분 걸립니다
docker compose exec django python manage.py createsuperuser   # 관리자 계정 만들기
```

- 모델 파일은 저장소에 들어 있습니다: `ai_service/models/action17/`, `ai_service/models/yolo11n-pose.pt`
- DB 마이그레이션은 `django` 컨테이너가 시작될 때 자동으로 적용됩니다(`python manage.py migrate`).

| 화면 | 주소 |
|---|---|
| 회원가입 / 로그인 | http://localhost:8000/accounts/signup/ · http://localhost:8000/accounts/login/ |
| 사용자 분석 화면 | http://localhost:8000/analyses/ |
| 관리자 대시보드 (관리자만) | http://localhost:8000/dashboard/ |
| Django Admin | http://localhost:8000/admin/ |
| 분석 API (DRF) | http://localhost:8000/api/v1/analyses/ |
| FastAPI Swagger (개발용) | http://localhost:8001/docs |

종료: `docker compose down` (데이터 유지) · 데이터까지 삭제: `docker compose down -v`

## 3. 전체 아키텍처

`compose.yaml`이 실행하는 서비스는 **postgres · redis · fastapi · django · celery 5개**입니다.

```mermaid
flowchart LR
    U["브라우저<br/>analyses.js"] -- "① 분석 요청 (영상/카메라 이름)" --> D["django<br/>Django + DRF"]
    D -- "② AnalysisJob 저장 (PENDING)" --> P[("postgres<br/>PostgreSQL")]
    D -- "③ 작업 ID 등록" --> R[("redis<br/>작업 대기열")]
    R -- "④ 작업 전달" --> C["celery<br/>Celery Worker"]
    CAM["IP Webcam<br/>(허용 목록)"] -. "⑤ 카메라일 때: 8초 프레임 수집" .-> C
    C -- "⑥ 영상 전송 (HTTP)" --> F["fastapi<br/>YOLO Pose + RandomForest"]
    F -- "⑦ 사람별 예측 JSON" --> C
    C -- "⑧ 상태·PersonPrediction 저장 (ORM)" --> P
    U -- "⑨ 상태·결과 조회 (2초마다)" --> D
    D -- "조회" --> P
    D <-. "업로드 영상 공유 볼륨 media_data" .-> C
```

- **FastAPI**는 추론만 하고 DB에 접근하지 않습니다. 결과 저장은 Celery Worker가 Django ORM으로 합니다.
- 컨테이너끼리는 `localhost`가 아니라 서비스 이름(`postgres`, `redis`, `fastapi`)으로 접속합니다.

## 4. DB 설계 (ERD)

직접 작성한 모델은 `backend/analyses/models.py`에 있습니다. 회원은 Django 기본 `auth_user`를 그대로 씁니다.

```mermaid
erDiagram
    auth_user ||--o{ AnalysisJob : "요청"
    AnalysisJob ||--o{ PersonPrediction : "포함"
    auth_user {
        int id PK
        string username
        string password "해시"
        bool is_staff "관리자 여부"
    }
    AnalysisJob {
        bigint id PK
        int user_id FK
        string title
        text memo
        string input_type "UPLOAD / CAMERA"
        string video "media 경로"
        string camera_name "허용 목록의 이름"
        float capture_seconds
        string expected_action "사용자 선택(정답 아님)"
        string status "PENDING/PROCESSING/COMPLETED/FAILED"
        string celery_task_id
        string error_code
        text error_message
        float video_duration_sec
        float ai_processing_sec
        datetime created_at
        datetime completed_at
    }
    PersonPrediction {
        bigint id PK
        bigint job_id FK
        int track_id "작업 안에서만 유효"
        string action_label "A001~A017"
        float confidence
        bool uncertain "신뢰도 0.3 미만"
        json top3
        int valid_frames
        float first_seen_sec
        float last_seen_sec
    }
```

- `auth_user`: 회원 계정과 관리자 여부
- `AnalysisJob`: 분석 요청 1건과 진행 상태·오류·처리 시간. 영상 파일은 DB가 아닌 media 폴더에 저장
- `PersonPrediction`: 한 작업 안에서 추적한 사람별 예측. 영상에 2명이 나오면 작업 1건에 2행

## 5. AI 분석 버튼 구현 흐름

```mermaid
sequenceDiagram
    actor User as 사용자
    participant JS as analyses.js
    participant V as views.py (DRF)
    participant DB as PostgreSQL
    participant Q as Redis
    participant T as tasks.py (Celery)
    participant CC as camera_capture.py
    participant S as services.py
    participant API as predictions.py (FastAPI)

    User->>JS: AI 분석 시작 클릭
    JS->>V: POST /api/v1/analyses/ (영상 또는 카메라 이름)
    V->>V: serializers.py 입력 검사 (형식·용량·허용 카메라)
    V->>DB: AnalysisJob 생성 (PENDING)
    V-->>JS: 202 + 작업 ID (분석 완료를 기다리지 않음)
    V->>Q: 커밋 후 run_analysis 등록 (transaction.on_commit)
    loop 2초마다
        JS->>V: GET /api/v1/analyses/{id}/status/
    end
    Q->>T: 작업 전달
    T->>DB: PROCESSING
    opt IP Webcam
        T->>CC: 8초 프레임 수집 → MP4 저장
    end
    T->>S: 영상 전달
    S->>API: POST /predict/video
    API->>API: pose_pipeline.py (YOLO Pose·추적) → action_features.py → action_inference.py
    API-->>S: 사람별 예측 JSON
    T->>DB: PersonPrediction 저장 + COMPLETED (실패 시 FAILED + 오류 코드)
    JS->>V: GET /api/v1/analyses/{id}/
    V-->>JS: 사람별 결과 → 화면 표시
```

## 6. 사용한 모델과 입력 형식

- **Pose**: `yolo11n-pose.pt` (COCO 17 관절). 약 30fps로 샘플링하고 ByteTrack으로 사람별 추적
- **동작 분류**: `ai_service/models/action17/action17_rf.joblib`. RandomForest 트리 300개, 17개 동작, 입력 1,543개
  - 6일차 38-D(정규화 좌표 34 + 팔꿈치·무릎 각도 4)에 고관절·어깨 각도, 상체 기울기, 몸 높이를 더해 프레임당 44개
  - 이를 30 Step 시퀀스(1,320)로 만들고, 영상 전체 요약 통계(220)와 골반 이동량(3)을 붙임
  - Feature 계산 코드 `ai_service/services/action_features.py`를 학습과 추론이 함께 씀
- **학습·평가**: 5명 × 17동작 × 최대 3방향, 영상 238개. 사람 단위 교차검증(Leave-One-Subject-Out)으로 정확도 **0.782**, Macro F1 **0.777** (자세한 내용은 [`training/README.md`](training/README.md))
- **FastAPI 입출력**: `POST /predict/video`에 영상 파일을 보내면 `persons[]`에 `track_id`, `action_label`, `confidence`, `uncertain`, `top3`, `valid_frames`를 담아 돌려줌. 처리할 수 없으면 오류 코드를 돌려줌: `no_person_detected`, `video_too_short`, `video_open_failed`, `insufficient_sequence`
- **동작 이름**: `ai_service/models/action17/action_labels.json`에 `{"A001": "이름", ...}`을 넣으면 화면에 함께 표시됨

> 모델은 학습한 17개 동작 중 하나로만 답합니다. 학습하지 않은 동작도 그중 하나로 나올 수 있으므로 신뢰도와 함께 봅니다. 학습에 참여한 사람이 5명뿐이라 새로운 사람·환경에서는 정확도가 낮을 수 있습니다.

## 7. 실행 결과와 검증

| 테스트 | 입력 | 결과 |
|---|---|---|
| T01 모델 불러오기 | `/health` | 17개 Class, Feature 1,543개 확인 |
| T02 영상 업로드 | S05_A009_T02.mp4 (11초) | 202 응답(0.4초) → PROCESSING → COMPLETED, A009 (0.70) |
| T03 IP Webcam | 가짜 MJPEG 카메라(A001 영상 반복), 4초 수집 | COMPLETED, A001 (0.29, 판단 불확실) |
| T04 사람별 결과 | 두 사람을 나란히 붙인 영상 | 작업 1건에 Track 1·2 두 행 저장 |
| T05 비동기 | 업로드 직후 | 웹 서버가 작업 ID를 바로 반환, 처리는 Worker가 함 |
| T06 영구 저장 | `docker compose down` → `up` | 작업·예측·영상 그대로 남음 |
| T07 CRUD 권한 | 회원 A·B | B가 A의 작업에 GET·PATCH·DELETE → 404, 상태 값은 PATCH로 바꿀 수 없음 |
| T08 관리자 | admin 계정 | 전체 작업 목록, 동작별 막대·상태별 원형 차트 |
| T09 예외 처리 | `.txt` 파일, 허용 안 된 카메라, 깨진 MP4, FastAPI 중단 | 400 / 400 / FAILED `video_open_failed` / 10초 간격 3번 재시도 후 FAILED `ai_unreachable` |
| T10 Docker | 5개 서비스 재시작 | PostgreSQL 데이터 유지, 서비스 재연결 |

자동 테스트: `docker compose exec django python manage.py test` (18개)

## 폴더 구조

```text
backend/      Django·DRF·Celery (accounts, analyses, templates, static)
ai_service/   FastAPI 추론 서비스 (routers, services, src, models)
training/     관절 추출·재학습 스크립트와 평가 결과
data/         학습용 축소 영상(videos_small)과 관절 데이터(keypoints)
compose.yaml  5개 서비스 정의
```
