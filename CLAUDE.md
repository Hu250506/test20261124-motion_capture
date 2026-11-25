# CLAUDE.md: 이 저장소에서 일하는 Claude가 따를 지침

## 사용자와 소통
- 사용자는 Django·DRF·FastAPI를 처음 배우는 학생(9~10일차 과제)이다. **한국어로**, 초보자가 따라 할 수 있게 쉬운 말로 설명한다.
- "단답으로" 같은 요청이 있으면 짧게 답한다. 결과를 보고할 때는 확인한 것과 확인하지 못한 것을 구분해서 말한다.
- 사용자가 실행할 명령은 한 블록에 하나씩 준다. 사용자 환경은 Windows + WSL Ubuntu + VS Code이고, `ai_service/.venv` 가상환경을 쓴다.
- `main`에 바로 푸시하지 않는다. 작업 브랜치는 `training/retrain-17class`이고, `main`에 합칠 때는 사용자에게 먼저 묻는다.
- 되돌리기 어려운 작업(`git reset --hard`, 강제 푸시, `docker compose down -v`로 DB 삭제, 파일 삭제)은 먼저 묻는다.

## 프로젝트 한 줄 요약
무인 매장 CCTV 영상에서 이상행동 17개를 인식하는 웹 서비스. 브라우저 → Django·DRF → PostgreSQL / Redis → Celery Worker → FastAPI(YOLO Pose + RandomForest) → Worker가 ORM으로 결과 저장.
전체 설명은 `README.md`, 모델 학습·평가는 `training/README.md`에 있다. 작업 전에 둘 다 읽는다.

## 과제 규칙 (강사 문서에서 온 것, 어기지 말 것)
- Docker Compose 서비스는 정확히 5개: `postgres`, `redis`, `fastapi`, `django`, `celery`
- FastAPI는 추론만 한다. DB·ORM·회원 기능을 FastAPI에 만들지 않는다. 결과 저장은 Celery Worker가 Django ORM으로 한다.
- 직접 만든 업무 모델은 2개(`AnalysisJob`, `PersonPrediction`)뿐이고, 회원은 Django 기본 `User`를 쓴다.
- 작업 상태는 `PENDING → PROCESSING → COMPLETED / FAILED`로 통일한다.
- 컨테이너끼리는 `localhost`가 아니라 서비스 이름으로 접속한다(`http://fastapi:8001`, `redis://redis:6379/0`, `postgres`).
- IP Webcam 주소는 `.env`의 `ALLOWED_CAMERAS` 허용 목록에만 둔다. 사용자가 입력한 임의 URL을 호출하지 않는다.
- 실제 비밀번호, `.env`, 카메라 주소는 커밋하지 않는다. `.env.example`에는 자리표시자만 둔다.
- README 7개 목차(소개, 실행 방법, 아키텍처 그림, ERD, 버튼 흐름도, 모델과 입력, 실행 결과)는 유지한다.

## 모델: 바꾸기 전에 반드시 알아야 할 것
- 강사 문서의 6클래스·1,140 Feature 모델이 아니다. **17클래스·1,543 Feature** 모델을 원본 영상으로 다시 학습했다. 기존 `clip_model.joblib`과 `action_model.joblib`은 Feature 계산 코드가 없어서 쓰지 못했다.
- `ai_service/services/action_features.py`는 **학습과 추론이 같이 쓰는 코드**다. 여기를 바꾸면 `training/train_action_model.py`로 반드시 다시 학습하고, `ai_service/models/action17/`을 함께 커밋한다. 바꾸지 않고 두는 것이 기본이다.
- 행동 이름은 두 곳에 있고 **같아야 한다**: 한국어는 `ai_service/models/action17/action_labels.json` ↔ `backend/analyses/models.py`의 `ACTION_NAMES`, 영어(`action_key`, 예: `kiosk_tap`)는 `action_keys.json` ↔ `ACTION_KEYS`. 같은지는 테스트(`ActionLabelTests`)가 검사한다. 번호(A001~A017)는 바꾸지 않는다(모델 클래스). 이름을 바꾸면 저장된 `PersonPrediction`도 데이터 마이그레이션으로 고친다(0003 참고).
- 평가는 사람 단위(Leave-One-Subject-Out)로 했다. 정확도 0.782, Macro F1 0.777. 같은 사람의 영상이 학습과 평가에 섞이면 안 된다.
- `uncertain`(판단 불확실) 기준은 신뢰도 0.3이다. 근거는 `training/results/loso_predictions.csv`의 구간별 정답률이다.
- 예측은 '이상행동 확정'이 아니라 '검토 대상 알림'으로 표현한다.
- 자주 헷갈리는 쌍: A009→A012, A004↔A005, A004→A013. 관절만으로는 대상 물체를 구분하지 못한다.

## 데이터
- `data/videos_small/`: 학습 영상 238개(긴 변 960px, 소리 없음). `data/keypoints/`: 영상에서 추출한 관절 데이터. `data/raw_videos/`: 원본 6.8GB로, `.gitignore` 대상이다. **절대 커밋하지 않는다**(GitHub는 파일 하나당 100MB, 푸시 한 번에 약 2GB가 한계).
- 파일 이름 규칙은 `S01_A003_T02`(S=사람, A=행동, T=촬영 방향). S02는 T03이 없다(34개).
- 사람이 나오는 영상이라 저장소는 Private로 둔다.

## 실행·검증 명령
```bash
cp .env.example .env                      # 처음 한 번만. 비밀번호·SECRET_KEY 변경
docker compose up --build -d
docker compose ps                         # postgres·redis·fastapi는 healthy, django·celery는 running
docker compose exec django python manage.py createsuperuser
docker compose exec django python manage.py test      # 백엔드 테스트
docker compose logs -f celery fastapi                 # 작업 흐름 확인
```
- Docker 없이 백엔드 테스트만 돌릴 때: `cd backend && USE_SQLITE=1 python manage.py test`
- FastAPI만 단독으로 실행할 때: `cd ai_service && uvicorn main:app --port 8001` → http://localhost:8001/docs
- 모델 재학습: `python training/extract_keypoints.py` → `python training/train_action_model.py`

**완료를 말하기 전에** 실제로 실행해서 확인한다. 코드만 보고 "될 것"이라고 하지 않는다. 화면을 바꿨으면 브라우저로 열어 보고, API를 바꿨으면 curl이나 Swagger로 호출해 본다.

## 이미 겪은 문제 (같은 실수 반복 금지)
- `ultralytics`가 화면용 `opencv-python`을 같이 설치해 `libxcb.so.1` 오류가 났다. `ai_service/Dockerfile`에서 그것을 지우고 headless 버전만 남긴다. 이 처리를 빼지 않는다.
- Celery의 Redis 연결에 `packaging` 패키지가 필요하다(`backend/requirements.txt`). 빠지면 업로드가 500 오류로 끝난다.
- 작업 등록은 `transaction.on_commit` 뒤에 한다(DB 저장 전에 Worker가 실행되는 것 방지). 등록에 실패하면 `queue_unavailable`로 FAILED 처리한다.
- FastAPI 연결 실패는 10초 간격으로 3번 다시 시도한다. 카메라 작업은 다시 시도할 때 다시 수집하지 않는다.
- 일반 회원이 `/dashboard/`에 들어가면 403을 보여 준다. `staff_member_required`를 쓰면 로그인 화면과 무한히 오간다.
- 집계 쿼리(`annotate(Count)`)에서는 모델의 기본 정렬이 무시된다. `order_by`를 직접 쓴다.
- 업로드 영상은 `/media/`로 공개하지 않는다. `protected_media`에서 주인과 관리자만 볼 수 있다.
- Chart.js는 CDN이 아니라 `backend/static/vendor/chart.umd.js`(4.4.1)를 쓴다. 발표장 인터넷이 막혀도 차트가 나온다.
- WSL 터미널이 지워진 폴더를 가리키면 `Unable to read current working directory`가 난다. `cd`로 다시 들어간다.

## 남은 일
- 집 컴퓨터에서 `docker compose up --build`로 PyTorch CPU 설치(download.pytorch.org)가 되는지 확인 (클라우드 세션에서는 그 주소가 막혀 PyPI 버전으로만 검증함)
- 휴대폰 IP Webcam 앱으로 실제 카메라 분석 확인 (같은 와이파이, Celery 컨테이너에서 휴대폰 IP로 접속 가능해야 함)
- README 7장 캡처 넣기: 정상 1건, 실패 1건, 관리자 화면·대시보드. 캡처에 카메라 주소나 비밀번호가 보이지 않게 한다.
- 발표 준비: README 순서(아키텍처 → ERD → 버튼 흐름 → 시연 → 겪은 오류 하나)
- 검수가 끝나면 사용자에게 물어보고 `main`에 합치기(Pull Request)
- (선택) S02 T03 영상 17개를 추가하고 다시 학습
