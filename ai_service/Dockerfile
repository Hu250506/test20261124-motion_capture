# FastAPI 추론 서비스: YOLO Pose + 17개 동작 RandomForest
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics
WORKDIR /app

# PyTorch는 CPU 전용 버전을 먼저 설치해 이미지 크기를 줄인다(GPU 버전은 수 GB).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch==2.15.0 torchvision==0.29.2

COPY requirements.txt .
# ultralytics가 화면용 opencv-python을 함께 설치하는데, 그 버전은 그래픽 라이브러리(libGL·libxcb)가 필요하다.
# 서버에는 화면이 없으므로 지우고 headless 버전만 남긴다(추가 시스템 패키지 불필요).
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==5.0.0.93 \
    && python -c "import cv2, ultralytics; print('cv2', cv2.__version__)"

COPY . .
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
