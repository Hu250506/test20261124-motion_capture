"""영상 → YOLO Pose 관절 추출 (학습용).

각 영상에서 프레임을 target_fps로 샘플링해 YOLO Pose를 돌리고,
가장 크게 보이는 사람(주 인물)의 COCO-17 관절을 npz로 저장한다.
출력: data/keypoints/<clip_id>.npz  (kp: [T,17,2] 픽셀, conf: [T,17], t: [T], ok: [T], wh)
"""
import argparse, glob, os, sys, time
import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="data/videos_small")
ap.add_argument("--out", default="data/keypoints")
ap.add_argument("--weights", default="ai_service/models/yolo11n-pose.pt")
ap.add_argument("--target-fps", type=float, default=30.0)
ap.add_argument("--conf", type=float, default=0.25)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
model = YOLO(a.weights)
files = sorted(glob.glob(os.path.join(a.videos, "**", "*.mp4"), recursive=True))
print(f"{len(files)} videos", flush=True)
t0 = time.time()
for i, f in enumerate(files):
    cid = os.path.splitext(os.path.basename(f))[0]
    dst = os.path.join(a.out, cid + ".npz")
    if os.path.exists(dst):
        continue
    cap = cv2.VideoCapture(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    step = max(1, round(fps / a.target_fps))
    kps, confs, ts, oks = [], [], [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            r = model.predict(frame, conf=a.conf, verbose=False)[0]
            k = np.full((17, 2), np.nan, np.float32); c = np.zeros(17, np.float32); ok = 0
            if r.keypoints is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                j = int(area.argmax())
                k = r.keypoints.xy[j].cpu().numpy().astype(np.float32)
                c = (r.keypoints.conf[j].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)).astype(np.float32)
                ok = 1
            kps.append(k); confs.append(c); ts.append(idx / fps); oks.append(ok)
        idx += 1
    cap.release()
    np.savez_compressed(dst, kp=np.array(kps), conf=np.array(confs), t=np.array(ts),
                        ok=np.array(oks), wh=np.array([W, H]), fps=fps, src=f)
    el = time.time() - t0
    print(f"[{i+1}/{len(files)}] {cid} frames={len(ts)} ok={np.mean(oks) if oks else 0:.2f} elapsed={el/60:.1f}m", flush=True)
print("DONE", flush=True)
