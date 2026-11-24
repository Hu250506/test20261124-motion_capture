"""17개 동작 Random Forest 재학습 + Subject 단위 평가.

입력 : data/keypoints/<S..>_<A...>_<T..>.npz  (training/extract_keypoints.py 결과)
출력 : ai_service/models/action17/  (모델·Feature 이름·학습 정보)
       training/results/            (지표·혼동행렬·오분류 목록)

평가 방법: Leave-One-Subject-Out. 한 사람을 통째로 빼고 나머지 사람으로 학습한 뒤
뺀 사람의 영상으로 평가하는 것을 5명 모두 반복한다(같은 사람이 학습·평가에 섞이지 않음).
"""
import glob, json, os, re, sys, time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai_service"))
from services import action_features as af  # noqa: E402
from src.feature_utils import LEFT_RIGHT_PAIRS  # noqa: E402

KP_DIR = os.path.join(ROOT, "data", "keypoints")
MODEL_DIR = os.path.join(ROOT, "ai_service", "models", "action17")
RES_DIR = os.path.join(ROOT, "training", "results")
SEED = 42
RF = dict(n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight="balanced",
          n_jobs=-1, random_state=SEED)


def flip(kp, w):
    """화면 좌우 반전 + Left/Right 관절 교환 (Train 데이터 증강용)."""
    f = kp.copy()
    f[..., 0] = w - f[..., 0]
    for l, r in LEFT_RIGHT_PAIRS:
        f[:, [l, r]] = f[:, [r, l]]
    return f


PERM = list(range(17))  # 좌우 반전 시 관절 번호 교환표 (신뢰도 배열에도 적용)
for _l, _r in LEFT_RIGHT_PAIRS:
    PERM[_l], PERM[_r] = _r, _l


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(KP_DIR, "*.npz"))):
        cid = os.path.splitext(os.path.basename(p))[0]
        m = re.match(r"(S\d+)_(A\d+)_(T\d+)$", cid)
        if not m:
            print("skip (이름 규칙 불일치):", cid)
            continue
        z = np.load(p)
        kp, conf, t, w = z["kp"], z["conf"], z["t"], float(z["wh"][0])
        x, info = af.clip_feature(kp, conf, t)
        xf, _ = af.clip_feature(flip(kp, w), conf[:, PERM], t)
        rows.append(dict(clip=cid, subject=m[1], action=m[2], take=m[3], x=x, xf=xf, **info))
    return rows


def fit(X, y):
    return RandomForestClassifier(**RF).fit(X, y)


def main():
    t0 = time.time()
    rows = load()
    bad = [r for r in rows if r["x"] is None]
    rows = [r for r in rows if r["x"] is not None]
    subjects = sorted({r["subject"] for r in rows})
    classes = sorted({r["action"] for r in rows})
    names = af.feature_names()
    seq38 = [i for i, n in enumerate(names) if n.startswith("s") and n[1:3].isdigit()
             and names[i].split("_", 1)[1] in af.FRAME_FEATURE_NAMES[:38]]
    print(f"clips={len(rows)} excluded={len(bad)} subjects={subjects} classes={len(classes)} feat={len(names)}")

    variants = {
        "baseline_30x38": dict(cols=seq38, aug=False),   # 6일차 방식(1,140 Feature)
        "full": dict(cols=None, aug=False),              # 44-D 시퀀스 + 요약 통계
        "full_flipaug": dict(cols=None, aug=True),       # + 좌우반전 증강(Train만)
    }
    results, preds_by_variant = {}, {}
    for vname, v in variants.items():
        sel = (lambda a: a[:, v["cols"]]) if v["cols"] is not None else (lambda a: a)
        y_true, y_pred, fold = [], [], {}
        pred_rows = []
        for s in subjects:
            tr = [r for r in rows if r["subject"] != s]
            te = [r for r in rows if r["subject"] == s]
            Xtr = np.stack([r["x"] for r in tr]); ytr = [r["action"] for r in tr]
            if v["aug"]:
                ex = [r for r in tr if r["xf"] is not None]
                Xtr = np.vstack([Xtr, np.stack([r["xf"] for r in ex])]); ytr += [r["action"] for r in ex]
            clf = fit(sel(Xtr), ytr)
            Xte = sel(np.stack([r["x"] for r in te]))
            p = clf.predict(Xte); pr = clf.predict_proba(Xte).max(1)
            fold[s] = round(accuracy_score([r["action"] for r in te], p), 3)
            y_true += [r["action"] for r in te]; y_pred += list(p)
            pred_rows += [dict(clip=r["clip"], subject=s, take=r["take"], true=r["action"], pred=pp,
                               confidence=round(float(c), 3)) for r, pp, c in zip(te, p, pr)]
        results[vname] = dict(accuracy=round(accuracy_score(y_true, y_pred), 3),
                              macro_f1=round(f1_score(y_true, y_pred, average="macro"), 3),
                              per_subject=fold, n_features=len(v["cols"]) if v["cols"] else len(names))
        preds_by_variant[vname] = pred_rows
        print(vname, results[vname], flush=True)

    best = max(results, key=lambda k: (results[k]["macro_f1"], results[k]["accuracy"]))
    pr = preds_by_variant[best]
    yt = [r["true"] for r in pr]; yp = [r["pred"] for r in pr]
    by_take = {}
    for tk in sorted({r["take"] for r in pr}):
        sub = [r for r in pr if r["take"] == tk]
        by_take[tk] = round(np.mean([r["true"] == r["pred"] for r in sub]), 3)

    os.makedirs(RES_DIR, exist_ok=True)
    cm = confusion_matrix(yt, yp, labels=classes)
    with open(os.path.join(RES_DIR, "confusion_matrix.csv"), "w") as f:
        f.write("true\\pred," + ",".join(classes) + "\n")
        for c, row in zip(classes, cm):
            f.write(c + "," + ",".join(map(str, row)) + "\n")
    with open(os.path.join(RES_DIR, "loso_predictions.csv"), "w") as f:
        f.write("clip,subject,take,true,pred,confidence,correct\n")
        for r in pr:
            f.write(f"{r['clip']},{r['subject']},{r['take']},{r['true']},{r['pred']},{r['confidence']},{int(r['true']==r['pred'])}\n")
    with open(os.path.join(RES_DIR, "misclassified.csv"), "w") as f:
        f.write("clip,true,pred,confidence\n")
        for r in sorted(pr, key=lambda r: r["clip"]):
            if r["true"] != r["pred"]:
                f.write(f"{r['clip']},{r['true']},{r['pred']},{r['confidence']}\n")
    report = classification_report(yt, yp, labels=classes, digits=3, zero_division=0)
    metrics = dict(evaluation="leave-one-subject-out", best_variant=best, variants=results,
                   accuracy_by_take=by_take, excluded_clips=[r["clip"] for r in bad],
                   n_clips=len(rows), subjects=subjects, classes=classes)
    with open(os.path.join(RES_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RES_DIR, "classification_report.txt"), "w") as f:
        f.write(f"variant={best} (LOSO)\n\n" + report)
    print(report)

    # 최종 모델: 가장 좋은 설정으로 5명 전체 데이터 학습
    v = variants[best]
    X = np.stack([r["x"] for r in rows]); y = [r["action"] for r in rows]
    if v["aug"]:
        ex = [r for r in rows if r["xf"] is not None]
        X = np.vstack([X, np.stack([r["xf"] for r in ex])]); y += [r["action"] for r in ex]
    cols = v["cols"] if v["cols"] is not None else list(range(len(names)))
    final = fit(X[:, cols], y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final, os.path.join(MODEL_DIR, "action17_rf.joblib"), compress=3)
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump([names[i] for i in cols], f, indent=0)
    import sklearn
    with open(os.path.join(MODEL_DIR, "training_info.json"), "w") as f:
        json.dump(dict(variant=best, classes=list(final.classes_), n_features=len(cols),
                       sequence_steps=af.SEQUENCE_STEPS, min_valid_frames=af.MIN_VALID_FRAMES,
                       min_core_conf=af.MIN_CORE_CONF, target_fps=30, pose_weights="yolo11n-pose.pt",
                       person_selection="largest bbox per frame", feature_module="ai_service/services/action_features.py",
                       feature_columns_subset=v["cols"] is not None, rf_params=RF,
                       train_subjects=subjects, train_clips=len(rows), flip_augmentation=v["aug"],
                       loso_accuracy=results[best]["accuracy"], loso_macro_f1=results[best]["macro_f1"],
                       sklearn_version=sklearn.__version__), f, ensure_ascii=False, indent=2)
    print(f"saved. best={best} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
