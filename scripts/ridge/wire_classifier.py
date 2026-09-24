"""Texture+ridge random-forest wire classifier (pipeline v3, 2026-07-24).

Why a classifier: the classical single-response pipeline hit its ceiling --
an accuracy ladder over hysteresis, percentile normalisation, background
flattening, blob suppression and shape filtering was neutral-to-worse than
v2 (sato thr 0.14 + dirclose) at matched coverage. A 17-feature RF beats v2
on every metric under CROSS-FRAME validation (train GT frame 1 -> test GT
frame 2 and vice versa, coverage-matched):

    v2 classical : recall 0.570  prec 0.505  F1 0.534  jRecall 0.672
    RF @ p>0.70  : recall 0.617  prec 0.550  F1 0.571  jRecall 0.729
    RF+close@0.70: recall 0.780  prec 0.525  F1 0.622  jRecall 0.909  (edit variant)

Feature importances: local std (sigma 2/4) carry ~44% -- wires are SMOOTH
dark bodies against mottled background, a texture signal no ridge filter
sees. sato contributes only ~4%.

Training data: the 2 hand-labeled 80k GT frames at 50k-equivalent scale
(x0.625). In-band transfer verified qualitatively at 50k/65k native; the
out-of-band caveats of the ridge pipeline stay (rim tracing on upscaled
bright-wire frames, over-fire risk far below 0.625).

The model is deterministic (fixed seed, fixed GT) and small; it is trained
on first use (~2 min) and cached in results/wire_classifier/.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import sato, frangi, meijering
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results" / "wire_classifier"
MODEL_PATH = CACHE_DIR / "rf_v3.joblib"

PROD_THR = 0.70         # coverage-matched operating point (ratio 0.93 vs GT)
N_TRAIN = 300_000       # balanced subsample per GT frame
SEED = 42

FEATURE_NAMES = ["I_s1", "I_s2", "I_s4", "std2", "std4", "grad1", "grad2",
                 "sato1", "sato2", "sato3", "sato123", "frangi", "meijering",
                 "H1.5_e1", "H1.5_e2", "H3_e1", "H3_e2"]


def feature_stack(img01: np.ndarray) -> np.ndarray:
    """17 per-pixel feature maps; img01 is min-max normalised to [0,1]."""
    F = []
    for s in (1, 2, 4):
        F.append(ndi.gaussian_filter(img01, s))
    for s in (2, 4):
        m = ndi.gaussian_filter(img01, s)
        m2 = ndi.gaussian_filter(img01 ** 2, s)
        F.append(np.sqrt(np.clip(m2 - m * m, 0, None)))
    for s in (1, 2):
        F.append(ndi.gaussian_gradient_magnitude(img01, s))
    for s in (1, 2, 3):
        F.append(sato(img01, sigmas=(s,), black_ridges=True))
    F.append(sato(img01, sigmas=(1, 2, 3), black_ridges=True))
    F.append(frangi(img01, sigmas=(1, 2, 3), black_ridges=True))
    F.append(meijering(img01, sigmas=(1, 2, 3), black_ridges=True))
    for s in (1.5, 3.0):
        H = hessian_matrix(img01, sigma=s, order="rc", use_gaussian_derivatives=False)
        e1, e2 = hessian_matrix_eigvals(H)
        F.extend([e1 * s ** 2, e2 * s ** 2])
    return np.stack([f.astype(np.float32) for f in F], axis=-1)


def _train() -> RandomForestClassifier:
    import cv2
    from gt_labels import gt_frames, parse_labeled, load_raw

    rng = np.random.default_rng(SEED)
    X, Y = [], []
    for fr in gt_frames():
        s = fr["scale_to_50k"]
        raw = load_raw(fr["raw"])
        h, w = raw.shape
        img01 = cv2.resize(raw, (int(round(w * s)), int(round(h * s))),
                           interpolation=cv2.INTER_AREA) / 255.0
        gtn = parse_labeled(fr["labeled"])["union"].astype(np.uint8)
        gt = cv2.resize(gtn, (int(round(w * s)), int(round(h * s))),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        feats = feature_stack(img01).reshape(-1, len(FEATURE_NAMES))
        y = gt.ravel()
        pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        k = N_TRAIN // 2
        idx = np.concatenate([rng.choice(pos, min(k, len(pos)), replace=False),
                              rng.choice(neg, min(k, len(neg)), replace=False)])
        X.append(feats[idx])
        Y.append(y[idx])

    clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=25,
                                 random_state=0, n_jobs=16)
    clf.fit(np.concatenate(X), np.concatenate(Y))
    return clf


def load_classifier(retrain: bool = False) -> RandomForestClassifier:
    """Cached deterministic model; trains from the GT frames if absent."""
    if MODEL_PATH.exists() and not retrain:
        return joblib.load(MODEL_PATH)
    print("training wire classifier from GT frames (one-off, ~2 min) ...",
          flush=True)
    clf = _train()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH, compress=3)
    (CACHE_DIR / "rf_v3_meta.json").write_text(json.dumps({
        "pipeline": "v3 texture+ridge RF",
        "trained": str(date.today()),
        "features": FEATURE_NAMES,
        "params": {"n_estimators": 150, "min_samples_leaf": 25,
                   "random_state": 0, "n_train_per_frame": N_TRAIN,
                   "sample_seed": SEED},
        "training_data": "2x TiS3_80000x GT frames at 50k-equivalent (x0.625)",
        "prod_thr": PROD_THR,
        "gt_cross_frame_cov_matched": {"recall": 0.617, "precision": 0.550,
                                       "f1": 0.571, "coverage_ratio": 0.93,
                                       "junction_recall": 0.729},
    }, indent=2))
    print(f"cached -> {MODEL_PATH.relative_to(ROOT)}")
    return clf


def wire_probability(clf: RandomForestClassifier, img01: np.ndarray,
                     chunk: int = 4_000_000) -> np.ndarray:
    """Per-pixel wire probability, predicted in chunks to bound memory."""
    feats = feature_stack(img01).reshape(-1, len(FEATURE_NAMES))
    prob = np.empty(feats.shape[0], dtype=np.float32)
    for i in range(0, feats.shape[0], chunk):
        prob[i:i + chunk] = clf.predict_proba(feats[i:i + chunk])[:, 1]
    return prob.reshape(img01.shape)
