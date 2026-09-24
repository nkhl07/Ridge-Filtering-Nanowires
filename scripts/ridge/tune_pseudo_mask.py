"""
Task 0.2b -- Tune a sato/frangi pseudo-mask against the 2 GT frames at the
50k-equivalent scale, and report recall / precision / coverage ratio.

Why ridge filters and not a threshold: the GT shows real wires are DARK bodies
with bright rims (75% of GT wire pixels sit below local background). The old
bright-only threshold scored 11% recall / 20% precision. sato/frangi with
black_ridges=True detect dark ridge structures directly.

The winning (filter, sigma, threshold) is written to pseudo_mask_params.json
and reused by the overlay prep and the all-frame pseudo-mask pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import sato, frangi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_labels import gt_frames, parse_labeled, load_raw

OUT = Path(__file__).resolve().parents[2] / "results" / "real_stats"
OUT.mkdir(parents=True, exist_ok=True)


def to_50k(img: np.ndarray, s: float, nearest: bool = False) -> np.ndarray:
    """Rescale an 80k-native array to 50k-equivalent pixel scale (s = 0.625)."""
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=interp)


def ridge_response(img01: np.ndarray, kind: str, sigmas) -> np.ndarray:
    fn = sato if kind == "sato" else frangi
    r = fn(img01, sigmas=sigmas, black_ridges=True)
    return (r - r.min()) / (r.max() - r.min() + 1e-12)


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = float((pred & gt).sum())
    rec = tp / max(gt.sum(), 1)
    prec = tp / max(pred.sum(), 1)
    f1 = 2 * rec * prec / max(rec + prec, 1e-9)
    return {"recall": rec, "precision": prec, "f1": f1,
            "pred_cov": float(pred.mean()), "gt_cov": float(gt.mean()),
            "coverage_ratio": float(gt.mean() / max(pred.mean(), 1e-9))}


def main() -> None:
    frames = gt_frames()
    data = []
    for fr in frames:
        s = fr["scale_to_50k"]
        raw = load_raw(fr["raw"])
        gt = parse_labeled(fr["labeled"])["union"]
        raw50 = to_50k(raw, s)
        gt50 = to_50k(gt.astype(np.uint8), s, nearest=True).astype(bool)
        img01 = raw50 / 255.0
        data.append((fr["name"], img01, gt50))
        print(f"{fr['name']}: 50k shape {img01.shape}  GT coverage {gt50.mean()*100:.2f}%")

    sigma_sets = {"s1-3": (1, 2, 3), "s1-4": (1, 2, 3, 4), "s2-5": (2, 3, 4, 5)}
    results = []
    print(f"\n{'filter':7s} {'sigmas':7s} {'thr':>5s} {'recall':>7s} {'prec':>7s} "
          f"{'F1':>6s} {'predCov':>8s} {'covRatio':>8s}")
    for kind in ("sato", "frangi"):
        for sname, sig in sigma_sets.items():
            resp = [ridge_response(im, kind, sig) for _, im, _ in data]
            for thr in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
                sc = []
                for r, (_, _, gt) in zip(resp, data):
                    sc.append(score(r > thr, gt))
                m = {k: float(np.mean([x[k] for x in sc])) for k in sc[0]}
                results.append({"filter": kind, "sigmas": sname, "thr": thr, **m})
                print(f"{kind:7s} {sname:7s} {thr:5.2f} {m['recall']:7.3f} "
                      f"{m['precision']:7.3f} {m['f1']:6.3f} {m['pred_cov']*100:7.2f}% "
                      f"{m['coverage_ratio']:8.2f}")

    best = max(results, key=lambda r: r["f1"])
    print(f"\nBEST by F1: {best['filter']} sigmas={best['sigmas']} thr={best['thr']} "
          f"-> recall {best['recall']:.3f} precision {best['precision']:.3f} "
          f"F1 {best['f1']:.3f} coverage_ratio {best['coverage_ratio']:.2f}")
    with open(OUT / "pseudo_mask_params.json", "w") as fh:
        json.dump({"best": best, "sigma_sets": {k: list(v) for k, v in sigma_sets.items()},
                   "all": results,
                   "note": "black_ridges=True; wires are dark bodies. coverage_ratio = "
                           "GT_coverage / pseudo_coverage (the correction factor, valid "
                           "ONLY for the 80k GT population)."}, fh, indent=2)
    print(f"wrote {OUT/'pseudo_mask_params.json'}")


if __name__ == "__main__":
    main()
